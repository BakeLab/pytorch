#include <c10/util/error.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/cuda/GdsFile.h>
#include <torch/csrc/utils/pybind.h>

#if defined(USE_CUFILE)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits>
#include <mutex>

#include <cufile.h>

namespace {
// To get error message for cuFileRead/Write APIs that return ssize_t (-1 for
// filesystem error and a negative CUfileOpError enum value otherwise).
template <
    class T>
std::string cuGDSFileGetErrorString(T status) requires (std::is_integral_v<T>) {
  status = std::abs(status);
  return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status))
                               : std::string(c10::utils::str_error(errno));
}

// To get error message for Buf/Handle registration APIs that return
// CUfileError_t
template <
    class T>
std::string cuGDSFileGetErrorString(T status) requires (!std::is_integral_v<T>) {
  std::string errStr = cuGDSFileGetErrorString(static_cast<int>(status.err));
  if (IS_CUDA_ERR(status))
    errStr.append(".").append(
        cudaGetErrorString(static_cast<cudaError_t>(status.cu_err)));
  return errStr;
}
} // namespace

static void gds_load_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Read the binary file
  ssize_t ret = cuFileRead(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileRead failed: ", cuGDSFileGetErrorString(ret));
  TORCH_CHECK(
      ret == static_cast<ssize_t>(nbytes),
      "cuFileRead handled only ",
      ret,
      " of ",
      nbytes,
      " bytes");
}

static void gds_save_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Write device memory contents to the file
  ssize_t ret = cuFileWrite(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuGDSFileGetErrorString(ret));
  TORCH_CHECK(
      ret == static_cast<ssize_t>(nbytes),
      "cuFileWrite handled only ",
      ret,
      " of ",
      nbytes,
      " bytes");
}

static void gds_register_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  CUfileError_t status = cuFileBufRegister(dataPtr, nbytes, 0);
  TORCH_CHECK(
      status.err == CU_FILE_SUCCESS,
      "cuFileBufRegister failed: ",
      cuGDSFileGetErrorString(status));
  return;
}

static void gds_deregister_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  CUfileError_t status = cuFileBufDeregister(dataPtr);
  TORCH_CHECK(
      status.err == CU_FILE_SUCCESS,
      "cuFileBufDeregister failed: ",
      cuGDSFileGetErrorString(status));
  return;
}

static int64_t gds_register_handle(int fd) {
  CUfileDescr_t cf_descr;
  CUfileHandle_t cf_handle{};
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    TORCH_CHECK(
        false,
        "cuFileHandleRegister failed: ",
        cuGDSFileGetErrorString(status));
  }

  // Returning cuFileHandle_t as int64_t
  return reinterpret_cast<int64_t>(cf_handle);
}

static void gds_deregister_handle(int64_t handle) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  cuFileHandleDeregister(cf_handle);
}


// Each operation owns a duplicated fd, registered handle and its buffers. Closing
// the Python GdsFile, or dropping the tensors, cannot invalidate pending I/O.
class GdsIO {
 public:
  GdsIO(int fd, std::vector<at::Storage> storages, std::vector<int64_t> offsets)
      : storages_(std::move(storages)), offsets_(std::move(offsets)) {
    TORCH_CHECK(!storages_.empty(), "GDS requires at least one storage");
    TORCH_CHECK(storages_.size() == offsets_.size(), "GDS storage/offset count mismatch");
    TORCH_CHECK(storages_[0].device().is_cuda(), "GDS requires CUDA/HIP storage");
    device_ = storages_[0].device();
    for (size_t i = 0; i < storages_.size(); ++i) {
      TORCH_CHECK(storages_[i].device() == device_, "GDS buffers must be on the same device");
      TORCH_CHECK(offsets_[i] >= 0, "GDS file offsets must be nonnegative");
      TORCH_CHECK(
          storages_[i].nbytes() > 0 &&
              storages_[i].nbytes() <= static_cast<size_t>(std::numeric_limits<ssize_t>::max()),
          "GDS requires a nonempty storage of at most SSIZE_MAX bytes");
      TORCH_CHECK(
          offsets_[i] <= std::numeric_limits<int64_t>::max() -
              static_cast<int64_t>(storages_[i].nbytes()),
          "GDS file range overflows");
    }
    c10::cuda::CUDAGuard guard(device_);
    fd_ = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    TORCH_CHECK(fd_ >= 0, "GDS dup failed: ", c10::utils::str_error(errno));
    try {
      handle_ = gds_register_handle(fd_);
    } catch (...) {
      ::close(fd_);
      fd_ = -1;
      throw;
    }
  }

  GdsIO(const GdsIO&) = delete;
  GdsIO& operator=(const GdsIO&) = delete;

  ~GdsIO() {
    // Dropping the result is safe but blocking. Explicit wait() reports errors.
    try {
      if (!done_) {
        wait();
      }
    } catch (const std::exception& e) {
      TORCH_WARN("GDS operation cleanup: ", e.what());
    }
    c10::cuda::CUDAGuard guard(device_);
    if (batch_) {
      cuFileBatchIODestroy(batch_);
    }
    if (event_) {
      C10_CUDA_CHECK_WARN(cudaEventDestroy(event_));
    }
    if (args_) {
      C10_CUDA_CHECK_WARN(cudaFreeHost(args_));
    }
    gds_deregister_handle(handle_);
    ::close(fd_);
  }

  void submit_async(bool write) {
    c10::cuda::CUDAGuard guard(device_);
    stream_ = c10::cuda::getCurrentCUDAStream(device_.index()).stream();
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&args_), sizeof(AsyncArgs), 0));
    *args_ = {storages_[0].nbytes(), offsets_[0], 0, -1};
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto handle = reinterpret_cast<CUfileHandle_t>(handle_);
    auto status = write
        ? cuFileWriteAsync(handle, storages_[0].mutable_data(), &args_->size,
              &args_->offset, &args_->buffer_offset, &args_->bytes, stream_)
        : cuFileReadAsync(handle, storages_[0].mutable_data(), &args_->size,
              &args_->offset, &args_->buffer_offset, &args_->bytes, stream_);
    TORCH_CHECK(status.err == CU_FILE_SUCCESS, "GDS async submit failed: ",
                cuGDSFileGetErrorString(status));
    submitted_ = true;
    // If recording fails, wait() falls back to synchronizing the stream.
    C10_CUDA_CHECK(cudaEventRecord(event_, stream_));
    event_recorded_ = true;
  }

  void submit_batch(bool write) {
    TORCH_CHECK(storages_.size() <= 128, "GDS batches support at most 128 operations");
    c10::cuda::CUDAGuard guard(device_);
    // Batch I/O is not stream ordered. Complete preceding producer work first.
    c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
    params_.resize(storages_.size());
    events_.resize(storages_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
      auto& param = params_[i];
      param.mode = CUFILE_BATCH;
      param.u.batch.devPtr_base = storages_[i].mutable_data();
      param.u.batch.file_offset = offsets_[i];
      param.u.batch.devPtr_offset = 0;
      param.u.batch.size = storages_[i].nbytes();
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      param.fh = reinterpret_cast<CUfileHandle_t>(handle_);
      param.opcode = write ? CUFILE_WRITE : CUFILE_READ;
      param.cookie = &param;
    }
    auto status = cuFileBatchIOSetUp(&batch_, params_.size());
    TORCH_CHECK(status.err == CU_FILE_SUCCESS, "GDS batch setup failed: ",
                cuGDSFileGetErrorString(status));
    status = cuFileBatchIOSubmit(batch_, params_.size(), params_.data(), 0);
    TORCH_CHECK(status.err == CU_FILE_SUCCESS, "GDS batch submit failed: ",
                cuGDSFileGetErrorString(status));
    submitted_ = true;
  }

  void wait() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!submitted_) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_);
    if (!done_) {
      if (batch_) {
        while (completed_ < params_.size()) {
          unsigned count = events_.size();
          auto status = cuFileBatchIOGetStatus(batch_, 1, &count, events_.data(), nullptr);
          TORCH_CHECK(status.err == CU_FILE_SUCCESS, "GDS batch wait failed: ",
                      cuGDSFileGetErrorString(status));
          for (unsigned i = 0; i < count; ++i) {
            const auto& event = events_[i];
            auto* param = static_cast<CUfileIOParams_t*>(event.cookie);
            if (event.status != CUFILE_COMPLETE || event.ret != param->u.batch.size) {
              error_ = "GDS batch operation failed or transferred fewer bytes than requested";
            }
          }
          completed_ += count;
        }
      } else {
        if (event_recorded_) {
          C10_CUDA_CHECK(cudaEventSynchronize(event_));
        } else {
          C10_CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        if (args_->bytes != static_cast<ssize_t>(storages_[0].nbytes())) {
          error_ = args_->bytes < 0
              ? "GDS async I/O failed: " + cuGDSFileGetErrorString(args_->bytes)
              : "GDS async I/O transferred fewer bytes than requested";
        }
      }
      done_ = true;
    }
    TORCH_CHECK(error_.empty(), error_);
  }

 private:
  struct AsyncArgs {
    size_t size;
    off_t offset;
    off_t buffer_offset;
    ssize_t bytes;
  };
  std::vector<at::Storage> storages_;
  std::vector<int64_t> offsets_;
  c10::Device device_{c10::DeviceType::CUDA, 0};
  int fd_ = -1;
  int64_t handle_ = 0;
  CUfileBatchHandle_t batch_ = nullptr;
  std::vector<CUfileIOParams_t> params_;
  std::vector<CUfileIOEvents_t> events_;
  size_t completed_ = 0;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
  AsyncArgs* args_ = nullptr;
  bool event_recorded_ = false;
  bool submitted_ = false;
  bool done_ = false;
  std::string error_;
  std::mutex mutex_;
};

static std::shared_ptr<GdsIO> gds_submit(
    int fd, std::vector<at::Storage> storages,
    std::vector<int64_t> offsets, bool write, bool batch) {
  TORCH_CHECK(batch || storages.size() == 1, "Async GDS requires one storage");
  auto request = std::make_shared<GdsIO>(fd, std::move(storages), std::move(offsets));
  if (batch) {
    request->submit_batch(write);
  } else {
    request->submit_async(write);
  }
  return request;
}

#endif

namespace torch::cuda::shared {

void initGdsBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#if defined(USE_CUFILE)
  py::class_<GdsIO, std::shared_ptr<GdsIO>>(m, "_GdsIO")
      .def("wait", &GdsIO::wait, py::call_guard<py::gil_scoped_release>());
  m.def("_gds_submit", &gds_submit, py::call_guard<py::gil_scoped_release>());
  m.def("_gds_register_handle", &gds_register_handle);
  m.def("_gds_deregister_handle", &gds_deregister_handle);
  m.def("_gds_register_buffer", &gds_register_buffer);
  m.def("_gds_deregister_buffer", &gds_deregister_buffer);
  m.def("_gds_load_storage", &gds_load_storage);
  m.def("_gds_save_storage", &gds_save_storage);
#endif
}

} // namespace torch::cuda::shared
