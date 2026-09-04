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
    class T,
    std::enable_if_t<std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::string cuGDSFileGetErrorString(T status) {
  status = std::abs(status);
  return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status))
                               : std::string(c10::utils::str_error(errno));
}

// To get error message for Buf/Handle registration APIs that return
// CUfileError_t
template <
    class T,
    std::enable_if_t<!std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::string cuGDSFileGetErrorString(T status) {
  std::string errStr = cuGDSFileGetErrorString(static_cast<int>(status.err));
  if (IS_CUDA_ERR(status))
    errStr.append(".").append(
        cudaGetErrorString(static_cast<cudaError_t>(status.cu_err)));
  return errStr;
}
} // namespace


// Resources are reused only after all submitted I/O has completed.
struct GdsResources {
  struct AsyncArgs {
    size_t size;
    off_t offset;
    off_t buffer_offset;
    ssize_t bytes;
  };

  explicit GdsResources(c10::Device device, bool batch)
      : device(device), is_batch(batch) {}
  GdsResources(const GdsResources&) = delete;
  GdsResources& operator=(const GdsResources&) = delete;

  void initialize() {
    c10::cuda::CUDAGuard guard(device);
    if (is_batch) {
      params.resize(128);
      events.resize(128);
      auto status = cuFileBatchIOSetUp(&batch, params.size());
      TORCH_CHECK(status.err == CU_FILE_SUCCESS, "GDS batch setup failed: ",
                  cuGDSFileGetErrorString(status));
    } else {
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&args), sizeof(AsyncArgs), 0));
    }
  }

  ~GdsResources() {
    try {
      c10::cuda::CUDAGuard guard(device);
      if (batch) {
        cuFileBatchIODestroy(batch);
      }
      if (event) {
        C10_CUDA_CHECK_WARN(cudaEventDestroy(event));
      }
      if (args) {
        C10_CUDA_CHECK_WARN(cudaFreeHost(args));
      }
    } catch (const std::exception& e) {
      TORCH_WARN("GDS resource cleanup: ", e.what());
    }
  }

  c10::Device device;
  bool is_batch;
  CUfileBatchHandle_t batch = nullptr;
  std::vector<CUfileIOParams_t> params;
  std::vector<CUfileIOEvents_t> events;
  cudaEvent_t event = nullptr;
  AsyncArgs* args = nullptr;
};

// Shared by synchronous calls and pending I/O. The duplicate descriptor keeps
// the registration valid even after the Python GdsFile closes its own fd.
class GdsHandle {
 public:
  explicit GdsHandle(int fd) {
    fd_ = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    TORCH_CHECK(fd_ >= 0, "GDS dup failed: ", c10::utils::str_error(errno));
    CUfileDescr_t descriptor{};
    descriptor.handle.fd = fd_;
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    auto status = cuFileHandleRegister(&handle_, &descriptor);
    if (status.err != CU_FILE_SUCCESS) {
      ::close(fd_);
      TORCH_CHECK(false, "cuFileHandleRegister failed: ", cuGDSFileGetErrorString(status));
    }
  }

  GdsHandle(const GdsHandle&) = delete;
  GdsHandle& operator=(const GdsHandle&) = delete;

  ~GdsHandle() {
    // Cached contexts must be destroyed before deregistering the file.
    idle_.clear();
    cuFileHandleDeregister(handle_);
    ::close(fd_);
  }

  CUfileHandle_t get() const {
    return handle_;
  }

  std::unique_ptr<GdsResources> acquire(c10::Device device, bool batch) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = idle_.begin(); it != idle_.end(); ++it) {
        if ((*it)->device == device && (*it)->is_batch == batch) {
          auto resources = std::move(*it);
          idle_.erase(it);
          return resources;
        }
      }
    }
    auto resources = std::make_unique<GdsResources>(device, batch);
    resources->initialize();
    return resources;
  }

  void release(std::unique_ptr<GdsResources> resources) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Bound retained resources per file; concurrent requests can still exceed
    // this count, but excess resources are freed when those requests finish.
    if (idle_.size() < 8) {
      idle_.push_back(std::move(resources));
    }
  }

 private:
  int fd_ = -1;
  CUfileHandle_t handle_ = nullptr;
  std::vector<std::unique_ptr<GdsResources>> idle_;
  std::mutex mutex_;
};

static std::shared_ptr<GdsHandle> gds_register_handle(int fd) {
  return std::make_shared<GdsHandle>(fd);
}

static void gds_load_storage(
    const std::shared_ptr<GdsHandle>& handle,
    const at::Storage& storage,
    off_t offset) {
  TORCH_CHECK(handle, "GDS file handle is not registered");
  CUfileHandle_t cf_handle = handle->get();
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
    const std::shared_ptr<GdsHandle>& handle,
    const at::Storage& storage,
    off_t offset) {
  TORCH_CHECK(handle, "GDS file handle is not registered");
  CUfileHandle_t cf_handle = handle->get();
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

class GdsIO {
 public:
  GdsIO(std::shared_ptr<GdsHandle> handle, std::vector<at::Storage> storages, std::vector<int64_t> offsets)
      : handle_(std::move(handle)), storages_(std::move(storages)), offsets_(std::move(offsets)) {
    TORCH_CHECK(handle_, "GDS file handle is not registered");
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
  }

  void submit_async(bool write) {
    c10::cuda::CUDAGuard guard(device_);
    stream_ = c10::cuda::getCurrentCUDAStream(device_.index()).stream();
    resources_ = handle_->acquire(device_, false);
    auto* args = resources_->args;
    *args = {storages_[0].nbytes(), offsets_[0], 0, -1};
    auto handle = handle_->get();
    auto status = write
        ? cuFileWriteAsync(handle, storages_[0].mutable_data(), &args->size,
              &args->offset, &args->buffer_offset, &args->bytes, stream_)
        : cuFileReadAsync(handle, storages_[0].mutable_data(), &args->size,
              &args->offset, &args->buffer_offset, &args->bytes, stream_);
    TORCH_CHECK(status.err == CU_FILE_SUCCESS, "GDS async submit failed: ",
                cuGDSFileGetErrorString(status));
    submitted_ = true;
    // If recording fails, wait() falls back to synchronizing the stream.
    C10_CUDA_CHECK(cudaEventRecord(resources_->event, stream_));
    event_recorded_ = true;
  }

  void submit_batch(bool write) {
    TORCH_CHECK(storages_.size() <= 128, "GDS batches support at most 128 operations");
    c10::cuda::CUDAGuard guard(device_);
    // Batch I/O is not stream ordered. Complete preceding producer work first.
    c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
    resources_ = handle_->acquire(device_, true);
    auto& params = resources_->params;
    for (size_t i = 0; i < storages_.size(); ++i) {
      auto& param = params[i];
      param.mode = CUFILE_BATCH;
      param.u.batch.devPtr_base = storages_[i].mutable_data();
      param.u.batch.file_offset = offsets_[i];
      param.u.batch.devPtr_offset = 0;
      param.u.batch.size = storages_[i].nbytes();
      param.fh = handle_->get();
      param.opcode = write ? CUFILE_WRITE : CUFILE_READ;
      param.cookie = &param;
    }
    auto status = cuFileBatchIOSubmit(resources_->batch, storages_.size(), params.data(), 0);
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
      if (resources_->is_batch) {
        while (completed_ < storages_.size()) {
          unsigned count = resources_->events.size();
          auto status = cuFileBatchIOGetStatus(resources_->batch, 1, &count, resources_->events.data(), nullptr);
          TORCH_CHECK(status.err == CU_FILE_SUCCESS, "GDS batch wait failed: ",
                      cuGDSFileGetErrorString(status));
          for (unsigned i = 0; i < count; ++i) {
            const auto& event = resources_->events[i];
            auto* param = static_cast<CUfileIOParams_t*>(event.cookie);
            if (event.status != CUFILE_COMPLETE || event.ret != param->u.batch.size) {
              error_ = "GDS batch operation failed or transferred fewer bytes than requested";
            }
          }
          completed_ += count;
        }
      } else {
        if (event_recorded_) {
          C10_CUDA_CHECK(cudaEventSynchronize(resources_->event));
        } else {
          C10_CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        if (resources_->args->bytes != static_cast<ssize_t>(storages_[0].nbytes())) {
          error_ = resources_->args->bytes < 0
              ? "GDS async I/O failed: " + cuGDSFileGetErrorString(resources_->args->bytes)
              : "GDS async I/O transferred fewer bytes than requested";
        }
      }
      done_ = true;
      // Keep the result (including any error) on this request, but return only
      // successfully completed resources to the pool. Old results can remain
      // alive and be waited on again without owning the next operation's state.
      storages_.clear();
      if (error_.empty()) {
        handle_->release(std::move(resources_));
      } else {
        resources_.reset();
      }
      handle_.reset();
    }
    TORCH_CHECK(error_.empty(), error_);
  }

 private:
  std::shared_ptr<GdsHandle> handle_;
  std::unique_ptr<GdsResources> resources_;
  std::vector<at::Storage> storages_;
  std::vector<int64_t> offsets_;
  c10::Device device_{c10::DeviceType::CUDA, 0};
  size_t completed_ = 0;
  cudaStream_t stream_ = nullptr;
  bool event_recorded_ = false;
  bool submitted_ = false;
  bool done_ = false;
  std::string error_;
  std::mutex mutex_;
};

static std::shared_ptr<GdsIO> gds_submit(
    std::shared_ptr<GdsHandle> handle, std::vector<at::Storage> storages,
    std::vector<int64_t> offsets, bool write, bool batch) {
  TORCH_CHECK(batch || storages.size() == 1, "Async GDS requires one storage");
  auto request = std::make_shared<GdsIO>(std::move(handle), std::move(storages), std::move(offsets));
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
  py::class_<GdsHandle, std::shared_ptr<GdsHandle>>(m, "_GdsHandle");
  py::class_<GdsIO, std::shared_ptr<GdsIO>>(m, "_GdsIO")
      .def("wait", &GdsIO::wait, py::call_guard<py::gil_scoped_release>());
  m.def("_gds_submit", &gds_submit, py::call_guard<py::gil_scoped_release>());
  m.def("_gds_register_handle", &gds_register_handle);
  m.def("_gds_register_buffer", &gds_register_buffer);
  m.def("_gds_deregister_buffer", &gds_deregister_buffer);
  m.def("_gds_load_storage", &gds_load_storage);
  m.def("_gds_save_storage", &gds_save_storage);
#endif
}

} // namespace torch::cuda::shared
