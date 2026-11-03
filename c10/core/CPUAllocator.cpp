#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/util/Logging.h>

// TODO: rename flag to C10
C10_DEFINE_bool(
    caffe2_report_cpu_memory_usage,
    false,
    "If set, print out detailed memory usage")

namespace c10 {

struct C10_API DefaultCPUAllocator final : at::Allocator {
  DefaultCPUAllocator() = default;
  at::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    try {
      data = c10::alloc_cpu(nbytes);
    } catch (c10::Error&) {
      profiledCPUMemoryReporter().OutOfMemory(nbytes);
      throw;
    }
    profiledCPUMemoryReporter().New(data, nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    profiledCPUMemoryReporter().Delete(ptr);
    free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }
};

ProfiledCPUMemoryReporter& profiledCPUMemoryReporter() {
  static ProfiledCPUMemoryReporter reporter_;
  return reporter_;
}

at::Allocator* GetCPUAllocator() {
  return GetAllocator(DeviceType::CPU);
}

void SetCPUAllocator(at::Allocator* alloc, uint8_t priority) {
  SetAllocator(DeviceType::CPU, alloc, priority);
}

// Global default CPU Allocator
static DefaultCPUAllocator g_cpu_alloc;

at::Allocator* GetDefaultCPUAllocator() {
  return &g_cpu_alloc;
}

REGISTER_ALLOCATOR(DeviceType::CPU, &g_cpu_alloc)

void ProfiledCPUMemoryReporter::New(void* ptr, size_t nbytes) {
  if (nbytes == 0) {
    return;
  }
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
    std::lock_guard<std::mutex> guard(mutex_);
    size_table_[ptr] = nbytes;
    allocated_ += nbytes;
    allocated = allocated_;
  }
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    LOG(INFO) << "C10 alloc " << nbytes << " bytes, total alloc " << allocated
              << " bytes.";
  }
  if (profile_memory) {
    reportMemoryUsageToProfiler(
        ptr,
        static_cast<int64_t>(nbytes),
        allocated,
        0,
        c10::Device(c10::DeviceType::CPU));
  }
}

void ProfiledCPUMemoryReporter::Delete(void* ptr) {
  size_t nbytes = 0;
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = size_table_.find(ptr);
    if (it != size_table_.end()) {
      allocated_ -= it->second;
      allocated = allocated_;
      nbytes = it->second;
      size_table_.erase(it);
    } else {
      // C10_LOG_EVERY_MS might log every time in some builds,
      // using a simple counter to avoid spammy logs
      if (log_cnt_++ % 1000 == 0) {
        LOG(WARNING) << "Memory block of unknown size was allocated before "
                     << "the profiling started, profiler results will not "
                     << "include the deallocation event";
      }
    }
  }
  if (nbytes == 0) {
    return;
  }
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    LOG(INFO) << "C10 deleted " << nbytes << " bytes, total alloc " << allocated
              << " bytes.";
  }
  if (profile_memory) {
    reportMemoryUsageToProfiler(
        ptr,
        -static_cast<int64_t>(nbytes),
        allocated,
        0,
        c10::Device(c10::DeviceType::CPU));
  }
}

void ProfiledCPUMemoryReporter::OutOfMemory(size_t nbytes) {
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
    std::lock_guard<std::mutex> guard(mutex_);

    allocated = allocated_;
  }
  if (nbytes == 0) {
    return;
  }
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    LOG(INFO) << "C10 Out of Memory. Trying to allocate " << nbytes
              << " bytes, total alloc " << allocated << " bytes.";
  }
  if (profile_memory) {
    reportOutOfMemoryToProfiler(
        static_cast<int64_t>(nbytes),
        allocated,
        0,
        c10::Device(c10::DeviceType::CPU));
  }
}

} // namespace c10
