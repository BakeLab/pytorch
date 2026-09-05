#pragma once
// Split out of CUDAContextLight.h; see CUDABlasHandle.h. cublasLt.h pulls in
// cublas_api.h but not the cublas_v2.h renaming layer, so this is not a
// superset of CUDABlasHandle.h.

#include <cublasLt.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <torch/headeronly/macros/Export.h>

namespace at::cuda {

TORCH_CUDA_CPP_API cublasLtHandle_t getCurrentCUDABlasLtHandle();

#ifdef USE_ROCM
// Like getCurrentCUDABlasLtHandle(), but returns nullptr instead of creating a
// handle when the current thread has none for the current (device, stream) and
// the pool has no spare. Safe to call under stream capture.
TORCH_CUDA_CPP_API cublasLtHandle_t getCurrentCUDABlasLtHandleIfAvailable();
// hipblaslt handles are pooled per (device, stream) and cannot be created
// while stream capture is active. Pre-stock the shared free list so other
// threads (e.g. autograd workers) can reserve one mid-capture.
TORCH_CUDA_CPP_API void ensureCublasLtHandlesAvailable(size_t n);
// Under capture, getCurrentCUDABlasHandle() returns a capture handle: one per
// (thread, device), with no rocBLAS arena, shared by every stream the thread
// uses in the capture. Each request points it at the current stream and binds
// that stream's workspace from the capture's memory pool. Once the public
// handle has been requested on the current device, reserve this thread's
// capture handle and pre-stock one spare for another thread, because creating
// a handle is illegal under capture.
TORCH_CUDA_CPP_API void prepareCaptureCublasHandles();
// Unbinds the capture handles used by this capture and frees their workspaces,
// and the hipBLASLt workspaces getCUDABlasLtWorkspace handed out during it,
// back to the capture's memory pool.
TORCH_CUDA_CPP_API void releaseCaptureCublasWorkspaces(
    c10::CaptureId_t capture_id);
#endif

} // namespace at::cuda
