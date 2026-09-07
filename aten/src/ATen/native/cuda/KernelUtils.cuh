#pragma once
#include <ATen/core/TensorAccessor.h>
#include <torch/headeronly/cuda/Atomic.h>
#include <torch/headeronly/cuda/KernelUtils.h>

#if !defined(USE_ROCM)
#include <cuda_bf16.h>
#endif

#if defined(USE_ROCM)
#include <device_functions.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <torch/headeronly/cuda/detail/ROCmMacros.h>
#endif

namespace at::native {

using torch::headeronly::fastAtomicAdd;
using torch::headeronly::fastSpecializedAtomicAdd;

// Scatter into a packed accessor without naming a bound.
//
// fastAtomicAdd needs to know how far it may look for a pairing partner -- see
// Note [Passing pointer and offset to fastAtomicAdd] in GridSampler.cu. A
// caller holding a raw pointer supplies that by hand, and too large a value
// lets the pairing write past the allocation. An accessor already carries what
// it would be derived from:
//
//   memory_span = 1 + sum over d of (size[d] - 1) * stride[d]
//
// the distance to the last addressable element rather than the element count,
// which is what at::native::storage_size_for computes on the host. The two
// differ for a strided view, whose gaps are legal pairing partners: inside the
// region, though belonging to whatever the view was taken from, and only ever
// receiving a zero.
template <
    typename scalar_t,
    size_t N,
    template <typename U> class PtrTraits,
    typename index_t,
    typename... Idx>
__device__ __forceinline__ void fastAtomicAdd(
    at::GenericPackedTensorAccessor<scalar_t, N, PtrTraits, index_t> accessor,
    scalar_t value,
    Idx... indices) {
  static_assert(
      sizeof...(Idx) == N, "fastAtomicAdd needs one index per dimension");
  const index_t index[N] = {static_cast<index_t>(indices)...};
  index_t offset = 0;
  index_t memory_span = 1;
#pragma unroll
  for (size_t d = 0; d < N; ++d) {
    offset += index[d] * accessor.stride(d);
    memory_span += (accessor.size(d) - 1) * accessor.stride(d);
  }
  fastAtomicAdd(accessor.data(), offset, memory_span, value, true);
}

__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t h,
    const size_t w) {
  return (nc * height + h) * width + w;
}

// for channels-last
template <typename index_t = size_t>
__device__ __forceinline__ index_t idx_cl(
    const index_t n,
    const index_t h,
    const index_t w,
    const index_t c,
    const index_t height,
    const index_t width,
    const index_t channel) {
  return ((n * height + h) * width + w) * channel + c;
}

#ifdef USE_ROCM
// This function implements a committed store.
// Upon returning, the store is committed to global memory.
// This is useful in avoiding the need for fences.
// If multiple stores are done in a row there is option to skip
// waiting for commit for all but the last store.
template <typename T, bool wait_for_commit = true>
__device__ inline void cmtdStore(void* address, T value) {
  int constexpr num_long_per_val = sizeof(value) / sizeof(long);
  int constexpr num_int_per_val = sizeof(value) / sizeof(int);
  int constexpr num_short_per_val = sizeof(value) / sizeof(short);
  int constexpr num_char_per_val = sizeof(value) / sizeof(char);
  union pnr {
    T v;
    long l[num_long_per_val];
    int i[num_int_per_val];
    short s[num_short_per_val];
    char c[num_char_per_val];
  } _pnr = {.v = value};
  if constexpr (num_long_per_val * sizeof(long) == sizeof(value))
    for (int i = 0; i < num_long_per_val; i++)
      __hip_atomic_store(
          reinterpret_cast<long*>(address) + i,
          _pnr.l[i],
          __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
  else if constexpr (num_int_per_val * sizeof(int) == sizeof(value))
    for (int i = 0; i < num_int_per_val; i++)
      __hip_atomic_store(
          reinterpret_cast<int*>(address) + i,
          _pnr.i[i],
          __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
  else if constexpr (num_short_per_val * sizeof(short) == sizeof(value))
    for (int i = 0; i < num_short_per_val; i++)
      __hip_atomic_store(
          reinterpret_cast<short*>(address) + i,
          _pnr.s[i],
          __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
  else if constexpr (num_char_per_val * sizeof(char) == sizeof(value))
    for (int i = 0; i < num_char_per_val; i++)
      __hip_atomic_store(
          reinterpret_cast<char*>(address) + i,
          _pnr.c[i],
          __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
  if constexpr (wait_for_commit) {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__GFX12__)
    asm volatile("s_wait_storecnt(0)" ::: "memory");
#elif defined(__GFX10__) || defined(__GFX11__)
    asm volatile("s_waitcnt_vscnt null, 0" ::: "memory");
#else
    // Older architectures have only 'vmcnt' counter.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
  }
}

#endif

} // namespace at::native
