#pragma once

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define __builtin_clear_padding(...) ((void)0)
#endif

#include <bits/c++config.h>

#if !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE != 16
#error "CUDA builds require libstdc++ 16"
#endif

#if defined(__CUDACC__) && __cplusplus >= 202302L
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wbuiltin-macro-redefined"
#endif
#pragma push_macro("__cplusplus")
#undef __cplusplus
#define __cplusplus 202100L
#include <string>
#pragma pop_macro("__cplusplus")
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
