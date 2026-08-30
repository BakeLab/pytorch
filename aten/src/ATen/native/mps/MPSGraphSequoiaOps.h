#pragma once

#include <c10/macros/Macros.h>

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#if !defined(__MAC_26_0) &&             \
    (!defined(MAC_OS_X_VERSION_26_0) || \
     (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_26_0))
#define MTLLanguageVersion4_0 ((MTLLanguageVersion)((4 << 16) + 0))
#endif // Building for target older than MacOS-26
