/* Shared helpers for the integration tests (backend-agnostic).
 *
 * The Metal runtime is the macOS default; on Linux the Vulkan backend is
 * selected by pointing H3_SHADER_SOURCE at h3_vulkan_shaders.comp. */
#ifndef H3_TEST_UTIL_H
#define H3_TEST_UTIL_H

#include <stdlib.h>

static inline const char *h3_test_shader(void) {
    const char *path = getenv("H3_SHADER_SOURCE");
    return path ? path : "h3_shaders.metal";
}

#endif
