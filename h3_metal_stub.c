/* Host h3_metal_probe(). Compiled on platforms without Metal (Linux). With a
 * Vulkan backend in the build it reports the selected Vulkan device; the
 * plain scaffold build fails cleanly. */
#include "h3_metal.h"

#include <stdio.h>
#include <string.h>

#if defined(H3_HAVE_VULKAN)
int h3_vulkan_probe(h3_device_info *info, char *error, size_t error_size);
#endif

int h3_metal_probe(h3_device_info *info, char *error, size_t error_size) {
#if defined(H3_HAVE_VULKAN)
    return h3_vulkan_probe(info, error, error_size);
#else
    if (info) memset(info, 0, sizeof(*info));
    if (error && error_size) {
        snprintf(error, error_size,
                 "no Metal device: this build has no Apple GPU backend "
                 "(Vulkan/CUDA backend scaffolding pending)");
    }
    return 0;
#endif
}
