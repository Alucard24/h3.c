/* Host stub for h3_metal_probe(). Compiled on platforms without Metal
 * (Linux); the real implementation lives in h3_metal.m (macOS only). */
#include "h3_metal.h"

#include <stdio.h>
#include <string.h>

int h3_metal_probe(h3_device_info *info, char *error, size_t error_size) {
    if (info) memset(info, 0, sizeof(*info));
    if (error && error_size) {
        snprintf(error, error_size,
                 "no Metal device: this build has no Apple GPU backend "
                 "(Vulkan/CUDA backend scaffolding pending)");
    }
    return 0;
}
