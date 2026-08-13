#ifndef H3_SCHEDULE_CACHE_H
#define H3_SCHEDULE_CACHE_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

int h3_schedule_cache_load(
    h3_gpu *gpu, const char *directory, const char *key,
    const char *weight_path, uint64_t weight_offset,
    const char *bias_path, uint64_t bias_offset, uint64_t time_hash,
    uint64_t rows, uint64_t columns, h3_gpu_tensor **output, int *hit,
    uint64_t *loaded_bytes, char *error, size_t error_size);

int h3_schedule_cache_save(
    const char *directory, const char *key,
    const char *weight_path, uint64_t weight_offset,
    const char *bias_path, uint64_t bias_offset, uint64_t time_hash,
    uint64_t rows, uint64_t columns, const h3_gpu_tensor *output,
    uint64_t *stored_bytes, char *error, size_t error_size);

#endif
