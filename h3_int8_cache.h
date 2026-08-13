#ifndef H3_INT8_CACHE_H
#define H3_INT8_CACHE_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

/* Persistent row-wise int8 weight cache. Cache entries are validated against
 * the source shard metadata and tensor schema before any GPU allocation. */
int h3_int8_cache_load(h3_gpu *gpu, const char *directory, const char *key,
                       const char *source_path, uint64_t source_offset,
                       uint64_t rows, uint64_t columns,
                       h3_gpu_tensor **weight, h3_gpu_tensor **scales,
                       int *hit, uint64_t *loaded_bytes,
                       char *error, size_t error_size);

int h3_int8_cache_save(const char *directory, const char *key,
                       const char *source_path, uint64_t source_offset,
                       uint64_t rows, uint64_t columns,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *scales,
                       uint64_t *stored_bytes,
                       char *error, size_t error_size);

#endif
