#ifndef H3_CUDA_ACCEL_H
#define H3_CUDA_ACCEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h3_cuda_accel h3_cuda_accel;

typedef struct {
    uint64_t matmul_launches;
    uint64_t epilogue_launches;
    uint64_t attention_launches;
    uint64_t convolution_launches;
} h3_cuda_accel_stats;

/* Optional cuBLASLt accelerator for the hot int8 DiT projections. Creation
 * failure is non-fatal: callers retain the native correctness-kernel path. */
h3_cuda_accel *h3_cuda_accel_create(int compute_major, int compute_minor);
void h3_cuda_accel_free(h3_cuda_accel *accel);
int h3_cuda_accel_available(const h3_cuda_accel *accel);
size_t h3_cuda_accel_device_bytes(const h3_cuda_accel *accel);

/* Return 1 when the request was handled without an API error. `used` is set to
 * one only when cuBLASLt and its fused CUDA epilogue were enqueued; a zero
 * value asks the backend to dispatch its portable kernel instead. */
int h3_cuda_accel_linear_f32(
    h3_cuda_accel *accel, void *stream, void *output, const void *input,
    const void *weight, const void *bias, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim, int *used, h3_cuda_accel_stats *stats,
    char *error, size_t error_size);

int h3_cuda_accel_linear_int8_bf16(
    h3_cuda_accel *accel, void *stream, void *output, const void *input,
    const void *weight, const void *input_scales, const void *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t output_dim, int *used,
    h3_cuda_accel_stats *stats, char *error, size_t error_size);

int h3_cuda_accel_fc1_swiglu_int8_bf16(
    h3_cuda_accel *accel, void *stream, void *output, const void *input,
    const void *weight, const void *input_scales, const void *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t hidden_dim, int *used,
    h3_cuda_accel_stats *stats, char *error, size_t error_size);

int h3_cuda_accel_linear_int8_grouped_bf16(
    h3_cuda_accel *accel, void *stream, void *output, const void *input,
    const void *weight, const void *input_scales, const void *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
    uint32_t group_size, uint32_t groups, int *used,
    h3_cuda_accel_stats *stats, char *error, size_t error_size);

/* Tiled online-softmax attention. Q/K/V are row-major
 * [sequence,heads,head_dim]; output may retain that layout or use the native
 * [heads,sequence,head_dim] form consumed by the int8 output projection. */
int h3_cuda_accel_sdpa_bf16(
    h3_cuda_accel *accel, void *stream, void *output, const void *query,
    const void *key, const void *value, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, float scale, int head_major_output, int *used,
    h3_cuda_accel_stats *stats, char *error, size_t error_size);

int h3_cuda_accel_sdpa_f32(
    h3_cuda_accel *accel, void *stream, void *output, const void *query,
    const void *key, const void *value, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, float scale, int *used, h3_cuda_accel_stats *stats,
    char *error, size_t error_size);

int h3_cuda_accel_conv3d_f32(
    h3_cuda_accel *accel, void *stream, void *output, const void *input,
    const void *weight, const void *bias, uint32_t batch, uint32_t depth,
    uint32_t height, uint32_t width, uint32_t input_channels,
    uint32_t output_channels, uint32_t kernel_depth, uint32_t kernel_height,
    uint32_t kernel_width, uint32_t stride_depth, uint32_t stride_height,
    uint32_t stride_width, int *used, h3_cuda_accel_stats *stats,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
