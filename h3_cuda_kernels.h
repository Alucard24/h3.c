#ifndef H3_CUDA_KERNELS_H
#define H3_CUDA_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t count;
    uint32_t elements;
    uint32_t sample_offset;
    uint32_t tokens;
    uint32_t width;
    uint32_t rows;
    uint32_t vocab_size;
    uint32_t approximate;
    float left_scale;
    float right_scale;
    float minimum;
    float maximum;
    float delta;
    float ratio;
    float epsilon;
    uint32_t has_bias;
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    uint32_t gate_slot;
    uint32_t grouped;
    uint32_t conv_batch;
    uint32_t conv_depth;
    uint32_t conv_height;
    uint32_t conv_width;
    uint32_t conv_in_channels;
    uint32_t conv_out_channels;
    uint32_t conv_kernel_depth;
    uint32_t conv_kernel_height;
    uint32_t conv_kernel_width;
    uint32_t conv_stride_depth;
    uint32_t conv_stride_height;
    uint32_t conv_stride_width;
    uint32_t pad_depth_front;
    uint32_t pad_height_before;
    uint32_t pad_height_after;
    uint32_t pad_width_before;
    uint32_t pad_width_after;
} h3_cuda_args;

typedef enum {
    H3_CUDA_KERNEL_CAST_F32_TO_BF16,
    H3_CUDA_KERNEL_CAST_BF16_TO_F32,
    H3_CUDA_KERNEL_SILU_F32,
    H3_CUDA_KERNEL_SILU_BF16,
    H3_CUDA_KERNEL_SILU_MUL_BF16,
    H3_CUDA_KERNEL_CLIP_F32,
    H3_CUDA_KERNEL_GELU_BF16,
    H3_CUDA_KERNEL_ADD_BF16,
    H3_CUDA_KERNEL_SUB_BF16,
    H3_CUDA_KERNEL_ADD_SCALED_F32,
    H3_CUDA_KERNEL_GEGLU_F32,
    H3_CUDA_KERNEL_EULER_BF16,
    H3_CUDA_KERNEL_EMBEDDING_BF16,
    H3_CUDA_KERNEL_RMS_NORM_F32,
    H3_CUDA_KERNEL_LAYER_NORM_F32,
    H3_CUDA_KERNEL_RMS_NORM_BF16,
    H3_CUDA_KERNEL_LAYER_NORM_BF16,
    H3_CUDA_KERNEL_LINEAR_BF16,
    H3_CUDA_KERNEL_SWIGLU_HALVES_BF16,
    H3_CUDA_KERNEL_ADALN_BF16,
    H3_CUDA_KERNEL_GATE_BF16,
    H3_CUDA_KERNEL_GATE_ADALN_BF16,
    H3_CUDA_KERNEL_RMS_INVERSE_BF16,
    H3_CUDA_KERNEL_ADALN_LINEAR_BF16,
    H3_CUDA_KERNEL_HEAD_RMS_NORM_BF16,
    H3_CUDA_KERNEL_GROUPED_QKV_ROPE_BF16,
    H3_CUDA_KERNEL_SDPA_BF16,
    H3_CUDA_KERNEL_SDPA_FLASH_BF16,
    H3_CUDA_KERNEL_PATCH_LINEAR_BF16,
    H3_CUDA_KERNEL_PATCH_LINEAR_BF16_MAP,
    H3_CUDA_KERNEL_TOKEN_POOL_BF16,
    H3_CUDA_KERNEL_TOKEN_POOL_ADALN_BF16,
    H3_CUDA_KERNEL_TOKEN_EXPAND_DELTA_BF16,
    H3_CUDA_KERNEL_TOKEN_EXPAND_ADALN_BF16,
    H3_CUDA_KERNEL_TEXT_QK_ROPE_BF16,
    H3_CUDA_KERNEL_ROPE_TEXT_BF16,
    H3_CUDA_KERNEL_GQA_CAUSAL_BF16,
    H3_CUDA_KERNEL_LINEAR_F32,
    H3_CUDA_KERNEL_SCALE_ADD_F32,
    H3_CUDA_KERNEL_SWIGLU_F32,
    H3_CUDA_KERNEL_VIDEO_QKV_ROPE_F32,
    H3_CUDA_KERNEL_VAE_ENCODER_PAD_F32,
    H3_CUDA_KERNEL_VAE_ENCODER_GROUP_NORM_SILU_F32,
    H3_CUDA_KERNEL_SDPA_F32,
    H3_CUDA_KERNEL_SDPA_FLASH_F32,
    H3_CUDA_KERNEL_CONV3D_F32,
    H3_CUDA_KERNEL_WEIGHT_NORM_F32,
    H3_CUDA_KERNEL_SNAKE1D_F32,
    H3_CUDA_KERNEL_ALIAS_FREE_SNAKE_F32,
    H3_CUDA_KERNEL_AUDIO_QKV_SPLIT_F32,
    H3_CUDA_KERNEL_AUDIO_ATTENTION_POOL_F32,
    H3_CUDA_KERNEL_CONV1D_STRIDE_F32,
    H3_CUDA_KERNEL_CONV_TRANSPOSE1D_F32,
    H3_CUDA_KERNEL_SDPA_CAUSAL_F32,
    H3_CUDA_KERNEL_VISION_QKV_ROPE_BF16,
    H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8,
    H3_CUDA_KERNEL_LINEAR_INT8_BF16,
    H3_CUDA_KERNEL_FC1_SWIGLU_INT8_BF16,
    H3_CUDA_KERNEL_GATE_ADALN_QUANTIZE_INT8,
    H3_CUDA_KERNEL_QUANTIZE_ROWS_GROUPS_BF16_I8,
    H3_CUDA_KERNEL_LINEAR_INT8_GROUPED_BF16,
    H3_CUDA_KERNEL_QUANTIZE_HEAD_MAJOR_ROWS_BF16_I8,
    H3_CUDA_KERNEL_COUNT
} h3_cuda_kernel;

int h3_cuda_launch(h3_cuda_kernel kernel, void *stream,
                   const h3_cuda_args *args, void *const buffers[10],
                   uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                   char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
