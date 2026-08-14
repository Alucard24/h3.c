/* Native CUDA GPU backend for the h3 inference engine.
 *
 * The public orchestration mirrors the Vulkan backend, while storage,
 * ordering, and dispatch use CUDA device memory and one ordered stream.
 * Correctness kernels are generated from h3_vulkan_shaders.comp so both
 * Linux backends retain the same arithmetic boundaries and binding order.
 */
#include "h3.h"
#include "h3_cuda_accel.h"
#include "h3_cuda_kernels.h"
#include "h3_gpu.h"

#include <cuda_runtime_api.h>

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define H3_CUDA_THREADS 256u
#define H3_CUDA_BINDINGS 10u

static const char *const h3_cuda_kernel_names[H3_CUDA_KERNEL_COUNT] = {
    "cast_f32_to_bf16",
    "cast_bf16_to_f32",
    "silu_f32",
    "silu_bf16",
    "silu_mul_bf16",
    "clip_f32",
    "gelu_bf16",
    "add_bf16",
    "sub_bf16",
    "add_scaled_f32",
    "geglu_f32",
    "euler_bf16",
    "embedding_bf16",
    "rms_norm_f32",
    "layer_norm_f32",
    "rms_norm_bf16",
    "layer_norm_bf16",
    "linear_bf16",
    "swiglu_halves_bf16",
    "adaln_bf16",
    "gate_bf16",
    "gate_adaln_bf16",
    "rms_inverse_bf16",
    "adaln_linear_bf16",
    "head_rms_norm_bf16",
    "grouped_qkv_rope_bf16",
    "sdpa_bf16",
    "sdpa_flash_bf16",
    "patch_linear_bf16",
    "patch_linear_bf16_map",
    "token_pool_bf16",
    "token_pool_adaln_bf16",
    "token_expand_delta_bf16",
    "token_expand_adaln_bf16",
    "text_qk_rope_bf16",
    "rope_text_bf16",
    "gqa_causal_bf16",
    "linear_f32",
    "scale_add_f32",
    "swiglu_f32",
    "video_qkv_rope_f32",
    "vae_encoder_pad_f32",
    "vae_encoder_group_norm_silu_f32",
    "sdpa_f32",
    "sdpa_flash_f32",
    "conv3d_f32",
    "weight_norm_f32",
    "snake1d_f32",
    "alias_free_snake_f32",
    "audio_qkv_split_f32",
    "audio_attention_pool_f32",
    "conv1d_stride_f32",
    "conv_transpose1d_f32",
    "sdpa_causal_f32",
    "vision_qkv_rope_bf16",
    "quantize_rows_bf16_i8",
    "linear_int8_bf16",
    "fc1_swiglu_int8_bf16",
    "gate_adaln_quantize_int8",
    "quantize_rows_groups_bf16_i8",
    "linear_int8_grouped_bf16",
    "quantize_head_major_rows_bf16_i8"};

static const uint32_t h3_cuda_kernel_bindings[H3_CUDA_KERNEL_COUNT] = {
    2, 2, 2, 2, 3, 2, 2, 3, 3, 3, 3,  3, 3,  3, 4, 3, 4, 4, 2, 5, 5,
    8, 2, 8, 2, 8, 4, 4, 4, 5, 6, 10, 6, 10, 8, 4, 4, 4, 4, 2, 6, 2,
    4, 4, 4, 4, 3, 3, 6, 7, 2, 4, 4,  4, 6,  3, 5, 5, 9, 3, 5, 3};

typedef struct {
    h3_cuda_args args;
    void *buffers[H3_CUDA_BINDINGS];
} h3_cuda_set;

struct h3_gpu_tensor {
    h3_gpu *gpu;
    void *data;
    size_t elements;
    h3_gpu_dtype dtype;
    size_t byte_size;
    struct h3_gpu_tensor *pending_next;
};

struct h3_gpu {
    char error[512];
    h3_gpu_stats stats;
    int device;
    cudaStream_t stream;
    cudaStream_t upload_stream;
    cudaEvent_t batch_start;
    cudaEvent_t batch_end;
    h3_cuda_accel *accel;
    size_t accel_accounted_bytes;
    int command_active;
    h3_cuda_args args_storage;
    h3_cuda_args *args;
    h3_cuda_set prepared;
    double encode_start;
    char profile_label[96];
    h3_gpu_stats profile_start_stats;
    h3_gpu_stats profile_mark_stats;
    double profile_start_wall;
    double profile_mark_wall;
    h3_gpu_tensor *pending_head;
    h3_gpu_tensor *pending_tail;
};

static double h3_cuda_now(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0)
        return 0.0;
    return (double)timestamp.tv_sec + (double)timestamp.tv_nsec * 1e-9;
}

static int h3_cuda_profile_enabled(void) {
    const char *value = getenv("H3_PROFILE");
    return value && *value && strcmp(value, "0") != 0;
}

static uint64_t h3_cuda_counter_delta(uint64_t value, uint64_t start) {
    return value >= start ? value - start : 0;
}

static void h3_cuda_profile_emit(h3_gpu *gpu, const char *phase,
                                 h3_gpu_stats start, double wall_start) {
    if (!gpu || !phase || !h3_cuda_profile_enabled())
        return;
    h3_gpu_stats value = gpu->stats;
    fprintf(stderr,
            "h3 profile: %-24s %-14s wall=%8.3fs encode=%7.3fs "
            "wait=%8.3fs gpu=%7.3fs peak=%7.3fGiB alloc=%7.3fGiB "
            "submissions=%llu kernels=%llu cublaslt=%llu attention=%llu "
            "cudnn=%llu\n",
            gpu->profile_label[0] ? gpu->profile_label : "CUDA context", phase,
            h3_cuda_now() - wall_start,
            value.command_encode_seconds - start.command_encode_seconds,
            value.command_wait_seconds - start.command_wait_seconds,
            value.gpu_seconds - start.gpu_seconds,
            (double)value.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)h3_cuda_counter_delta(value.allocated_bytes,
                                          start.allocated_bytes) /
                (1024.0 * 1024.0 * 1024.0),
            (unsigned long long)h3_cuda_counter_delta(value.submissions,
                                                       start.submissions),
            (unsigned long long)h3_cuda_counter_delta(
                value.direct_dispatches, start.direct_dispatches),
            (unsigned long long)h3_cuda_counter_delta(
                value.mps_linear_dispatches, start.mps_linear_dispatches),
            (unsigned long long)h3_cuda_counter_delta(
                value.mps_sdpa_dispatches, start.mps_sdpa_dispatches),
            (unsigned long long)h3_cuda_counter_delta(
                value.mps_conv_dispatches, start.mps_conv_dispatches));
}

static void h3_cuda_account_accel(h3_gpu *gpu,
                                  h3_cuda_accel_stats accelerated) {
    if (!gpu)
        return;
    size_t bytes = h3_cuda_accel_device_bytes(gpu->accel);
    if (bytes > gpu->accel_accounted_bytes) {
        size_t added = bytes - gpu->accel_accounted_bytes;
        gpu->stats.allocated_bytes += added;
        gpu->stats.live_bytes += added;
        if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
            gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
        gpu->accel_accounted_bytes = bytes;
    }
    gpu->stats.mps_linear_dispatches += accelerated.matmul_launches;
    gpu->stats.mps_sdpa_dispatches += accelerated.attention_launches;
    gpu->stats.mps_conv_dispatches += accelerated.convolution_launches;
    gpu->stats.direct_dispatches += accelerated.epilogue_launches;
}

static void h3_cuda_set_error(h3_gpu *gpu, const char *format, ...) {
    if (!gpu)
        return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(gpu->error, sizeof(gpu->error), format, arguments);
    va_end(arguments);
}

static int h3_cuda_result(h3_gpu *gpu, cudaError_t result,
                          const char *operation) {
    if (result == cudaSuccess)
        return 1;
    h3_cuda_set_error(gpu, "%s failed: %s", operation,
                      cudaGetErrorString(result));
    return 0;
}

static size_t h3_cuda_dtype_size(h3_gpu_dtype dtype) {
    switch (dtype) {
    case H3_GPU_BF16:
        return 2;
    case H3_GPU_I8:
        return 1;
    case H3_GPU_F32:
    case H3_GPU_U32:
        return 4;
    }
    return 0;
}

static int h3_cuda_require_command(h3_gpu *gpu) {
    if (!gpu || !gpu->command_active) {
        h3_cuda_set_error(gpu,
                          "no active CUDA stream batch (call h3_gpu_begin)");
        return 0;
    }
    return 1;
}

static int h3_cuda_check_tensors(h3_gpu *gpu, size_t count, ...) {
    va_list arguments;
    va_start(arguments, count);
    for (size_t index = 0; index < count; index++) {
        const h3_gpu_tensor *tensor = va_arg(arguments, const h3_gpu_tensor *);
        if (!tensor || tensor->gpu != gpu) {
            h3_cuda_set_error(gpu,
                              tensor ? "tensor belongs to another CUDA context"
                                     : "null tensor argument");
            va_end(arguments);
            return 0;
        }
    }
    va_end(arguments);
    return 1;
}

static int h3_cuda_selected_device(char *error, size_t error_size,
                                   int *selected) {
    int count = 0;
    cudaError_t result = cudaGetDeviceCount(&count);
    if (result != cudaSuccess || count == 0) {
        if (error && error_size)
            snprintf(error, error_size, "no CUDA device: %s",
                     cudaGetErrorString(result));
        return 0;
    }
    int device = 0;
    const char *value = getenv("H3_CUDA_DEVICE");
    if (value && *value) {
        char *end = NULL;
        errno = 0;
        long parsed = strtol(value, &end, 10);
        if (errno || end == value || *end || parsed < 0 || parsed >= count) {
            if (error && error_size)
                snprintf(error, error_size, "invalid H3_CUDA_DEVICE=%s", value);
            return 0;
        }
        device = (int)parsed;
    }
    *selected = device;
    return 1;
}

h3_gpu *h3_gpu_create(const char *shader_source_path, char *error,
                      size_t error_size) {
    if (!shader_source_path) {
        if (error && error_size)
            snprintf(error, error_size, "no CUDA kernel source path");
        return NULL;
    }
    h3_gpu *gpu = calloc(1, sizeof(*gpu));
    if (!gpu) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory");
        return NULL;
    }
    gpu->args = &gpu->args_storage;
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "CUDA context");
    gpu->profile_start_wall = h3_cuda_now();
    gpu->profile_mark_wall = gpu->profile_start_wall;
    struct cudaDeviceProp properties;
    memset(&properties, 0, sizeof(properties));
    if (!h3_cuda_selected_device(error, error_size, &gpu->device) ||
        !h3_cuda_result(gpu, cudaSetDevice(gpu->device), "cudaSetDevice") ||
        !h3_cuda_result(gpu,
                        cudaGetDeviceProperties(&properties, gpu->device),
                        "cudaGetDeviceProperties") ||
        !h3_cuda_result(
            gpu, cudaStreamCreateWithFlags(&gpu->stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags") ||
        !h3_cuda_result(
            gpu, cudaStreamCreateWithFlags(&gpu->upload_stream,
                                           cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags upload") ||
        !h3_cuda_result(gpu, cudaEventCreate(&gpu->batch_start),
                        "cudaEventCreate batch start") ||
        !h3_cuda_result(gpu, cudaEventCreate(&gpu->batch_end),
                        "cudaEventCreate batch end")) {
        if (error && error_size && gpu->error[0])
            snprintf(error, error_size, "%s", gpu->error);
        h3_gpu_free(gpu);
        return NULL;
    }
    gpu->accel = h3_cuda_accel_create(properties.major, properties.minor);
    return gpu;
}

static void h3_cuda_tensor_destroy(h3_gpu_tensor *tensor) {
    if (!tensor)
        return;
    h3_gpu *gpu = tensor->gpu;
    if (tensor->data)
        (void)cudaFree(tensor->data);
    if (gpu && gpu->stats.live_bytes >= tensor->byte_size)
        gpu->stats.live_bytes -= tensor->byte_size;
    free(tensor);
}

static void h3_cuda_drain_pending(h3_gpu *gpu) {
    h3_gpu_tensor *tensor = gpu ? gpu->pending_head : NULL;
    while (tensor) {
        h3_gpu_tensor *next = tensor->pending_next;
        h3_cuda_tensor_destroy(tensor);
        tensor = next;
    }
    if (gpu) {
        gpu->pending_head = NULL;
        gpu->pending_tail = NULL;
    }
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu)
        return;
    if (gpu->stream)
        (void)cudaStreamSynchronize(gpu->stream);
    if (gpu->upload_stream)
        (void)cudaStreamSynchronize(gpu->upload_stream);
    h3_cuda_profile_emit(gpu, "total", gpu->profile_start_stats,
                         gpu->profile_start_wall);
    h3_cuda_drain_pending(gpu);
    h3_cuda_accel_free(gpu->accel);
    gpu->accel = NULL;
    if (gpu->stats.live_bytes >= gpu->accel_accounted_bytes)
        gpu->stats.live_bytes -= gpu->accel_accounted_bytes;
    if (gpu->batch_start)
        (void)cudaEventDestroy(gpu->batch_start);
    if (gpu->batch_end)
        (void)cudaEventDestroy(gpu->batch_end);
    if (gpu->upload_stream)
        (void)cudaStreamDestroy(gpu->upload_stream);
    if (gpu->stream)
        (void)cudaStreamDestroy(gpu->stream);
    free(gpu);
}

int h3_gpu_is_m5(const h3_gpu *gpu) {
    (void)gpu;
    return 1;
}

int h3_gpu_has_nax_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 1;
}

int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 1;
}

int h3_gpu_has_int8_streaming(const h3_gpu *gpu) {
    (void)gpu;
    return 1;
}

const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu ? gpu->error : "null CUDA context";
}

static h3_gpu_tensor *h3_cuda_tensor_new(h3_gpu *gpu, size_t elements,
                                         h3_gpu_dtype dtype) {
    size_t element_size = h3_cuda_dtype_size(dtype);
    if (!gpu || !element_size || elements > SIZE_MAX / element_size) {
        h3_cuda_set_error(gpu, "invalid CUDA tensor size");
        return NULL;
    }
    h3_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3_cuda_set_error(gpu, "out of memory");
        return NULL;
    }
    tensor->gpu = gpu;
    tensor->elements = elements;
    tensor->dtype = dtype;
    tensor->byte_size = elements * element_size;
    if (tensor->byte_size &&
        !h3_cuda_result(gpu, cudaMalloc(&tensor->data, tensor->byte_size),
                        "cudaMalloc")) {
        free(tensor);
        return NULL;
    }
    gpu->stats.tensor_allocations++;
    gpu->stats.allocated_bytes += tensor->byte_size;
    gpu->stats.live_bytes += tensor->byte_size;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_cuda_tensor_new(gpu, elements, H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_cuda_tensor_new(gpu, elements, H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_cuda_tensor_new(gpu, elements, H3_GPU_I8);
}

static h3_gpu_tensor *h3_cuda_tensor_from(h3_gpu *gpu, const void *values,
                                          size_t elements, h3_gpu_dtype dtype) {
    if (elements && !values) {
        h3_cuda_set_error(gpu, "null tensor source");
        return NULL;
    }
    h3_gpu_tensor *tensor = h3_cuda_tensor_new(gpu, elements, dtype);
    if (!tensor || !tensor->byte_size)
        return tensor;
    if (!h3_cuda_result(gpu,
                        cudaMemcpyAsync(tensor->data, values, tensor->byte_size,
                                        cudaMemcpyHostToDevice, gpu->stream),
                        "cudaMemcpyAsync host to device") ||
        !h3_cuda_result(gpu, cudaStreamSynchronize(gpu->stream),
                        "cudaStreamSynchronize tensor upload")) {
        h3_cuda_tensor_destroy(tensor);
        return NULL;
    }
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_cuda_tensor_from(gpu, values, elements, H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return h3_cuda_tensor_from(gpu, values, elements, H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return h3_cuda_tensor_from(gpu, values, elements, H3_GPU_U32);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor)
        return;
    h3_gpu *gpu = tensor->gpu;
    if (gpu && gpu->command_active) {
        tensor->pending_next = NULL;
        if (gpu->pending_tail)
            gpu->pending_tail->pending_next = tensor;
        else
            gpu->pending_head = tensor;
        gpu->pending_tail = tensor;
        return;
    }
    h3_cuda_tensor_destroy(tensor);
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->elements : 0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->dtype : H3_GPU_F32;
}

static int h3_cuda_tensor_read(const h3_gpu_tensor *tensor,
                               size_t source_offset, void *values,
                               size_t elements) {
    if (!tensor || (!values && elements) || source_offset > tensor->elements ||
        elements > tensor->elements - source_offset)
        return 0;
    if (!elements)
        return 1;
    h3_gpu *gpu = tensor->gpu;
    size_t element_size = h3_cuda_dtype_size(tensor->dtype);
    const uint8_t *source =
        (const uint8_t *)tensor->data + source_offset * element_size;
    if (!h3_cuda_result(gpu,
                        cudaMemcpyAsync(values, source, elements * element_size,
                                        cudaMemcpyDeviceToHost, gpu->stream),
                        "cudaMemcpyAsync device to host"))
        return 0;
    return h3_cuda_result(gpu, cudaStreamSynchronize(gpu->stream),
                          "cudaStreamSynchronize readback");
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_cuda_tensor_read(tensor, 0, values, elements);
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    return h3_cuda_tensor_read(tensor, source_offset, values, elements);
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    return h3_cuda_tensor_read(tensor, 0, values, elements);
}

int h3_gpu_tensor_read_i8(const h3_gpu_tensor *tensor, int8_t *values,
                          size_t elements) {
    if (!tensor || tensor->dtype != H3_GPU_I8)
        return 0;
    return h3_cuda_tensor_read(tensor, 0, values, elements);
}

static int h3_cuda_tensor_write(h3_gpu_tensor *tensor,
                                size_t destination_offset, const void *values,
                                size_t elements) {
    if (!tensor || (!values && elements) ||
        destination_offset > tensor->elements ||
        elements > tensor->elements - destination_offset)
        return 0;
    if (!elements)
        return 1;
    h3_gpu *gpu = tensor->gpu;
    size_t element_size = h3_cuda_dtype_size(tensor->dtype);
    uint8_t *destination =
        (uint8_t *)tensor->data + destination_offset * element_size;
    if (!h3_cuda_result(gpu,
                        cudaMemcpyAsync(destination, values,
                                        elements * element_size,
                                        cudaMemcpyHostToDevice, gpu->stream),
                        "cudaMemcpyAsync host to device"))
        return 0;
    return h3_cuda_result(gpu, cudaStreamSynchronize(gpu->stream),
                          "cudaStreamSynchronize upload");
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_cuda_tensor_write(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    return h3_cuda_tensor_write(tensor, destination_offset, values, elements);
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_cuda_tensor_write(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    return h3_cuda_tensor_write(tensor, destination_offset, values, elements);
}

static int h3_cuda_pread_all(int fd, void *destination, size_t bytes,
                             uint64_t offset) {
    uint8_t *cursor = destination;
    while (bytes) {
        ssize_t got = pread(fd, cursor, bytes, (off_t)offset);
        if (got <= 0)
            return 0;
        cursor += (size_t)got;
        bytes -= (size_t)got;
        offset += (uint64_t)got;
    }
    return 1;
}

static h3_gpu_tensor *h3_cuda_tensor_load(h3_gpu *gpu, const char *path,
                                          uint64_t file_offset, size_t elements,
                                          h3_gpu_dtype dtype) {
    h3_gpu_tensor *tensor = h3_cuda_tensor_new(gpu, elements, dtype);
    if (!tensor || !tensor->byte_size)
        return tensor;
    void *staging = malloc(tensor->byte_size);
    int fd = open(path, O_RDONLY);
    if (!staging || fd < 0 ||
        !h3_cuda_pread_all(fd, staging, tensor->byte_size, file_offset) ||
        !h3_cuda_result(gpu,
                        cudaMemcpyAsync(tensor->data, staging,
                                        tensor->byte_size,
                                        cudaMemcpyHostToDevice, gpu->stream),
                        "CUDA weight upload") ||
        !h3_cuda_result(gpu, cudaStreamSynchronize(gpu->stream),
                        "CUDA weight upload synchronization")) {
        if (fd < 0)
            h3_cuda_set_error(gpu, "cannot open %s", path);
        else if (staging && !gpu->error[0])
            h3_cuda_set_error(gpu, "short read from %s", path);
        if (fd >= 0)
            close(fd);
        free(staging);
        h3_cuda_tensor_destroy(tensor);
        return NULL;
    }
    close(fd);
    free(staging);
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
    return h3_cuda_tensor_load(gpu, path, file_offset, elements, H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
    return h3_cuda_tensor_load(gpu, path, file_offset, elements, H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_load_i8(h3_gpu *gpu, const char *path,
                                     uint64_t file_offset, size_t elements) {
    return h3_cuda_tensor_load(gpu, path, file_offset, elements, H3_GPU_I8);
}

static int h3_cuda_tensor_read_file(h3_gpu_tensor *tensor, const char *path,
                                    uint64_t file_offset, size_t elements,
                                    char *error, size_t error_size) {
    if (!tensor || elements > tensor->elements)
        return 0;
    size_t bytes = elements * h3_cuda_dtype_size(tensor->dtype);
    void *staging = bytes ? malloc(bytes) : NULL;
    int fd = open(path, O_RDONLY);
    int ok = !bytes || (staging && fd >= 0 &&
                        h3_cuda_pread_all(fd, staging, bytes, file_offset));
    if (fd >= 0)
        close(fd);
    if (ok && bytes)
        ok = h3_cuda_result(tensor->gpu,
                            cudaMemcpyAsync(tensor->data, staging, bytes,
                                            cudaMemcpyHostToDevice,
                                            tensor->gpu->upload_stream),
                            "CUDA streamed weight upload") &&
             h3_cuda_result(
                 tensor->gpu,
                 cudaStreamSynchronize(tensor->gpu->upload_stream),
                 "CUDA streamed weight synchronization");
    if (!ok && error && error_size)
        snprintf(error, error_size, "%s",
                 tensor->gpu->error[0] ? tensor->gpu->error
                                       : "cannot read CUDA tensor file");
    free(staging);
    return ok ? 1 : 0;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_cuda_tensor_read_file(tensor, path, file_offset, elements, error,
                                    error_size);
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    if (!tensor || tensor->dtype != H3_GPU_BF16)
        return 0;
    return h3_cuda_tensor_read_file(tensor, path, file_offset, elements, error,
                                    error_size);
}

int h3_gpu_tensor_stream_file(h3_gpu_tensor *tensor, const char *path,
                              uint64_t file_offset, size_t elements,
                              char *error, size_t error_size) {
    return h3_cuda_tensor_read_file(tensor, path, file_offset, elements, error,
                                    error_size);
}

int h3_gpu_begin(h3_gpu *gpu) {
    if (!gpu || gpu->command_active) {
        h3_cuda_set_error(gpu, gpu ? "CUDA stream batch already active"
                                   : "null CUDA context");
        return 0;
    }
    if (!h3_cuda_result(gpu, cudaEventRecord(gpu->batch_start, gpu->stream),
                        "cudaEventRecord batch start"))
        return 0;
    gpu->command_active = 1;
    gpu->encode_start = h3_cuda_now();
    return 1;
}

int h3_gpu_continue(h3_gpu *gpu) {
    if (!h3_cuda_require_command(gpu))
        return 0;
    gpu->stats.command_encode_seconds += h3_cuda_now() - gpu->encode_start;
    gpu->stats.submissions++;
    gpu->encode_start = h3_cuda_now();
    return 1;
}

int h3_gpu_submit(h3_gpu *gpu) {
    if (!h3_cuda_require_command(gpu))
        return 0;
    gpu->stats.command_encode_seconds += h3_cuda_now() - gpu->encode_start;
    if (!h3_cuda_result(gpu, cudaEventRecord(gpu->batch_end, gpu->stream),
                        "cudaEventRecord batch end"))
        return 0;
    double wait_start = h3_cuda_now();
    if (!h3_cuda_result(gpu, cudaStreamSynchronize(gpu->stream),
                        "cudaStreamSynchronize"))
        return 0;
    double waited = h3_cuda_now() - wait_start;
    float elapsed_ms = 0.0f;
    if (!h3_cuda_result(
            gpu, cudaEventElapsedTime(&elapsed_ms, gpu->batch_start,
                                      gpu->batch_end),
            "cudaEventElapsedTime"))
        return 0;
    gpu->stats.command_wait_seconds += waited;
    gpu->stats.gpu_seconds += (double)elapsed_ms * 1e-3;
    gpu->stats.submissions++;
    gpu->command_active = 0;
    h3_cuda_drain_pending(gpu);
    return 1;
}

int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats)
        return 0;
    *stats = gpu->stats;
    return 1;
}

static h3_cuda_set *h3_cuda_prepare_off(h3_gpu *gpu, h3_cuda_kernel kernel,
                                        const h3_gpu_tensor *const *tensors,
                                        uint32_t tensor_count,
                                        size_t first_offset,
                                        size_t last_offset) {
    if (!gpu || kernel >= H3_CUDA_KERNEL_COUNT ||
        tensor_count > H3_CUDA_BINDINGS) {
        h3_cuda_set_error(gpu, "invalid CUDA kernel binding request");
        return NULL;
    }
    if (tensor_count != h3_cuda_kernel_bindings[kernel]) {
        h3_cuda_set_error(gpu, "CUDA kernel %s expects %u bindings, got %u",
                          h3_cuda_kernel_names[kernel],
                          h3_cuda_kernel_bindings[kernel], tensor_count);
        return NULL;
    }
    memset(&gpu->prepared, 0, sizeof(gpu->prepared));
    gpu->prepared.args = *gpu->args;
    /* Match Vulkan's per-dispatch argument slots: every following operation
     * starts from a zeroed block, so fields unused by its wrapper cannot leak
     * from the previous kernel. */
    memset(gpu->args, 0, sizeof(*gpu->args));
    for (uint32_t index = 0; index < tensor_count; index++) {
        size_t offset = index == 0 ? first_offset : 0;
        if (index + 1 == tensor_count)
            offset = last_offset;
        gpu->prepared.buffers[index] = (uint8_t *)tensors[index]->data + offset;
    }
    return &gpu->prepared;
}

static h3_cuda_set *h3_cuda_prepare(h3_gpu *gpu, h3_cuda_kernel kernel,
                                    const h3_gpu_tensor *const *tensors,
                                    uint32_t tensor_count) {
    return h3_cuda_prepare_off(gpu, kernel, tensors, tensor_count, 0, 0);
}

static int h3_cuda_dispatch(h3_gpu *gpu, h3_cuda_kernel kernel,
                            h3_cuda_set *set, uint32_t groups_x,
                            uint32_t groups_y, uint32_t groups_z) {
    if (!h3_cuda_require_command(gpu) || !set)
        return 0;
    if (!h3_cuda_launch(kernel, (void *)gpu->stream, &set->args, set->buffers,
                        groups_x, groups_y, groups_z, gpu->error,
                        sizeof(gpu->error)))
        return 0;
    gpu->stats.direct_dispatches++;
    return 1;
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_cuda_check_tensors(gpu, 2, output, input))
        return 0;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[2] = {input, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_CAST_F32_TO_BF16, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_CAST_F32_TO_BF16, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_cuda_check_tensors(gpu, 2, output, input))
        return 0;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[2] = {input, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_CAST_BF16_TO_F32, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_CAST_BF16_TO_F32, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_cuda_check_tensors(gpu, 2, output, input))
        return 0;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[2] = {input, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SILU_F32, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_SILU_F32, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_cuda_check_tensors(gpu, 2, output, input))
        return 0;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[2] = {input, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SILU_BF16, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_SILU_BF16, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate, const h3_gpu_tensor *up,
                         uint32_t elements) {
    if (!h3_cuda_check_tensors(gpu, 3, output, gate, up))
        return 0;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[3] = {gate, up, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SILU_MUL_BF16, tensors, 3);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_SILU_MUL_BF16, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    if (!h3_cuda_check_tensors(gpu, 2, output, input))
        return 0;
    gpu->args->elements = elements;
    gpu->args->minimum = minimum;
    gpu->args->maximum = maximum;
    const h3_gpu_tensor *tensors[2] = {input, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_CLIP_F32, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_CLIP_F32, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    if (!h3_cuda_check_tensors(gpu, 2, output, input))
        return 0;
    gpu->args->elements = elements;
    gpu->args->approximate = approximate ? 1u : 0u;
    const h3_gpu_tensor *tensors[2] = {input, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_GELU_BF16, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_GELU_BF16, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!h3_cuda_check_tensors(gpu, 3, output, left, right))
        return 0;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[3] = {left, right, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_ADD_BF16, tensors, 3);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_ADD_BF16, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!h3_cuda_check_tensors(gpu, 3, output, left, right))
        return 0;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[3] = {left, right, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SUB_BF16, tensors, 3);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_SUB_BF16, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                          float left_scale, float right_scale,
                          uint32_t elements) {
    if (!h3_cuda_check_tensors(gpu, 3, output, left, right))
        return 0;
    gpu->args->elements = elements;
    gpu->args->left_scale = left_scale;
    gpu->args->right_scale = right_scale;
    const h3_gpu_tensor *tensors[3] = {left, right, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_ADD_SCALED_F32, tensors, 3);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_ADD_SCALED_F32, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate, const h3_gpu_tensor *linear,
                     uint32_t elements) {
    if (!h3_cuda_check_tensors(gpu, 3, output, gate, linear))
        return 0;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[3] = {gate, linear, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_GEGLU_F32, tensors, 3);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_GEGLU_F32, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample, size_t sample_offset,
                      const h3_gpu_tensor *last, const h3_gpu_tensor *previous,
                      uint32_t elements, float delta, float ratio) {
    if (!h3_cuda_check_tensors(gpu, 3, sample, last, previous))
        return 0;
    gpu->args->sample_offset = (uint32_t)sample_offset;
    gpu->args->elements = elements;
    gpu->args->delta = delta;
    gpu->args->ratio = ratio;
    const h3_gpu_tensor *tensors[3] = {sample, last, previous};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_EULER_BF16, tensors, 3);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_EULER_BF16, set,
                            (elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            1, 1);
}

int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    if (!h3_cuda_check_tensors(gpu, 3, output, weight, token_ids))
        return 0;
    gpu->args->tokens = tokens;
    gpu->args->width = width;
    gpu->args->vocab_size = vocab_size;
    const h3_gpu_tensor *tensors[3] = {weight, token_ids, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_EMBEDDING_BF16, tensors, 3);
    if (set == NULL)
        return 0;
    uint64_t elements = (uint64_t)tokens * (uint64_t)width;
    return h3_cuda_dispatch(
        gpu, H3_CUDA_KERNEL_EMBEDDING_BF16, set,
        (uint32_t)((elements + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS), 1, 1);
}

static int h3_cuda_norm(h3_gpu *gpu, h3_cuda_kernel kernel,
                        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                        const h3_gpu_tensor *bias, h3_gpu_tensor *output,
                        uint32_t rows, uint32_t width, float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 3, output, input, weight))
        return 0;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->epsilon = epsilon;
    uint32_t count = 3;
    const h3_gpu_tensor *tensors[4] = {input, weight, output, bias};
    if (bias)
        count = 4;
    h3_cuda_set *set = h3_cuda_prepare(gpu, kernel, tensors, count);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, kernel, set, rows, 1, 1);
}

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                        uint32_t rows, uint32_t width, float epsilon) {
    return h3_cuda_norm(gpu, H3_CUDA_KERNEL_RMS_NORM_F32, input, weight, NULL,
                        output, rows, width, epsilon);
}

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    return h3_cuda_norm(gpu, H3_CUDA_KERNEL_LAYER_NORM_F32, input, weight, bias,
                        output, rows, width, epsilon);
}

int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    return h3_cuda_norm(gpu, H3_CUDA_KERNEL_RMS_NORM_BF16, input, weight, NULL,
                        output, rows, width, epsilon);
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    return h3_cuda_norm(gpu, H3_CUDA_KERNEL_LAYER_NORM_BF16, input, weight,
                        bias, output, rows, width, epsilon);
}

/* ------------------------------------------------------------- linear/mlp */

static int h3_cuda_linear(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t input_dim, uint32_t output_dim) {
    if (!h3_cuda_check_tensors(gpu, 3, output, input, weight))
        return 0;
    if ((size_t)rows * input_dim > h3_gpu_tensor_elements(input) ||
        (size_t)output_dim * input_dim > h3_gpu_tensor_elements(weight) ||
        (size_t)rows * output_dim > h3_gpu_tensor_elements(output) ||
        (bias && output_dim > h3_gpu_tensor_elements(bias))) {
        h3_cuda_set_error(gpu, "linear tensor size mismatch");
        return 0;
    }
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = {input, weight, bias_buffer, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_LINEAR_BF16, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_LINEAR_BF16, set,
                            (output_dim + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    return h3_cuda_linear(gpu, output, input, weight, bias, rows, input_dim,
                          output_dim);
}

/* Fused fc1 -> SwiGLU -> fc2. The FC1 weight is [2*hidden, input_dim] with
 * gate in the first half and up in the second, matching the checkpoint
 * layout. Each boundary rounds to BF16 exactly once (fc1, silu*gate, fc2),
 * the same numerical contract as the Metal silu_mul kernel. */
int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    if (!h3_cuda_check_tensors(gpu, 4, output, input, fc1_weight, fc2_weight))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    h3_gpu_tensor *fc1 =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim * 2);
    h3_gpu_tensor *activated =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim);
    if (!fc1 || !activated) {
        h3_gpu_tensor_free(fc1);
        h3_gpu_tensor_free(activated);
        h3_cuda_set_error(gpu, "MLP scratch allocation failed");
        return 0;
    }
    int ok = 0;
    if (h3_cuda_linear(gpu, fc1, input, fc1_weight, NULL, rows, input_dim,
                       hidden_dim * 2) == 1) {
        gpu->args->elements = rows * hidden_dim;
        gpu->args->width = hidden_dim;
        const h3_gpu_tensor *tensors[2] = {fc1, activated};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SWIGLU_HALVES_BF16, tensors, 2);
        if (set != NULL &&
            h3_cuda_dispatch(
                gpu, H3_CUDA_KERNEL_SWIGLU_HALVES_BF16, set,
                (uint32_t)(((uint64_t)rows * hidden_dim + H3_CUDA_THREADS - 1) /
                           H3_CUDA_THREADS),
                1, 1) == 1 &&
            h3_cuda_linear(gpu, output, activated, fc2_weight, NULL, rows,
                           hidden_dim, output_dim) == 1) {
            ok = 1;
        }
    }
    h3_gpu_tensor_free(fc1);
    h3_gpu_tensor_free(activated);
    return ok ? 1 : 0;
}

/* ------------------------------------------------------------ adaln/gate */

int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *norm_weight,
                             const h3_gpu_tensor *modulation,
                             const h3_gpu_tensor *row_map, uint32_t rows,
                             uint32_t width, uint32_t slots,
                             uint32_t shift_slot, uint32_t scale_slot,
                             float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 5, output, input, norm_weight, modulation,
                               row_map))
        return 0;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->slots = slots;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[5] = {input, norm_weight, modulation, row_map,
                                       output};
    h3_cuda_set *set =
        h3_cuda_prepare_off(gpu, H3_CUDA_KERNEL_ADALN_BF16, tensors, 5,
                            (size_t)input_offset * 2, 0);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_ADALN_BF16, set, rows, 1, 1);
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    return h3_gpu_adaln_bf16_offset(gpu, output, input, 0, norm_weight,
                                    modulation, row_map, rows, width, slots,
                                    shift_slot, scale_slot, epsilon);
}

int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    if (!h3_cuda_check_tensors(gpu, 5, output, residual, branch, modulation,
                               row_map))
        return 0;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->slots = slots;
    gpu->args->gate_slot = gate_slot;
    const h3_gpu_tensor *tensors[5] = {residual, branch, modulation, row_map,
                                       output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_GATE_BF16, tensors, 5);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_GATE_BF16, set,
                            (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_gate_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation,
    const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot,
    uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 8, gated_residual, output, residual, branch,
                               norm_weight, gate_modulation, norm_modulation,
                               row_map))
        return 0;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->slots = slots;
    gpu->args->gate_slot = gate_slot;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[8] = {
        residual,    branch,          gate_modulation, row_map,
        norm_weight, norm_modulation, gated_residual,  output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_GATE_ADALN_BF16, tensors, 8);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_GATE_ADALN_BF16, set, rows, 1,
                            1);
}

int h3_gpu_adaln_linear_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *inverse,
    const h3_gpu_tensor *input, size_t input_offset,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
    const h3_gpu_tensor *row_map, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *bias, uint32_t rows, uint32_t width,
    uint32_t output_dim, uint32_t slots, uint32_t shift_slot,
    uint32_t scale_slot, float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 8, output, inverse, input, norm_weight,
                               modulation, row_map, weight, bias))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    /* Inverse RMS scalars first (rows F32). */
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *rms_tensors[2] = {input, inverse};
    h3_cuda_set *rms_set =
        h3_cuda_prepare_off(gpu, H3_CUDA_KERNEL_RMS_INVERSE_BF16, rms_tensors,
                            2, (size_t)input_offset * 2, 0);
    if (rms_set == NULL ||
        h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_RMS_INVERSE_BF16, rms_set, rows, 1,
                         1) == 0)
        return 0;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->output_dim = output_dim;
    gpu->args->slots = slots;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[8] = {input,       inverse, norm_weight,
                                       modulation,  row_map, weight,
                                       bias_buffer, output};
    h3_cuda_set *set =
        h3_cuda_prepare_off(gpu, H3_CUDA_KERNEL_ADALN_LINEAR_BF16, tensors, 8,
                            (size_t)input_offset * 2, 0);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_ADALN_LINEAR_BF16, set,
                            (output_dim + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim,
                              float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 2, tensor, weight))
        return 0;
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[2] = {tensor, weight};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_HEAD_RMS_NORM_BF16, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_HEAD_RMS_NORM_BF16, set,
                            (sequence + 15) / 16, (heads + 15) / 16, 1);
}

/* ------------------------------------------------------------ qkv/sdpa */

static int h3_cuda_patch_linear(h3_gpu *gpu, h3_gpu_tensor *output,
                                size_t output_offset,
                                const h3_gpu_tensor *input, size_t input_offset,
                                const h3_gpu_tensor *weight,
                                const h3_gpu_tensor *bias, uint32_t rows,
                                uint32_t input_dim, uint32_t output_dim) {
    if (!h3_cuda_check_tensors(gpu, 3, output, input, weight))
        return 0;
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = {input, weight, bias_buffer, output};
    h3_cuda_set *set = h3_cuda_prepare_off(
        gpu, H3_CUDA_KERNEL_PATCH_LINEAR_BF16, tensors, 4,
        (size_t)input_offset * 4, (size_t)output_offset * 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_PATCH_LINEAR_BF16, set,
                            (output_dim + 15) / 16, (rows + 15) / 16, 1);
}

static int
h3_cuda_qkv_rope(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
                 h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                 const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
                 const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
                 uint32_t sequence, uint32_t heads, uint32_t head_dim,
                 uint32_t rope_half, float epsilon, int grouped) {
    if (!h3_cuda_check_tensors(gpu, 8, query, key, value, qkv, q_norm, k_norm,
                               rope_cos, rope_sin))
        return 0;
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->output_dim = rope_half;
    gpu->args->epsilon = epsilon;
    gpu->args->grouped = grouped ? 1u : 0u;
    const h3_gpu_tensor *tensors[8] = {qkv,      q_norm, k_norm, rope_cos,
                                       rope_sin, query,  key,    value};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_GROUPED_QKV_ROPE_BF16, tensors, 8);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_GROUPED_QKV_ROPE_BF16, set,
                            (head_dim + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            heads, sequence);
}

static int h3_cuda_sdpa(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                        const h3_gpu_tensor *value, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim, float scale,
                        int head_major_output) {
    if (!h3_cuda_check_tensors(gpu, 4, output, query, key, value))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t required = (size_t)sequence * heads * head_dim;
    if (required > h3_gpu_tensor_elements(output) ||
        required > h3_gpu_tensor_elements(query) ||
        required > h3_gpu_tensor_elements(key) ||
        required > h3_gpu_tensor_elements(value)) {
        h3_cuda_set_error(gpu, "BF16 SDPA tensor size mismatch");
        return 0;
    }
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->left_scale = scale;
    gpu->args->grouped = head_major_output ? 1u : 0u;
    /* Long sequences use the 128-thread flash kernel; short ones keep the
     * one-thread-per-output naive kernel, whose bit-exact reference order
     * is what the parity tests compare against. */
    int flash = sequence >= 128 && head_dim <= 128;
    if (flash) {
        h3_cuda_accel_stats accelerated = {0};
        int used = 0;
        int ok = h3_cuda_accel_sdpa_bf16(
            gpu->accel, (void *)gpu->stream, output->data, query->data,
            key->data, value->data, sequence, heads, head_dim, scale,
            head_major_output, &used, &accelerated, gpu->error,
            sizeof(gpu->error));
        h3_cuda_account_accel(gpu, accelerated);
        if (!ok)
            return 0;
        if (used)
            return 1;
    }
    h3_cuda_kernel kernel =
        flash ? H3_CUDA_KERNEL_SDPA_FLASH_BF16 : H3_CUDA_KERNEL_SDPA_BF16;
    const h3_gpu_tensor *tensors[4] = {query, key, value, output};
    h3_cuda_set *set = h3_cuda_prepare(gpu, kernel, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, kernel, set, heads, sequence, 1);
}

int h3_gpu_grouped_qkv_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value,
    const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_norm,
    const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
    const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, uint32_t rope_half, float epsilon) {
    return h3_cuda_qkv_rope(gpu, query, key, value, qkv, q_norm, k_norm,
                            rope_cos, rope_sin, sequence, heads, head_dim,
                            rope_half, epsilon, 1);
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
                         h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim, uint32_t rope_half,
                         float epsilon) {
    return h3_cuda_qkv_rope(gpu, query, key, value, qkv, q_norm, k_norm,
                            rope_cos, rope_sin, sequence, heads, head_dim,
                            rope_half, epsilon, 0);
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_cuda_sdpa(gpu, output, query, key, value, sequence, heads,
                        head_dim, scale, 0);
}

int h3_gpu_sdpa_bf16_head_major_output(h3_gpu *gpu, h3_gpu_tensor *output,
                                       const h3_gpu_tensor *query,
                                       const h3_gpu_tensor *key,
                                       const h3_gpu_tensor *value,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, float scale) {
    return h3_cuda_sdpa(gpu, output, query, key, value, sequence, heads,
                        head_dim, scale, 1);
}

/* ------------------------------------------------------------ patch */

int h3_gpu_patch_linear_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                                    size_t output_offset,
                                    const h3_gpu_tensor *input,
                                    size_t input_offset,
                                    const h3_gpu_tensor *weight,
                                    const h3_gpu_tensor *bias, uint32_t rows,
                                    uint32_t input_dim, uint32_t output_dim) {
    return h3_cuda_patch_linear(gpu, output, output_offset, input, input_offset,
                                weight, bias, rows, input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(gpu, output, 0, input, 0, weight,
                                           bias, rows, input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16_map(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *bias,
                                 const h3_gpu_tensor *row_map,
                                 uint32_t output_rows, uint32_t rows,
                                 uint32_t input_dim, uint32_t output_dim) {
    if (!h3_cuda_check_tensors(gpu, 4, output, input, weight, row_map))
        return 0;
    if (bias && !h3_cuda_check_tensors(gpu, 1, bias))
        return 0;
    (void)output_rows;
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[5] = {input, weight, bias_buffer, output,
                                       row_map};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_PATCH_LINEAR_BF16_MAP, tensors, 5);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_PATCH_LINEAR_BF16_MAP, set,
                            (output_dim + 15) / 16, (rows + 15) / 16, 1);
}

/* ---------------------------------------------------------- token pool */

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    if (!h3_cuda_check_tensors(gpu, 6, output, input, original, baseline,
                               baseline_indices, pairs))
        return 0;
    (void)input_rows;
    (void)baseline_rows;
    gpu->args->sample_offset = (uint32_t)input_offset;
    gpu->args->tokens = (uint32_t)original_offset;
    gpu->args->vocab_size = (uint32_t)baseline_offset;
    gpu->args->rows = rows;
    gpu->args->width = width;
    const h3_gpu_tensor *tensors[6] = {
        input, pairs, output, baseline, baseline_indices, original};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_TOKEN_POOL_BF16, tensors, 6);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_TOKEN_POOL_BF16, set,
                            (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_token_pool_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *input, size_t input_offset, h3_gpu_tensor *original,
    size_t original_offset, h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
    const h3_gpu_tensor *row_map, uint32_t input_rows, uint32_t rows,
    uint32_t baseline_rows, uint32_t width, uint32_t slots, uint32_t shift_slot,
    uint32_t scale_slot, float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 10, residual, output, input, original,
                               baseline, baseline_indices, pairs, norm_weight,
                               modulation, row_map))
        return 0;
    (void)input_rows;
    (void)baseline_rows;
    gpu->args->sample_offset = (uint32_t)input_offset;
    gpu->args->tokens = (uint32_t)original_offset;
    gpu->args->vocab_size = (uint32_t)baseline_offset;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->slots = slots;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[10] = {
        input,    pairs,       residual,   baseline, baseline_indices,
        original, norm_weight, modulation, row_map,  output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_TOKEN_POOL_ADALN_BF16, tensors, 10);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_TOKEN_POOL_ADALN_BF16, set,
                            rows, 1, 1);
}

int h3_gpu_token_expand_delta_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *original,
    size_t original_offset, const h3_gpu_tensor *reduced,
    const h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
    uint32_t width, uint32_t exact_prefix_rows, float update_scale) {
    if (!h3_cuda_check_tensors(gpu, 6, output, original, reduced, baseline,
                               baseline_indices, parents))
        return 0;
    (void)reduced_rows;
    (void)baseline_rows;
    gpu->args->tokens = (uint32_t)original_offset;
    gpu->args->vocab_size = (uint32_t)baseline_offset;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->elements = exact_prefix_rows;
    gpu->args->left_scale = update_scale;
    const h3_gpu_tensor *tensors[6] = {original,         reduced, baseline,
                                       baseline_indices, parents, output};
    h3_cuda_set *set = h3_cuda_prepare(
        gpu, H3_CUDA_KERNEL_TOKEN_EXPAND_DELTA_BF16, tensors, 6);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_TOKEN_EXPAND_DELTA_BF16, set,
                            (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_token_expand_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *original, size_t original_offset,
    const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline,
    size_t baseline_offset, const h3_gpu_tensor *baseline_indices,
    const h3_gpu_tensor *parents, const h3_gpu_tensor *norm_weight,
    const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
    uint32_t width, uint32_t exact_prefix_rows, float update_scale,
    uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 10, residual, output, original, reduced,
                               baseline, baseline_indices, parents, norm_weight,
                               modulation, row_map))
        return 0;
    (void)reduced_rows;
    (void)baseline_rows;
    gpu->args->tokens = (uint32_t)original_offset;
    gpu->args->vocab_size = (uint32_t)baseline_offset;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->elements = exact_prefix_rows;
    gpu->args->left_scale = update_scale;
    gpu->args->slots = slots;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[10] = {
        original, reduced,     baseline,   baseline_indices, parents,
        residual, norm_weight, modulation, row_map,          output};
    h3_cuda_set *set = h3_cuda_prepare(
        gpu, H3_CUDA_KERNEL_TOKEN_EXPAND_ADALN_BF16, tensors, 10);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_TOKEN_EXPAND_ADALN_BF16, set,
                            rows, 1, 1);
}

/* ------------------------------------------------------------ qwen text */

int h3_gpu_text_qk_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query_output, h3_gpu_tensor *key_output,
    const h3_gpu_tensor *query_input, const h3_gpu_tensor *key_input,
    const h3_gpu_tensor *q_weight, const h3_gpu_tensor *k_weight,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t sequence, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_dim, float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 8, query_output, key_output, query_input,
                               key_input, q_weight, k_weight, rope_cos,
                               rope_sin))
        return 0;
    gpu->args->rows = sequence;
    gpu->args->width = query_heads;
    gpu->args->input_dim = kv_heads;
    gpu->args->output_dim = head_dim;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[8] = {query_input,  key_input, q_weight,
                                       k_weight,     rope_cos,  rope_sin,
                                       query_output, key_output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_TEXT_QK_ROPE_BF16, tensors, 8);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_TEXT_QK_ROPE_BF16, set,
                            (head_dim + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS,
                            query_heads, sequence);
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32, uint32_t sequence,
                          uint32_t query_heads, uint32_t kv_heads,
                          uint32_t head_dim) {
    if (!h3_cuda_check_tensors(gpu, 4, query, key, rope_cos_f32, rope_sin_f32))
        return 0;
    gpu->args->rows = sequence;
    gpu->args->width = query_heads;
    gpu->args->input_dim = kv_heads;
    gpu->args->output_dim = head_dim;
    uint32_t maximum_heads = query_heads > kv_heads ? query_heads : kv_heads;
    const h3_gpu_tensor *tensors[4] = {query, key, rope_cos_f32, rope_sin_f32};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_ROPE_TEXT_BF16, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_ROPE_TEXT_BF16, set,
                            (sequence + 15) / 16, (maximum_heads + 15) / 16, 1);
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale) {
    if (!h3_cuda_check_tensors(gpu, 4, output, query, key, value))
        return 0;
    gpu->args->rows = sequence;
    gpu->args->width = query_heads;
    gpu->args->input_dim = kv_heads;
    gpu->args->output_dim = head_dim;
    gpu->args->left_scale = scale;
    const h3_gpu_tensor *tensors[4] = {query, key, value, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_GQA_CAUSAL_BF16, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_GQA_CAUSAL_BF16, set, sequence,
                            query_heads, 1);
}

void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    if (!gpu || !label || !*label)
        return;
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%s", label);
}

void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    if (!gpu || !phase || !*phase || !h3_cuda_profile_enabled())
        return;
    h3_cuda_profile_emit(gpu, phase, gpu->profile_mark_stats,
                         gpu->profile_mark_wall);
    gpu->profile_mark_stats = gpu->stats;
    gpu->profile_mark_wall = h3_cuda_now();
}

/* ------------------------------------------------------------------ */
/* Kernels not yet ported to CUDA. They fail cleanly with a
 * descriptive error; the callers' existing error paths handle it. */
static int h3_cuda_not_ported(h3_gpu *gpu, const char *name) {
    h3_cuda_set_error(gpu, "kernel %s is not ported to the CUDA backend yet",
                      name);
    return 0;
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    if (!h3_cuda_check_tensors(gpu, 3, output, input, weight))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if ((size_t)rows * input_dim > h3_gpu_tensor_elements(input) ||
        (size_t)output_dim * input_dim > h3_gpu_tensor_elements(weight) ||
        (size_t)rows * output_dim > h3_gpu_tensor_elements(output) ||
        (bias && output_dim > h3_gpu_tensor_elements(bias))) {
        h3_cuda_set_error(gpu, "linear f32 tensor size mismatch");
        return 0;
    }
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    gpu->args->has_bias = bias ? 1u : 0u;
    h3_cuda_accel_stats accelerated = {0};
    int accelerated_linear = 0;
    if (!h3_cuda_accel_linear_f32(
            gpu->accel, (void *)gpu->stream, output->data, input->data,
            weight->data, bias ? bias->data : NULL, rows, input_dim,
            output_dim, &accelerated_linear, &accelerated, gpu->error,
            sizeof(gpu->error)))
        return 0;
    h3_cuda_account_accel(gpu, accelerated);
    if (accelerated_linear)
        return 1;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = {input, weight, bias_buffer, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_LINEAR_F32, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_LINEAR_F32, set,
                            (output_dim + 15) / 16, (rows + 15) / 16, 1);
}

static int h3_cuda_copy(h3_gpu *gpu, h3_gpu_tensor *destination,
                        size_t destination_offset, const h3_gpu_tensor *source,
                        size_t source_offset, size_t elements,
                        h3_gpu_dtype dtype) {
    if (!h3_cuda_check_tensors(gpu, 2, destination, source))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if (destination->dtype != dtype || source->dtype != dtype) {
        h3_cuda_set_error(gpu, "copy dtype mismatch");
        return 0;
    }
    if (source_offset > source->elements ||
        elements > source->elements - source_offset ||
        destination_offset > destination->elements ||
        elements > destination->elements - destination_offset) {
        h3_cuda_set_error(gpu, "invalid CUDA copy range");
        return 0;
    }
    size_t element_size = h3_cuda_dtype_size(dtype);
    if (elements &&
        !h3_cuda_result(gpu,
                        cudaMemcpyAsync((uint8_t *)destination->data +
                                            destination_offset * element_size,
                                        (const uint8_t *)source->data +
                                            source_offset * element_size,
                                        elements * element_size,
                                        cudaMemcpyDeviceToDevice, gpu->stream),
                        "cudaMemcpyAsync device to device"))
        return 0;
    gpu->stats.blit_copies++;
    return 1;
}

int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset, const h3_gpu_tensor *source,
                     size_t source_offset, size_t elements) {
    return h3_cuda_copy(gpu, destination, destination_offset, source,
                        source_offset, elements, H3_GPU_BF16);
}

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset, const h3_gpu_tensor *source,
                    size_t source_offset, size_t elements) {
    return h3_cuda_copy(gpu, destination, destination_offset, source,
                        source_offset, elements, H3_GPU_F32);
}

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    (void)gpu;
    (void)output;
    (void)input;
    (void)norm_weight;
    (void)modulation;
    (void)row_map;
    (void)rows;
    (void)width;
    (void)slots;
    (void)shift_slot;
    (void)scale_slot;
    (void)epsilon;
    return h3_cuda_not_ported(gpu, "h3_gpu_adaln_f32");
}

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width,
                    uint32_t slots, uint32_t gate_slot) {
    (void)gpu;
    (void)output;
    (void)residual;
    (void)branch;
    (void)modulation;
    (void)row_map;
    (void)rows;
    (void)width;
    (void)slots;
    (void)gate_slot;
    return h3_cuda_not_ported(gpu, "h3_gpu_gate_f32");
}

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
                        h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim, uint32_t rope_half,
                        float epsilon) {
    (void)gpu;
    (void)query;
    (void)key;
    (void)value;
    (void)qkv;
    (void)q_norm;
    (void)k_norm;
    (void)rope_cos;
    (void)rope_sin;
    (void)sequence;
    (void)heads;
    (void)head_dim;
    (void)rope_half;
    (void)epsilon;
    return h3_cuda_not_ported(gpu, "h3_gpu_qkv_rope_f32");
}

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    if (!h3_cuda_check_tensors(gpu, 4, output, query, key, value))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t required = (size_t)sequence * heads * head_dim;
    if (required > h3_gpu_tensor_elements(output) ||
        required > h3_gpu_tensor_elements(query) ||
        required > h3_gpu_tensor_elements(key) ||
        required > h3_gpu_tensor_elements(value)) {
        h3_cuda_set_error(gpu, "F32 SDPA tensor size mismatch");
        return 0;
    }
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->left_scale = scale;
    /* Long sequences use the tiled online-softmax accelerator; unsupported
     * dimensions retain the generated correctness kernels. */
    h3_cuda_accel_stats accelerated = {0};
    int accelerated_attention = 0;
    if (!h3_cuda_accel_sdpa_f32(
            gpu->accel, (void *)gpu->stream, output->data, query->data,
            key->data, value->data, sequence, heads, head_dim, scale,
            &accelerated_attention, &accelerated, gpu->error,
            sizeof(gpu->error)))
        return 0;
    h3_cuda_account_accel(gpu, accelerated);
    if (accelerated_attention)
        return 1;
    h3_cuda_kernel kernel = (sequence >= 128 && head_dim <= 128)
                                ? H3_CUDA_KERNEL_SDPA_FLASH_F32
                                : H3_CUDA_KERNEL_SDPA_F32;
    const h3_gpu_tensor *tensors[4] = {query, key, value, output};
    h3_cuda_set *set = h3_cuda_prepare(gpu, kernel, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, kernel, set, heads, sequence, 1);
}

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    if (!h3_cuda_check_tensors(gpu, 2, output, fused))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if ((size_t)rows * width * 2 > h3_gpu_tensor_elements(fused) ||
        (size_t)rows * width > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "swiglu f32 tensor size mismatch");
        return 0;
    }
    gpu->args->rows = rows;
    gpu->args->width = width;
    const h3_gpu_tensor *tensors[2] = {fused, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SWIGLU_F32, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_SWIGLU_F32, set,
                            (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    if (!h3_cuda_check_tensors(gpu, 4, output, residual, branch, scale))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t count = (size_t)rows * width;
    if (count > h3_gpu_tensor_elements(residual) ||
        count > h3_gpu_tensor_elements(branch) ||
        width > h3_gpu_tensor_elements(scale) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "scale-add tensor size mismatch");
        return 0;
    }
    gpu->args->rows = rows;
    gpu->args->width = width;
    const h3_gpu_tensor *tensors[4] = {residual, branch, scale, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SCALE_ADD_F32, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_SCALE_ADD_F32, set,
                            (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim,
                              uint32_t rope_half, float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 6, query, key, value, qkv, rope_cos,
                               rope_sin))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t inner = (size_t)heads * head_dim;
    if ((size_t)sequence * inner * 3 > h3_gpu_tensor_elements(qkv) ||
        (size_t)sequence * rope_half > h3_gpu_tensor_elements(rope_cos) ||
        (size_t)sequence * rope_half > h3_gpu_tensor_elements(rope_sin) ||
        (size_t)sequence * inner > h3_gpu_tensor_elements(query) ||
        (size_t)sequence * inner > h3_gpu_tensor_elements(key) ||
        (size_t)sequence * inner > h3_gpu_tensor_elements(value)) {
        h3_cuda_set_error(gpu, "video qkv rope tensor size mismatch");
        return 0;
    }
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->output_dim = rope_half;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[6] = {qkv,   rope_cos, rope_sin,
                                       query, key,      value};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_VIDEO_QKV_ROPE_F32, tensors, 6);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_VIDEO_QKV_ROPE_F32, set,
                            (head_dim + 15) / 16, (heads + 15) / 16, sequence);
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(gpu, output, input, weight, bias, batch,
                                    length, input_channels, output_channels,
                                    kernel, 1, padding, dilation);
}

int h3_gpu_conv1d_stride_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch,
    uint32_t length, uint32_t input_channels, uint32_t output_channels,
    uint32_t kernel, uint32_t stride, uint32_t padding, uint32_t dilation) {
    if (!h3_cuda_check_tensors(gpu, 3, output, input, weight))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || !dilation || (uint64_t)length + 2 * padding < effective) {
        h3_cuda_set_error(gpu, "conv1d invalid dimensions");
        return 0;
    }
    uint32_t output_length =
        (uint32_t)(((uint64_t)length + 2 * padding - effective) / stride + 1);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (input_count > h3_gpu_tensor_elements(input) ||
        weight_count > h3_gpu_tensor_elements(weight) ||
        output_count > h3_gpu_tensor_elements(output) ||
        (bias && output_channels > h3_gpu_tensor_elements(bias))) {
        h3_cuda_set_error(gpu, "conv1d tensor size mismatch");
        return 0;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = length;
    gpu->args->conv_in_channels = input_channels;
    gpu->args->conv_out_channels = output_channels;
    gpu->args->conv_kernel_width = kernel;
    gpu->args->conv_stride_width = stride;
    gpu->args->conv_stride_height = padding;
    gpu->args->conv_stride_depth = dilation;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = {input, weight, bias_buffer, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_CONV1D_STRIDE_F32, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_CONV1D_STRIDE_F32, set,
                            (output_channels + 15) / 16,
                            (output_length + 15) / 16, batch);
}

int h3_gpu_conv_transpose1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input,
                                const h3_gpu_tensor *weight,
                                const h3_gpu_tensor *bias, uint32_t batch,
                                uint32_t length, uint32_t input_channels,
                                uint32_t output_channels, uint32_t kernel,
                                uint32_t stride, uint32_t padding) {
    if (!h3_cuda_check_tensors(gpu, 3, output, input, weight))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || (uint64_t)(length - 1) * stride + kernel < 2 * padding) {
        h3_cuda_set_error(gpu, "conv-transpose1d invalid dimensions");
        return 0;
    }
    uint32_t output_length =
        (uint32_t)((uint64_t)(length - 1) * stride + kernel - 2 * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (input_count > h3_gpu_tensor_elements(input) ||
        weight_count > h3_gpu_tensor_elements(weight) ||
        output_count > h3_gpu_tensor_elements(output) ||
        (bias && output_channels > h3_gpu_tensor_elements(bias))) {
        h3_cuda_set_error(gpu, "conv-transpose1d tensor size mismatch");
        return 0;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = length;
    gpu->args->conv_in_channels = input_channels;
    gpu->args->conv_out_channels = output_channels;
    gpu->args->conv_kernel_width = kernel;
    gpu->args->conv_stride_width = stride;
    gpu->args->conv_stride_height = padding;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = {input, weight, bias_buffer, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_CONV_TRANSPOSE1D_F32, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_CONV_TRANSPOSE1D_F32, set,
                            (output_channels + 15) / 16,
                            (output_length + 15) / 16, batch);
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude, uint32_t outer,
                           uint32_t inner) {
    if (!h3_cuda_check_tensors(gpu, 3, output, vector, magnitude))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if ((size_t)outer * inner > h3_gpu_tensor_elements(vector) ||
        outer > h3_gpu_tensor_elements(magnitude) ||
        (size_t)outer * inner > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "weight-norm tensor size mismatch");
        return 0;
    }
    gpu->args->rows = outer;
    gpu->args->width = inner;
    const h3_gpu_tensor *tensors[3] = {vector, magnitude, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_WEIGHT_NORM_F32, tensors, 3);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_WEIGHT_NORM_F32, set,
                            (outer + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS, 1,
                            1);
}

int h3_gpu_alias_free_snake_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input,
                                const h3_gpu_tensor *alpha_log,
                                const h3_gpu_tensor *beta_log,
                                const h3_gpu_tensor *upsample_filter,
                                const h3_gpu_tensor *downsample_filter,
                                uint32_t batch, uint32_t length,
                                uint32_t channels) {
    if (!h3_cuda_check_tensors(gpu, 6, output, input, alpha_log, beta_log,
                               upsample_filter, downsample_filter))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t count = (size_t)batch * length * channels;
    if (count > h3_gpu_tensor_elements(input) ||
        channels > h3_gpu_tensor_elements(alpha_log) ||
        channels > h3_gpu_tensor_elements(beta_log) ||
        12 > h3_gpu_tensor_elements(upsample_filter) ||
        12 > h3_gpu_tensor_elements(downsample_filter) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "alias-free snake tensor size mismatch");
        return 0;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = length;
    gpu->args->conv_in_channels = channels;
    const h3_gpu_tensor *tensors[6] = {
        input, alpha_log, beta_log, upsample_filter, downsample_filter, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_ALIAS_FREE_SNAKE_F32, tensors, 6);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_ALIAS_FREE_SNAKE_F32, set,
                            (channels + 15) / 16, (length + 15) / 16, batch);
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input, const h3_gpu_tensor *alpha,
                       uint32_t batch, uint32_t length, uint32_t channels) {
    if (!h3_cuda_check_tensors(gpu, 3, output, input, alpha))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t count = (size_t)batch * length * channels;
    if (count > h3_gpu_tensor_elements(input) ||
        channels > h3_gpu_tensor_elements(alpha) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "snake1d tensor size mismatch");
        return 0;
    }
    gpu->args->elements = (uint32_t)count;
    gpu->args->conv_in_channels = channels;
    const h3_gpu_tensor *tensors[3] = {input, alpha, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SNAKE1D_F32, tensors, 3);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(
        gpu, H3_CUDA_KERNEL_SNAKE1D_F32, set,
        (uint32_t)((count + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS), 1, 1);
}

int h3_gpu_audio_qkv_split_f32(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value,
    const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_bias,
    const h3_gpu_tensor *k_bias, const h3_gpu_tensor *v_bias, uint32_t batch,
    uint32_t length, uint32_t heads, uint32_t head_dim) {
    if (!h3_cuda_check_tensors(gpu, 7, query, key, value, qkv, q_bias, k_bias,
                               v_bias))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (count * 3 > h3_gpu_tensor_elements(qkv) ||
        width > h3_gpu_tensor_elements(q_bias) ||
        width > h3_gpu_tensor_elements(k_bias) ||
        width > h3_gpu_tensor_elements(v_bias) ||
        count > h3_gpu_tensor_elements(query) ||
        count > h3_gpu_tensor_elements(key) ||
        count > h3_gpu_tensor_elements(value)) {
        h3_cuda_set_error(gpu, "audio qkv split tensor size mismatch");
        return 0;
    }
    gpu->args->elements = (uint32_t)count;
    gpu->args->conv_width = heads;
    gpu->args->conv_height = head_dim;
    const h3_gpu_tensor *tensors[7] = {qkv,   q_bias, k_bias, v_bias,
                                       query, key,    value};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_AUDIO_QKV_SPLIT_F32, tensors, 7);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(
        gpu, H3_CUDA_KERNEL_AUDIO_QKV_SPLIT_F32, set,
        (uint32_t)((count + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS), 1, 1);
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t batch,
                           uint32_t sequence, uint32_t heads, uint32_t head_dim,
                           float scale) {
    if (!h3_cuda_check_tensors(gpu, 4, output, query, key, value))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t count = (size_t)batch * sequence * heads * head_dim;
    if (count > h3_gpu_tensor_elements(query) ||
        count > h3_gpu_tensor_elements(key) ||
        count > h3_gpu_tensor_elements(value) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "causal sdpa tensor size mismatch");
        return 0;
    }
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->left_scale = scale;
    gpu->args->conv_batch = batch;
    const h3_gpu_tensor *tensors[4] = {query, key, value, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SDPA_CAUSAL_F32, tensors, 4);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_SDPA_CAUSAL_F32, set, heads,
                            sequence, batch);
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                    const h3_gpu_tensor *attended,
                                    uint32_t batch, uint32_t length,
                                    uint32_t heads, uint32_t head_dim,
                                    uint32_t output_dim) {
    if (!h3_cuda_check_tensors(gpu, 2, output, attended))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if (!output_dim || head_dim % output_dim) {
        h3_cuda_set_error(gpu, "audio pool invalid dimensions");
        return 0;
    }
    size_t count = (size_t)batch * length * output_dim;
    if ((size_t)batch * length * heads * head_dim >
            h3_gpu_tensor_elements(attended) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "audio pool tensor size mismatch");
        return 0;
    }
    gpu->args->elements = (uint32_t)count;
    gpu->args->conv_width = heads;
    gpu->args->conv_height = head_dim;
    gpu->args->conv_in_channels = output_dim;
    const h3_gpu_tensor *tensors[2] = {attended, output};
    h3_cuda_set *set = h3_cuda_prepare(
        gpu, H3_CUDA_KERNEL_AUDIO_ATTENTION_POOL_F32, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(
        gpu, H3_CUDA_KERNEL_AUDIO_ATTENTION_POOL_F32, set,
        (uint32_t)((count + H3_CUDA_THREADS - 1) / H3_CUDA_THREADS), 1, 1);
}

int h3_gpu_vae_encoder_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *input, uint32_t batch,
                               uint32_t depth, uint32_t height, uint32_t width,
                               uint32_t channels, uint32_t depth_front,
                               uint32_t height_before, uint32_t height_after,
                               uint32_t width_before, uint32_t width_after) {
    if (!h3_cuda_check_tensors(gpu, 2, output, input))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * (depth + depth_front) *
                          (height + height_before + height_after) *
                          (width + width_before + width_after) * channels;
    if (input_count > h3_gpu_tensor_elements(input) ||
        output_count > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "vae pad tensor size mismatch");
        return 0;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = depth;
    gpu->args->conv_height = height;
    gpu->args->conv_width = width;
    gpu->args->conv_in_channels = channels;
    gpu->args->pad_depth_front = depth_front;
    gpu->args->pad_height_before = height_before;
    gpu->args->pad_height_after = height_after;
    gpu->args->pad_width_before = width_before;
    gpu->args->pad_width_after = width_after;
    const h3_gpu_tensor *tensors[2] = {input, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_VAE_ENCODER_PAD_F32, tensors, 2);
    if (set == NULL)
        return 0;
    uint32_t out_height = height + height_before + height_after;
    uint32_t out_width = width + width_before + width_after;
    uint32_t out_depth = depth + depth_front;
    uint32_t planes = batch * out_depth * out_height;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_VAE_ENCODER_PAD_F32, set,
                            (channels + 15) / 16, (out_width + 15) / 16,
                            planes);
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth,
                      uint32_t height, uint32_t width, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel_depth,
                      uint32_t kernel_height, uint32_t kernel_width,
                      uint32_t stride_depth, uint32_t stride_height,
                      uint32_t stride_width) {
    if (!h3_cuda_check_tensors(gpu, 3, output, input, weight))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if (!batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height || !kernel_width ||
        !stride_depth || !stride_height || !stride_width ||
        depth < kernel_depth || height < kernel_height ||
        width < kernel_width) {
        h3_cuda_set_error(gpu, "conv3d invalid dimensions");
        return 0;
    }
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    size_t input_count =
        (size_t)batch * depth * height * width * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels *
                          kernel_depth * kernel_height * kernel_width;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * output_channels;
    if (input_count > h3_gpu_tensor_elements(input) ||
        weight_count > h3_gpu_tensor_elements(weight) ||
        output_count > h3_gpu_tensor_elements(output) ||
        (bias && output_channels > h3_gpu_tensor_elements(bias))) {
        h3_cuda_set_error(gpu, "conv3d tensor size mismatch");
        return 0;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = depth;
    gpu->args->conv_height = height;
    gpu->args->conv_width = width;
    gpu->args->conv_in_channels = input_channels;
    gpu->args->conv_out_channels = output_channels;
    gpu->args->conv_kernel_depth = kernel_depth;
    gpu->args->conv_kernel_height = kernel_height;
    gpu->args->conv_kernel_width = kernel_width;
    gpu->args->conv_stride_depth = stride_depth;
    gpu->args->conv_stride_height = stride_height;
    gpu->args->conv_stride_width = stride_width;
    gpu->args->has_bias = bias ? 1u : 0u;
    h3_cuda_accel_stats accelerated = {0};
    int accelerated_conv = 0;
    if (!h3_cuda_accel_conv3d_f32(
            gpu->accel, (void *)gpu->stream, output->data, input->data,
            weight->data, bias ? bias->data : NULL, batch, depth, height,
            width, input_channels, output_channels, kernel_depth,
            kernel_height, kernel_width, stride_depth, stride_height,
            stride_width, &accelerated_conv, &accelerated, gpu->error,
            sizeof(gpu->error)))
        return 0;
    h3_cuda_account_accel(gpu, accelerated);
    if (accelerated_conv)
        return 1;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = {input, weight, bias_buffer, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_CONV3D_F32, tensors, 4);
    if (set == NULL)
        return 0;
    uint32_t planes = batch * output_depth * output_height;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_CONV3D_F32, set,
                            (output_width + 15) / 16, (output_height + 15) / 16,
                            planes);
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch,
    uint32_t depth, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t groups, float epsilon) {
    if (!h3_cuda_check_tensors(gpu, 4, output, input, weight, bias))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t count = (size_t)batch * depth * height * width * channels;
    if (count > h3_gpu_tensor_elements(input) ||
        channels > h3_gpu_tensor_elements(weight) ||
        channels > h3_gpu_tensor_elements(bias) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "vae group norm tensor size mismatch");
        return 0;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = depth;
    gpu->args->conv_height = height;
    gpu->args->conv_width = width;
    gpu->args->conv_in_channels = channels;
    gpu->args->conv_out_channels = groups;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[4] = {input, weight, bias, output};
    h3_cuda_set *set = h3_cuda_prepare(
        gpu, H3_CUDA_KERNEL_VAE_ENCODER_GROUP_NORM_SILU_F32, tensors, 4);
    if (set == NULL)
        return 0;
    uint32_t rows = batch * depth * groups;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_VAE_ENCODER_GROUP_NORM_SILU_F32,
                            set, rows, 1, 1);
}

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated, const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    /* The Metal NAX path is a tensor-ops matmul (Apple-only); CUDA uses
     * the same fused fc1 -> SwiGLU -> fc2 contract as h3_gpu_mlp_bf16,
     * writing through the caller-provided activated scratch when usable. */
    if (!h3_cuda_check_tensors(gpu, 4, output, input, fc1_weight, fc2_weight))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    h3_gpu_tensor *fc1 =
        h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim * 2);
    h3_gpu_tensor *scratch = activated;
    int own_scratch = 0;
    if (!scratch ||
        h3_gpu_tensor_elements(scratch) < (size_t)rows * hidden_dim) {
        scratch = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim);
        own_scratch = 1;
    }
    if (!fc1 || !scratch) {
        h3_gpu_tensor_free(fc1);
        if (own_scratch)
            h3_gpu_tensor_free(scratch);
        h3_cuda_set_error(gpu, "NAX MLP scratch allocation failed");
        return 0;
    }
    int ok = 0;
    if (h3_cuda_linear(gpu, fc1, input, fc1_weight, NULL, rows, input_dim,
                       hidden_dim * 2) == 1) {
        gpu->args->elements = rows * hidden_dim;
        gpu->args->width = hidden_dim;
        const h3_gpu_tensor *tensors[2] = {fc1, scratch};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SWIGLU_HALVES_BF16, tensors, 2);
        if (set != NULL &&
            h3_cuda_dispatch(
                gpu, H3_CUDA_KERNEL_SWIGLU_HALVES_BF16, set,
                (uint32_t)(((uint64_t)rows * hidden_dim + H3_CUDA_THREADS - 1) /
                           H3_CUDA_THREADS),
                1, 1) == 1 &&
            h3_cuda_linear(gpu, output, scratch, fc2_weight, NULL, rows,
                           hidden_dim, output_dim) == 1)
            ok = 1;
    }
    h3_gpu_tensor_free(fc1);
    if (own_scratch)
        h3_gpu_tensor_free(scratch);
    return ok;
}
int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {

    if (!h3_cuda_check_tensors(gpu, 3, output, scales, input))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if ((size_t)rows * columns > h3_gpu_tensor_elements(input) ||
        (size_t)rows * columns > h3_gpu_tensor_elements(output) ||
        rows > h3_gpu_tensor_elements(scales)) {
        h3_cuda_set_error(gpu, "int8 quantize tensor size mismatch");
        return 0;
    }
    gpu->args->rows = rows;
    gpu->args->width = columns;
    gpu->args->left_scale = 1.0f;
    const h3_gpu_tensor *tensors[3] = {input, output, scales};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, tensors, 3);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, set,
                            rows, 1, 1);
}
static int h3_cuda_accelerated_linear_int8(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim, int *used) {
    h3_cuda_accel_stats accelerated = {0};
    int ok = h3_cuda_accel_linear_int8_bf16(
        gpu->accel, (void *)gpu->stream, output->data, input->data,
        weight->data, input_scales->data, weight_scales->data, rows, input_dim,
        output_dim, used, &accelerated, gpu->error, sizeof(gpu->error));
    h3_cuda_account_accel(gpu, accelerated);
    return ok;
}

static int h3_cuda_accelerated_fc1_swiglu_int8(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t input_dim,
    uint32_t hidden_dim, int *used) {
    h3_cuda_accel_stats accelerated = {0};
    int ok = h3_cuda_accel_fc1_swiglu_int8_bf16(
        gpu->accel, (void *)gpu->stream, output->data, input->data,
        weight->data, input_scales->data, weight_scales->data, rows, input_dim,
        hidden_dim, used, &accelerated, gpu->error, sizeof(gpu->error));
    h3_cuda_account_accel(gpu, accelerated);
    return ok;
}

static int h3_cuda_accelerated_grouped_linear_int8(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim, uint32_t group_size, uint32_t groups, int *used) {
    h3_cuda_accel_stats accelerated = {0};
    int ok = h3_cuda_accel_linear_int8_grouped_bf16(
        gpu->accel, (void *)gpu->stream, output->data, input->data,
        weight->data, input_scales->data, weight_scales->data, rows, input_dim,
        output_dim, group_size, groups, used, &accelerated, gpu->error,
        sizeof(gpu->error));
    h3_cuda_account_accel(gpu, accelerated);
    return ok;
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {

    if (!h3_cuda_check_tensors(gpu, 4, output, quantized_input, input_scales,
                               input) ||
        !h3_cuda_check_tensors(gpu, 3, output, weight, weight_scales))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    (void)use_slower_uncached_int8_scales;
    if ((size_t)rows * input_dim > h3_gpu_tensor_elements(input) ||
        (size_t)rows * input_dim > h3_gpu_tensor_elements(quantized_input) ||
        rows > h3_gpu_tensor_elements(input_scales) ||
        (size_t)output_dim * input_dim > h3_gpu_tensor_elements(weight) ||
        output_dim > h3_gpu_tensor_elements(weight_scales) ||
        (size_t)rows * output_dim > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "int8 linear tensor size mismatch");
        return 0;
    }
    gpu->args->rows = rows;
    gpu->args->width = input_dim;
    gpu->args->left_scale = 1.0f;
    {
        const h3_gpu_tensor *q[3] = {input, quantized_input, input_scales};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, q, 3);
        if (set == NULL)
            return 0;
        if (h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, set,
                             rows, 1, 1) == 0)
            return 0;
    }
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    int accelerated = 0;
    if (!h3_cuda_accelerated_linear_int8(
            gpu, output, quantized_input, weight, input_scales, weight_scales,
            rows, input_dim, output_dim, &accelerated))
        return 0;
    if (accelerated)
        return 1;
    {
        const h3_gpu_tensor *l[5] = {quantized_input, weight, input_scales,
                                     weight_scales, output};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_LINEAR_INT8_BF16, l, 5);
        if (set == NULL)
            return 0;
        return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_LINEAR_INT8_BF16, set,
                                (output_dim + 15) / 16, (rows + 15) / 16, 1);
    }
}
int h3_gpu_linear_int8_head_major_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t output_dim) {
    if (!h3_cuda_check_tensors(gpu, 4, output, quantized_input, input_scales,
                               input) ||
        !h3_cuda_check_tensors(gpu, 3, output, weight, weight_scales))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if (!heads || !head_dim || head_dim > UINT32_MAX / heads) {
        h3_cuda_set_error(gpu, "head-major int8 linear invalid dimensions");
        return 0;
    }
    uint32_t input_dim = heads * head_dim;
    if ((size_t)rows * input_dim > h3_gpu_tensor_elements(input) ||
        (size_t)rows * input_dim > h3_gpu_tensor_elements(quantized_input) ||
        rows > h3_gpu_tensor_elements(input_scales) ||
        (size_t)output_dim * input_dim > h3_gpu_tensor_elements(weight) ||
        output_dim > h3_gpu_tensor_elements(weight_scales) ||
        (size_t)rows * output_dim > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "head-major int8 linear tensor size mismatch");
        return 0;
    }
    /* SDPA stores [head,row,dimension]. Gather and quantize directly into
     * the row-major activation consumed by the projection. */
    gpu->args->rows = rows;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->left_scale = 1.0f;
    {
        const h3_gpu_tensor *q[3] = {input, quantized_input, input_scales};
        h3_cuda_set *set = h3_cuda_prepare(
            gpu, H3_CUDA_KERNEL_QUANTIZE_HEAD_MAJOR_ROWS_BF16_I8, q, 3);
        if (set == NULL)
            return 0;
        if (h3_cuda_dispatch(gpu,
                             H3_CUDA_KERNEL_QUANTIZE_HEAD_MAJOR_ROWS_BF16_I8,
                             set, rows, 1, 1) == 0)
            return 0;
    }
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    int accelerated = 0;
    if (!h3_cuda_accelerated_linear_int8(
            gpu, output, quantized_input, weight, input_scales, weight_scales,
            rows, input_dim, output_dim, &accelerated))
        return 0;
    if (accelerated)
        return 1;
    {
        const h3_gpu_tensor *l[5] = {quantized_input, weight, input_scales,
                                     weight_scales, output};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_LINEAR_INT8_BF16, l, 5);
        if (set == NULL)
            return 0;
        return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_LINEAR_INT8_BF16, set,
                                (output_dim + 15) / 16, (rows + 15) / 16, 1);
    }
}

int h3_gpu_mlp_int8_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *activated,
    h3_gpu_tensor *quantized_activation, h3_gpu_tensor *activation_scales,
    const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight,
    const h3_gpu_tensor *fc1_scales, const h3_gpu_tensor *fc2_weight,
    const h3_gpu_tensor *fc2_scales, const h3_gpu_tensor *fc1_bf16,
    const h3_gpu_tensor *fc2_bf16, uint32_t rows, uint32_t input_dim,
    uint32_t hidden_dim, uint32_t output_dim, int use_slower_grouped_quantizer,
    int use_slower_dynamic_fc1_k, int use_int8_row_fc2,
    int input_is_quantized) {

    if (!h3_cuda_check_tensors(gpu, 5, output, activated, quantized_activation,
                               activation_scales, input) ||
        !h3_cuda_check_tensors(gpu, 4, output, fc1_weight, fc1_scales,
                               fc2_weight) ||
        !h3_cuda_check_tensors(gpu, 2, output, fc2_scales))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    (void)fc1_bf16;
    (void)fc2_bf16;
    (void)use_slower_grouped_quantizer;
    (void)use_slower_dynamic_fc1_k;
    uint32_t groups = use_int8_row_fc2 ? 0u : hidden_dim / 1024u;
    if (!use_int8_row_fc2 && hidden_dim % 1024u) {
        h3_cuda_set_error(gpu,
                          "grouped int8 FC2 requires hidden_dim % 1024 == 0");
        return 0;
    }
    size_t fc1_count = (size_t)hidden_dim * 2 * input_dim;
    size_t fc2_count = (size_t)output_dim * hidden_dim;
    if ((size_t)rows * input_dim > h3_gpu_tensor_elements(input) ||
        fc1_count > h3_gpu_tensor_elements(fc1_weight) ||
        hidden_dim * 2 > h3_gpu_tensor_elements(fc1_scales) ||
        fc2_count > h3_gpu_tensor_elements(fc2_weight) ||
        output_dim > h3_gpu_tensor_elements(fc2_scales) ||
        (size_t)rows * (use_int8_row_fc2 ? input_dim : hidden_dim) >
            h3_gpu_tensor_elements(quantized_activation) ||
        (size_t)rows * (use_int8_row_fc2 ? 1u : groups) >
            h3_gpu_tensor_elements(activation_scales) ||
        (size_t)rows * hidden_dim > h3_gpu_tensor_elements(activated) ||
        (size_t)rows * output_dim > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "int8 MLP tensor size mismatch");
        return 0;
    }
    /* FC1 input: reuse the caller's quantized buffer when prequantized. */
    const h3_gpu_tensor *fc1_input = quantized_activation;
    if (!input_is_quantized) {
        gpu->args->rows = rows;
        gpu->args->width = input_dim;
        gpu->args->left_scale = 1.0f;
        const h3_gpu_tensor *q[3] = {input, quantized_activation,
                                     activation_scales};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, q, 3);
        if (set == NULL)
            return 0;
        if (h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, set,
                             rows, 1, 1) == 0)
            return 0;
    }
    /* Fused FC1 + SwiGLU (gate | up halves). cuBLASLt computes both exact
     * INT32 projections in bounded chunks; the portable tile remains the
     * fallback for unaligned or unsupported dimensions. */
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = hidden_dim;
    int accelerated_fc1 = 0;
    if (!h3_cuda_accelerated_fc1_swiglu_int8(
            gpu, activated, fc1_input, fc1_weight, activation_scales,
            fc1_scales, rows, input_dim, hidden_dim, &accelerated_fc1))
        return 0;
    if (!accelerated_fc1) {
        const h3_gpu_tensor *f[5] = {fc1_input, fc1_weight, activation_scales,
                                     fc1_scales, activated};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_FC1_SWIGLU_INT8_BF16, f, 5);
        if (set == NULL)
            return 0;
        if (h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_FC1_SWIGLU_INT8_BF16, set,
                             (hidden_dim + 15) / 16, (rows + 15) / 16, 1) == 0)
            return 0;
    }
    if (use_int8_row_fc2) {
        /* Per-row activation scales for FC2 (Metal row-fc2 path). */
        gpu->args->rows = rows;
        gpu->args->width = hidden_dim;
        gpu->args->left_scale = 1.0f;
        {
            const h3_gpu_tensor *q[3] = {activated, quantized_activation,
                                         activation_scales};
            h3_cuda_set *set = h3_cuda_prepare(
                gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, q, 3);
            if (set == NULL)
                return 0;
            if (h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, set,
                                 rows, 1, 1) == 0)
                return 0;
        }
        gpu->args->rows = rows;
        gpu->args->input_dim = hidden_dim;
        gpu->args->output_dim = output_dim;
        int accelerated_fc2 = 0;
        if (!h3_cuda_accelerated_linear_int8(
                gpu, output, quantized_activation, fc2_weight,
                activation_scales, fc2_scales, rows, hidden_dim, output_dim,
                &accelerated_fc2))
            return 0;
        if (accelerated_fc2)
            return 1;
        {
            const h3_gpu_tensor *l[5] = {quantized_activation, fc2_weight,
                                         activation_scales, fc2_scales, output};
            h3_cuda_set *set =
                h3_cuda_prepare(gpu, H3_CUDA_KERNEL_LINEAR_INT8_BF16, l, 5);
            if (set == NULL)
                return 0;
            return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_LINEAR_INT8_BF16, set,
                                    (output_dim + 15) / 16, (rows + 15) / 16,
                                    1);
        }
    }
    /* Default grouped FC2 (Metal grouped_nax path): one max-abs scale per
     * 1024-wide activation group, dequantized per group with a single
     * BF16 rounding. */
    gpu->args->rows = rows;
    gpu->args->width = hidden_dim;
    gpu->args->output_dim = groups;
    gpu->args->elements = 1024u;
    {
        const h3_gpu_tensor *q[3] = {activated, quantized_activation,
                                     activation_scales};
        h3_cuda_set *set = h3_cuda_prepare(
            gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_GROUPS_BF16_I8, q, 3);
        if (set == NULL)
            return 0;
        if (h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_GROUPS_BF16_I8,
                             set, rows, 1, 1) == 0)
            return 0;
    }
    gpu->args->rows = rows;
    gpu->args->input_dim = hidden_dim;
    gpu->args->output_dim = output_dim;
    gpu->args->width = 1024u;
    gpu->args->elements = groups;
    int accelerated_fc2 = 0;
    if (!h3_cuda_accelerated_grouped_linear_int8(
            gpu, output, quantized_activation, fc2_weight, activation_scales,
            fc2_scales, rows, hidden_dim, output_dim, 1024u, groups,
            &accelerated_fc2))
        return 0;
    if (accelerated_fc2)
        return 1;
    {
        const h3_gpu_tensor *l[5] = {quantized_activation, fc2_weight,
                                     activation_scales, fc2_scales, output};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_LINEAR_INT8_GROUPED_BF16, l, 5);
        if (set == NULL)
            return 0;
        return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_LINEAR_INT8_GROUPED_BF16,
                                set, (output_dim + 15) / 16, (rows + 15) / 16,
                                1);
    }
}

int h3_gpu_vision_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                h3_gpu_tensor *key, h3_gpu_tensor *value,
                                const h3_gpu_tensor *qkv,
                                const h3_gpu_tensor *rope_cos,
                                const h3_gpu_tensor *rope_sin,
                                uint32_t sequence, uint32_t heads,
                                uint32_t head_dim, uint32_t rope_half) {
    if (!h3_cuda_check_tensors(gpu, 6, query, key, value, qkv, rope_cos,
                               rope_sin))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (count * 3 > h3_gpu_tensor_elements(qkv) ||
        rope_count > h3_gpu_tensor_elements(rope_cos) ||
        rope_count > h3_gpu_tensor_elements(rope_sin) ||
        count > h3_gpu_tensor_elements(query) ||
        count > h3_gpu_tensor_elements(key) ||
        count > h3_gpu_tensor_elements(value)) {
        h3_cuda_set_error(gpu, "vision qkv rope tensor size mismatch");
        return 0;
    }
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->output_dim = rope_half;
    const h3_gpu_tensor *tensors[6] = {qkv,   rope_cos, rope_sin,
                                       query, key,      value};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_VISION_QKV_ROPE_BF16, tensors, 6);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_VISION_QKV_ROPE_BF16, set,
                            (head_dim + 15) / 16, (heads + 15) / 16, sequence);
}
int h3_gpu_gate_adaln_quantize_int8(
    h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *quantized_output,
    h3_gpu_tensor *quantized_scales, const h3_gpu_tensor *residual,
    const h3_gpu_tensor *branch, const h3_gpu_tensor *norm_weight,
    const h3_gpu_tensor *gate_modulation, const h3_gpu_tensor *norm_modulation,
    const h3_gpu_tensor *row_map, uint32_t rows, uint32_t padded_rows,
    uint32_t width, uint32_t slots, uint32_t gate_slot, uint32_t shift_slot,
    uint32_t scale_slot, float epsilon) {

    if (!h3_cuda_check_tensors(gpu, 9, gated_residual, quantized_output,
                               quantized_scales, residual, branch, norm_weight,
                               gate_modulation, norm_modulation, row_map))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    (void)padded_rows;
    size_t elements = (size_t)rows * width;
    if (!rows || !width || width > 5376 || gate_slot >= slots ||
        shift_slot >= slots || scale_slot >= slots ||
        elements > h3_gpu_tensor_elements(residual) ||
        elements > h3_gpu_tensor_elements(branch) ||
        width > h3_gpu_tensor_elements(norm_weight) ||
        rows > h3_gpu_tensor_elements(row_map) ||
        elements > h3_gpu_tensor_elements(gated_residual) ||
        elements > h3_gpu_tensor_elements(quantized_output) ||
        rows > h3_gpu_tensor_elements(quantized_scales)) {
        h3_cuda_set_error(gpu, "gate-AdaLN-quantize tensor size mismatch");
        return 0;
    }
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->slots = slots;
    gpu->args->gate_slot = gate_slot;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[9] = {
        residual,       branch,           gate_modulation,
        row_map,        norm_weight,      norm_modulation,
        gated_residual, quantized_output, quantized_scales};
    h3_cuda_set *set = h3_cuda_prepare(
        gpu, H3_CUDA_KERNEL_GATE_ADALN_QUANTIZE_INT8, tensors, 9);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_GATE_ADALN_QUANTIZE_INT8, set,
                            rows, 1, 1);
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value,
    h3_gpu_tensor *qkv, const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t rows,
    uint32_t input_dim, uint32_t heads, uint32_t head_dim, uint32_t rope_half,
    float epsilon) {
    /* Keep the portable BF16 projection followed by grouped norm/RoPE.
     * The CUDA accelerator currently specializes the resident INT8 QKV path. */
    uint32_t inner = heads * head_dim;
    if (h3_gpu_linear_bf16(gpu, qkv, input, weight, NULL, rows, input_dim,
                           inner * 3) == 0)
        return 0;
    return h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm,
                                        k_norm, rope_cos, rope_sin, rows, heads,
                                        head_dim, rope_half, epsilon);
}
int h3_gpu_grouped_qkv_linear_rope_int8(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value,
    h3_gpu_tensor *qkv_scratch, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *weight_scales, const h3_gpu_tensor *q_norm,
    const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
    const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim,
    uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon,
    int input_is_quantized, int use_slower_unfused_qkv_rope,
    int use_slower_scalar_qkv_rms, int use_slower_uncached_int8_scales) {

    if (!h3_cuda_check_tensors(gpu, 7, query, key, value, qkv_scratch,
                               quantized_input, input_scales, input) ||
        !h3_cuda_check_tensors(gpu, 5, query, weight, weight_scales, q_norm,
                               k_norm) ||
        !h3_cuda_check_tensors(gpu, 3, query, rope_cos, rope_sin))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    (void)use_slower_unfused_qkv_rope;
    (void)use_slower_scalar_qkv_rms;
    (void)use_slower_uncached_int8_scales;
    uint32_t inner = heads * head_dim;
    size_t projected = (size_t)rows * inner;
    if ((size_t)rows * input_dim > h3_gpu_tensor_elements(input) ||
        (size_t)rows * input_dim > h3_gpu_tensor_elements(quantized_input) ||
        rows > h3_gpu_tensor_elements(input_scales) ||
        (size_t)inner * 3 * input_dim > h3_gpu_tensor_elements(weight) ||
        inner * 3 > h3_gpu_tensor_elements(weight_scales) ||
        projected > h3_gpu_tensor_elements(query) ||
        projected > h3_gpu_tensor_elements(key) ||
        projected > h3_gpu_tensor_elements(value) ||
        projected * 3 > h3_gpu_tensor_elements(qkv_scratch)) {
        h3_cuda_set_error(gpu, "int8 QKV projection tensor size mismatch");
        return 0;
    }
    if (!input_is_quantized) {
        gpu->args->rows = rows;
        gpu->args->width = input_dim;
        gpu->args->left_scale = 1.0f;
        const h3_gpu_tensor *q[3] = {input, quantized_input, input_scales};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, q, 3);
        if (set == NULL)
            return 0;
        if (h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_QUANTIZE_ROWS_BF16_I8, set,
                             rows, 1, 1) == 0)
            return 0;
    }
    /* Reuse the caller's persistent BF16 QKV arena. Allocating this per block
     * creates a large transient peak at full video resolution. */
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = inner * 3;
    int accelerated = 0;
    if (!h3_cuda_accelerated_linear_int8(
            gpu, qkv_scratch, quantized_input, weight, input_scales,
            weight_scales, rows, input_dim, inner * 3, &accelerated))
        return 0;
    if (!accelerated) {
        const h3_gpu_tensor *l[5] = {quantized_input, weight, input_scales,
                                     weight_scales, qkv_scratch};
        h3_cuda_set *set =
            h3_cuda_prepare(gpu, H3_CUDA_KERNEL_LINEAR_INT8_BF16, l, 5);
        if (set == NULL)
            return 0;
        if (h3_cuda_dispatch(gpu, H3_CUDA_KERNEL_LINEAR_INT8_BF16, set,
                             (inner * 3 + 15) / 16, (rows + 15) / 16, 1) == 0)
            return 0;
    }
    return h3_gpu_grouped_qkv_rope_bf16(
        gpu, query, key, value, qkv_scratch, q_norm, k_norm, rope_cos, rope_sin,
        rows, heads, head_dim, rope_half, epsilon);
}

int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    if (!h3_cuda_check_tensors(gpu, 2, output, fused))
        return 0;
    if (!h3_cuda_require_command(gpu))
        return 0;
    if ((size_t)rows * width * 2 > h3_gpu_tensor_elements(fused) ||
        (size_t)rows * width > h3_gpu_tensor_elements(output)) {
        h3_cuda_set_error(gpu, "swiglu bf16 tensor size mismatch");
        return 0;
    }
    gpu->args->elements = rows * width;
    gpu->args->rows = rows;
    gpu->args->width = width;
    const h3_gpu_tensor *tensors[2] = {fused, output};
    h3_cuda_set *set =
        h3_cuda_prepare(gpu, H3_CUDA_KERNEL_SWIGLU_HALVES_BF16, tensors, 2);
    if (set == NULL)
        return 0;
    return h3_cuda_dispatch(
        gpu, H3_CUDA_KERNEL_SWIGLU_HALVES_BF16, set,
        (uint32_t)(((uint64_t)rows * width + H3_CUDA_THREADS - 1) /
                   H3_CUDA_THREADS),
        1, 1);
}

/* Device probe used by h3_metal_probe() on non-Apple platforms. */
int h3_cuda_probe(h3_device_info *info, char *error, size_t error_size) {
    if (!info)
        return 0;
    memset(info, 0, sizeof(*info));
    int device = 0;
    if (!h3_cuda_selected_device(error, error_size, &device))
        return 0;
    struct cudaDeviceProp properties;
    cudaError_t result = cudaGetDeviceProperties(&properties, device);
    if (result != cudaSuccess) {
        if (error && error_size)
            snprintf(error, error_size, "cudaGetDeviceProperties failed: %s",
                     cudaGetErrorString(result));
        return 0;
    }
    snprintf(info->name, sizeof(info->name), "%s", properties.name);
    snprintf(info->architecture, sizeof(info->architecture), "CUDA sm_%d%d",
             properties.major, properties.minor);
    info->physical_memory = (uint64_t)properties.totalGlobalMem;
    info->recommended_working_set = (uint64_t)properties.totalGlobalMem;
    info->max_buffer_length = (uint64_t)properties.totalGlobalMem;
    info->unified_memory = properties.integrated ? 1 : 0;
    return 1;
}
