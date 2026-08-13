/* cuBLASLt-backed hot paths for the native CUDA backend.
 *
 * cuBLASLt performs exact int8 x int8 -> int32 Tensor Core GEMMs. Small CUDA
 * epilogues apply H3's per-row/per-channel scales and its precise BF16/SwiGLU
 * boundaries. Output columns are chunked through a bounded scratch arena, so
 * accelerating the 14K-wide MLP does not require a multi-gigabyte INT32
 * intermediate on 16 GB cards.
 */
#include "h3_cuda_accel.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#if defined(H3_HAVE_CUDNN)
#include <cudnn.h>
#endif

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <new>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_LT_PLAN_CACHE 512u
#define H3_LT_DEFAULT_SCRATCH_MIB 64u
#define H3_LT_MIN_SCRATCH_MIB 8u
#define H3_LT_MAX_SCRATCH_MIB 512u
#define H3_CUDNN_DEFAULT_WORKSPACE_MIB 256u
#define H3_CUDNN_MAX_WORKSPACE_MIB 1024u
#define H3_LT_THREADS 256u

typedef struct {
    int occupied;
    int supported;
    uint32_t rows;
    uint32_t columns;
    uint32_t inner;
    uint32_t input_stride;
    uint32_t weight_stride;
    cublasLtMatrixLayout_t input_layout;
    cublasLtMatrixLayout_t weight_layout;
    cublasLtMatrixLayout_t output_layout;
    cublasLtMatmulAlgo_t algorithm;
} h3_lt_plan;

struct h3_cuda_accel {
    int available;
    int scratch_failed;
    int compute_major;
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t operation;
    cublasLtMatmulDesc_t operation_f32;
    cublasLtMatmulPreference_t preference;
    h3_lt_plan plans[H3_LT_PLAN_CACHE];
    h3_lt_plan f32_plans[H3_LT_PLAN_CACHE];
    uint32_t plan_count;
    uint32_t f32_plan_count;
    void *scratch;
    size_t scratch_bytes;
    size_t scratch_limit;
#if defined(H3_HAVE_CUDNN)
    cudnnHandle_t cudnn;
    void *cudnn_workspace;
    size_t cudnn_workspace_bytes;
    size_t cudnn_workspace_limit;
#endif
};

static void h3_lt_error(char *error, size_t error_size, const char *format,
                        ...) {
    if (!error || !error_size)
        return;
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int h3_lt_disabled(void) {
    const char *value = getenv("H3_DISABLE_CUBLASLT");
    return value && *value && strcmp(value, "0") != 0;
}

static size_t h3_lt_scratch_limit(void) {
    unsigned long long mib = H3_LT_DEFAULT_SCRATCH_MIB;
    const char *value = getenv("H3_CUDA_LT_SCRATCH_MB");
    if (value && *value) {
        char *end = NULL;
        errno = 0;
        unsigned long long parsed = strtoull(value, &end, 10);
        if (!errno && end != value && !*end)
            mib = parsed;
    }
    if (mib < H3_LT_MIN_SCRATCH_MIB)
        mib = H3_LT_MIN_SCRATCH_MIB;
    if (mib > H3_LT_MAX_SCRATCH_MIB)
        mib = H3_LT_MAX_SCRATCH_MIB;
    return (size_t)mib * 1024u * 1024u;
}

#if defined(H3_HAVE_CUDNN)
static size_t h3_cudnn_workspace_limit(void) {
    unsigned long long mib = H3_CUDNN_DEFAULT_WORKSPACE_MIB;
    const char *value = getenv("H3_CUDA_CUDNN_WORKSPACE_MB");
    if (value && *value) {
        char *end = NULL;
        errno = 0;
        unsigned long long parsed = strtoull(value, &end, 10);
        if (!errno && end != value && !*end)
            mib = parsed;
    }
    if (mib > H3_CUDNN_MAX_WORKSPACE_MIB)
        mib = H3_CUDNN_MAX_WORKSPACE_MIB;
    return (size_t)mib * 1024u * 1024u;
}
#endif

static int h3_lt_ensure_scratch(h3_cuda_accel *accel) {
    if (!accel || !accel->available || accel->scratch_failed)
        return 0;
    if (accel->scratch)
        return 1;
    size_t candidate = accel->scratch_limit;
    while (candidate >= (size_t)H3_LT_MIN_SCRATCH_MIB * 1024u * 1024u) {
        cudaError_t result = cudaMalloc(&accel->scratch, candidate);
        if (result == cudaSuccess) {
            accel->scratch_bytes = candidate;
            return 1;
        }
        accel->scratch = NULL;
        (void)cudaGetLastError();
        candidate /= 2u;
    }
    accel->scratch_failed = 1;
    return 0;
}

static void h3_lt_destroy_plan(h3_lt_plan *plan) {
    if (!plan)
        return;
    if (plan->input_layout)
        (void)cublasLtMatrixLayoutDestroy(plan->input_layout);
    if (plan->weight_layout)
        (void)cublasLtMatrixLayoutDestroy(plan->weight_layout);
    if (plan->output_layout)
        (void)cublasLtMatrixLayoutDestroy(plan->output_layout);
    memset(plan, 0, sizeof(*plan));
}

h3_cuda_accel *h3_cuda_accel_create(int compute_major, int compute_minor) {
    (void)compute_minor;
    h3_cuda_accel *accel = new (std::nothrow) h3_cuda_accel();
    if (!accel)
        return NULL;
    memset(accel, 0, sizeof(*accel));
    accel->compute_major = compute_major;
    accel->scratch_limit = h3_lt_scratch_limit();
#if defined(H3_HAVE_CUDNN)
    accel->cudnn_workspace_limit = h3_cudnn_workspace_limit();
    if (compute_major >= 7)
        (void)cudnnCreate(&accel->cudnn);
#endif
    if (compute_major < 7)
        return accel;
    if (cublasLtCreate(&accel->handle) != CUBLAS_STATUS_SUCCESS)
        return accel;
    if (cublasLtMatmulDescCreate(&accel->operation, CUBLAS_COMPUTE_32I,
                                 CUDA_R_32I) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescCreate(&accel->operation_f32,
                                 CUBLAS_COMPUTE_32F_PEDANTIC,
                                 CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
        return accel;
    cublasOperation_t input_operation = CUBLAS_OP_N;
    cublasOperation_t weight_operation = CUBLAS_OP_T;
    if (cublasLtMatmulDescSetAttribute(
            accel->operation, CUBLASLT_MATMUL_DESC_TRANSA, &input_operation,
            sizeof(input_operation)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(
            accel->operation, CUBLASLT_MATMUL_DESC_TRANSB, &weight_operation,
            sizeof(weight_operation)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(
            accel->operation_f32, CUBLASLT_MATMUL_DESC_TRANSA,
            &input_operation, sizeof(input_operation)) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(
            accel->operation_f32, CUBLASLT_MATMUL_DESC_TRANSB,
            &weight_operation, sizeof(weight_operation)) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceCreate(&accel->preference) !=
            CUBLAS_STATUS_SUCCESS)
        return accel;
    size_t no_workspace = 0;
    if (cublasLtMatmulPreferenceSetAttribute(
            accel->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &no_workspace, sizeof(no_workspace)) != CUBLAS_STATUS_SUCCESS)
        return accel;
    accel->available = 1;
    return accel;
}

void h3_cuda_accel_free(h3_cuda_accel *accel) {
    if (!accel)
        return;
    if (accel->scratch)
        (void)cudaFree(accel->scratch);
#if defined(H3_HAVE_CUDNN)
    if (accel->cudnn_workspace)
        (void)cudaFree(accel->cudnn_workspace);
    if (accel->cudnn)
        (void)cudnnDestroy(accel->cudnn);
#endif
    for (uint32_t index = 0; index < accel->plan_count; index++)
        h3_lt_destroy_plan(&accel->plans[index]);
    for (uint32_t index = 0; index < accel->f32_plan_count; index++)
        h3_lt_destroy_plan(&accel->f32_plans[index]);
    if (accel->preference)
        (void)cublasLtMatmulPreferenceDestroy(accel->preference);
    if (accel->operation)
        (void)cublasLtMatmulDescDestroy(accel->operation);
    if (accel->operation_f32)
        (void)cublasLtMatmulDescDestroy(accel->operation_f32);
    if (accel->handle)
        (void)cublasLtDestroy(accel->handle);
    delete accel;
}

int h3_cuda_accel_available(const h3_cuda_accel *accel) {
    return accel && accel->available && !accel->scratch_failed &&
           !h3_lt_disabled();
}

size_t h3_cuda_accel_device_bytes(const h3_cuda_accel *accel) {
    if (!accel)
        return 0;
    size_t bytes = accel->scratch_bytes;
#if defined(H3_HAVE_CUDNN)
    if (bytes <= SIZE_MAX - accel->cudnn_workspace_bytes)
        bytes += accel->cudnn_workspace_bytes;
#endif
    return bytes;
}

/* 1 = plan available, 0 = unsupported/fallback, -1 = descriptor API error. */
static int h3_lt_get_typed_plan(
    h3_cuda_accel *accel, cublasLtMatmulDesc_t operation,
    cudaDataType_t input_type, cudaDataType_t output_type,
    h3_lt_plan plans[H3_LT_PLAN_CACHE], uint32_t *plan_count,
    uint32_t rows, uint32_t columns, uint32_t inner,
    uint32_t input_stride, uint32_t weight_stride, h3_lt_plan **result) {
    *result = NULL;
    for (uint32_t index = 0; index < *plan_count; index++) {
        h3_lt_plan *plan = &plans[index];
        if (plan->rows == rows && plan->columns == columns &&
            plan->inner == inner && plan->input_stride == input_stride &&
            plan->weight_stride == weight_stride) {
            if (!plan->supported)
                return 0;
            *result = plan;
            return 1;
        }
    }
    if (*plan_count >= H3_LT_PLAN_CACHE)
        return 0;
    h3_lt_plan *plan = &plans[(*plan_count)++];
    memset(plan, 0, sizeof(*plan));
    plan->occupied = 1;
    plan->rows = rows;
    plan->columns = columns;
    plan->inner = inner;
    plan->input_stride = input_stride;
    plan->weight_stride = weight_stride;
    cublasStatus_t status = cublasLtMatrixLayoutCreate(
        &plan->input_layout, input_type, rows, inner, input_stride);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &plan->weight_layout, input_type, columns, inner, weight_stride);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &plan->output_layout, output_type, rows, columns, columns);
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutSetAttribute(
            plan->input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutSetAttribute(
            plan->weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutSetAttribute(
            plan->output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    if (status != CUBLAS_STATUS_SUCCESS) {
        h3_lt_destroy_plan(plan);
        plan->occupied = 1;
        plan->rows = rows;
        plan->columns = columns;
        plan->inner = inner;
        plan->input_stride = input_stride;
        plan->weight_stride = weight_stride;
        return status == CUBLAS_STATUS_NOT_SUPPORTED ? 0 : -1;
    }
    cublasLtMatmulHeuristicResult_t heuristic[8];
    int returned = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        accel->handle, operation, plan->input_layout, plan->weight_layout,
        plan->output_layout, plan->output_layout, accel->preference, 8,
        heuristic, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        h3_lt_destroy_plan(plan);
        plan->occupied = 1;
        plan->rows = rows;
        plan->columns = columns;
        plan->inner = inner;
        plan->input_stride = input_stride;
        plan->weight_stride = weight_stride;
        return status == CUBLAS_STATUS_SUCCESS ||
                       status == CUBLAS_STATUS_NOT_SUPPORTED
                   ? 0
                   : -1;
    }
    plan->algorithm = heuristic[0].algo;
    plan->supported = 1;
    *result = plan;
    return 1;
}

static int h3_lt_get_plan(h3_cuda_accel *accel, uint32_t rows,
                          uint32_t columns, uint32_t inner,
                          uint32_t input_stride, uint32_t weight_stride,
                          h3_lt_plan **result) {
    return h3_lt_get_typed_plan(
        accel, accel->operation, CUDA_R_8I, CUDA_R_32I, accel->plans,
        &accel->plan_count, rows, columns, inner, input_stride, weight_stride,
        result);
}

static int h3_lt_get_f32_plan(h3_cuda_accel *accel, uint32_t rows,
                              uint32_t columns, uint32_t inner,
                              h3_lt_plan **result) {
    return h3_lt_get_typed_plan(
        accel, accel->operation_f32, CUDA_R_32F, CUDA_R_32F,
        accel->f32_plans, &accel->f32_plan_count, rows, columns, inner, inner,
        inner, result);
}

static int h3_lt_matmul(h3_cuda_accel *accel, cudaStream_t stream,
                        const int8_t *input, const int8_t *weight,
                        int32_t *output, h3_lt_plan *plan) {
    const int32_t alpha = 1;
    const int32_t beta = 0;
    cublasStatus_t status = cublasLtMatmul(
        accel->handle, accel->operation, &alpha, input, plan->input_layout,
        weight, plan->weight_layout, &beta, output, plan->output_layout, output,
        plan->output_layout, &plan->algorithm, NULL, 0, stream);
    if (status == CUBLAS_STATUS_SUCCESS)
        return 1;
    if (status == CUBLAS_STATUS_NOT_SUPPORTED ||
        status == CUBLAS_STATUS_INVALID_VALUE)
        return 0;
    return -1;
}

static int h3_lt_matmul_f32(h3_cuda_accel *accel, cudaStream_t stream,
                            const float *input, const float *weight,
                            float *output, h3_lt_plan *plan) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasLtMatmul(
        accel->handle, accel->operation_f32, &alpha, input,
        plan->input_layout, weight, plan->weight_layout, &beta, output,
        plan->output_layout, output, plan->output_layout, &plan->algorithm,
        NULL, 0, stream);
    if (status == CUBLAS_STATUS_SUCCESS)
        return 1;
    if (status == CUBLAS_STATUS_NOT_SUPPORTED ||
        status == CUBLAS_STATUS_INVALID_VALUE)
        return 0;
    return -1;
}

__global__ static void h3_lt_bias_f32(float *output, const float *bias,
                                      size_t elements,
                                      uint32_t output_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        output[index] += bias[index % output_dim];
}

__device__ __forceinline__ uint16_t h3_lt_f32_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

__device__ __forceinline__ float h3_lt_bf16_to_f32(uint16_t value) {
    return __uint_as_float((uint32_t)value << 16);
}

/* Flash-style attention tile: one block owns 16 query rows for one head.
 * Tensor Cores form each 16x16 QK tile; online softmax and P*V stay in F32 so
 * only the public output boundary rounds to BF16. Compared with the portable
 * kernel this replaces one reduction/barrier sequence per key with four
 * barriers per 16-key tile. */
template <int HEAD_DIM>
__global__ static void h3_lt_sdpa_bf16_kernel(
    uint16_t *output, const uint16_t *query, const uint16_t *key,
    const uint16_t *value, uint32_t sequence, uint32_t heads, float scale,
    int head_major_output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr uint32_t query_tile = 16u;
    constexpr uint32_t key_tile = 16u;
    constexpr uint32_t tile_elements = query_tile * (uint32_t)HEAD_DIM;
    constexpr uint32_t outputs_per_thread =
        (tile_elements + H3_LT_THREADS - 1u) / H3_LT_THREADS;
    __shared__ __align__(16) __nv_bfloat16 query_shared[query_tile][HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 key_shared[key_tile][HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 value_shared[key_tile][HEAD_DIM];
    __shared__ __align__(16) float scores[query_tile][key_tile];
    __shared__ float probabilities[query_tile][key_tile];
    __shared__ float row_maximum[query_tile];
    __shared__ float row_sum[query_tile];
    __shared__ float row_alpha[query_tile];

    uint32_t tid = threadIdx.x;
    uint32_t query_start = blockIdx.x * query_tile;
    uint32_t head = blockIdx.y;
    float output_values[outputs_per_thread];
#pragma unroll
    for (uint32_t slot = 0; slot < outputs_per_thread; slot++)
        output_values[slot] = 0.0f;
    for (uint32_t index = tid; index < tile_elements;
         index += H3_LT_THREADS) {
        uint32_t local_row = index / (uint32_t)HEAD_DIM;
        uint32_t dimension = index - local_row * (uint32_t)HEAD_DIM;
        uint32_t row = query_start + local_row;
        uint16_t bits = 0;
        if (row < sequence && head < heads)
            bits = query[((size_t)row * heads + head) * HEAD_DIM + dimension];
        query_shared[local_row][dimension] = __ushort_as_bfloat16(bits);
    }
    if (tid < query_tile) {
        row_maximum[tid] = -3.402823466e+38f;
        row_sum[tid] = 0.0f;
        row_alpha[tid] = 0.0f;
    }
    __syncthreads();

    for (uint32_t key_start = 0; key_start < sequence;
         key_start += key_tile) {
        for (uint32_t index = tid; index < key_tile * (uint32_t)HEAD_DIM;
             index += H3_LT_THREADS) {
            uint32_t local_row = index / (uint32_t)HEAD_DIM;
            uint32_t dimension = index - local_row * (uint32_t)HEAD_DIM;
            uint32_t row = key_start + local_row;
            uint16_t key_bits = 0;
            uint16_t value_bits = 0;
            if (row < sequence && head < heads) {
                size_t source =
                    ((size_t)row * heads + head) * HEAD_DIM + dimension;
                key_bits = key[source];
                value_bits = value[source];
            }
            key_shared[local_row][dimension] =
                __ushort_as_bfloat16(key_bits);
            value_shared[local_row][dimension] =
                __ushort_as_bfloat16(value_bits);
        }
        __syncthreads();

        if (tid < 32u) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                                    float>
                score_fragment;
            nvcuda::wmma::fill_fragment(score_fragment, 0.0f);
#pragma unroll
            for (uint32_t inner = 0; inner < (uint32_t)HEAD_DIM;
                 inner += 16u) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                        __nv_bfloat16,
                                        nvcuda::wmma::row_major>
                    query_fragment;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                        __nv_bfloat16,
                                        nvcuda::wmma::col_major>
                    key_fragment;
                nvcuda::wmma::load_matrix_sync(
                    query_fragment, &query_shared[0][inner], HEAD_DIM);
                nvcuda::wmma::load_matrix_sync(
                    key_fragment, &key_shared[0][inner], HEAD_DIM);
                nvcuda::wmma::mma_sync(score_fragment, query_fragment,
                                        key_fragment, score_fragment);
            }
            nvcuda::wmma::store_matrix_sync(
                &scores[0][0], score_fragment, key_tile,
                nvcuda::wmma::mem_row_major);
        }
        __syncthreads();

        if (tid < query_tile) {
            uint32_t valid_keys = sequence - key_start;
            if (valid_keys > key_tile)
                valid_keys = key_tile;
            float tile_maximum = -3.402823466e+38f;
            for (uint32_t local_key = 0; local_key < valid_keys; local_key++) {
                float score = scores[tid][local_key] * scale;
                scores[tid][local_key] = score;
                if (score > tile_maximum)
                    tile_maximum = score;
            }
            float old_maximum = row_maximum[tid];
            float new_maximum = old_maximum > tile_maximum ? old_maximum
                                                            : tile_maximum;
            float alpha = row_sum[tid] > 0.0f
                              ? expf(old_maximum - new_maximum)
                              : 0.0f;
            float tile_sum = 0.0f;
            for (uint32_t local_key = 0; local_key < key_tile; local_key++) {
                float probability =
                    local_key < valid_keys
                        ? expf(scores[tid][local_key] - new_maximum)
                        : 0.0f;
                probabilities[tid][local_key] = probability;
                tile_sum += probability;
            }
            row_alpha[tid] = alpha;
            row_sum[tid] = row_sum[tid] * alpha + tile_sum;
            row_maximum[tid] = new_maximum;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t slot = 0; slot < outputs_per_thread; slot++) {
            uint32_t index = tid + slot * H3_LT_THREADS;
            if (index < tile_elements) {
                uint32_t local_row = index / (uint32_t)HEAD_DIM;
                uint32_t dimension = index - local_row * (uint32_t)HEAD_DIM;
                float partial = 0.0f;
#pragma unroll
                for (uint32_t local_key = 0; local_key < key_tile;
                     local_key++) {
                    uint16_t bits = __bfloat16_as_ushort(
                        value_shared[local_key][dimension]);
                    partial = fmaf(probabilities[local_row][local_key],
                                   h3_lt_bf16_to_f32(bits), partial);
                }
                output_values[slot] =
                    fmaf(output_values[slot], row_alpha[local_row], partial);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t slot = 0; slot < outputs_per_thread; slot++) {
        uint32_t index = tid + slot * H3_LT_THREADS;
        if (index < tile_elements) {
            uint32_t local_row = index / (uint32_t)HEAD_DIM;
            uint32_t dimension = index - local_row * (uint32_t)HEAD_DIM;
            uint32_t row = query_start + local_row;
            if (row < sequence && head < heads) {
                size_t destination = head_major_output
                                         ? ((size_t)head * sequence + row) *
                                               HEAD_DIM +
                                               dimension
                                         : ((size_t)row * heads + head) *
                                               HEAD_DIM +
                                               dimension;
                output[destination] = h3_lt_f32_to_bf16(
                    output_values[slot] / row_sum[local_row]);
            }
        }
    }
#else
    (void)output;
    (void)query;
    (void)key;
    (void)value;
    (void)sequence;
    (void)heads;
    (void)scale;
    (void)head_major_output;
#endif
}

template <int HEAD_DIM>
__global__ static void h3_lt_sdpa_f32_kernel(
    float *output, const float *query, const float *key, const float *value,
    uint32_t sequence, uint32_t heads, float scale) {
    constexpr uint32_t query_tile = 16u;
    constexpr uint32_t key_tile = 16u;
    constexpr uint32_t tile_elements = query_tile * (uint32_t)HEAD_DIM;
    constexpr uint32_t outputs_per_thread =
        (tile_elements + H3_LT_THREADS - 1u) / H3_LT_THREADS;
    __shared__ float query_shared[query_tile][HEAD_DIM];
    __shared__ float key_shared[key_tile][HEAD_DIM];
    __shared__ float value_shared[key_tile][HEAD_DIM];
    __shared__ float scores[query_tile][key_tile];
    __shared__ float probabilities[query_tile][key_tile];
    __shared__ float row_maximum[query_tile];
    __shared__ float row_sum[query_tile];
    __shared__ float row_alpha[query_tile];

    uint32_t tid = threadIdx.x;
    uint32_t warp = tid >> 5;
    uint32_t lane = tid & 31u;
    uint32_t query_start = blockIdx.x * query_tile;
    uint32_t head = blockIdx.y;
    float output_values[outputs_per_thread];
#pragma unroll
    for (uint32_t slot = 0; slot < outputs_per_thread; slot++)
        output_values[slot] = 0.0f;
    for (uint32_t index = tid; index < tile_elements;
         index += H3_LT_THREADS) {
        uint32_t local_row = index / (uint32_t)HEAD_DIM;
        uint32_t dimension = index - local_row * (uint32_t)HEAD_DIM;
        uint32_t row = query_start + local_row;
        query_shared[local_row][dimension] =
            row < sequence
                ? query[((size_t)row * heads + head) * HEAD_DIM + dimension]
                : 0.0f;
    }
    if (tid < query_tile) {
        row_maximum[tid] = -3.402823466e+38f;
        row_sum[tid] = 0.0f;
        row_alpha[tid] = 0.0f;
    }
    __syncthreads();

    for (uint32_t key_start = 0; key_start < sequence;
         key_start += key_tile) {
        for (uint32_t index = tid; index < key_tile * (uint32_t)HEAD_DIM;
             index += H3_LT_THREADS) {
            uint32_t local_row = index / (uint32_t)HEAD_DIM;
            uint32_t dimension = index - local_row * (uint32_t)HEAD_DIM;
            uint32_t row = key_start + local_row;
            size_t source =
                ((size_t)row * heads + head) * HEAD_DIM + dimension;
            key_shared[local_row][dimension] =
                row < sequence ? key[source] : 0.0f;
            value_shared[local_row][dimension] =
                row < sequence ? value[source] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t row_slot = 0; row_slot < 2u; row_slot++) {
            uint32_t local_query = warp + row_slot * 8u;
#pragma unroll
            for (uint32_t local_key = 0; local_key < key_tile; local_key++) {
                float dot = 0.0f;
#pragma unroll
                for (uint32_t dimension = lane;
                     dimension < (uint32_t)HEAD_DIM; dimension += 32u)
                    dot = fmaf(query_shared[local_query][dimension],
                               key_shared[local_key][dimension], dot);
#pragma unroll
                for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
                    dot += __shfl_down_sync(0xffffffffu, dot, offset);
                if (lane == 0u)
                    scores[local_query][local_key] = dot * scale;
            }
        }
        __syncthreads();

        if (tid < query_tile) {
            uint32_t valid_keys = sequence - key_start;
            if (valid_keys > key_tile)
                valid_keys = key_tile;
            float tile_maximum = -3.402823466e+38f;
            for (uint32_t local_key = 0; local_key < valid_keys; local_key++)
                if (scores[tid][local_key] > tile_maximum)
                    tile_maximum = scores[tid][local_key];
            float old_maximum = row_maximum[tid];
            float new_maximum = old_maximum > tile_maximum ? old_maximum
                                                            : tile_maximum;
            float alpha = row_sum[tid] > 0.0f
                              ? expf(old_maximum - new_maximum)
                              : 0.0f;
            float tile_sum = 0.0f;
            for (uint32_t local_key = 0; local_key < key_tile; local_key++) {
                float probability =
                    local_key < valid_keys
                        ? expf(scores[tid][local_key] - new_maximum)
                        : 0.0f;
                probabilities[tid][local_key] = probability;
                tile_sum += probability;
            }
            row_alpha[tid] = alpha;
            row_sum[tid] = row_sum[tid] * alpha + tile_sum;
            row_maximum[tid] = new_maximum;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t slot = 0; slot < outputs_per_thread; slot++) {
            uint32_t index = tid + slot * H3_LT_THREADS;
            if (index < tile_elements) {
                uint32_t local_row = index / (uint32_t)HEAD_DIM;
                uint32_t dimension = index - local_row * (uint32_t)HEAD_DIM;
                float partial = 0.0f;
#pragma unroll
                for (uint32_t local_key = 0; local_key < key_tile;
                     local_key++)
                    partial = fmaf(probabilities[local_row][local_key],
                                   value_shared[local_key][dimension], partial);
                output_values[slot] =
                    fmaf(output_values[slot], row_alpha[local_row], partial);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t slot = 0; slot < outputs_per_thread; slot++) {
        uint32_t index = tid + slot * H3_LT_THREADS;
        if (index < tile_elements) {
            uint32_t local_row = index / (uint32_t)HEAD_DIM;
            uint32_t dimension = index - local_row * (uint32_t)HEAD_DIM;
            uint32_t row = query_start + local_row;
            if (row < sequence)
                output[((size_t)row * heads + head) * HEAD_DIM + dimension] =
                    output_values[slot] / row_sum[local_row];
        }
    }
}

__global__ static void h3_lt_linear_epilogue(
    const int32_t *accumulator, uint16_t *output, const float *input_scales,
    const float *weight_scales, uint32_t rows, uint32_t output_dim,
    uint32_t column_offset, uint32_t columns) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t elements = (size_t)rows * columns;
    if (index >= elements)
        return;
    uint32_t row = (uint32_t)(index / columns);
    uint32_t local_column = (uint32_t)(index - (size_t)row * columns);
    uint32_t column = column_offset + local_column;
    float value = __fmul_rn(__int2float_rn(accumulator[index]),
                            input_scales[row]);
    value = __fmul_rn(value, weight_scales[column]);
    output[(size_t)row * output_dim + column] = h3_lt_f32_to_bf16(value);
}

__global__ static void h3_lt_fc1_swiglu_epilogue(
    const int32_t *gate_accumulator, const int32_t *up_accumulator,
    uint16_t *output, const float *input_scales, const float *weight_scales,
    uint32_t rows, uint32_t hidden_dim, uint32_t column_offset,
    uint32_t columns) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t elements = (size_t)rows * columns;
    if (index >= elements)
        return;
    uint32_t row = (uint32_t)(index / columns);
    uint32_t local_column = (uint32_t)(index - (size_t)row * columns);
    uint32_t column = column_offset + local_column;
    float input_scale = input_scales[row];
    float gate = __fmul_rn(__int2float_rn(gate_accumulator[index]),
                           input_scale);
    gate = __fmul_rn(gate, weight_scales[column]);
    gate = h3_lt_bf16_to_f32(h3_lt_f32_to_bf16(gate));
    float up = __fmul_rn(__int2float_rn(up_accumulator[index]), input_scale);
    up = __fmul_rn(up, weight_scales[hidden_dim + column]);
    float activated = gate / (1.0f + expf(-gate));
    output[(size_t)row * hidden_dim + column] =
        h3_lt_f32_to_bf16(__fmul_rn(activated, up));
}

__global__ static void h3_lt_grouped_epilogue(
    const int32_t *group_accumulator, float *scaled_accumulator,
    uint16_t *output, const float *input_scales, const float *weight_scales,
    uint32_t rows, uint32_t output_dim, uint32_t column_offset,
    uint32_t columns, uint32_t group, uint32_t groups) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t elements = (size_t)rows * columns;
    if (index >= elements)
        return;
    uint32_t row = (uint32_t)(index / columns);
    uint32_t local_column = (uint32_t)(index - (size_t)row * columns);
    uint32_t column = column_offset + local_column;
    float term = __fmul_rn(__int2float_rn(group_accumulator[index]),
                           input_scales[(size_t)row * groups + group]);
    term = __fmul_rn(term, weight_scales[column]);
    float total = group == 0u ? term
                              : __fadd_rn(scaled_accumulator[index], term);
    if (group + 1u == groups)
        output[(size_t)row * output_dim + column] =
            h3_lt_f32_to_bf16(total);
    else
        scaled_accumulator[index] = total;
}

static uint32_t h3_lt_chunk_columns(size_t scratch_bytes, uint32_t rows,
                                    uint32_t output_dim,
                                    size_t bytes_per_element) {
    if (!rows || !output_dim || !bytes_per_element)
        return 0;
    size_t denominator = (size_t)rows * bytes_per_element;
    size_t maximum = denominator ? scratch_bytes / denominator : 0;
    if (!maximum)
        return 0;
    uint32_t columns = maximum > output_dim ? output_dim : (uint32_t)maximum;
    if (columns < output_dim && columns >= 16u)
        columns &= ~15u;
    return columns;
}

static int h3_lt_launch_result(cudaError_t result, char *error,
                               size_t error_size, const char *operation) {
    if (result == cudaSuccess)
        return 1;
    h3_lt_error(error, error_size, "%s failed: %s", operation,
                cudaGetErrorString(result));
    return 0;
}

static void h3_lt_reset_outputs(int *used, h3_cuda_accel_stats *stats) {
    if (used)
        *used = 0;
    if (stats)
        memset(stats, 0, sizeof(*stats));
}

int h3_cuda_accel_linear_f32(
    h3_cuda_accel *accel, void *stream_handle, void *output,
    const void *input, const void *weight, const void *bias, uint32_t rows,
    uint32_t input_dim, uint32_t output_dim, int *used,
    h3_cuda_accel_stats *stats, char *error, size_t error_size) {
    h3_lt_reset_outputs(used, stats);
    const char *disabled = getenv("H3_DISABLE_CUBLASLT_F32");
    /* Preserve bit-exact small-kernel tests and avoid heuristic overhead for
     * projections too small to benefit. */
    if (!used || !stats || !accel || !accel->operation_f32 || rows < 16u ||
        input_dim < 64u || output_dim < 64u ||
        (disabled && *disabled && strcmp(disabled, "0") != 0))
        return 1;
    h3_lt_plan *plan = NULL;
    int planned = h3_lt_get_f32_plan(accel, rows, output_dim, input_dim,
                                     &plan);
    if (planned < 0) {
        h3_lt_error(error, error_size, "cuBLASLt F32 plan creation failed");
        return 0;
    }
    if (!planned)
        return 1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    int launched = h3_lt_matmul_f32(
        accel, stream, static_cast<const float *>(input),
        static_cast<const float *>(weight), static_cast<float *>(output), plan);
    if (launched < 0) {
        h3_lt_error(error, error_size, "cuBLASLt F32 linear failed");
        return 0;
    }
    if (!launched)
        return 1;
    stats->matmul_launches++;
    if (bias) {
        size_t elements = (size_t)rows * output_dim;
        unsigned int blocks =
            (unsigned int)((elements + H3_LT_THREADS - 1u) / H3_LT_THREADS);
        h3_lt_bias_f32<<<blocks, H3_LT_THREADS, 0, stream>>>(
            static_cast<float *>(output), static_cast<const float *>(bias),
            elements, output_dim);
        if (!h3_lt_launch_result(cudaPeekAtLastError(), error, error_size,
                                 "CUDA F32 bias epilogue"))
            return 0;
        stats->epilogue_launches++;
    }
    *used = 1;
    return 1;
}

int h3_cuda_accel_linear_int8_bf16(
    h3_cuda_accel *accel, void *stream_handle, void *output,
    const void *input, const void *weight, const void *input_scales,
    const void *weight_scales, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim, int *used, h3_cuda_accel_stats *stats, char *error,
    size_t error_size) {
    h3_lt_reset_outputs(used, stats);
    if (!used || !stats || !h3_cuda_accel_available(accel) || !rows ||
        !input_dim || !output_dim || input_dim % 4u != 0u)
        return 1;
    if (!h3_lt_ensure_scratch(accel))
        return 1;
    uint32_t chunk = h3_lt_chunk_columns(accel->scratch_bytes, rows,
                                         output_dim, sizeof(int32_t));
    if (!chunk)
        return 1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    int32_t *accumulator = static_cast<int32_t *>(accel->scratch);
    for (uint32_t offset = 0; offset < output_dim;) {
        uint32_t columns = output_dim - offset;
        if (columns > chunk)
            columns = chunk;
        h3_lt_plan *plan = NULL;
        int planned = h3_lt_get_plan(accel, rows, columns, input_dim, input_dim,
                                     input_dim, &plan);
        if (planned < 0) {
            h3_lt_error(error, error_size,
                        "cuBLASLt int8 linear plan creation failed");
            return 0;
        }
        if (!planned)
            return 1;
        const int8_t *weight_chunk =
            static_cast<const int8_t *>(weight) + (size_t)offset * input_dim;
        int launched = h3_lt_matmul(
            accel, stream, static_cast<const int8_t *>(input), weight_chunk,
            accumulator, plan);
        if (launched < 0) {
            h3_lt_error(error, error_size, "cuBLASLt int8 linear failed");
            return 0;
        }
        if (!launched)
            return 1;
        stats->matmul_launches++;
        size_t elements = (size_t)rows * columns;
        unsigned int blocks =
            (unsigned int)((elements + H3_LT_THREADS - 1u) / H3_LT_THREADS);
        h3_lt_linear_epilogue<<<blocks, H3_LT_THREADS, 0, stream>>>(
            accumulator, static_cast<uint16_t *>(output),
            static_cast<const float *>(input_scales),
            static_cast<const float *>(weight_scales), rows, output_dim, offset,
            columns);
        if (!h3_lt_launch_result(cudaPeekAtLastError(), error, error_size,
                                 "CUDA int8 linear epilogue"))
            return 0;
        stats->epilogue_launches++;
        offset += columns;
    }
    *used = 1;
    return 1;
}

int h3_cuda_accel_fc1_swiglu_int8_bf16(
    h3_cuda_accel *accel, void *stream_handle, void *output,
    const void *input, const void *weight, const void *input_scales,
    const void *weight_scales, uint32_t rows, uint32_t input_dim,
    uint32_t hidden_dim, int *used, h3_cuda_accel_stats *stats, char *error,
    size_t error_size) {
    h3_lt_reset_outputs(used, stats);
    if (!used || !stats || !h3_cuda_accel_available(accel) || !rows ||
        !input_dim || !hidden_dim || input_dim % 4u != 0u)
        return 1;
    if (!h3_lt_ensure_scratch(accel))
        return 1;
    uint32_t chunk = h3_lt_chunk_columns(accel->scratch_bytes, rows,
                                         hidden_dim, sizeof(int32_t) * 2u);
    if (!chunk)
        return 1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    int32_t *gate_accumulator = static_cast<int32_t *>(accel->scratch);
    size_t half = accel->scratch_bytes / 2u;
    half -= half % alignof(int32_t);
    int32_t *up_accumulator = reinterpret_cast<int32_t *>(
        static_cast<unsigned char *>(accel->scratch) + half);
    for (uint32_t offset = 0; offset < hidden_dim;) {
        uint32_t columns = hidden_dim - offset;
        if (columns > chunk)
            columns = chunk;
        h3_lt_plan *plan = NULL;
        int planned = h3_lt_get_plan(accel, rows, columns, input_dim, input_dim,
                                     input_dim, &plan);
        if (planned < 0) {
            h3_lt_error(error, error_size,
                        "cuBLASLt FC1 plan creation failed");
            return 0;
        }
        if (!planned)
            return 1;
        const int8_t *weights = static_cast<const int8_t *>(weight);
        const int8_t *gate_weight = weights + (size_t)offset * input_dim;
        const int8_t *up_weight =
            weights + (size_t)(hidden_dim + offset) * input_dim;
        int gate_launched = h3_lt_matmul(
            accel, stream, static_cast<const int8_t *>(input), gate_weight,
            gate_accumulator, plan);
        int up_launched = gate_launched > 0
                              ? h3_lt_matmul(
                                    accel, stream,
                                    static_cast<const int8_t *>(input),
                                    up_weight, up_accumulator, plan)
                              : gate_launched;
        if (gate_launched < 0 || up_launched < 0) {
            h3_lt_error(error, error_size, "cuBLASLt FC1 SwiGLU failed");
            return 0;
        }
        if (!gate_launched || !up_launched)
            return 1;
        stats->matmul_launches += 2u;
        size_t elements = (size_t)rows * columns;
        unsigned int blocks =
            (unsigned int)((elements + H3_LT_THREADS - 1u) / H3_LT_THREADS);
        h3_lt_fc1_swiglu_epilogue<<<blocks, H3_LT_THREADS, 0, stream>>>(
            gate_accumulator, up_accumulator, static_cast<uint16_t *>(output),
            static_cast<const float *>(input_scales),
            static_cast<const float *>(weight_scales), rows, hidden_dim, offset,
            columns);
        if (!h3_lt_launch_result(cudaPeekAtLastError(), error, error_size,
                                 "CUDA FC1 SwiGLU epilogue"))
            return 0;
        stats->epilogue_launches++;
        offset += columns;
    }
    *used = 1;
    return 1;
}

int h3_cuda_accel_linear_int8_grouped_bf16(
    h3_cuda_accel *accel, void *stream_handle, void *output,
    const void *input, const void *weight, const void *input_scales,
    const void *weight_scales, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim, uint32_t group_size, uint32_t groups, int *used,
    h3_cuda_accel_stats *stats, char *error, size_t error_size) {
    h3_lt_reset_outputs(used, stats);
    if (!used || !stats || !h3_cuda_accel_available(accel) || !rows ||
        !input_dim || !output_dim || !group_size || !groups ||
        group_size % 4u != 0u || (uint64_t)group_size * groups != input_dim)
        return 1;
    if (!h3_lt_ensure_scratch(accel))
        return 1;
    uint32_t chunk = h3_lt_chunk_columns(accel->scratch_bytes, rows,
                                         output_dim, sizeof(int32_t) * 2u);
    if (!chunk)
        return 1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    int32_t *group_accumulator = static_cast<int32_t *>(accel->scratch);
    size_t half = accel->scratch_bytes / 2u;
    half -= half % alignof(float);
    float *scaled_accumulator = reinterpret_cast<float *>(
        static_cast<unsigned char *>(accel->scratch) + half);
    const int8_t *inputs = static_cast<const int8_t *>(input);
    const int8_t *weights = static_cast<const int8_t *>(weight);
    for (uint32_t offset = 0; offset < output_dim;) {
        uint32_t columns = output_dim - offset;
        if (columns > chunk)
            columns = chunk;
        h3_lt_plan *plan = NULL;
        int planned = h3_lt_get_plan(accel, rows, columns, group_size,
                                     input_dim, input_dim, &plan);
        if (planned < 0) {
            h3_lt_error(error, error_size,
                        "cuBLASLt grouped FC2 plan creation failed");
            return 0;
        }
        if (!planned)
            return 1;
        size_t elements = (size_t)rows * columns;
        unsigned int blocks =
            (unsigned int)((elements + H3_LT_THREADS - 1u) / H3_LT_THREADS);
        for (uint32_t group = 0; group < groups; group++) {
            size_t group_offset = (size_t)group * group_size;
            const int8_t *input_group = inputs + group_offset;
            const int8_t *weight_group =
                weights + (size_t)offset * input_dim + group_offset;
            int launched = h3_lt_matmul(accel, stream, input_group,
                                        weight_group, group_accumulator, plan);
            if (launched < 0) {
                h3_lt_error(error, error_size,
                            "cuBLASLt grouped FC2 failed");
                return 0;
            }
            if (!launched)
                return 1;
            stats->matmul_launches++;
            h3_lt_grouped_epilogue<<<blocks, H3_LT_THREADS, 0, stream>>>(
                group_accumulator, scaled_accumulator,
                static_cast<uint16_t *>(output),
                static_cast<const float *>(input_scales),
                static_cast<const float *>(weight_scales), rows, output_dim,
                offset, columns, group, groups);
            if (!h3_lt_launch_result(cudaPeekAtLastError(), error, error_size,
                                     "CUDA grouped FC2 epilogue"))
                return 0;
            stats->epilogue_launches++;
        }
        offset += columns;
    }
    *used = 1;
    return 1;
}

int h3_cuda_accel_sdpa_bf16(
    h3_cuda_accel *accel, void *stream_handle, void *output,
    const void *query, const void *key, const void *value, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale, int head_major_output,
    int *used, h3_cuda_accel_stats *stats, char *error, size_t error_size) {
    h3_lt_reset_outputs(used, stats);
    const char *disabled = getenv("H3_DISABLE_CUDA_FLASH2");
    if (!used || !stats || !accel || accel->compute_major < 8 || !sequence ||
        !heads || (head_dim != 64u && head_dim != 128u) ||
        (disabled && *disabled && strcmp(disabled, "0") != 0))
        return 1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    dim3 grid((sequence + 15u) / 16u, heads, 1u);
    if (head_dim == 64u) {
        h3_lt_sdpa_bf16_kernel<64><<<grid, H3_LT_THREADS, 0, stream>>>(
            static_cast<uint16_t *>(output),
            static_cast<const uint16_t *>(query),
            static_cast<const uint16_t *>(key),
            static_cast<const uint16_t *>(value), sequence, heads, scale,
            head_major_output);
    } else {
        h3_lt_sdpa_bf16_kernel<128><<<grid, H3_LT_THREADS, 0, stream>>>(
            static_cast<uint16_t *>(output),
            static_cast<const uint16_t *>(query),
            static_cast<const uint16_t *>(key),
            static_cast<const uint16_t *>(value), sequence, heads, scale,
            head_major_output);
    }
    if (!h3_lt_launch_result(cudaPeekAtLastError(), error, error_size,
                             "CUDA tiled BF16 attention"))
        return 0;
    stats->attention_launches++;
    *used = 1;
    return 1;
}

int h3_cuda_accel_sdpa_f32(
    h3_cuda_accel *accel, void *stream_handle, void *output,
    const void *query, const void *key, const void *value, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale, int *used,
    h3_cuda_accel_stats *stats, char *error, size_t error_size) {
    h3_lt_reset_outputs(used, stats);
    const char *disabled = getenv("H3_DISABLE_CUDA_FLASH2_F32");
    if (!used || !stats || !accel || accel->compute_major < 7 ||
        sequence < 128u || !heads ||
        (head_dim != 64u && head_dim != 128u) ||
        (disabled && *disabled && strcmp(disabled, "0") != 0))
        return 1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    dim3 grid((sequence + 15u) / 16u, heads, 1u);
    if (head_dim == 64u) {
        h3_lt_sdpa_f32_kernel<64><<<grid, H3_LT_THREADS, 0, stream>>>(
            static_cast<float *>(output), static_cast<const float *>(query),
            static_cast<const float *>(key), static_cast<const float *>(value),
            sequence, heads, scale);
    } else {
        h3_lt_sdpa_f32_kernel<128><<<grid, H3_LT_THREADS, 0, stream>>>(
            static_cast<float *>(output), static_cast<const float *>(query),
            static_cast<const float *>(key), static_cast<const float *>(value),
            sequence, heads, scale);
    }
    if (!h3_lt_launch_result(cudaPeekAtLastError(), error, error_size,
                             "CUDA tiled F32 attention"))
        return 0;
    stats->attention_launches++;
    *used = 1;
    return 1;
}

#if defined(H3_HAVE_CUDNN)
static int h3_cudnn_ensure_workspace(h3_cuda_accel *accel, size_t bytes) {
    if (!bytes)
        return 1;
    if (bytes <= accel->cudnn_workspace_bytes)
        return 1;
    if (bytes > accel->cudnn_workspace_limit)
        return 0;
    void *workspace = NULL;
    if (cudaMalloc(&workspace, bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    if (accel->cudnn_workspace)
        (void)cudaFree(accel->cudnn_workspace);
    accel->cudnn_workspace = workspace;
    accel->cudnn_workspace_bytes = bytes;
    return 1;
}

static int h3_cudnn_dimension(uint64_t value, int *result) {
    if (!result || value > (uint64_t)INT_MAX)
        return 0;
    *result = (int)value;
    return 1;
}
#endif

int h3_cuda_accel_conv3d_f32(
    h3_cuda_accel *accel, void *stream_handle, void *output,
    const void *input, const void *weight, const void *bias, uint32_t batch,
    uint32_t depth, uint32_t height, uint32_t width,
    uint32_t input_channels, uint32_t output_channels,
    uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width,
    uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width,
    int *used, h3_cuda_accel_stats *stats, char *error, size_t error_size) {
    h3_lt_reset_outputs(used, stats);
#if !defined(H3_HAVE_CUDNN)
    (void)accel;
    (void)stream_handle;
    (void)output;
    (void)input;
    (void)weight;
    (void)bias;
    (void)batch;
    (void)depth;
    (void)height;
    (void)width;
    (void)input_channels;
    (void)output_channels;
    (void)kernel_depth;
    (void)kernel_height;
    (void)kernel_width;
    (void)stride_depth;
    (void)stride_height;
    (void)stride_width;
    (void)error;
    (void)error_size;
    return 1;
#else
    const char *disabled = getenv("H3_DISABLE_CUDNN");
    /* Small/irregular channel counts retain the arithmetic-oracle kernel;
     * production VAE blocks use channel multiples of eight. */
    if (!used || !stats || !accel || !accel->cudnn || !batch || !depth ||
        !height || !width || input_channels < 8u || output_channels < 8u ||
        input_channels % 8u || output_channels % 8u || !kernel_depth ||
        !kernel_height || !kernel_width || !stride_depth || !stride_height ||
        !stride_width || depth < kernel_depth || height < kernel_height ||
        width < kernel_width ||
        (disabled && *disabled && strcmp(disabled, "0") != 0))
        return 1;
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1u;
    uint32_t output_height = (height - kernel_height) / stride_height + 1u;
    uint32_t output_width = (width - kernel_width) / stride_width + 1u;
    int input_dimensions[5], input_strides[5], output_dimensions[5];
    int output_strides[5], filter_dimensions[5], bias_dimensions[5];
    int bias_strides[5];
    uint64_t input_spatial = (uint64_t)depth * height * width;
    uint64_t output_spatial =
        (uint64_t)output_depth * output_height * output_width;
    if (!h3_cudnn_dimension(batch, &input_dimensions[0]) ||
        !h3_cudnn_dimension(input_channels, &input_dimensions[1]) ||
        !h3_cudnn_dimension(depth, &input_dimensions[2]) ||
        !h3_cudnn_dimension(height, &input_dimensions[3]) ||
        !h3_cudnn_dimension(width, &input_dimensions[4]) ||
        !h3_cudnn_dimension(input_spatial * input_channels,
                            &input_strides[0]) ||
        !h3_cudnn_dimension(1u, &input_strides[1]) ||
        !h3_cudnn_dimension((uint64_t)height * width * input_channels,
                            &input_strides[2]) ||
        !h3_cudnn_dimension((uint64_t)width * input_channels,
                            &input_strides[3]) ||
        !h3_cudnn_dimension(input_channels, &input_strides[4]) ||
        !h3_cudnn_dimension(batch, &output_dimensions[0]) ||
        !h3_cudnn_dimension(output_channels, &output_dimensions[1]) ||
        !h3_cudnn_dimension(output_depth, &output_dimensions[2]) ||
        !h3_cudnn_dimension(output_height, &output_dimensions[3]) ||
        !h3_cudnn_dimension(output_width, &output_dimensions[4]) ||
        !h3_cudnn_dimension(output_spatial * output_channels,
                            &output_strides[0]) ||
        !h3_cudnn_dimension(1u, &output_strides[1]) ||
        !h3_cudnn_dimension(
            (uint64_t)output_height * output_width * output_channels,
            &output_strides[2]) ||
        !h3_cudnn_dimension((uint64_t)output_width * output_channels,
                            &output_strides[3]) ||
        !h3_cudnn_dimension(output_channels, &output_strides[4]))
        return 1;
    filter_dimensions[0] = (int)output_channels;
    filter_dimensions[1] = (int)input_channels;
    filter_dimensions[2] = (int)kernel_depth;
    filter_dimensions[3] = (int)kernel_height;
    filter_dimensions[4] = (int)kernel_width;
    bias_dimensions[0] = 1;
    bias_dimensions[1] = (int)output_channels;
    bias_dimensions[2] = 1;
    bias_dimensions[3] = 1;
    bias_dimensions[4] = 1;
    bias_strides[0] = (int)output_channels;
    bias_strides[1] = 1;
    bias_strides[2] = 1;
    bias_strides[3] = 1;
    bias_strides[4] = 1;
    int padding[3] = {0, 0, 0};
    int strides[3] = {(int)stride_depth, (int)stride_height,
                      (int)stride_width};
    int dilations[3] = {1, 1, 1};
    cudnnTensorDescriptor_t input_descriptor = NULL;
    cudnnTensorDescriptor_t output_descriptor = NULL;
    cudnnTensorDescriptor_t bias_descriptor = NULL;
    cudnnFilterDescriptor_t filter_descriptor = NULL;
    cudnnConvolutionDescriptor_t convolution_descriptor = NULL;
    cudnnStatus_t status = cudnnSetStream(
        accel->cudnn, static_cast<cudaStream_t>(stream_handle));
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnCreateTensorDescriptor(&input_descriptor);
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnCreateTensorDescriptor(&output_descriptor);
    if (status == CUDNN_STATUS_SUCCESS && bias)
        status = cudnnCreateTensorDescriptor(&bias_descriptor);
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnCreateFilterDescriptor(&filter_descriptor);
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnSetTensorNdDescriptor(
            input_descriptor, CUDNN_DATA_FLOAT, 5, input_dimensions,
            input_strides);
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnSetTensorNdDescriptor(
            output_descriptor, CUDNN_DATA_FLOAT, 5, output_dimensions,
            output_strides);
    if (status == CUDNN_STATUS_SUCCESS && bias)
        status = cudnnSetTensorNdDescriptor(
            bias_descriptor, CUDNN_DATA_FLOAT, 5, bias_dimensions,
            bias_strides);
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnSetFilterNdDescriptor(
            filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5,
            filter_dimensions);
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnSetConvolutionNdDescriptor(
            convolution_descriptor, 3, padding, strides, dilations,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    const char *tf32 = getenv("H3_CUDA_CUDNN_TF32");
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnSetConvolutionMathType(
            convolution_descriptor,
            tf32 && *tf32 && strcmp(tf32, "0") != 0
                ? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
                : CUDNN_FMA_MATH);
    cudnnConvolutionFwdAlgoPerf_t performances[8];
    int returned = 0;
    if (status == CUDNN_STATUS_SUCCESS)
        status = cudnnGetConvolutionForwardAlgorithm_v7(
            accel->cudnn, input_descriptor, filter_descriptor,
            convolution_descriptor, output_descriptor, 8, &returned,
            performances);
    int selected = -1;
    if (status == CUDNN_STATUS_SUCCESS) {
        for (int index = 0; index < returned; index++) {
            if (performances[index].status == CUDNN_STATUS_SUCCESS &&
                performances[index].determinism == CUDNN_DETERMINISTIC &&
                performances[index].memory <= accel->cudnn_workspace_limit) {
                selected = index;
                break;
            }
        }
    }
    if (selected >= 0 &&
        !h3_cudnn_ensure_workspace(accel, performances[selected].memory))
        selected = -1;
    if (status == CUDNN_STATUS_SUCCESS && selected >= 0) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        status = cudnnConvolutionForward(
            accel->cudnn, &alpha, input_descriptor, input, filter_descriptor,
            weight, convolution_descriptor, performances[selected].algo,
            accel->cudnn_workspace, performances[selected].memory, &beta,
            output_descriptor, output);
        if (status == CUDNN_STATUS_SUCCESS && bias)
            status = cudnnAddTensor(accel->cudnn, &alpha, bias_descriptor,
                                    bias, &alpha, output_descriptor, output);
    }
    if (convolution_descriptor)
        (void)cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    if (filter_descriptor)
        (void)cudnnDestroyFilterDescriptor(filter_descriptor);
    if (bias_descriptor)
        (void)cudnnDestroyTensorDescriptor(bias_descriptor);
    if (output_descriptor)
        (void)cudnnDestroyTensorDescriptor(output_descriptor);
    if (input_descriptor)
        (void)cudnnDestroyTensorDescriptor(input_descriptor);
    if (status != CUDNN_STATUS_SUCCESS) {
        h3_lt_error(error, error_size, "cuDNN Conv3D failed: %s",
                    cudnnGetErrorString(status));
        return 0;
    }
    if (selected < 0)
        return 1;
    stats->convolution_launches++;
    *used = 1;
    return 1;
#endif
}
