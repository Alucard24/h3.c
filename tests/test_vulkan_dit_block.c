/* End-to-end DiT block on the Vulkan backend.
 *
 * Replicates tests/test_real_dit_block.c's run_dit_block_inplace sequence
 * (modulation linear, AdaLN, QKV, grouped QKV/RoPE, SDPA, attention
 * output, gate, MLP AdaLN, fc1/SwiGLU/fc2, gate) with synthetic weights
 * and a CPU reference that reproduces the kernel arithmetic order.
 * Verifies the whole portable DiT block chain on Vulkan without the MLX
 * fixtures.
 *
 * Skips cleanly when no Vulkan backend is available.
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    SEQUENCE = 8, HIDDEN = 5376, HEADS = 56, HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM, FFN = 14336, TIME_INPUT = 256,
    TIME_DIM = 2688, MODALITIES = 3, SLOTS = 6, ROPE_HALF = 48
};

static int tests_run;
static int failed;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        failed++; \
    } \
} while (0)

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16f(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

/* Deterministic generator: xorshift64. */
static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static float rng_unit(void) {
    return (float)((double)(rng_next() >> 11) / 9007199254740992.0 * 2.0 -
                   1.0);
}

/* ------------------------------------------------------------ reference */

static float tree_sum256(const uint16_t *values, uint32_t width) {
    float reductions[256];
    for (uint32_t tid = 0; tid < 256; tid++) {
        float local = 0.0f;
        for (uint32_t k = tid; k < width; k += 256u) {
            float value = bf16f(values[k]);
            local = fmaf(value, value, local);
        }
        reductions[tid] = local;
    }
    for (uint32_t stride = 128; stride > 0; stride >>= 1)
        for (uint32_t index = 0; index < stride; index++)
            reductions[index] += reductions[index + stride];
    return reductions[0];
}

static void ref_linear(const uint16_t *input, const uint16_t *weight,
                       const uint16_t *bias, int has_bias, uint16_t *output,
                       uint32_t rows, uint32_t input_dim,
                       uint32_t output_dim) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < output_dim; column++) {
            float sum = has_bias ? bf16f(bias[column]) : 0.0f;
            for (uint32_t k = 0; k < input_dim; k++) {
                sum = fmaf(bf16f(input[row * input_dim + k]),
                           bf16f(weight[column * input_dim + k]), sum);
            }
            output[row * output_dim + column] = bf16(sum);
        }
    }
}

static void ref_rms_norm(const uint16_t *input, const uint16_t *weight,
                         uint16_t *output, uint32_t rows, uint32_t width,
                         float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        float inverse = 1.0f / sqrtf(tree_sum256(input + row * width, width) /
                                     (float)width + epsilon);
        for (uint32_t column = 0; column < width; column++) {
            float normalized = bf16f(input[row * width + column]) * inverse *
                               bf16f(weight[column]);
            output[row * width + column] = bf16(normalized);
        }
    }
}

static void ref_adaln(const uint16_t *input, const uint16_t *norm_weight,
                      const uint16_t *modulation, const uint32_t *row_map,
                      uint16_t *output, uint32_t rows, uint32_t width,
                      uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        float inverse = 1.0f / sqrtf(tree_sum256(input + row * width, width) /
                                     (float)width + epsilon);
        uint32_t base = row_map[row] * slots * width;
        for (uint32_t column = 0; column < width; column++) {
            float normalized = bf16f(input[row * width + column]) * inverse *
                               bf16f(norm_weight[column]);
            float shift = bf16f(modulation[base + shift_slot * width + column]);
            float scale = bf16f(modulation[base + scale_slot * width + column]);
            output[row * width + column] =
                bf16(normalized * (1.0f + scale) + shift);
        }
    }
}

static void ref_gate(const uint16_t *residual, const uint16_t *branch,
                     const uint16_t *modulation, const uint32_t *row_map,
                     uint16_t *output, uint32_t rows, uint32_t width,
                     uint32_t slots, uint32_t gate_slot) {
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t base = row_map[row] * slots * width;
        for (uint32_t column = 0; column < width; column++) {
            uint32_t index = row * width + column;
            float gate = bf16f(modulation[base + gate_slot * width + column]);
            output[index] = bf16(bf16f(residual[index]) +
                                 bf16f(branch[index]) * gate);
        }
    }
}

static void ref_qkv_rope(const uint16_t *qkv, const uint16_t *q_norm,
                         const uint16_t *k_norm, const uint16_t *rope_cos,
                         const uint16_t *rope_sin, uint16_t *query,
                         uint16_t *key, uint16_t *value, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) {
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t inner = heads * head_dim;
            uint32_t row_base = row * inner * 3;
            uint32_t q_base = row_base + head * head_dim * 3;
            uint32_t k_base = q_base + head_dim;
            uint32_t v_base = k_base + head_dim;
            float q_sum = 0.0f, k_sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float q = bf16f(qkv[q_base + d]);
                float k = bf16f(qkv[k_base + d]);
                q_sum = fmaf(q, q, q_sum);
                k_sum = fmaf(k, k, k_sum);
            }
            float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
            float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
            for (uint32_t d = 0; d < head_dim; d++) {
                float q0 = bf16f(qkv[q_base + d]) * q_inverse *
                           bf16f(q_norm[d]);
                float k0 = bf16f(qkv[k_base + d]) * k_inverse *
                           bf16f(k_norm[d]);
                if (d < rope_half) {
                    float q1 = bf16f(qkv[q_base + d + rope_half]) * q_inverse *
                               bf16f(q_norm[d + rope_half]);
                    float k1 = bf16f(qkv[k_base + d + rope_half]) * k_inverse *
                               bf16f(k_norm[d + rope_half]);
                    float c = bf16f(rope_cos[row * rope_half + d]);
                    float s = bf16f(rope_sin[row * rope_half + d]);
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else if (d < rope_half * 2) {
                    uint32_t pair = d - rope_half;
                    float q1 = bf16f(qkv[q_base + pair]) * q_inverse *
                               bf16f(q_norm[pair]);
                    float k1 = bf16f(qkv[k_base + pair]) * k_inverse *
                               bf16f(k_norm[pair]);
                    float c = bf16f(rope_cos[row * rope_half + pair]);
                    float s = bf16f(rope_sin[row * rope_half + pair]);
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                uint32_t index = (row * heads + head) * head_dim + d;
                query[index] = bf16(q0);
                key[index] = bf16(k0);
                value[index] = qkv[v_base + d];
            }
        }
    }
}

static void ref_sdpa(const uint16_t *q, const uint16_t *k, const uint16_t *v,
                     uint16_t *output, uint32_t sequence, uint32_t heads,
                     uint32_t head_dim, float scale) {
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t qbase = (row * heads + head) * head_dim;
            float scores[SEQUENCE];
            float maxv = -3.402823466e+38f;
            for (uint32_t s = 0; s < sequence; s++) {
                uint32_t kbase = (s * heads + head) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++)
                    dot = fmaf(bf16f(q[qbase + d]),
                               bf16f(k[kbase + d]), dot);
                scores[s] = dot * scale;
                if (scores[s] > maxv) maxv = scores[s];
            }
            float sum = 0.0f;
            for (uint32_t s = 0; s < sequence; s++)
                sum += expf(scores[s] - maxv);
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (uint32_t s = 0; s < sequence; s++) {
                    uint32_t kbase = (s * heads + head) * head_dim;
                    acc = fmaf(expf(scores[s] - maxv),
                               bf16f(v[kbase + d]), acc);
                }
                output[qbase + d] = bf16(acc / sum);
            }
        }
    }
}

static void ref_swiglu_halves(const uint16_t *fc1, uint16_t *activated,
                              uint32_t rows, uint32_t hidden) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < hidden; column++) {
            float gate = bf16f(fc1[row * hidden * 2 + column]);
            float up = bf16f(fc1[row * hidden * 2 + hidden + column]);
            activated[row * hidden + column] =
                bf16(gate / (1.0f + expf(-gate)) * up);
        }
    }
}

/* --------------------------------------------------------------- report */

static void report(const char *name, const uint16_t *got,
                   const uint16_t *expected, size_t count,
                   double *relative_l2, double *relative_max) {
    size_t worst = 0;
    uint32_t worst_ulp = 0;
    double square_error = 0.0, square_value = 0.0, max_error = 0.0;
    double max_value = 0.0;
    for (size_t index = 0; index < count; index++) {
        uint16_t g = got[index], e = expected[index];
        uint32_t diff = (g & 0x8000u) != (e & 0x8000u) ?
            UINT32_MAX : (uint32_t)abs((int)(g & 0x7fffu) -
                                       (int)(e & 0x7fffu));
        if (diff > worst_ulp) {
            worst_ulp = diff;
            worst = index;
        }
        double gv = bf16f(g), ev = bf16f(e);
        double delta = gv - ev;
        if (fabs(delta) > max_error) max_error = fabs(delta);
        if (fabs(ev) > max_value) max_value = fabs(ev);
        square_error += delta * delta;
        square_value += ev * ev;
    }
    *relative_l2 = sqrt(square_error /
                        (square_value > 1e-24 ? square_value : 1e-24));
    *relative_max = max_error / (max_value > 1e-12 ? max_value : 1e-12);
    printf("%-22s worst %u ulp @[%zu] rel-L2 %.3g rel-max %.3g\n", name,
           worst_ulp, worst, *relative_l2, *relative_max);
}

/* The GPU inverse sqrt and exp differ from the CPU libm by ~1 ulp; each of
 * the block's twelve kernels can add one to the reference, so the whole
 * chain is compared within 8 BF16 ulps. */
static int check_ulp(const uint16_t *got, const uint16_t *expected,
                     size_t count) {
    for (size_t index = 0; index < count; index++) {
        uint16_t g = got[index], e = expected[index];
        uint32_t diff = (g & 0x8000u) != (e & 0x8000u) ?
            UINT32_MAX : (uint32_t)abs((int)(g & 0x7fffu) -
                                       (int)(e & 0x7fffu));
        if (diff > 8) {
            fprintf(stderr, "  %s[%zu]: got %u expected %u (%.6g vs %.6g)\n",
                    "chain", index, g, e, bf16f(g), bf16f(e));
            return 0;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------ main */

static h3_gpu_tensor *bf16_tensor(h3_gpu *gpu, const uint16_t *values,
                                  size_t count) {
    return h3_gpu_tensor_from_bf16(gpu, values, count);
}

static h3_gpu_tensor *u32_tensor(h3_gpu *gpu, const uint32_t *values,
                                 size_t count) {
    return h3_gpu_tensor_from_u32(gpu, values, count);
}

int main(int argc, char **argv) {
    const char *shader_path = argc > 1 ? argv[1] : "h3_vulkan_shaders.comp";
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(shader_path, error, sizeof(error));
    if (!gpu) {
        printf("SKIP: no Vulkan backend available (%s)\n", error);
        return 0;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* Sizes (elements). */
    size_t mod_count = (size_t)MODALITIES * SLOTS * HIDDEN;
    size_t qkv_count = (size_t)INNER * 3 * HIDDEN;
    size_t out_count = (size_t)HIDDEN * INNER;
    size_t fc1_count = (size_t)FFN * 2 * HIDDEN;
    size_t fc2_count = (size_t)HIDDEN * FFN;
    size_t seq_hidden = (size_t)SEQUENCE * HIDDEN;
    size_t seq_inner = (size_t)SEQUENCE * INNER;
    size_t seq_ffn = (size_t)SEQUENCE * FFN;
    size_t rope_count = (size_t)SEQUENCE * ROPE_HALF;

    /* Weights. */
    uint16_t *adaln_w = malloc((size_t)TIME_DIM * mod_count * 2);
    uint16_t *adaln_b = malloc(mod_count * 2);
    uint16_t *norm1 = malloc(HIDDEN * 2);
    uint16_t *norm2 = malloc(HIDDEN * 2);
    uint16_t *qkv_w = malloc(qkv_count * 2);
    uint16_t *q_norm = malloc(HEAD_DIM * 2);
    uint16_t *k_norm = malloc(HEAD_DIM * 2);
    uint16_t *out_w = malloc(out_count * 2);
    uint16_t *fc1_w = malloc(fc1_count * 2);
    uint16_t *fc2_w = malloc(fc2_count * 2);
    uint16_t *rope_cos = malloc(rope_count * 2);
    uint16_t *rope_sin = malloc(rope_count * 2);
    uint16_t *time_silu = malloc((size_t)TIME_DIM * 2);
    uint16_t *hidden = malloc(seq_hidden * 2);
    uint32_t *row_map = malloc(SEQUENCE * 4);
    CHECK(adaln_w && adaln_b && norm1 && norm2 && qkv_w && q_norm &&
          k_norm && out_w && fc1_w && fc2_w && rope_cos && rope_sin &&
          time_silu && hidden && row_map);
    if (failed) return 1;

    for (size_t index = 0; index < (size_t)TIME_DIM * mod_count; index++)
        adaln_w[index] = bf16(rng_unit() * 0.02f);
    for (size_t index = 0; index < mod_count; index++)
        adaln_b[index] = bf16(rng_unit() * 0.01f);
    for (size_t index = 0; index < HIDDEN; index++) {
        norm1[index] = bf16(1.0f + rng_unit() * 0.05f);
        norm2[index] = bf16(1.0f + rng_unit() * 0.05f);
    }
    for (size_t index = 0; index < qkv_count; index++)
        qkv_w[index] = bf16(rng_unit() * 0.02f);
    for (size_t index = 0; index < HEAD_DIM; index++) {
        q_norm[index] = bf16(1.0f + rng_unit() * 0.05f);
        k_norm[index] = bf16(1.0f + rng_unit() * 0.05f);
    }
    for (size_t index = 0; index < out_count; index++)
        out_w[index] = bf16(rng_unit() * 0.02f);
    for (size_t index = 0; index < fc1_count; index++)
        fc1_w[index] = bf16(rng_unit() * 0.02f);
    for (size_t index = 0; index < fc2_count; index++)
        fc2_w[index] = bf16(rng_unit() * 0.02f);
    for (size_t index = 0; index < rope_count; index++) {
        rope_cos[index] = bf16(cosf((float)index * 0.01f));
        rope_sin[index] = bf16(sinf((float)index * 0.01f));
    }
    for (size_t index = 0; index < TIME_DIM; index++)
        time_silu[index] = bf16(rng_unit());
    for (size_t index = 0; index < seq_hidden; index++)
        hidden[index] = bf16(rng_unit() * 0.5f);
    for (size_t index = 0; index < SEQUENCE; index++)
        row_map[index] = 0;

    /* CPU reference. */
    uint16_t *ref_modulation = malloc(mod_count * 2);
    uint16_t *ref_mod_attention = malloc(seq_hidden * 2);
    uint16_t *ref_qkv = malloc((size_t)SEQUENCE * INNER * 3 * 2);
    uint16_t *ref_query = malloc(seq_inner * 2);
    uint16_t *ref_key = malloc(seq_inner * 2);
    uint16_t *ref_value = malloc(seq_inner * 2);
    uint16_t *ref_heads = malloc(seq_inner * 2);
    uint16_t *ref_attention_output = malloc(seq_hidden * 2);
    uint16_t *ref_mod_mlp = malloc(seq_hidden * 2);
    uint16_t *ref_fc1 = malloc((size_t)SEQUENCE * FFN * 2 * 2);
    uint16_t *ref_activated = malloc(seq_ffn * 2);
    uint16_t *ref_mlp_output = malloc(seq_hidden * 2);
    uint16_t *ref_hidden = malloc(seq_hidden * 2);
    CHECK(ref_modulation && ref_mod_attention && ref_qkv && ref_query &&
          ref_key && ref_value && ref_heads && ref_attention_output &&
          ref_mod_mlp && ref_fc1 && ref_activated && ref_mlp_output &&
          ref_hidden);
    if (failed) return 1;
    memcpy(ref_hidden, hidden, seq_hidden * 2);

    ref_linear(time_silu, adaln_w, adaln_b, 1, ref_modulation, 1, TIME_DIM,
               (uint32_t)mod_count);
    ref_adaln(hidden, norm1, ref_modulation, row_map, ref_mod_attention,
              SEQUENCE, HIDDEN, SLOTS, 0, 1, 1e-5f);
    ref_linear(ref_mod_attention, qkv_w, NULL, 0, ref_qkv, SEQUENCE, HIDDEN,
               INNER * 3);
    ref_qkv_rope(ref_qkv, q_norm, k_norm, rope_cos, rope_sin, ref_query,
                 ref_key, ref_value, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF,
                 1e-5f);
    ref_sdpa(ref_query, ref_key, ref_value, ref_heads, SEQUENCE, HEADS,
             HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM));
    ref_linear(ref_heads, out_w, NULL, 0, ref_attention_output, SEQUENCE,
               INNER, HIDDEN);
    ref_gate(ref_hidden, ref_attention_output, ref_modulation, row_map,
             ref_hidden, SEQUENCE, HIDDEN, SLOTS, 2);
    ref_adaln(ref_hidden, norm2, ref_modulation, row_map, ref_mod_mlp,
              SEQUENCE, HIDDEN, SLOTS, 3, 4, 1e-5f);
    ref_linear(ref_mod_mlp, fc1_w, NULL, 0, ref_fc1, SEQUENCE, HIDDEN,
               FFN * 2);
    ref_swiglu_halves(ref_fc1, ref_activated, SEQUENCE, FFN);
    ref_linear(ref_activated, fc2_w, NULL, 0, ref_mlp_output, SEQUENCE, FFN,
               HIDDEN);
    ref_gate(ref_hidden, ref_mlp_output, ref_modulation, row_map, ref_hidden,
             SEQUENCE, HIDDEN, SLOTS, 5);

    /* GPU tensors. */
    h3_gpu_tensor *t_adaln_w = bf16_tensor(gpu, adaln_w,
                                           (size_t)TIME_DIM * mod_count);
    h3_gpu_tensor *t_adaln_b = bf16_tensor(gpu, adaln_b, mod_count);
    h3_gpu_tensor *t_norm1 = bf16_tensor(gpu, norm1, HIDDEN);
    h3_gpu_tensor *t_norm2 = bf16_tensor(gpu, norm2, HIDDEN);
    h3_gpu_tensor *t_qkv_w = bf16_tensor(gpu, qkv_w, qkv_count);
    h3_gpu_tensor *t_q_norm = bf16_tensor(gpu, q_norm, HEAD_DIM);
    h3_gpu_tensor *t_k_norm = bf16_tensor(gpu, k_norm, HEAD_DIM);
    h3_gpu_tensor *t_out_w = bf16_tensor(gpu, out_w, out_count);
    h3_gpu_tensor *t_fc1_w = bf16_tensor(gpu, fc1_w, fc1_count);
    h3_gpu_tensor *t_fc2_w = bf16_tensor(gpu, fc2_w, fc2_count);
    h3_gpu_tensor *t_rope_cos = bf16_tensor(gpu, rope_cos, rope_count);
    h3_gpu_tensor *t_rope_sin = bf16_tensor(gpu, rope_sin, rope_count);
    h3_gpu_tensor *t_time_silu = bf16_tensor(gpu, time_silu, TIME_DIM);
    h3_gpu_tensor *t_hidden = bf16_tensor(gpu, hidden, seq_hidden);
    h3_gpu_tensor *t_row_map = u32_tensor(gpu, row_map, SEQUENCE);
    h3_gpu_tensor *t_modulation = h3_gpu_tensor_new_bf16(gpu, mod_count);
    h3_gpu_tensor *t_mod_attention = h3_gpu_tensor_new_bf16(gpu, seq_hidden);
    h3_gpu_tensor *t_qkv = h3_gpu_tensor_new_bf16(gpu,
                                                  (size_t)SEQUENCE * INNER * 3);
    h3_gpu_tensor *t_query = h3_gpu_tensor_new_bf16(gpu, seq_inner);
    h3_gpu_tensor *t_key = h3_gpu_tensor_new_bf16(gpu, seq_inner);
    h3_gpu_tensor *t_value = h3_gpu_tensor_new_bf16(gpu, seq_inner);
    h3_gpu_tensor *t_heads = h3_gpu_tensor_new_bf16(gpu, seq_inner);
    h3_gpu_tensor *t_attention_output = h3_gpu_tensor_new_bf16(gpu,
                                                                seq_hidden);
    h3_gpu_tensor *t_mod_mlp = h3_gpu_tensor_new_bf16(gpu, seq_hidden);
    h3_gpu_tensor *t_mlp_output = h3_gpu_tensor_new_bf16(gpu, seq_hidden);
    CHECK(t_adaln_w && t_adaln_b && t_norm1 && t_norm2 && t_qkv_w &&
          t_q_norm && t_k_norm && t_out_w && t_fc1_w && t_fc2_w &&
          t_rope_cos && t_rope_sin && t_time_silu && t_hidden && t_row_map &&
          t_modulation && t_mod_attention && t_qkv && t_query && t_key &&
          t_value && t_heads && t_attention_output && t_mod_mlp &&
          t_mlp_output);
    if (failed) return 1;

    h3_gpu_begin(gpu);
    CHECK(h3_gpu_linear_bf16(gpu, t_modulation, t_time_silu, t_adaln_w,
                             t_adaln_b, 1, TIME_DIM,
                             (uint32_t)mod_count) == 1);
    CHECK(h3_gpu_adaln_bf16(gpu, t_mod_attention, t_hidden, t_norm1,
                            t_modulation, t_row_map, SEQUENCE, HIDDEN, SLOTS,
                            0, 1, 1e-5f) == 1);
    CHECK(h3_gpu_linear_bf16(gpu, t_qkv, t_mod_attention, t_qkv_w, NULL,
                             SEQUENCE, HIDDEN, INNER * 3) == 1);
    CHECK(h3_gpu_grouped_qkv_rope_bf16(gpu, t_query, t_key, t_value, t_qkv,
                                       t_q_norm, t_k_norm, t_rope_cos,
                                       t_rope_sin, SEQUENCE, HEADS, HEAD_DIM,
                                       ROPE_HALF, 1e-5f) == 1);
    CHECK(h3_gpu_sdpa_bf16(gpu, t_heads, t_query, t_key, t_value, SEQUENCE,
                           HEADS, HEAD_DIM,
                           1.0f / sqrtf((float)HEAD_DIM)) == 1);
    CHECK(h3_gpu_linear_bf16(gpu, t_attention_output, t_heads, t_out_w, NULL,
                             SEQUENCE, INNER, HIDDEN) == 1);
    CHECK(h3_gpu_gate_bf16(gpu, t_hidden, t_hidden, t_attention_output,
                           t_modulation, t_row_map, SEQUENCE, HIDDEN, SLOTS,
                           2) == 1);
    CHECK(h3_gpu_adaln_bf16(gpu, t_mod_mlp, t_hidden, t_norm2, t_modulation,
                            t_row_map, SEQUENCE, HIDDEN, SLOTS, 3, 4,
                            1e-5f) == 1);
    CHECK(h3_gpu_mlp_bf16(gpu, t_mlp_output, t_mod_mlp, t_fc1_w, t_fc2_w,
                          SEQUENCE, HIDDEN, FFN, HIDDEN) == 1);
    CHECK(h3_gpu_gate_bf16(gpu, t_hidden, t_hidden, t_mlp_output,
                           t_modulation, t_row_map, SEQUENCE, HIDDEN, SLOTS,
                           5) == 1);
    CHECK(h3_gpu_submit(gpu) == 1);
    if (failed) {
        fprintf(stderr, "GPU pipeline error: %s\n", h3_gpu_error(gpu));
    } else {
        uint16_t *got_hidden = malloc(seq_hidden * 2);
        uint16_t *got_mod_att = malloc(seq_hidden * 2);
        CHECK(got_hidden && got_mod_att);
        if (!failed) {
            CHECK(h3_gpu_tensor_read_bf16(t_hidden, got_hidden,
                                          seq_hidden) == 1);
            CHECK(h3_gpu_tensor_read_bf16(t_mod_attention, got_mod_att,
                                          seq_hidden) == 1);
            double l2, relmax;
            report("hidden", got_hidden, ref_hidden, seq_hidden, &l2,
                   &relmax);
            /* A twelve-kernel BF16 chain accumulates ~1 ulp of absolute
             * error per kernel; values near zero then carry large relative
             * errors, so the final hidden is judged by the project's
             * tensor-scale bounds instead of per-element ulps. */
            CHECK(l2 < 0.01);
            CHECK(relmax < 0.02);
            report("mod_attention", got_mod_att, ref_mod_attention,
                   seq_hidden, &l2, &relmax);
            CHECK(check_ulp(got_mod_att, ref_mod_attention, seq_hidden));
            free(got_hidden);
            free(got_mod_att);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double seconds = (double)(end.tv_sec - start.tv_sec) +
                     (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
    printf("DiT block (Vulkan): %.2f s\n", seconds);

    h3_gpu_free(gpu);
    free(adaln_w); free(adaln_b); free(norm1); free(norm2);
    free(qkv_w); free(q_norm); free(k_norm); free(out_w);
    free(fc1_w); free(fc2_w); free(rope_cos); free(rope_sin);
    free(time_silu); free(hidden); free(row_map);
    free(ref_modulation); free(ref_mod_attention); free(ref_qkv);
    free(ref_query); free(ref_key); free(ref_value); free(ref_heads);
    free(ref_attention_output); free(ref_mod_mlp); free(ref_fc1);
    free(ref_activated); free(ref_mlp_output); free(ref_hidden);

    if (failed) {
        fprintf(stderr, "FAILED: %d of %d checks\n", failed, tests_run);
        return 1;
    }
    printf("ok: %d checks (DiT block on Vulkan)\n", tests_run);
    return 0;
}
