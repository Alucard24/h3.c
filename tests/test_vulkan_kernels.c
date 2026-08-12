/* Vulkan backend kernel tests: parity against CPU references computed with
 * the same arithmetic as h3_shaders.metal. Skips cleanly when no Vulkan
 * device is available. */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run;
static int failed;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        failed++; \
    } \
} while (0)

static uint16_t bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_value(uint16_t bits) {
    uint32_t raw = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &raw, 4);
    return value;
}

static float bf16_values(const uint16_t *bits) {
    return bf16_value(bits[0]);
}

static void bf16_row_values(const uint16_t *bits, float *values,
                            size_t count) {
    for (size_t index = 0; index < count; index++)
        values[index] = bf16_value(bits[index]);
}

static int close_relative(float got, float expected, float tolerance) {
    if (got == expected) return 1;
    float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    return fabsf(got - expected) <= tolerance * scale;
}

static int check_f32(const float *got, const float *expected, size_t count,
                     float tolerance, const char *name) {
    int ok = 1;
    for (size_t index = 0; index < count; index++) {
        if (!close_relative(got[index], expected[index], tolerance)) {
            fprintf(stderr, "  %s[%zu]: got %.9g expected %.9g\n", name,
                    index, got[index], expected[index]);
            ok = 0;
            break;
        }
    }
    return ok;
}

/* Norm kernels reduce with a threadgroup tree and an approximate
 * inversesqrt; both are within ~1 ulp of the CPU reference but not
 * bit-exact, and BF16 rounds that ulp. Compare BF16 values numerically
 * instead of bitwise. */
static int check_bf16_rel(const uint16_t *got, const uint16_t *expected,
                          size_t count, float tolerance, const char *name) {
    int ok = 1;
    for (size_t index = 0; index < count; index++) {
        float g = bf16_value(got[index]);
        float e = bf16_value(expected[index]);
        if (!close_relative(g, e, tolerance)) {
            fprintf(stderr, "  %s[%zu]: got %.9g expected %.9g\n", name,
                    index, g, e);
            ok = 0;
            break;
        }
    }
    return ok;
}

static void test_cast_and_unary(h3_gpu *gpu) {
    enum { N = 1000 };
    float input[N];
    uint16_t bf16[N], expected_bf16[N];
    float output_f32[N], expected_f32[N];
    for (size_t index = 0; index < N; index++) {
        double scale = (index % 5 == 0) ? 1e-6 : 1.0;
        input[index] = (float)(sin((double)index * 0.37) * 3.0 * scale);
        expected_bf16[index] = bf16_bits(input[index]);
        bf16[index] = expected_bf16[index];
        expected_f32[index] = bf16_value(expected_bf16[index]);
    }
    h3_gpu_tensor *in = h3_gpu_tensor_from_f32(gpu, input, N);
    h3_gpu_tensor *out_b = h3_gpu_tensor_new_bf16(gpu, N);
    h3_gpu_tensor *out_f = h3_gpu_tensor_new_f32(gpu, N);
    h3_gpu_tensor *in_b = h3_gpu_tensor_from_bf16(gpu, bf16, N);
    CHECK(in && out_b && out_f && in_b);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_cast_f32_to_bf16(gpu, out_b, in, N) == 0);
        CHECK(h3_gpu_cast_bf16_to_f32(gpu, out_f, in_b, N) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got_b[N];
        CHECK(h3_gpu_tensor_read_bf16(out_b, got_b, N) == 0);
        CHECK(memcmp(got_b, expected_bf16, sizeof(got_b)) == 0);
        float got_f[N];
        CHECK(h3_gpu_tensor_read_f32(out_f, got_f, N) == 0);
        CHECK(check_f32(got_f, expected_f32, N, 1e-7, "cast_bf16"));
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(out_b);
    h3_gpu_tensor_free(out_f);
    h3_gpu_tensor_free(in_b);
}

static void test_silu_bf16(h3_gpu *gpu) {
    enum { N = 777 };
    uint16_t input[N], expected[N];
    for (size_t index = 0; index < N; index++) {
        input[index] = bf16_bits((float)(sin((double)index * 0.91) * 4.0));
        float value = bf16_value(input[index]);
        expected[index] = bf16_bits(value / (1.0f + expf(-value)));
    }
    h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, input, N);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, N);
    CHECK(in && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_silu_bf16(gpu, out, in, N) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[N];
        CHECK(h3_gpu_tensor_read_bf16(out, got, N) == 0);
        CHECK(memcmp(got, expected, sizeof(got)) == 0);
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(out);
}

static void test_add_sub_silu_mul(h3_gpu *gpu) {
    enum { N = 640 };
    uint16_t left[N], right[N], expected_add[N], expected_sub[N],
             expected_mul[N];
    for (size_t index = 0; index < N; index++) {
        left[index] = bf16_bits((float)(sin((double)index * 0.41) * 2.5));
        right[index] = bf16_bits((float)(cos((double)index * 0.23) * 1.5));
        float a = bf16_value(left[index]);
        float b = bf16_value(right[index]);
        expected_add[index] = bf16_bits(a + b);
        expected_sub[index] = bf16_bits(a - b);
        expected_mul[index] = bf16_bits(a / (1.0f + expf(-a)) * b);
    }
    h3_gpu_tensor *a_t = h3_gpu_tensor_from_bf16(gpu, left, N);
    h3_gpu_tensor *b_t = h3_gpu_tensor_from_bf16(gpu, right, N);
    h3_gpu_tensor *add = h3_gpu_tensor_new_bf16(gpu, N);
    h3_gpu_tensor *sub = h3_gpu_tensor_new_bf16(gpu, N);
    h3_gpu_tensor *mul = h3_gpu_tensor_new_bf16(gpu, N);
    CHECK(a_t && b_t && add && sub && mul);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_add_bf16(gpu, add, a_t, b_t, N) == 0);
        CHECK(h3_gpu_sub_bf16(gpu, sub, a_t, b_t, N) == 0);
        CHECK(h3_gpu_silu_mul_bf16(gpu, mul, a_t, b_t, N) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[N];
        CHECK(h3_gpu_tensor_read_bf16(add, got, N) == 0);
        CHECK(memcmp(got, expected_add, sizeof(got)) == 0);
        CHECK(h3_gpu_tensor_read_bf16(sub, got, N) == 0);
        CHECK(memcmp(got, expected_sub, sizeof(got)) == 0);
        CHECK(h3_gpu_tensor_read_bf16(mul, got, N) == 0);
        CHECK(memcmp(got, expected_mul, sizeof(got)) == 0);
    }
    h3_gpu_tensor_free(a_t);
    h3_gpu_tensor_free(b_t);
    h3_gpu_tensor_free(add);
    h3_gpu_tensor_free(sub);
    h3_gpu_tensor_free(mul);
}

static void test_scaled_clip_geglu(h3_gpu *gpu) {
    enum { N = 512 };
    float left[N], right[N], expected_scaled[N], expected_clip[N],
          expected_geglu[N];
    for (size_t index = 0; index < N; index++) {
        left[index] = (float)(sin((double)index * 0.17) * 2.0);
        right[index] = (float)(cos((double)index * 0.31) * 2.0);
        expected_scaled[index] = left[index] * 0.75f + right[index] * -0.25f;
        expected_clip[index] = fmaxf(-0.5f, fminf(0.5f, left[index]));
        float x = left[index];
        float cube = x * x * x;
        float gelu = 0.5f * x *
            (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * cube)));
        expected_geglu[index] = gelu * right[index];
    }
    h3_gpu_tensor *a = h3_gpu_tensor_from_f32(gpu, left, N);
    h3_gpu_tensor *b = h3_gpu_tensor_from_f32(gpu, right, N);
    h3_gpu_tensor *scaled = h3_gpu_tensor_new_f32(gpu, N);
    h3_gpu_tensor *clipped = h3_gpu_tensor_new_f32(gpu, N);
    h3_gpu_tensor *geglu = h3_gpu_tensor_new_f32(gpu, N);
    CHECK(a && b && scaled && clipped && geglu);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_add_scaled_f32(gpu, scaled, a, b, 0.75f, -0.25f, N) == 0);
        CHECK(h3_gpu_clip_f32(gpu, clipped, a, N, -0.5f, 0.5f) == 0);
        CHECK(h3_gpu_geglu_f32(gpu, geglu, a, b, N) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        float got[N];
        CHECK(h3_gpu_tensor_read_f32(scaled, got, N) == 0);
        CHECK(check_f32(got, expected_scaled, N, 1e-7, "add_scaled"));
        CHECK(h3_gpu_tensor_read_f32(clipped, got, N) == 0);
        CHECK(check_f32(got, expected_clip, N, 1e-7, "clip"));
        CHECK(h3_gpu_tensor_read_f32(geglu, got, N) == 0);
        CHECK(check_f32(got, expected_geglu, N, 1e-6, "geglu"));
    }
    h3_gpu_tensor_free(a);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(scaled);
    h3_gpu_tensor_free(clipped);
    h3_gpu_tensor_free(geglu);
}

static void test_euler(h3_gpu *gpu) {
    enum { N = 400, OFFSET = 5 };
    float sample[N + OFFSET], expected[N + OFFSET];
    uint16_t last[N], previous[N];
    for (size_t index = 0; index < N + OFFSET; index++)
        sample[index] = expected[index] = (float)index * 0.1f;
    for (size_t index = 0; index < N; index++) {
        last[index] = bf16_bits((float)sin((double)index * 0.53) * 2.0);
        previous[index] = bf16_bits((float)cos((double)index * 0.61) * 2.0);
        float last_value = bf16_value(last[index]);
        float velocity = fmaf(0.3f,
                              last_value - bf16_value(previous[index]),
                              last_value);
        expected[OFFSET + index] = fmaf(0.05f, velocity,
                                        expected[OFFSET + index]);
    }
    h3_gpu_tensor *s = h3_gpu_tensor_from_f32(gpu, sample, N + OFFSET);
    h3_gpu_tensor *l = h3_gpu_tensor_from_bf16(gpu, last, N);
    h3_gpu_tensor *p = h3_gpu_tensor_from_bf16(gpu, previous, N);
    CHECK(s && l && p);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_euler_bf16(gpu, s, OFFSET, l, p, N, 0.05f, 0.3f) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        float got[N + OFFSET];
        CHECK(h3_gpu_tensor_read_f32(s, got, N + OFFSET) == 0);
        CHECK(check_f32(got, expected, N + OFFSET, 1e-7, "euler"));
    }
    h3_gpu_tensor_free(s);
    h3_gpu_tensor_free(l);
    h3_gpu_tensor_free(p);
}

static void test_embedding(h3_gpu *gpu) {
    enum { TOKENS = 40, WIDTH = 64, VOCAB = 1000 };
    uint16_t weight[VOCAB * WIDTH];
    for (size_t index = 0; index < VOCAB * WIDTH; index++)
        weight[index] = bf16_bits((float)((index * 2654435761u) % 65536u) /
                                  32768.0f - 1.0f);
    uint32_t ids[TOKENS];
    uint16_t expected[TOKENS * WIDTH];
    for (size_t token = 0; token < TOKENS; token++) {
        ids[token] = (token % 7 == 0) ? VOCAB + token : token * 17 % VOCAB;
        for (size_t column = 0; column < WIDTH; column++) {
            uint32_t id = ids[token];
            expected[token * WIDTH + column] = id < VOCAB ?
                weight[id * WIDTH + column] : 0;
        }
    }
    h3_gpu_tensor *w = h3_gpu_tensor_from_bf16(gpu, weight, VOCAB * WIDTH);
    h3_gpu_tensor *t = h3_gpu_tensor_from_u32(gpu, ids, TOKENS);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, TOKENS * WIDTH);
    CHECK(w && t && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_embedding_bf16(gpu, out, w, t, TOKENS, VOCAB, WIDTH) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[TOKENS * WIDTH];
        CHECK(h3_gpu_tensor_read_bf16(out, got, TOKENS * WIDTH) == 0);
        CHECK(memcmp(got, expected, sizeof(got)) == 0);
    }
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(t);
    h3_gpu_tensor_free(out);
}

/* Norm kernels reduce per row with a 256-thread tree; reproduce that order
 * so the F32 references match to rounding (the GPU inverse sqrt itself stays
 * within an ulp). */
static float tree_sum256(const float *values) {
    float reductions[256];
    for (size_t index = 0; index < 256; index++)
        reductions[index] = values[index];
    for (uint32_t stride = 128; stride > 0; stride >>= 1) {
        for (uint32_t index = 0; index < stride; index++)
            reductions[index] += reductions[index + stride];
    }
    return reductions[0];
}

static void test_norms(h3_gpu *gpu) {
    enum { ROWS = 3, WIDTH = 256 };
    const float epsilon = 1e-5f;
    float input[ROWS * WIDTH], weight[WIDTH], bias[WIDTH];
    uint16_t input_b[ROWS * WIDTH], weight_b[WIDTH], bias_b[WIDTH];
    float expected_rms[ROWS * WIDTH], expected_layer[ROWS * WIDTH];
    uint16_t expected_rms_b[ROWS * WIDTH], expected_layer_b[ROWS * WIDTH];
    for (size_t column = 0; column < WIDTH; column++) {
        weight[column] = (float)sin((double)column * 0.11) * 0.5f + 1.0f;
        bias[column] = (float)cos((double)column * 0.07) * 0.25f;
        weight_b[column] = bf16_bits(weight[column]);
        bias_b[column] = bf16_bits(bias[column]);
    }
    for (size_t row = 0; row < ROWS; row++) {
        float squared[WIDTH], squared_b[WIDTH];
        float layer_sq[WIDTH], layer_sq_b[WIDTH];
        for (size_t column = 0; column < WIDTH; column++) {
            input[row * WIDTH + column] =
                (float)(sin((double)(row * WIDTH + column) * 0.31) * 3.0);
            input_b[row * WIDTH + column] = bf16_bits(input[row * WIDTH + column]);
        }
        float mean = tree_sum256(input + row * WIDTH) / (float)WIDTH;
        float bf16_row[WIDTH];
        bf16_row_values(input_b + row * WIDTH, bf16_row, WIDTH);
        float mean_b = tree_sum256(bf16_row) / (float)WIDTH;
        for (size_t column = 0; column < WIDTH; column++) {
            float x = input[row * WIDTH + column];
            float xb = bf16_value(input_b[row * WIDTH + column]);
            /* RMS norm does not center; layer norm does. */
            squared[column] = fmaf(x, x, 0.0f);
            squared_b[column] = fmaf(xb, xb, 0.0f);
            float centered = x - mean;
            float centered_b = xb - mean_b;
            layer_sq[column] = fmaf(centered, centered, 0.0f);
            layer_sq_b[column] = fmaf(centered_b, centered_b, 0.0f);
        }
        float inverse = 1.0f / sqrtf(tree_sum256(squared) / (float)WIDTH +
                                     epsilon);
        float inverse_b = 1.0f / sqrtf(tree_sum256(squared_b) / (float)WIDTH +
                                       epsilon);
        float inverse_layer = 1.0f / sqrtf(tree_sum256(layer_sq) /
                                           (float)WIDTH + epsilon);
        float inverse_layer_b = 1.0f / sqrtf(tree_sum256(layer_sq_b) /
                                            (float)WIDTH + epsilon);
        for (size_t column = 0; column < WIDTH; column++) {
            float x = input[row * WIDTH + column];
            float xb = bf16_value(input_b[row * WIDTH + column]);
            expected_rms[row * WIDTH + column] = x * inverse * weight[column];
            expected_layer[row * WIDTH + column] =
                (x - mean) * inverse_layer * weight[column] + bias[column];
            expected_rms_b[row * WIDTH + column] =
                bf16_bits(xb * inverse_b * bf16_value(weight_b[column]));
            float normalized_layer = (xb - mean_b) * inverse_layer_b;
            expected_layer_b[row * WIDTH + column] = bf16_bits(
                fmaf(normalized_layer, bf16_value(weight_b[column]),
                     bf16_value(bias_b[column])));
        }
    }
    h3_gpu_tensor *in = h3_gpu_tensor_from_f32(gpu, input, ROWS * WIDTH);
    h3_gpu_tensor *w = h3_gpu_tensor_from_f32(gpu, weight, WIDTH);
    h3_gpu_tensor *b = h3_gpu_tensor_from_f32(gpu, bias, WIDTH);
    h3_gpu_tensor *rms = h3_gpu_tensor_new_f32(gpu, ROWS * WIDTH);
    h3_gpu_tensor *layer = h3_gpu_tensor_new_f32(gpu, ROWS * WIDTH);
    h3_gpu_tensor *in_b = h3_gpu_tensor_from_bf16(gpu, input_b, ROWS * WIDTH);
    h3_gpu_tensor *w_b = h3_gpu_tensor_from_bf16(gpu, weight_b, WIDTH);
    h3_gpu_tensor *b_b = h3_gpu_tensor_from_bf16(gpu, bias_b, WIDTH);
    h3_gpu_tensor *rms_b = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    h3_gpu_tensor *layer_b = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(in && w && b && rms && layer && in_b && w_b && b_b && rms_b &&
          layer_b);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_rms_norm_f32(gpu, rms, in, w, ROWS, WIDTH, epsilon) == 0);
        CHECK(h3_gpu_layer_norm_f32(gpu, layer, in, w, b, ROWS, WIDTH,
                                    epsilon) == 0);
        CHECK(h3_gpu_rms_norm_bf16(gpu, rms_b, in_b, w_b, ROWS, WIDTH,
                                   epsilon) == 0);
        CHECK(h3_gpu_layer_norm_bf16(gpu, layer_b, in_b, w_b, b_b, ROWS,
                                     WIDTH, epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        float got[ROWS * WIDTH];
        uint16_t got_b[ROWS * WIDTH];
        CHECK(h3_gpu_tensor_read_f32(rms, got, ROWS * WIDTH) == 0);
        CHECK(check_f32(got, expected_rms, ROWS * WIDTH, 1e-4, "rms_f32"));
        CHECK(h3_gpu_tensor_read_f32(layer, got, ROWS * WIDTH) == 0);
        CHECK(check_f32(got, expected_layer, ROWS * WIDTH, 1e-4,
                        "layer_f32"));
        CHECK(h3_gpu_tensor_read_bf16(rms_b, got_b, ROWS * WIDTH) == 0);
        CHECK(check_bf16_rel(got_b, expected_rms_b, ROWS * WIDTH, 1e-3,
                             "rms_bf16"));
        CHECK(h3_gpu_tensor_read_bf16(layer_b, got_b, ROWS * WIDTH) == 0);
        CHECK(check_bf16_rel(got_b, expected_layer_b, ROWS * WIDTH, 1e-3,
                             "layer_bf16"));
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(rms);
    h3_gpu_tensor_free(layer);
    h3_gpu_tensor_free(in_b);
    h3_gpu_tensor_free(w_b);
    h3_gpu_tensor_free(b_b);
    h3_gpu_tensor_free(rms_b);
    h3_gpu_tensor_free(layer_b);
}

static void test_continue_chain(h3_gpu *gpu) {
    /* Two command buffers chained without a wait must preserve order:
     * add, then silu of the add result, all inside one begin/submit. */
    enum { N = 256 };
    uint16_t left[N], right[N], expected[N];
    for (size_t index = 0; index < N; index++) {
        left[index] = bf16_bits((float)(sin((double)index * 0.19) * 2.0));
        right[index] = bf16_bits((float)(cos((double)index * 0.37) * 2.0));
        float a = bf16_value(left[index]);
        float b = bf16_value(right[index]);
        /* The add kernel rounds its BF16 output first; silu sees that
         * rounded value, not the exact float sum. */
        float sum = bf16_value(bf16_bits(a + b));
        expected[index] = bf16_bits(sum / (1.0f + expf(-sum)));
    }
    h3_gpu_tensor *a = h3_gpu_tensor_from_bf16(gpu, left, N);
    h3_gpu_tensor *b = h3_gpu_tensor_from_bf16(gpu, right, N);
    h3_gpu_tensor *sum = h3_gpu_tensor_new_bf16(gpu, N);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, N);
    CHECK(a && b && sum && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_add_bf16(gpu, sum, a, b, N) == 0);
        CHECK(h3_gpu_continue(gpu) == 0);
        CHECK(h3_gpu_silu_bf16(gpu, out, sum, N) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[N];
        CHECK(h3_gpu_tensor_read_bf16(out, got, N) == 0);
        CHECK(memcmp(got, expected, sizeof(got)) == 0);
    }
    h3_gpu_tensor_free(a);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(sum);
    h3_gpu_tensor_free(out);
}

int main(int argc, char **argv) {
    const char *shader_path = argc > 1 ? argv[1] : "h3_vulkan_shaders.comp";
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(shader_path, error, sizeof(error));
    if (!gpu) {
        printf("SKIP: no Vulkan backend available (%s)\n", error);
        return 0;
    }
    test_cast_and_unary(gpu);
    test_silu_bf16(gpu);
    test_add_sub_silu_mul(gpu);
    test_scaled_clip_geglu(gpu);
    test_euler(gpu);
    test_embedding(gpu);
    test_norms(gpu);
    test_continue_chain(gpu);
    h3_gpu_free(gpu);
    if (failed) {
        fprintf(stderr, "FAILED: %d of %d checks\n", failed, tests_run);
        return 1;
    }
    printf("ok: %d checks (Vulkan backend)\n", tests_run);
    return 0;
}
