/* Vulkan backend kernel tests: parity against CPU references computed with
 * the same arithmetic as h3_shaders.metal. Skips cleanly when no Vulkan
 * device is available. */
#include "h3_gpu.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
/* BF16-aware comparison: within `max_ulp` BF16 units, or within an
 * absolute bound (for denormal-scale outputs where a few ulps are
 * meaningless). */
static int check_bf16_ulp_abs(const uint16_t *got, const uint16_t *expected,
                              size_t count, uint32_t max_ulp,
                              float max_absolute, const char *name) {
    int ok = 1;
    for (size_t index = 0; index < count; index++) {
        uint16_t g = got[index], e = expected[index];
        uint32_t diff = (g & 0x8000u) != (e & 0x8000u) ?
            UINT32_MAX : (uint32_t)abs((int)(g & 0x7fffu) -
                                       (int)(e & 0x7fffu));
        float gv = bf16_value(g), ev = bf16_value(e);
        if (diff > max_ulp && fabsf(gv - ev) > max_absolute) {
            fprintf(stderr, "  %s[%zu]: got %u expected %u (%.9g vs %.9g)\n",
                    name, index, g, e, gv, ev);
            ok = 0;
            break;
        }
    }
    return ok;
}

/* BF16-aware comparison: within `max_ulp` BF16 units. */
static int check_bf16_ulp(const uint16_t *got, const uint16_t *expected,
                          size_t count, uint32_t max_ulp, const char *name) {
    int ok = 1;
    for (size_t index = 0; index < count; index++) {
        uint16_t g = got[index], e = expected[index];
        uint32_t diff = (g & 0x8000u) != (e & 0x8000u) ?
            UINT32_MAX : (uint32_t)abs((int)(g & 0x7fffu) -
                                       (int)(e & 0x7fffu));
        if (diff > max_ulp) {
            fprintf(stderr, "  %s[%zu]: got %u expected %u (%.9g vs %.9g)\n",
                    name, index, g, e, bf16_value(g), bf16_value(e));
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
        CHECK(check_f32(got_f, expected_f32, N, 2, "cast_bf16"));
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
static float tree_sum256(const float *values, uint32_t count) {
    float reductions[256];
    for (uint32_t index = 0; index < 256; index++)
        reductions[index] = index < count ? values[index] : 0.0f;
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
        float mean = tree_sum256(input + row * WIDTH, WIDTH) / (float)WIDTH;
        float bf16_row[WIDTH];
        bf16_row_values(input_b + row * WIDTH, bf16_row, WIDTH);
        float mean_b = tree_sum256(bf16_row, WIDTH) / (float)WIDTH;
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
        float inverse = 1.0f / sqrtf(tree_sum256(squared, WIDTH) / (float)WIDTH +
                                     epsilon);
        float inverse_b = 1.0f / sqrtf(tree_sum256(squared_b, WIDTH) / (float)WIDTH +
                                       epsilon);
        float inverse_layer = 1.0f / sqrtf(tree_sum256(layer_sq, WIDTH) /
                                           (float)WIDTH + epsilon);
        float inverse_layer_b = 1.0f / sqrtf(tree_sum256(layer_sq_b, WIDTH) /
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
        CHECK(check_bf16_ulp(got_b, expected_rms_b, ROWS * WIDTH, 2, "rms_bf16"));
        CHECK(h3_gpu_tensor_read_bf16(layer_b, got_b, ROWS * WIDTH) == 0);
        CHECK(check_bf16_ulp(got_b, expected_layer_b, ROWS * WIDTH, 2, "layer_bf16"));
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

/* CPU reference with the exact kernel order: bias init, then one fused
 * FMA per k in ascending k, rounded to BF16 once at the end. */
static void linear_reference(const uint16_t *input, const uint16_t *weight,
                             const uint16_t *bias, int has_bias,
                             uint16_t *output, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < output_dim; column++) {
            float sum = has_bias ? bf16_value(bias[column]) : 0.0f;
            for (uint32_t k = 0; k < input_dim; k++) {
                sum = fmaf(bf16_value(input[row * input_dim + k]),
                           bf16_value(weight[column * input_dim + k]), sum);
            }
            output[row * output_dim + column] = bf16_bits(sum);
        }
    }
}

static void test_linear_bf16(h3_gpu *gpu) {
    struct { uint32_t rows, input_dim, output_dim; int has_bias; } cases[] = {
        {20, 32, 48, 1}, {3, 5, 7, 0}, {16, 16, 16, 1}, {17, 33, 31, 1}
    };
    for (size_t case_index = 0; case_index < sizeof(cases) / sizeof(cases[0]);
         case_index++) {
        uint32_t rows = cases[case_index].rows;
        uint32_t input_dim = cases[case_index].input_dim;
        uint32_t output_dim = cases[case_index].output_dim;
        int has_bias = cases[case_index].has_bias;
        size_t input_count = (size_t)rows * input_dim;
        size_t weight_count = (size_t)output_dim * input_dim;
        uint16_t *input = malloc(input_count * 2);
        uint16_t *weight = malloc(weight_count * 2);
        uint16_t *bias = malloc(output_dim * 2);
        uint16_t *expected = malloc((size_t)rows * output_dim * 2);
        uint16_t *got = malloc((size_t)rows * output_dim * 2);
        for (size_t index = 0; index < input_count; index++)
            input[index] = bf16_bits((float)(sin((double)index * 0.53) * 2.0));
        for (size_t index = 0; index < weight_count; index++)
            weight[index] = bf16_bits(
                (float)(cos((double)index * 0.71) * 0.5));
        for (size_t index = 0; index < output_dim; index++)
            bias[index] = bf16_bits((float)(sin((double)index * 0.29) * 0.1));
        linear_reference(input, weight, has_bias ? bias : NULL, has_bias,
                         expected, rows, input_dim, output_dim);
        h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, input, input_count);
        h3_gpu_tensor *w = h3_gpu_tensor_from_bf16(gpu, weight, weight_count);
        h3_gpu_tensor *b = h3_gpu_tensor_from_bf16(gpu, bias, output_dim);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu,
                                                    (size_t)rows * output_dim);
        CHECK(in && w && b && out);
        if (!failed) {
            h3_gpu_begin(gpu);
            CHECK(h3_gpu_linear_bf16(gpu, out, in, w,
                                     has_bias ? b : NULL, rows, input_dim,
                                     output_dim) == 0);
            CHECK(h3_gpu_submit(gpu) == 0);
            CHECK(h3_gpu_tensor_read_bf16(out, got,
                                          (size_t)rows * output_dim) == 0);
            int same = memcmp(got, expected,
                              (size_t)rows * output_dim * 2) == 0;
            if (!same) {
                for (size_t index = 0;
                     index < (size_t)rows * output_dim; index++) {
                    if (got[index] != expected[index]) {
                        fprintf(stderr,
                                "  linear %ux%ux%u[%zu]: got %u expected %u\n",
                                rows, input_dim, output_dim, index,
                                got[index], expected[index]);
                        break;
                    }
                }
            }
            CHECK(same);
        }
        h3_gpu_tensor_free(in);
        h3_gpu_tensor_free(w);
        h3_gpu_tensor_free(b);
        h3_gpu_tensor_free(out);
        free(input); free(weight); free(bias); free(expected); free(got);
    }
}

static void test_linear_f32(h3_gpu *gpu) {
    struct { uint32_t rows, input_dim, output_dim; int has_bias; } cases[] = {
        {20, 32, 48, 1}, {3, 5, 7, 0}, {16, 16, 16, 1}, {17, 33, 31, 1}
    };
    for (size_t case_index = 0; case_index < sizeof(cases) / sizeof(cases[0]);
         case_index++) {
        uint32_t rows = cases[case_index].rows;
        uint32_t input_dim = cases[case_index].input_dim;
        uint32_t output_dim = cases[case_index].output_dim;
        int has_bias = cases[case_index].has_bias;
        size_t input_count = (size_t)rows * input_dim;
        size_t weight_count = (size_t)output_dim * input_dim;
        float *input = malloc(input_count * sizeof(float));
        float *weight = malloc(weight_count * sizeof(float));
        float *bias = malloc(output_dim * sizeof(float));
        float *expected = malloc((size_t)rows * output_dim * sizeof(float));
        float *got = malloc((size_t)rows * output_dim * sizeof(float));
        for (size_t index = 0; index < input_count; index++)
            input[index] = (float)(sin((double)index * 0.53) * 2.0);
        for (size_t index = 0; index < weight_count; index++)
            weight[index] = (float)(cos((double)index * 0.71) * 0.5);
        for (size_t index = 0; index < output_dim; index++)
            bias[index] = (float)(sin((double)index * 0.29) * 0.1);
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t column = 0; column < output_dim; column++) {
                float sum = has_bias ? bias[column] : 0.0f;
                for (uint32_t k = 0; k < input_dim; k++) {
                    sum = fmaf(input[row * input_dim + k],
                               weight[column * input_dim + k], sum);
                }
                expected[row * output_dim + column] = sum;
            }
        }
        h3_gpu_tensor *in = h3_gpu_tensor_from_f32(gpu, input, input_count);
        h3_gpu_tensor *w = h3_gpu_tensor_from_f32(gpu, weight, weight_count);
        h3_gpu_tensor *b = h3_gpu_tensor_from_f32(gpu, bias, output_dim);
        h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu,
                                                   (size_t)rows * output_dim);
        CHECK(in && w && b && out);
        if (!failed) {
            h3_gpu_begin(gpu);
            CHECK(h3_gpu_linear_f32(gpu, out, in, w, has_bias ? b : NULL,
                                    rows, input_dim, output_dim) == 0);
            CHECK(h3_gpu_submit(gpu) == 0);
            CHECK(h3_gpu_tensor_read_f32(out, got,
                                         (size_t)rows * output_dim) == 0);
            CHECK(memcmp(got, expected,
                         (size_t)rows * output_dim * sizeof(float)) == 0);
            if (memcmp(got, expected,
                       (size_t)rows * output_dim * sizeof(float)) != 0) {
                for (size_t index = 0;
                     index < (size_t)rows * output_dim; index++) {
                    if (got[index] != expected[index]) {
                        fprintf(stderr,
                                "  linear f32 %ux%ux%u[%zu]: got %.9g "
                                "expected %.9g\n",
                                rows, input_dim, output_dim, index,
                                got[index], expected[index]);
                        break;
                    }
                }
            }
        }
        h3_gpu_tensor_free(in);
        h3_gpu_tensor_free(w);
        h3_gpu_tensor_free(b);
        h3_gpu_tensor_free(out);
        free(input); free(weight); free(bias); free(expected); free(got);
    }
}

static void test_scale_add_f32(h3_gpu *gpu) {
    uint32_t rows = 13, width = 37;
    size_t count = (size_t)rows * width;
    float *residual = malloc(count * sizeof(float));
    float *branch = malloc(count * sizeof(float));
    float *scale = malloc(width * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *got = malloc(count * sizeof(float));
    for (size_t index = 0; index < count; index++)
        residual[index] = (float)(sin((double)index * 0.31) * 3.0);
    for (size_t index = 0; index < count; index++)
        branch[index] = (float)(cos((double)index * 0.17) * 2.0);
    for (size_t index = 0; index < width; index++)
        scale[index] = (float)(sin((double)index * 0.43) * 0.5);
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < width; column++) {
            size_t index = (size_t)row * width + column;
            expected[index] = fmaf(branch[index], scale[column],
                                   residual[index]);
        }
    }
    h3_gpu_tensor *res = h3_gpu_tensor_from_f32(gpu, residual, count);
    h3_gpu_tensor *br = h3_gpu_tensor_from_f32(gpu, branch, count);
    h3_gpu_tensor *sc = h3_gpu_tensor_from_f32(gpu, scale, width);
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, count);
    CHECK(res && br && sc && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_scale_add_f32(gpu, out, res, br, sc, rows, width) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        CHECK(h3_gpu_tensor_read_f32(out, got, count) == 0);
        CHECK(memcmp(got, expected, count * sizeof(float)) == 0);
        if (memcmp(got, expected, count * sizeof(float)) != 0) {
            for (size_t index = 0; index < count; index++) {
                if (got[index] != expected[index]) {
                    fprintf(stderr, "  scale-add[%zu]: got %.9g expected %.9g\n",
                            index, got[index], expected[index]);
                    break;
                }
            }
        }
    }
    h3_gpu_tensor_free(res);
    h3_gpu_tensor_free(br);
    h3_gpu_tensor_free(sc);
    h3_gpu_tensor_free(out);
    free(residual); free(branch); free(scale); free(expected); free(got);
}

static void test_swiglu_bf16(h3_gpu *gpu) {
    uint32_t rows = 9, width = 41;
    size_t count = (size_t)rows * width;
    uint16_t *fused = malloc(count * 2 * 2);
    uint16_t *expected = malloc(count * 2);
    uint16_t *got = malloc(count * 2);
    for (size_t index = 0; index < count * 2; index++)
        fused[index] = bf16_bits((float)(sin((double)index * 0.63) * 1.5));
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < width; column++) {
            float gate = bf16_value(fused[row * width * 2 + column]);
            float up = bf16_value(
                fused[row * width * 2 + width + column]);
            expected[row * width + column] =
                bf16_bits(gate / (1.0f + expf(-gate)) * up);
        }
    }
    h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, fused, count * 2);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
    CHECK(in && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_swiglu_bf16(gpu, out, in, rows, width) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        CHECK(h3_gpu_tensor_read_bf16(out, got, count) == 0);
        CHECK(memcmp(got, expected, count * 2) == 0);
        if (memcmp(got, expected, count * 2) != 0) {
            for (size_t index = 0; index < count; index++) {
                if (got[index] != expected[index]) {
                    fprintf(stderr, "  swiglu bf16[%zu]: got %u expected %u\n",
                            index, got[index], expected[index]);
                    break;
                }
            }
        }
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(out);
    free(fused); free(expected); free(got);
}

static void test_copy(h3_gpu *gpu) {
    size_t count = 1000;
    uint16_t *bf16_values = malloc(count * 2);
    float *f32_values = malloc(count * sizeof(float));
    for (size_t index = 0; index < count; index++) {
        bf16_values[index] = bf16_bits((float)(sin((double)index * 0.37)));
        f32_values[index] = (float)(cos((double)index * 0.11) * 2.0);
    }
    h3_gpu_tensor *src16 = h3_gpu_tensor_from_bf16(gpu, bf16_values, count);
    h3_gpu_tensor *dst16 = h3_gpu_tensor_new_bf16(gpu, count + 32);
    h3_gpu_tensor *src32 = h3_gpu_tensor_from_f32(gpu, f32_values, count);
    h3_gpu_tensor *dst32 = h3_gpu_tensor_new_f32(gpu, count + 32);
    CHECK(src16 && dst16 && src32 && dst32);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_copy_bf16(gpu, dst16, 16, src16, 8, count - 24) == 0);
        CHECK(h3_gpu_copy_f32(gpu, dst32, 16, src32, 8, count - 24) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t *got16 = malloc((count + 32) * 2);
        float *got32 = malloc((count + 32) * sizeof(float));
        CHECK(h3_gpu_tensor_read_bf16(dst16, got16, count + 32) == 0);
        CHECK(h3_gpu_tensor_read_f32_range(dst32, 16, got32,
                                           count - 24) == 0);
        CHECK(memcmp(got16 + 16, bf16_values + 8, (count - 24) * 2) == 0);
        CHECK(memcmp(got32, f32_values + 8, (count - 24) * sizeof(float)) == 0);
        free(got16);
        free(got32);
    }
    h3_gpu_tensor_free(src16);
    h3_gpu_tensor_free(dst16);
    h3_gpu_tensor_free(src32);
    h3_gpu_tensor_free(dst32);
    free(bf16_values);
    free(f32_values);
}

static void test_mlp_bf16(h3_gpu *gpu) {
    enum { ROWS = 8, INPUT_DIM = 16, HIDDEN = 12, OUTPUT_DIM = 20 };
    size_t fc1_count = (size_t)HIDDEN * 2 * INPUT_DIM;
    size_t fc2_count = (size_t)OUTPUT_DIM * HIDDEN;
    uint16_t input[ROWS * INPUT_DIM], fc1_w[fc1_count], fc2_w[fc2_count];
    uint16_t expected[ROWS * OUTPUT_DIM];
    for (size_t index = 0; index < ROWS * INPUT_DIM; index++)
        input[index] = bf16_bits((float)(sin((double)index * 0.37) * 2.0));
    for (size_t index = 0; index < fc1_count; index++)
        fc1_w[index] = bf16_bits((float)(cos((double)index * 0.83) * 0.4));
    for (size_t index = 0; index < fc2_count; index++)
        fc2_w[index] = bf16_bits((float)(sin((double)index * 0.47) * 0.4));
    uint16_t fc1[ROWS * HIDDEN * 2], act[ROWS * HIDDEN];
    linear_reference(input, fc1_w, NULL, 0, fc1, ROWS, INPUT_DIM,
                     HIDDEN * 2);
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t column = 0; column < HIDDEN; column++) {
            float gate = bf16_value(fc1[row * HIDDEN * 2 + column]);
            float up = bf16_value(fc1[row * HIDDEN * 2 + HIDDEN + column]);
            act[row * HIDDEN + column] =
                bf16_bits(gate / (1.0f + expf(-gate)) * up);
        }
    }
    linear_reference(act, fc2_w, NULL, 0, expected, ROWS, HIDDEN, OUTPUT_DIM);
    h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, input, ROWS * INPUT_DIM);
    h3_gpu_tensor *w1 = h3_gpu_tensor_from_bf16(gpu, fc1_w, fc1_count);
    h3_gpu_tensor *w2 = h3_gpu_tensor_from_bf16(gpu, fc2_w, fc2_count);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    CHECK(in && w1 && w2 && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_mlp_bf16(gpu, out, in, w1, w2, ROWS, INPUT_DIM, HIDDEN,
                              OUTPUT_DIM) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[ROWS * OUTPUT_DIM];
        CHECK(h3_gpu_tensor_read_bf16(out, got, ROWS * OUTPUT_DIM) == 0);
        int same = memcmp(got, expected, sizeof(got)) == 0;
        if (!same) {
            for (size_t index = 0; index < ROWS * OUTPUT_DIM; index++) {
                if (got[index] != expected[index]) {
                    fprintf(stderr,
                            "  mlp[%zu]: got %u expected %u\n", index,
                            got[index], expected[index]);
                    break;
                }
            }
        }
        CHECK(same);
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(w1);
    h3_gpu_tensor_free(w2);
    h3_gpu_tensor_free(out);
}

static void test_adaln_bf16(h3_gpu *gpu) {
    enum { ROWS = 4, WIDTH = 128, SLOTS = 3 };
    const float epsilon = 1e-5f;
    uint16_t input[ROWS * WIDTH], norm_w[WIDTH];
    uint16_t mod[ROWS * SLOTS * WIDTH];
    uint32_t row_map[ROWS] = {2, 0, 1, 2};
    uint16_t expected[ROWS * WIDTH];
    for (size_t index = 0; index < ROWS * WIDTH; index++)
        input[index] = bf16_bits((float)(sin((double)index * 0.41) * 2.5));
    for (size_t index = 0; index < WIDTH; index++)
        norm_w[index] = bf16_bits((float)cos((double)index * 0.17) * 0.5f + 1.0f);
    for (size_t index = 0; index < ROWS * SLOTS * WIDTH; index++)
        mod[index] = bf16_bits((float)sin((double)index * 0.29) * 0.3f);
    for (uint32_t row = 0; row < ROWS; row++) {
        float squared[WIDTH];
        float base = (float)row * WIDTH;
        for (size_t column = 0; column < WIDTH; column++) {
            float value = bf16_value(input[(size_t)row * WIDTH + column]);
            squared[column] = fmaf(value, value, 0.0f);
        }
        float inverse = 1.0f / sqrtf(tree_sum256(squared, WIDTH) / (float)WIDTH +
                                     epsilon);
        uint32_t mrow = row_map[row];
        size_t mbase = (size_t)mrow * SLOTS * WIDTH;
        for (size_t column = 0; column < WIDTH; column++) {
            float normalized = bf16_value(input[(size_t)row * WIDTH + column]) *
                               inverse * bf16_value(norm_w[column]);
            float shift = bf16_value(mod[mbase + 1 * WIDTH + column]);
            float scale = bf16_value(mod[mbase + 2 * WIDTH + column]);
            expected[(size_t)row * WIDTH + column] =
                bf16_bits(normalized * (1.0f + scale) + shift);
        }
    }
    h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, input, ROWS * WIDTH);
    h3_gpu_tensor *w = h3_gpu_tensor_from_bf16(gpu, norm_w, WIDTH);
    h3_gpu_tensor *m = h3_gpu_tensor_from_bf16(gpu, mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(in && w && m && map && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_adaln_bf16(gpu, out, in, w, m, map, ROWS, WIDTH, SLOTS,
                                1, 2, epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[ROWS * WIDTH];
        CHECK(h3_gpu_tensor_read_bf16(out, got, ROWS * WIDTH) == 0);
        CHECK(check_bf16_ulp(got, expected, ROWS * WIDTH, 2, "adaln"));
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(m);
    h3_gpu_tensor_free(map);
    h3_gpu_tensor_free(out);
}

static void test_adaln_offset(h3_gpu *gpu) {
    /* Offset view: run rows 2..3 of a 4-row buffer through AdaLN. */
    enum { ROWS = 4, WIDTH = 64, SLOTS = 2 };
    const float epsilon = 1e-5f;
    uint16_t input[ROWS * WIDTH], norm_w[WIDTH];
    uint16_t mod[ROWS * SLOTS * WIDTH];
    uint32_t row_map[2] = {0, 1};
    uint16_t expected[2 * WIDTH];
    for (size_t index = 0; index < ROWS * WIDTH; index++)
        input[index] = bf16_bits((float)(sin((double)index * 0.33) * 2.0));
    for (size_t index = 0; index < WIDTH; index++)
        norm_w[index] = bf16_bits(1.0f);
    for (size_t index = 0; index < ROWS * SLOTS * WIDTH; index++)
        mod[index] = bf16_bits((float)cos((double)index * 0.21) * 0.2f);
    for (uint32_t row = 0; row < 2; row++) {
        float squared[WIDTH];
        uint32_t src = 2 + row;
        for (size_t column = 0; column < WIDTH; column++) {
            float value = bf16_value(input[(size_t)src * WIDTH + column]);
            squared[column] = fmaf(value, value, 0.0f);
        }
        float inverse = 1.0f / sqrtf(tree_sum256(squared, WIDTH) / (float)WIDTH +
                                     epsilon);
        size_t mbase = (size_t)row_map[row] * SLOTS * WIDTH;
        for (size_t column = 0; column < WIDTH; column++) {
            float normalized = bf16_value(input[(size_t)src * WIDTH + column]) *
                               inverse * bf16_value(norm_w[column]);
            float shift = bf16_value(mod[mbase + column]);
            float scale = bf16_value(mod[mbase + WIDTH + column]);
            expected[(size_t)row * WIDTH + column] =
                bf16_bits(normalized * (1.0f + scale) + shift);
        }
    }
    h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, input, ROWS * WIDTH);
    h3_gpu_tensor *w = h3_gpu_tensor_from_bf16(gpu, norm_w, WIDTH);
    h3_gpu_tensor *m = h3_gpu_tensor_from_bf16(gpu, mod,
                                                ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(gpu, row_map, 2);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, 2 * WIDTH);
    CHECK(in && w && m && map && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_adaln_bf16_offset(gpu, out, in, 2 * WIDTH, w, m, map,
                                       2, WIDTH, SLOTS, 0, 1, epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[2 * WIDTH];
        CHECK(h3_gpu_tensor_read_bf16(out, got, 2 * WIDTH) == 0);
        CHECK(check_bf16_ulp(got, expected, 2 * WIDTH, 2, "adaln_offset"));
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(m);
    h3_gpu_tensor_free(map);
    h3_gpu_tensor_free(out);
}

static void test_gate_bf16(h3_gpu *gpu) {
    enum { ROWS = 3, WIDTH = 64, SLOTS = 2 };
    uint16_t residual[ROWS * WIDTH], branch[ROWS * WIDTH];
    uint16_t mod[ROWS * SLOTS * WIDTH];
    uint32_t row_map[ROWS] = {1, 0, 1};
    uint16_t expected[ROWS * WIDTH];
    for (size_t index = 0; index < ROWS * WIDTH; index++) {
        residual[index] = bf16_bits((float)(sin((double)index * 0.19) * 2.0));
        branch[index] = bf16_bits((float)(cos((double)index * 0.27) * 1.5));
    }
    for (size_t index = 0; index < ROWS * SLOTS * WIDTH; index++)
        mod[index] = bf16_bits((float)sin((double)index * 0.13) * 0.4f);
    for (uint32_t row = 0; row < ROWS; row++) {
        size_t mbase = (size_t)row_map[row] * SLOTS * WIDTH;
        for (size_t column = 0; column < WIDTH; column++) {
            size_t index = (size_t)row * WIDTH + column;
            float gate = bf16_value(mod[mbase + WIDTH + column]);
            expected[index] = bf16_bits(bf16_value(residual[index]) +
                                        bf16_value(branch[index]) * gate);
        }
    }
    h3_gpu_tensor *r = h3_gpu_tensor_from_bf16(gpu, residual, ROWS * WIDTH);
    h3_gpu_tensor *b = h3_gpu_tensor_from_bf16(gpu, branch, ROWS * WIDTH);
    h3_gpu_tensor *m = h3_gpu_tensor_from_bf16(gpu, mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(r && b && m && map && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_gate_bf16(gpu, out, r, b, m, map, ROWS, WIDTH, SLOTS,
                               1) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[ROWS * WIDTH];
        CHECK(h3_gpu_tensor_read_bf16(out, got, ROWS * WIDTH) == 0);
        CHECK(memcmp(got, expected, sizeof(got)) == 0);
    }
    h3_gpu_tensor_free(r);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(m);
    h3_gpu_tensor_free(map);
    h3_gpu_tensor_free(out);
}

static void test_gate_adaln_bf16(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 256, SLOTS = 2 };
    const float epsilon = 1e-5f;
    uint16_t residual[ROWS * WIDTH], branch[ROWS * WIDTH], nw[WIDTH];
    uint16_t gmod[ROWS * SLOTS * WIDTH], nmod[ROWS * SLOTS * WIDTH];
    uint32_t row_map[ROWS] = {0, 1};
    uint16_t expected[ROWS * WIDTH];
    uint16_t gated_expected[ROWS * WIDTH];
    for (size_t index = 0; index < ROWS * WIDTH; index++) {
        residual[index] = bf16_bits((float)(sin((double)index * 0.23) * 2.0));
        branch[index] = bf16_bits((float)(cos((double)index * 0.31) * 1.5));
    }
    for (size_t index = 0; index < WIDTH; index++)
        nw[index] = bf16_bits((float)sin((double)index * 0.11) * 0.5f + 1.0f);
    for (size_t index = 0; index < ROWS * SLOTS * WIDTH; index++) {
        gmod[index] = bf16_bits((float)cos((double)index * 0.07) * 0.3f);
        nmod[index] = bf16_bits((float)sin((double)index * 0.17) * 0.2f);
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        float squared[WIDTH];
        size_t mbase = (size_t)row_map[row] * SLOTS * WIDTH;
        for (size_t column = 0; column < WIDTH; column++) {
            size_t index = (size_t)row * WIDTH + column;
            float gate = bf16_value(gmod[mbase + WIDTH + column]);
            float gated = bf16_value(residual[index]) +
                          bf16_value(branch[index]) * gate;
            uint16_t gated_b = bf16_bits(gated);
            float value = bf16_value(gated_b);
            squared[column] = fmaf(value, value, 0.0f);
        }
        float inverse = 1.0f / sqrtf(tree_sum256(squared, WIDTH) / (float)WIDTH +
                                     epsilon);
        for (size_t column = 0; column < WIDTH; column++) {
            size_t index = (size_t)row * WIDTH + column;
            float gate = bf16_value(gmod[mbase + WIDTH + column]);
            uint16_t gated_b = bf16_bits(bf16_value(residual[index]) +
                                         bf16_value(branch[index]) * gate);
            float normalized = bf16_value(gated_b) * inverse *
                               bf16_value(nw[column]);
            float shift = bf16_value(nmod[mbase + column]);
            float scale = bf16_value(nmod[mbase + WIDTH + column]);
            expected[index] = bf16_bits(normalized * (1.0f + scale) + shift);
        }
    }
    h3_gpu_tensor *r = h3_gpu_tensor_from_bf16(gpu, residual, ROWS * WIDTH);
    h3_gpu_tensor *b = h3_gpu_tensor_from_bf16(gpu, branch, ROWS * WIDTH);
    h3_gpu_tensor *n = h3_gpu_tensor_from_bf16(gpu, nw, WIDTH);
    h3_gpu_tensor *g = h3_gpu_tensor_from_bf16(gpu, gmod,
                                                ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *m = h3_gpu_tensor_from_bf16(gpu, nmod,
                                                ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *gated = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(r && b && n && g && m && map && gated && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_gate_adaln_bf16(gpu, gated, out, r, b, n, g, m, map,
                                     ROWS, WIDTH, SLOTS, 1, 0, 1,
                                     epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[ROWS * WIDTH];
        CHECK(h3_gpu_tensor_read_bf16(gated, got, ROWS * WIDTH) == 0);
        for (uint32_t row = 0; row < ROWS; row++) {
            size_t mbase = (size_t)row_map[row] * SLOTS * WIDTH;
            for (size_t column = 0; column < WIDTH; column++) {
                size_t index = (size_t)row * WIDTH + column;
                float gate = bf16_value(gmod[mbase + WIDTH + column]);
                gated_expected[index] = bf16_bits(
                    bf16_value(residual[index]) +
                    bf16_value(branch[index]) * gate);
            }
        }
        CHECK(memcmp(got, gated_expected, sizeof(got)) == 0);
        CHECK(h3_gpu_tensor_read_bf16(out, got, ROWS * WIDTH) == 0);
        CHECK(check_bf16_ulp(got, expected, ROWS * WIDTH, 2, "gate_adaln"));
    }
    h3_gpu_tensor_free(r);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(n);
    h3_gpu_tensor_free(g);
    h3_gpu_tensor_free(m);
    h3_gpu_tensor_free(map);
    h3_gpu_tensor_free(gated);
    h3_gpu_tensor_free(out);
}

static void test_adaln_linear_bf16(h3_gpu *gpu) {
    enum { ROWS = 3, WIDTH = 64, OUT = 32, SLOTS = 2 };
    const float epsilon = 1e-5f;
    uint16_t input[ROWS * WIDTH], nw[WIDTH], mod[ROWS * SLOTS * WIDTH],
             weight[OUT * WIDTH], bias[OUT];
    uint32_t row_map[ROWS] = {1, 0, 1};
    uint16_t expected[ROWS * OUT];
    for (size_t index = 0; index < ROWS * WIDTH; index++)
        input[index] = bf16_bits((float)(sin((double)index * 0.37) * 2.0));
    for (size_t index = 0; index < WIDTH; index++)
        nw[index] = bf16_bits(1.0f);
    for (size_t index = 0; index < ROWS * SLOTS * WIDTH; index++)
        mod[index] = bf16_bits((float)cos((double)index * 0.19) * 0.2f);
    for (size_t index = 0; index < OUT * WIDTH; index++)
        weight[index] = bf16_bits((float)sin((double)index * 0.43) * 0.4f);
    for (size_t index = 0; index < OUT; index++)
        bias[index] = bf16_bits((float)cos((double)index * 0.29) * 0.1f);
    for (uint32_t row = 0; row < ROWS; row++) {
        float squared[WIDTH];
        for (size_t column = 0; column < WIDTH; column++) {
            float value = bf16_value(input[(size_t)row * WIDTH + column]);
            squared[column] = fmaf(value, value, 0.0f);
        }
        float inverse = 1.0f / sqrtf(tree_sum256(squared, WIDTH) / (float)WIDTH +
                                     epsilon);
        size_t mbase = (size_t)row_map[row] * SLOTS * WIDTH;
        for (size_t column = 0; column < OUT; column++) {
            float sum = bf16_value(bias[column]);
            for (size_t k = 0; k < WIDTH; k++) {
                float value = bf16_value(input[(size_t)row * WIDTH + k]);
                float shift = bf16_value(mod[mbase + k]);
                float scale = bf16_value(mod[mbase + WIDTH + k]);
                float normed = value * inverse * bf16_value(nw[k]);
                uint16_t normed_b = bf16_bits(normed * (1.0f + scale) + shift);
                sum = fmaf(bf16_value(normed_b),
                           bf16_value(weight[(size_t)column * WIDTH + k]),
                           sum);
            }
            expected[(size_t)row * OUT + column] = bf16_bits(sum);
        }
    }
    h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, input, ROWS * WIDTH);
    h3_gpu_tensor *n = h3_gpu_tensor_from_bf16(gpu, nw, WIDTH);
    h3_gpu_tensor *m = h3_gpu_tensor_from_bf16(gpu, mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *w = h3_gpu_tensor_from_bf16(gpu, weight, OUT * WIDTH);
    h3_gpu_tensor *b = h3_gpu_tensor_from_bf16(gpu, bias, OUT);
    h3_gpu_tensor *inv = h3_gpu_tensor_new_f32(gpu, ROWS);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, ROWS * OUT);
    CHECK(in && n && m && map && w && b && inv && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_adaln_linear_bf16(gpu, out, inv, in, 0, n, m, map, w, b,
                                       ROWS, WIDTH, OUT, SLOTS, 0, 1,
                                       epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[ROWS * OUT];
        CHECK(h3_gpu_tensor_read_bf16(out, got, ROWS * OUT) == 0);
        CHECK(check_bf16_ulp(got, expected, ROWS * OUT, 2, "adaln_linear"));
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(n);
    h3_gpu_tensor_free(m);
    h3_gpu_tensor_free(map);
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(inv);
    h3_gpu_tensor_free(out);
}

static void test_head_rms_norm_bf16(h3_gpu *gpu) {
    enum { SEQ = 4, HEADS = 6, DIM = 32 };
    const float epsilon = 1e-5f;
    uint16_t tensor[SEQ * HEADS * DIM], weight[DIM];
    uint16_t expected[SEQ * HEADS * DIM];
    for (size_t index = 0; index < SEQ * HEADS * DIM; index++)
        tensor[index] = bf16_bits((float)(sin((double)index * 0.51) * 1.5));
    for (size_t index = 0; index < DIM; index++)
        weight[index] = bf16_bits((float)cos((double)index * 0.13) * 0.5f + 1.0f);
    for (uint32_t row = 0; row < SEQ; row++) {
        for (uint32_t head = 0; head < HEADS; head++) {
            size_t base = ((size_t)row * HEADS + head) * DIM;
            float sum = 0.0f;
            for (uint32_t d = 0; d < DIM; d++) {
                float value = bf16_value(tensor[base + d]);
                sum = fmaf(value, value, sum);
            }
            float inverse = 1.0f / sqrtf(sum / (float)DIM + epsilon);
            for (uint32_t d = 0; d < DIM; d++) {
                float value = bf16_value(tensor[base + d]);
                expected[base + d] =
                    bf16_bits(value * inverse * bf16_value(weight[d]));
            }
        }
    }
    h3_gpu_tensor *t = h3_gpu_tensor_from_bf16(gpu, tensor, SEQ * HEADS * DIM);
    h3_gpu_tensor *w = h3_gpu_tensor_from_bf16(gpu, weight, DIM);
    CHECK(t && w);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_head_rms_norm_bf16(gpu, t, w, SEQ, HEADS, DIM,
                                        epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[SEQ * HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(t, got, SEQ * HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(got, expected, SEQ * HEADS * DIM, 2, "head_rms"));
    }
    h3_gpu_tensor_free(t);
    h3_gpu_tensor_free(w);
}

static void test_grouped_qkv_rope_bf16(h3_gpu *gpu) {
    enum { SEQ = 4, HEADS = 3, DIM = 8, HALF = 4 };
    const float epsilon = 1e-5f;
    uint16_t qkv[SEQ * HEADS * 3 * DIM], qn[DIM], kn[DIM];
    uint16_t rcos[SEQ * HALF], rsin[SEQ * HALF];
    uint16_t expected_q[SEQ * HEADS * DIM], expected_k[SEQ * HEADS * DIM];
    for (size_t index = 0; index < SEQ * HEADS * 3 * DIM; index++)
        qkv[index] = bf16_bits((float)(sin((double)index * 0.37) * 1.5));
    for (size_t index = 0; index < DIM; index++) {
        qn[index] = bf16_bits((float)cos((double)index * 0.19) * 0.5f + 1.0f);
        kn[index] = bf16_bits((float)sin((double)index * 0.23) * 0.5f + 1.0f);
    }
    for (size_t index = 0; index < SEQ * HALF; index++) {
        rcos[index] = bf16_bits(cosf((float)index * 0.31f));
        rsin[index] = bf16_bits(sinf((float)index * 0.31f));
    }
    for (uint32_t row = 0; row < SEQ; row++) {
        for (uint32_t head = 0; head < HEADS; head++) {
            uint32_t q_base = (row * HEADS * 3 + head * 3) * DIM;
            uint32_t k_base = q_base + DIM;
            float q_sum = 0.0f, k_sum = 0.0f;
            for (uint32_t d = 0; d < DIM; d++) {
                float q = bf16_value(qkv[q_base + d]);
                float k = bf16_value(qkv[k_base + d]);
                q_sum = fmaf(q, q, q_sum);
                k_sum = fmaf(k, k, k_sum);
            }
            float qi = 1.0f / sqrtf(q_sum / (float)DIM + epsilon);
            float ki = 1.0f / sqrtf(k_sum / (float)DIM + epsilon);
            for (uint32_t d = 0; d < DIM; d++) {
                float q0 = bf16_value(qkv[q_base + d]) * qi * bf16_value(qn[d]);
                float k0 = bf16_value(qkv[k_base + d]) * ki * bf16_value(kn[d]);
                if (d < HALF) {
                    float q1 = bf16_value(qkv[q_base + d + HALF]) * qi *
                               bf16_value(qn[d + HALF]);
                    float k1 = bf16_value(qkv[k_base + d + HALF]) * ki *
                               bf16_value(kn[d + HALF]);
                    float c = bf16_value(rcos[row * HALF + d]);
                    float s = bf16_value(rsin[row * HALF + d]);
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else {
                    uint32_t pair = d - HALF;
                    float q1 = bf16_value(qkv[q_base + pair]) * qi *
                               bf16_value(qn[pair]);
                    float k1 = bf16_value(qkv[k_base + pair]) * ki *
                               bf16_value(kn[pair]);
                    float c = bf16_value(rcos[row * HALF + pair]);
                    float s = bf16_value(rsin[row * HALF + pair]);
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                uint32_t index = (row * HEADS + head) * DIM + d;
                expected_q[index] = bf16_bits(q0);
                expected_k[index] = bf16_bits(k0);
            }
        }
    }
    h3_gpu_tensor *t = h3_gpu_tensor_from_bf16(gpu, qkv, SEQ * HEADS * 3 * DIM);
    h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, qn, DIM);
    h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, kn, DIM);
    h3_gpu_tensor *c = h3_gpu_tensor_from_bf16(gpu, rcos, SEQ * HALF);
    h3_gpu_tensor *s = h3_gpu_tensor_from_bf16(gpu, rsin, SEQ * HALF);
    h3_gpu_tensor *oq = h3_gpu_tensor_new_bf16(gpu, SEQ * HEADS * DIM);
    h3_gpu_tensor *ok = h3_gpu_tensor_new_bf16(gpu, SEQ * HEADS * DIM);
    h3_gpu_tensor *ov = h3_gpu_tensor_new_bf16(gpu, SEQ * HEADS * DIM);
    CHECK(t && q && k && c && s && oq && ok && ov);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_grouped_qkv_rope_bf16(gpu, oq, ok, ov, t, q, k, c, s,
                                           SEQ, HEADS, DIM, HALF,
                                           epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[SEQ * HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(oq, got, SEQ * HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(got, expected_q, SEQ * HEADS * DIM, 2,
                             "qkv_q"));
        CHECK(h3_gpu_tensor_read_bf16(ok, got, SEQ * HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(got, expected_k, SEQ * HEADS * DIM, 2,
                             "qkv_k"));
        /* value is copied verbatim from the grouped layout */
        uint16_t gotv[SEQ * HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(ov, gotv, SEQ * HEADS * DIM) == 0);
        int same = 1;
        for (uint32_t row = 0; row < SEQ && same; row++)
            for (uint32_t head = 0; head < HEADS && same; head++)
                for (uint32_t d = 0; d < DIM; d++)
                    if (gotv[(row * HEADS + head) * DIM + d] !=
                        qkv[(row * HEADS * 3 + head * 3 + 2) * DIM + d])
                        same = 0;
        CHECK(same);
    }
    h3_gpu_tensor_free(t);
    h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(c);
    h3_gpu_tensor_free(s);
    h3_gpu_tensor_free(oq);
    h3_gpu_tensor_free(ok);
    h3_gpu_tensor_free(ov);
}

static void test_qkv_rope_plain_bf16(h3_gpu *gpu) {
    /* Non-grouped checkpoint layout: [row][q/k/v][head][dim]. */
    enum { SEQ = 3, HEADS = 2, DIM = 8, HALF = 4 };
    const float epsilon = 1e-5f;
    uint16_t qkv[SEQ * 3 * HEADS * DIM], qn[DIM], kn[DIM];
    uint16_t rcos[SEQ * HALF], rsin[SEQ * HALF];
    uint16_t expected_q[SEQ * HEADS * DIM], expected_k[SEQ * HEADS * DIM];
    for (size_t index = 0; index < SEQ * 3 * HEADS * DIM; index++)
        qkv[index] = bf16_bits((float)(cos((double)index * 0.41) * 1.2));
    for (size_t index = 0; index < DIM; index++) {
        qn[index] = bf16_bits(1.0f);
        kn[index] = bf16_bits(1.0f);
    }
    for (size_t index = 0; index < SEQ * HALF; index++) {
        rcos[index] = bf16_bits(cosf((float)index * 0.17f));
        rsin[index] = bf16_bits(sinf((float)index * 0.17f));
    }
    for (uint32_t row = 0; row < SEQ; row++) {
        for (uint32_t head = 0; head < HEADS; head++) {
            uint32_t inner = HEADS * DIM;
            uint32_t q_base = row * inner * 3 + head * DIM;
            uint32_t k_base = q_base + inner;
            float q_sum = 0.0f, k_sum = 0.0f;
            for (uint32_t d = 0; d < DIM; d++) {
                float q = bf16_value(qkv[q_base + d]);
                float k = bf16_value(qkv[k_base + d]);
                q_sum = fmaf(q, q, q_sum);
                k_sum = fmaf(k, k, k_sum);
            }
            float qi = 1.0f / sqrtf(q_sum / (float)DIM + epsilon);
            float ki = 1.0f / sqrtf(k_sum / (float)DIM + epsilon);
            for (uint32_t d = 0; d < DIM; d++) {
                float q0 = bf16_value(qkv[q_base + d]) * qi;
                float k0 = bf16_value(qkv[k_base + d]) * ki;
                if (d < HALF) {
                    float q1 = bf16_value(qkv[q_base + d + HALF]) * qi;
                    float k1 = bf16_value(qkv[k_base + d + HALF]) * ki;
                    float c = bf16_value(rcos[row * HALF + d]);
                    float s = bf16_value(rsin[row * HALF + d]);
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else {
                    uint32_t pair = d - HALF;
                    float q1 = bf16_value(qkv[q_base + pair]) * qi;
                    float k1 = bf16_value(qkv[k_base + pair]) * ki;
                    float c = bf16_value(rcos[row * HALF + pair]);
                    float s = bf16_value(rsin[row * HALF + pair]);
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                uint32_t index = (row * HEADS + head) * DIM + d;
                expected_q[index] = bf16_bits(q0);
                expected_k[index] = bf16_bits(k0);
            }
        }
    }
    h3_gpu_tensor *t = h3_gpu_tensor_from_bf16(gpu, qkv, SEQ * 3 * HEADS * DIM);
    h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, qn, DIM);
    h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, kn, DIM);
    h3_gpu_tensor *c = h3_gpu_tensor_from_bf16(gpu, rcos, SEQ * HALF);
    h3_gpu_tensor *s = h3_gpu_tensor_from_bf16(gpu, rsin, SEQ * HALF);
    h3_gpu_tensor *oq = h3_gpu_tensor_new_bf16(gpu, SEQ * HEADS * DIM);
    h3_gpu_tensor *ok = h3_gpu_tensor_new_bf16(gpu, SEQ * HEADS * DIM);
    h3_gpu_tensor *ov = h3_gpu_tensor_new_bf16(gpu, SEQ * HEADS * DIM);
    CHECK(t && q && k && c && s && oq && ok && ov);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_qkv_rope_bf16(gpu, oq, ok, ov, t, q, k, c, s, SEQ, HEADS,
                                   DIM, HALF, epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[SEQ * HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(oq, got, SEQ * HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(got, expected_q, SEQ * HEADS * DIM, 2,
                             "qkv_plain_q"));
        CHECK(h3_gpu_tensor_read_bf16(ok, got, SEQ * HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(got, expected_k, SEQ * HEADS * DIM, 2,
                             "qkv_plain_k"));
    }
    h3_gpu_tensor_free(t);
    h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(c);
    h3_gpu_tensor_free(s);
    h3_gpu_tensor_free(oq);
    h3_gpu_tensor_free(ok);
    h3_gpu_tensor_free(ov);
}

static void test_sdpa_bf16(h3_gpu *gpu) {
    enum { SEQ = 8, HEADS = 4, DIM = 16 };
    uint16_t q[SEQ * HEADS * DIM], k[SEQ * HEADS * DIM],
             v[SEQ * HEADS * DIM];
    uint16_t expected[SEQ * HEADS * DIM];
    float scale = 1.0f / sqrtf((float)DIM);
    for (size_t index = 0; index < SEQ * HEADS * DIM; index++) {
        q[index] = bf16_bits((float)(sin((double)index * 0.53) * 1.5));
        k[index] = bf16_bits((float)(cos((double)index * 0.47) * 1.5));
        v[index] = bf16_bits((float)(sin((double)index * 0.61) * 1.0));
    }
    for (uint32_t row = 0; row < SEQ; row++) {
        for (uint32_t head = 0; head < HEADS; head++) {
            uint32_t qbase = (row * HEADS + head) * DIM;
            float scores[SEQ];
            float maxv = -3.402823466e+38f;
            for (uint32_t s = 0; s < SEQ; s++) {
                uint32_t kbase = (s * HEADS + head) * DIM;
                float dot = 0.0f;
                for (uint32_t d = 0; d < DIM; d++)
                    dot = fmaf(bf16_value(q[qbase + d]),
                               bf16_value(k[kbase + d]), dot);
                scores[s] = dot * scale;
                if (scores[s] > maxv) maxv = scores[s];
            }
            float sum = 0.0f;
            for (uint32_t s = 0; s < SEQ; s++)
                sum += expf(scores[s] - maxv);
            for (uint32_t d = 0; d < DIM; d++) {
                float acc = 0.0f;
                for (uint32_t s = 0; s < SEQ; s++) {
                    uint32_t kbase = (s * HEADS + head) * DIM;
                    acc = fmaf(expf(scores[s] - maxv),
                               bf16_value(v[kbase + d]), acc);
                }
                expected[qbase + d] = bf16_bits(acc / sum);
            }
        }
    }
    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, SEQ * HEADS * DIM);
    h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, SEQ * HEADS * DIM);
    h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, v, SEQ * HEADS * DIM);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, SEQ * HEADS * DIM);
    CHECK(tq && tk && tv && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_sdpa_bf16(gpu, out, tq, tk, tv, SEQ, HEADS, DIM,
                               scale) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[SEQ * HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(out, got, SEQ * HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(got, expected, SEQ * HEADS * DIM, 4, "sdpa"));
        /* head-major output variant: same values, transposed layout */
        h3_gpu_tensor *out2 = h3_gpu_tensor_new_bf16(gpu, SEQ * HEADS * DIM);
        CHECK(out2);
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_sdpa_bf16_head_major_output(gpu, out2, tq, tk, tv, SEQ,
                                                 HEADS, DIM, scale) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got2[SEQ * HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(out2, got2, SEQ * HEADS * DIM) == 0);
        int same = 1;
        for (uint32_t row = 0; row < SEQ && same; row++)
            for (uint32_t head = 0; head < HEADS && same; head++)
                for (uint32_t d = 0; d < DIM; d++)
                    if (got2[(head * SEQ + row) * DIM + d] !=
                        got[(row * HEADS + head) * DIM + d]) {
                        fprintf(stderr,
                                "  sdpa head-major mismatch row=%u head=%u "
                                "d=%u: %u vs %u\n",
                                row, head, d,
                                got2[(head * SEQ + row) * DIM + d],
                                got[(row * HEADS + head) * DIM + d]);
                        same = 0;
                        break;
                    }
        CHECK(same);
        h3_gpu_tensor_free(out2);
    }
    h3_gpu_tensor_free(tq);
    h3_gpu_tensor_free(tk);
    h3_gpu_tensor_free(tv);
    h3_gpu_tensor_free(out);
}

/* CPU reference for the F32 patch projection with the exact FMA order. */
static void patch_reference(const float *input, const float *weight,
                            const float *bias, int has_bias,
                            uint16_t *output, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < output_dim; column++) {
            float sum = has_bias ? bias[column] : 0.0f;
            for (uint32_t k = 0; k < input_dim; k++) {
                sum = fmaf(input[row * input_dim + k],
                           weight[column * input_dim + k], sum);
            }
            output[row * output_dim + column] = bf16_bits(sum);
        }
    }
}

static void test_patch_linear_bf16(h3_gpu *gpu) {
    enum { OUT = 5376 };
    struct { uint32_t rows, input_dim; int has_bias; } cases[] = {
        {6, 96, 1}, {9, 32, 0}, {3, 96, 0}
    };
    for (size_t case_index = 0; case_index < sizeof(cases) / sizeof(cases[0]);
         case_index++) {
        uint32_t rows = cases[case_index].rows;
        uint32_t input_dim = cases[case_index].input_dim;
        int has_bias = cases[case_index].has_bias;
        size_t input_count = (size_t)rows * input_dim;
        size_t weight_count = (size_t)OUT * input_dim;
        float *input = malloc(input_count * 4);
        float *weight = malloc(weight_count * 4);
        float *bias = malloc(OUT * 4);
        uint16_t *expected = malloc((size_t)rows * OUT * 2);
        uint16_t *got = malloc((size_t)rows * OUT * 2);
        for (size_t index = 0; index < input_count; index++)
            input[index] = (float)(sin((double)index * 0.37) * 0.8);
        for (size_t index = 0; index < weight_count; index++)
            weight[index] = (float)(cos((double)index * 0.61) * 0.05);
        for (size_t index = 0; index < OUT; index++)
            bias[index] = (float)sin((double)index * 0.29) * 0.01f;
        patch_reference(input, weight, has_bias ? bias : NULL, has_bias,
                        expected, rows, input_dim, OUT);
        h3_gpu_tensor *in = h3_gpu_tensor_from_f32(gpu, input, input_count);
        h3_gpu_tensor *w = h3_gpu_tensor_from_f32(gpu, weight, weight_count);
        h3_gpu_tensor *b = h3_gpu_tensor_from_f32(gpu, bias, OUT);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * OUT);
        CHECK(in && w && b && out);
        if (!failed) {
            h3_gpu_begin(gpu);
            CHECK(h3_gpu_patch_linear_bf16(gpu, out, in, w,
                                           has_bias ? b : NULL, rows,
                                           input_dim, OUT) == 0);
            CHECK(h3_gpu_submit(gpu) == 0);
            CHECK(h3_gpu_tensor_read_bf16(out, got,
                                          (size_t)rows * OUT) == 0);
            int same = memcmp(got, expected, (size_t)rows * OUT * 2) == 0;
            if (!same) {
                for (size_t index = 0; index < (size_t)rows * OUT; index++) {
                    if (got[index] != expected[index]) {
                        fprintf(stderr,
                                "  patch %ux%u[%zu]: got %u expected %u\n",
                                rows, input_dim, index, got[index],
                                expected[index]);
                        break;
                    }
                }
            }
            CHECK(same);
        }
        h3_gpu_tensor_free(in);
        h3_gpu_tensor_free(w);
        h3_gpu_tensor_free(b);
        h3_gpu_tensor_free(out);
        free(input); free(weight); free(bias); free(expected); free(got);
    }
}

static void test_patch_linear_offset_bf16(h3_gpu *gpu) {
    enum { ROWS = 4, IN = 32, OUT = 5376, IN_OFFSET = 2, OUT_OFFSET = 1 };
    size_t in_total = (size_t)(ROWS + IN_OFFSET) * IN;
    size_t out_total = (size_t)(ROWS + OUT_OFFSET) * OUT;
    float *input = malloc(in_total * 4);
    float *weight = malloc((size_t)OUT * IN * 4);
    uint16_t *expected = malloc((size_t)ROWS * OUT * 2);
    uint16_t *got = malloc(out_total * 2);
    for (size_t index = 0; index < in_total; index++)
        input[index] = (float)(cos((double)index * 0.43) * 0.7);
    for (size_t index = 0; index < (size_t)OUT * IN; index++)
        weight[index] = (float)(sin((double)index * 0.73) * 0.05);
    patch_reference(input + (size_t)IN_OFFSET * IN, weight, NULL, 0,
                    expected, ROWS, IN, OUT);
    h3_gpu_tensor *in = h3_gpu_tensor_from_f32(gpu, input, in_total);
    h3_gpu_tensor *w = h3_gpu_tensor_from_f32(gpu, weight, (size_t)OUT * IN);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, out_total);
    CHECK(in && w && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_patch_linear_bf16_offset(gpu, out, OUT_OFFSET * OUT, in,
                                              IN_OFFSET * IN, w, NULL, ROWS,
                                              IN, OUT) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        CHECK(h3_gpu_tensor_read_bf16(out, got, out_total) == 0);
        int same = memcmp(got + (size_t)OUT_OFFSET * OUT, expected,
                          (size_t)ROWS * OUT * 2) == 0;
        if (!same) {
            for (size_t index = 0; index < (size_t)ROWS * OUT; index++) {
                if (got[(size_t)OUT_OFFSET * OUT + index] != expected[index]) {
                    fprintf(stderr, "  patch_offset[%zu]: got %u expected %u\n",
                            index, got[(size_t)OUT_OFFSET * OUT + index],
                            expected[index]);
                    break;
                }
            }
        }
        CHECK(same);
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(out);
    free(input); free(weight); free(expected); free(got);
}

static void test_patch_linear_map_bf16(h3_gpu *gpu) {
    enum { ROWS = 5, IN = 96, OUT = 5376, OUT_ROWS = 7 };
    uint32_t row_map[ROWS] = {6, 2, 0, 4, 5};
    float *input = malloc((size_t)ROWS * IN * 4);
    float *weight = malloc((size_t)OUT * IN * 4);
    uint16_t *expected = malloc((size_t)OUT_ROWS * OUT * 2);
    uint16_t *got = malloc((size_t)OUT_ROWS * OUT * 2);
    for (size_t index = 0; index < (size_t)ROWS * IN; index++)
        input[index] = (float)(sin((double)index * 0.51) * 0.6);
    for (size_t index = 0; index < (size_t)OUT * IN; index++)
        weight[index] = (float)(cos((double)index * 0.83) * 0.05);
    memset(expected, 0, (size_t)OUT_ROWS * OUT * 2);
    uint16_t *row_out = malloc((size_t)OUT * 2);
    for (uint32_t row = 0; row < ROWS; row++) {
        patch_reference(input + (size_t)row * IN, weight, NULL, 0, row_out,
                        1, IN, OUT);
        memcpy(expected + (size_t)row_map[row] * OUT, row_out,
               (size_t)OUT * 2);
    }
    free(row_out);
    h3_gpu_tensor *in = h3_gpu_tensor_from_f32(gpu, input, (size_t)ROWS * IN);
    h3_gpu_tensor *w = h3_gpu_tensor_from_f32(gpu, weight, (size_t)OUT * IN);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, (size_t)OUT_ROWS * OUT);
    CHECK(in && w && map && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_patch_linear_bf16_map(gpu, out, in, w, NULL, map,
                                           OUT_ROWS, ROWS, IN, OUT) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        CHECK(h3_gpu_tensor_read_bf16(out, got, (size_t)OUT_ROWS * OUT) == 0);
        int same = memcmp(got, expected, (size_t)OUT_ROWS * OUT * 2) == 0;
        if (!same) {
            for (size_t index = 0; index < (size_t)OUT_ROWS * OUT; index++) {
                if (got[index] != expected[index]) {
                    fprintf(stderr, "  patch_map[%zu]: got %u expected %u\n",
                            index, got[index], expected[index]);
                    break;
                }
            }
        }
        CHECK(same);
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(map);
    h3_gpu_tensor_free(out);
    free(input); free(weight); free(expected); free(got);
}

static void test_sdpa_flash_bf16(h3_gpu *gpu) {
    /* Long sequence exercises the 128-thread flash kernel (selected
     * automatically for sequence >= 128). */
    enum { SEQ = 256, HEADS = 4, DIM = 128 };
    size_t count = (size_t)SEQ * HEADS * DIM;
    uint16_t *q = malloc(count * 2), *k = malloc(count * 2),
             *v = malloc(count * 2), *expected = malloc(count * 2);
    float scale = 1.0f / sqrtf((float)DIM);
    CHECK(q && k && v && expected);
    if (!failed) {
        for (size_t index = 0; index < count; index++) {
            q[index] = bf16_bits((float)(sin((double)index * 0.31) * 0.8));
            k[index] = bf16_bits((float)(cos((double)index * 0.43) * 0.8));
            v[index] = bf16_bits((float)(sin((double)index * 0.57) * 0.6));
        }
        for (uint32_t row = 0; row < SEQ; row++) {
            for (uint32_t head = 0; head < HEADS; head++) {
                uint32_t qbase = (row * HEADS + head) * DIM;
                float scores[SEQ];
                float maxv = -3.402823466e+38f;
                for (uint32_t s = 0; s < SEQ; s++) {
                    uint32_t kbase = (s * HEADS + head) * DIM;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < DIM; d++)
                        dot = fmaf(bf16_value(q[qbase + d]),
                                   bf16_value(k[kbase + d]), dot);
                    scores[s] = dot * scale;
                    if (scores[s] > maxv) maxv = scores[s];
                }
                float sum = 0.0f;
                for (uint32_t s = 0; s < SEQ; s++)
                    sum += expf(scores[s] - maxv);
                for (uint32_t d = 0; d < DIM; d++) {
                    float acc = 0.0f;
                    for (uint32_t s = 0; s < SEQ; s++) {
                        uint32_t kbase = (s * HEADS + head) * DIM;
                        acc = fmaf(expf(scores[s] - maxv),
                                   bf16_value(v[kbase + d]), acc);
                    }
                    expected[qbase + d] = bf16_bits(acc / sum);
                }
            }
        }
        h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, count);
        h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, count);
        h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, v, count);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
        CHECK(tq && tk && tv && out);
        if (!failed) {
            h3_gpu_begin(gpu);
            CHECK(h3_gpu_sdpa_bf16(gpu, out, tq, tk, tv, SEQ, HEADS, DIM,
                                   scale) == 0);
            CHECK(h3_gpu_submit(gpu) == 0);
            uint16_t *got = malloc(count * 2);
            CHECK(got);
            CHECK(h3_gpu_tensor_read_bf16(out, got, count) == 0);
            /* The flash kernel reduces dots with a tree instead of the
             * linear FMA order, so allow a few more ulps. */
            CHECK(check_bf16_ulp_abs(got, expected, count, 8, 2e-7f,
                                      "sdpa_flash"));
            free(got);
            h3_gpu_tensor_free(tq);
            h3_gpu_tensor_free(tk);
            h3_gpu_tensor_free(tv);
            h3_gpu_tensor_free(out);
        }
    }
    free(q); free(k); free(v); free(expected);
}

static void test_sdpa_flash_bench(h3_gpu *gpu) {
    /* Informational timing: naive vs flash on a production-like sequence. */
    enum { SEQ = 2048, HEADS = 8, DIM = 128 };
    size_t count = (size_t)SEQ * HEADS * DIM;
    uint16_t *q = malloc(count * 2), *k = malloc(count * 2),
             *v = malloc(count * 2);
    CHECK(q && k && v);
    if (!failed) {
        for (size_t index = 0; index < count; index++) {
            q[index] = bf16_bits((float)(sin((double)index * 0.11) * 0.5));
            k[index] = bf16_bits((float)(cos((double)index * 0.17) * 0.5));
            v[index] = bf16_bits((float)(sin((double)index * 0.23) * 0.5));
        }
        h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, count);
        h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, count);
        h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, v, count);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
        CHECK(tq && tk && tv && out);
        if (!failed) {
            h3_gpu_stats stats;
            /* Naive (force by temporary threshold override is not exposed;
             * the flash path is the one used at this size). */
            h3_gpu_begin(gpu);
            CHECK(h3_gpu_sdpa_bf16(gpu, out, tq, tk, tv, SEQ, HEADS, DIM,
                                   1.0f / sqrtf((float)DIM)) == 0);
            CHECK(h3_gpu_submit(gpu) == 0);
            h3_gpu_get_stats(gpu, &stats);
            printf("SDPA seq=%d heads=%d: flash kernel %.1f ms (GPU wait)\n",
                   SEQ, HEADS, stats.command_wait_seconds * 1e3);
        }
        h3_gpu_tensor_free(tq);
        h3_gpu_tensor_free(tk);
        h3_gpu_tensor_free(tv);
        h3_gpu_tensor_free(out);
    }
    free(q); free(k); free(v);
}

static void test_token_pool_expand(h3_gpu *gpu) {
    enum { INPUT_ROWS = 6, ROWS = 3, WIDTH = 64, BASELINE_ROWS = 2 };
    uint16_t input[INPUT_ROWS * WIDTH];
    uint32_t pairs[ROWS * 2] = {0, 1, 2, 3, 4, 4};
    uint32_t baseline_indices[ROWS] = {0, 0xffffffffu, 1};
    uint16_t expected_out[ROWS * WIDTH];
    uint16_t expected_orig[INPUT_ROWS * WIDTH];
    uint16_t expected_base[BASELINE_ROWS * WIDTH];
    for (size_t index = 0; index < INPUT_ROWS * WIDTH; index++)
        input[index] = bf16_bits((float)(sin((double)index * 0.29) * 1.5));
    memset(expected_orig, 0, sizeof(expected_orig));
    memset(expected_base, 0, sizeof(expected_base));
    for (uint32_t row = 0; row < ROWS; row++) {
        uint32_t first = pairs[row * 2], second = pairs[row * 2 + 1];
        for (uint32_t column = 0; column < WIDTH; column++) {
            expected_orig[first * WIDTH + column] =
                input[first * WIDTH + column];
            uint16_t pooled = input[first * WIDTH + column];
            if (first != second) {
                expected_orig[second * WIDTH + column] =
                    input[second * WIDTH + column];
                float average = (bf16_value(input[first * WIDTH + column]) +
                                 bf16_value(input[second * WIDTH + column])) *
                                0.5f;
                pooled = bf16_bits(average);
            }
            expected_out[row * WIDTH + column] = pooled;
            if (baseline_indices[row] != 0xffffffffu)
                expected_base[baseline_indices[row] * WIDTH + column] =
                    pooled;
        }
    }
    h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, input, INPUT_ROWS * WIDTH);
    h3_gpu_tensor *pr = h3_gpu_tensor_from_u32(gpu, pairs, ROWS * 2);
    h3_gpu_tensor *bi = h3_gpu_tensor_from_u32(gpu, baseline_indices, ROWS);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    h3_gpu_tensor *orig = h3_gpu_tensor_new_bf16(gpu, INPUT_ROWS * WIDTH);
    h3_gpu_tensor *base = h3_gpu_tensor_new_bf16(gpu, BASELINE_ROWS * WIDTH);
    CHECK(in && pr && bi && out && orig && base);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_token_pool_bf16(gpu, out, in, 0, orig, 0, base, 0, bi,
                                     pr, INPUT_ROWS, ROWS, BASELINE_ROWS,
                                     WIDTH) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[ROWS * WIDTH], got_orig[INPUT_ROWS * WIDTH],
                 got_base[BASELINE_ROWS * WIDTH];
        CHECK(h3_gpu_tensor_read_bf16(out, got, ROWS * WIDTH) == 0);
        CHECK(memcmp(got, expected_out, sizeof(got)) == 0);
        CHECK(h3_gpu_tensor_read_bf16(orig, got_orig,
                                      INPUT_ROWS * WIDTH) == 0);
        CHECK(memcmp(got_orig, expected_orig, sizeof(got_orig)) == 0);
        CHECK(h3_gpu_tensor_read_bf16(base, got_base,
                                      BASELINE_ROWS * WIDTH) == 0);
        CHECK(memcmp(got_base, expected_base, sizeof(got_base)) == 0);

        /* expand: parents map every full row to its pooled row; reduced
         * rows 0..2 hold the pooled values. */
        enum { FULL_ROWS = 6, REDUCED = 3 };
        uint32_t parents[FULL_ROWS] = {0, 0, 1, 1, 2, 2};
        uint32_t parent_bi[REDUCED] = {0, 1, 0xffffffffu};
        uint16_t expected_full[FULL_ROWS * WIDTH];
        for (uint32_t row = 0; row < FULL_ROWS; row++) {
            uint32_t parent = parents[row];
            for (uint32_t column = 0; column < WIDTH; column++) {
                uint32_t baseline_row = parent_bi[parent];
                float update;
                if (baseline_row == 0xffffffffu) {
                    expected_full[row * WIDTH + column] =
                        expected_out[parent * WIDTH + column];
                    continue;
                }
                update = bf16_value(expected_out[parent * WIDTH + column]) -
                         bf16_value(expected_base[baseline_row * WIDTH +
                                                  column]);
                expected_full[row * WIDTH + column] = bf16_bits(
                    bf16_value(expected_orig[row * WIDTH + column]) +
                    0.5f * update);
            }
        }
        h3_gpu_tensor *pr2 = h3_gpu_tensor_from_u32(gpu, parents, FULL_ROWS);
        h3_gpu_tensor *bi2 = h3_gpu_tensor_from_u32(gpu, parent_bi, REDUCED);
        h3_gpu_tensor *full = h3_gpu_tensor_new_bf16(gpu, FULL_ROWS * WIDTH);
        CHECK(pr2 && bi2 && full);
        if (!failed) {
            h3_gpu_begin(gpu);
            CHECK(h3_gpu_token_expand_delta_bf16(gpu, full, orig, 0, out,
                                                 base, 0, bi2, pr2, FULL_ROWS,
                                                 REDUCED, BASELINE_ROWS, WIDTH,
                                                 0, 0.5f) == 0);
            CHECK(h3_gpu_submit(gpu) == 0);
            uint16_t got_full[FULL_ROWS * WIDTH];
            CHECK(h3_gpu_tensor_read_bf16(full, got_full,
                                          FULL_ROWS * WIDTH) == 0);
            CHECK(memcmp(got_full, expected_full, sizeof(got_full)) == 0);
        }
        h3_gpu_tensor_free(pr2);
        h3_gpu_tensor_free(bi2);
        h3_gpu_tensor_free(full);
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(pr);
    h3_gpu_tensor_free(bi);
    h3_gpu_tensor_free(out);
    h3_gpu_tensor_free(orig);
    h3_gpu_tensor_free(base);
}

static void test_token_pool_adaln(h3_gpu *gpu) {
    enum { INPUT_ROWS = 4, ROWS = 2, FULL_ROWS_ALIAS = 4, WIDTH = 256,
           SLOTS = 2, BASELINE_ROWS = 1 };
    const float epsilon = 1e-5f;
    uint16_t input[INPUT_ROWS * WIDTH], nw[WIDTH], mod[ROWS * SLOTS * WIDTH];
    uint32_t pairs[ROWS * 2] = {0, 1, 2, 3};
    uint32_t baseline_indices[ROWS] = {0, 0xffffffffu};
    /* 4 entries: the expansion phase maps all four full rows. */
    uint32_t row_map[FULL_ROWS_ALIAS] = {1, 0, 1, 0};
    uint16_t expected_out[ROWS * WIDTH], expected_res[ROWS * WIDTH];
    for (size_t index = 0; index < INPUT_ROWS * WIDTH; index++)
        input[index] = bf16_bits((float)(cos((double)index * 0.23) * 1.5));
    for (size_t index = 0; index < WIDTH; index++)
        nw[index] = bf16_bits(1.0f);
    for (size_t index = 0; index < ROWS * SLOTS * WIDTH; index++)
        mod[index] = bf16_bits((float)sin((double)index * 0.13) * 0.2f);
    for (uint32_t row = 0; row < ROWS; row++) {
        uint32_t first = pairs[row * 2], second = pairs[row * 2 + 1];
        float squared[WIDTH];
        for (uint32_t column = 0; column < WIDTH; column++) {
            uint16_t pooled = input[first * WIDTH + column];
            if (first != second) {
                float average = (bf16_value(input[first * WIDTH + column]) +
                                 bf16_value(input[second * WIDTH + column])) *
                                0.5f;
                pooled = bf16_bits(average);
            }
            expected_res[row * WIDTH + column] = pooled;
            float value = bf16_value(pooled);
            squared[column] = fmaf(value, value, 0.0f);
        }
        float inverse = 1.0f / sqrtf(tree_sum256(squared, WIDTH) /
                                     (float)WIDTH + epsilon);
        uint32_t base = row_map[row] * SLOTS * WIDTH;
        for (uint32_t column = 0; column < WIDTH; column++) {
            float normalized = bf16_value(expected_res[row * WIDTH + column]) *
                               inverse * bf16_value(nw[column]);
            float shift = bf16_value(mod[base + column]);
            float scale = bf16_value(mod[base + WIDTH + column]);
            expected_out[row * WIDTH + column] =
                bf16_bits(normalized * (1.0f + scale) + shift);
        }
    }
    h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, input, INPUT_ROWS * WIDTH);
    h3_gpu_tensor *pr = h3_gpu_tensor_from_u32(gpu, pairs, ROWS * 2);
    h3_gpu_tensor *bi = h3_gpu_tensor_from_u32(gpu, baseline_indices, ROWS);
    h3_gpu_tensor *w = h3_gpu_tensor_from_bf16(gpu, nw, WIDTH);
    h3_gpu_tensor *m = h3_gpu_tensor_from_bf16(gpu, mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(gpu, row_map,
                                                FULL_ROWS_ALIAS);
    h3_gpu_tensor *res = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    h3_gpu_tensor *orig = h3_gpu_tensor_new_bf16(gpu, INPUT_ROWS * WIDTH);
    h3_gpu_tensor *base = h3_gpu_tensor_new_bf16(gpu, BASELINE_ROWS * WIDTH);
    CHECK(in && pr && bi && w && m && map && res && out && orig && base);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_token_pool_adaln_bf16(gpu, res, out, in, 0, orig, 0,
                                           base, 0, bi, pr, w, m, map,
                                           INPUT_ROWS, ROWS, BASELINE_ROWS,
                                           WIDTH, SLOTS, 0, 1, epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[ROWS * WIDTH];
        CHECK(h3_gpu_tensor_read_bf16(res, got, ROWS * WIDTH) == 0);
        CHECK(memcmp(got, expected_res, sizeof(got)) == 0);
        CHECK(h3_gpu_tensor_read_bf16(out, got, ROWS * WIDTH) == 0);
        CHECK(check_bf16_ulp(got, expected_out, ROWS * WIDTH, 2,
                             "pool_adaln"));

        /* expand_adaln */
        enum { FULL_ROWS = 4, REDUCED = 2 };
        uint32_t parents[FULL_ROWS] = {0, 0, 1, 1};
        uint32_t parent_bi[REDUCED] = {0, 0xffffffffu};
        uint16_t expected_full[FULL_ROWS * WIDTH];
        uint16_t expected_full_out[FULL_ROWS * WIDTH];
        for (uint32_t row = 0; row < FULL_ROWS; row++) {
            uint32_t parent = parents[row];
            uint32_t baseline_row = parent_bi[parent];
            float squared[WIDTH];
            for (uint32_t column = 0; column < WIDTH; column++) {
                uint16_t restored = expected_res[parent * WIDTH + column];
                if (!(row < 0 || baseline_row == 0xffffffffu)) {
                    float update =
                        bf16_value(expected_res[parent * WIDTH + column]) -
                        bf16_value(expected_res[baseline_row * WIDTH +
                                                column]);
                    restored = bf16_bits(
                        bf16_value(input[row * WIDTH + column]) +
                        0.5f * update);
                }
                expected_full[row * WIDTH + column] = restored;
                float value = bf16_value(restored);
                squared[column] = fmaf(value, value, 0.0f);
            }
            float inverse = 1.0f / sqrtf(tree_sum256(squared, WIDTH) /
                                         (float)WIDTH + epsilon);
            uint32_t base = row_map[row] * SLOTS * WIDTH;
            for (uint32_t column = 0; column < WIDTH; column++) {
                float normalized =
                    bf16_value(expected_full[row * WIDTH + column]) *
                    inverse * bf16_value(nw[column]);
                float shift = bf16_value(mod[base + column]);
                float scale = bf16_value(mod[base + WIDTH + column]);
                expected_full_out[row * WIDTH + column] =
                    bf16_bits(normalized * (1.0f + scale) + shift);
            }
        }
        h3_gpu_tensor *pr2 = h3_gpu_tensor_from_u32(gpu, parents, FULL_ROWS);
        h3_gpu_tensor *bi2 = h3_gpu_tensor_from_u32(gpu, parent_bi, REDUCED);
        h3_gpu_tensor *full = h3_gpu_tensor_new_bf16(gpu, FULL_ROWS * WIDTH);
        h3_gpu_tensor *full_out = h3_gpu_tensor_new_bf16(gpu,
                                                         FULL_ROWS * WIDTH);
        CHECK(pr2 && bi2 && full && full_out);
        if (!failed) {
            h3_gpu_begin(gpu);
            CHECK(h3_gpu_token_expand_adaln_bf16(
                      gpu, full, full_out, in, 0, res, base, 0, bi2, pr2,
                      w, m, map, FULL_ROWS, REDUCED, BASELINE_ROWS, WIDTH, 0,
                      0.5f, SLOTS, 0, 1, epsilon) == 0);
            CHECK(h3_gpu_submit(gpu) == 0);
            uint16_t got2[FULL_ROWS * WIDTH];
            CHECK(h3_gpu_tensor_read_bf16(full, got2,
                                          FULL_ROWS * WIDTH) == 0);
            CHECK(memcmp(got2, expected_full, sizeof(got2)) == 0);
            CHECK(h3_gpu_tensor_read_bf16(full_out, got2,
                                          FULL_ROWS * WIDTH) == 0);
            CHECK(check_bf16_ulp(got2, expected_full_out, FULL_ROWS * WIDTH,
                                 2, "expand_adaln"));
        }
        h3_gpu_tensor_free(pr2);
        h3_gpu_tensor_free(bi2);
        h3_gpu_tensor_free(full);
        h3_gpu_tensor_free(full_out);
    }
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(pr);
    h3_gpu_tensor_free(bi);
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(m);
    h3_gpu_tensor_free(map);
    h3_gpu_tensor_free(res);
    h3_gpu_tensor_free(out);
    h3_gpu_tensor_free(orig);
    h3_gpu_tensor_free(base);
}

static void test_device_local_load(h3_gpu *gpu) {
    /* Weight tensors are loaded into device-local memory through staging;
     * verify a kernel can consume them and the readback round-trips. */
    enum { N = 4096 };
    uint16_t data[N], expected[N];
    for (size_t index = 0; index < N; index++) {
        data[index] = bf16_bits((float)(sin((double)index * 0.21) * 1.5));
        expected[index] = bf16_bits(bf16_value(data[index]) +
                                     bf16_value(data[index]));
    }
    char path[] = "/tmp/h3-vk-device-local-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) {
        CHECK(write(fd, data, sizeof(data)) == (ssize_t)sizeof(data));
        close(fd);
        h3_gpu_tensor *w = h3_gpu_tensor_load_bf16(gpu, path, 0, N);
        h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, data, N);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, N);
        CHECK(w && in && out);
        if (!failed) {
            h3_gpu_begin(gpu);
            CHECK(h3_gpu_add_bf16(gpu, out, w, in, N) == 0);
            CHECK(h3_gpu_submit(gpu) == 0);
            uint16_t got[N];
            CHECK(h3_gpu_tensor_read_bf16(out, got, N) == 0);
            CHECK(memcmp(got, expected, sizeof(got)) == 0);
            /* device-local readback */
            uint16_t wgot[N];
            CHECK(h3_gpu_tensor_read_bf16(w, wgot, N) == 0);
            CHECK(memcmp(wgot, data, sizeof(wgot)) == 0);
        }
        h3_gpu_tensor_free(w);
        h3_gpu_tensor_free(in);
        h3_gpu_tensor_free(out);
        unlink(path);
    }
}

static void test_text_qk_rope_bf16(h3_gpu *gpu) {
    enum { SEQ = 4, Q_HEADS = 3, KV_HEADS = 2, DIM = 8, HALF = 4 };
    const float epsilon = 1e-5f;
    uint16_t qin[SEQ * Q_HEADS * DIM], kin[SEQ * KV_HEADS * DIM];
    uint16_t qw[DIM], kw[DIM], rcos[SEQ * HALF], rsin[SEQ * HALF];
    uint16_t expected_q[SEQ * Q_HEADS * DIM], expected_k[SEQ * KV_HEADS * DIM];
    for (size_t index = 0; index < SEQ * Q_HEADS * DIM; index++)
        qin[index] = bf16_bits((float)(sin((double)index * 0.41) * 1.2));
    for (size_t index = 0; index < SEQ * KV_HEADS * DIM; index++)
        kin[index] = bf16_bits((float)(cos((double)index * 0.37) * 1.2));
    for (size_t index = 0; index < DIM; index++) {
        qw[index] = bf16_bits((float)sin((double)index * 0.19) * 0.5f + 1.0f);
        kw[index] = bf16_bits((float)cos((double)index * 0.23) * 0.5f + 1.0f);
    }
    for (size_t index = 0; index < SEQ * HALF; index++) {
        rcos[index] = bf16_bits(cosf((float)index * 0.13f));
        rsin[index] = bf16_bits(sinf((float)index * 0.13f));
    }
    for (uint32_t row = 0; row < SEQ; row++) {
        for (uint32_t head = 0; head < Q_HEADS; head++) {
            uint32_t q_base = (row * Q_HEADS + head) * DIM;
            float q_sum = 0.0f;
            for (uint32_t d = 0; d < DIM; d++) {
                float v = bf16_value(qin[q_base + d]);
                q_sum = fmaf(v, v, q_sum);
            }
            float qi = 1.0f / sqrtf(q_sum / (float)DIM + epsilon);
            for (uint32_t d = 0; d < DIM; d++) {
                uint32_t pair = d < HALF ? d + HALF : d - HALF;
                float c = bf16_value(rcos[row * HALF + (d % HALF)]);
                float s = bf16_value(rsin[row * HALF + (d % HALF)]);
                float q0 = bf16_value(qin[q_base + d]) * qi * bf16_value(qw[d]);
                float q1 = bf16_value(qin[q_base + pair]) * qi *
                           bf16_value(qw[pair]);
                float rotated = d < HALF ? q0 * c - q1 * s : q0 * c + q1 * s;
                expected_q[q_base + d] = bf16_bits(rotated);
            }
        }
        for (uint32_t head = 0; head < KV_HEADS; head++) {
            uint32_t k_base = (row * KV_HEADS + head) * DIM;
            float k_sum = 0.0f;
            for (uint32_t d = 0; d < DIM; d++) {
                float v = bf16_value(kin[k_base + d]);
                k_sum = fmaf(v, v, k_sum);
            }
            float ki = 1.0f / sqrtf(k_sum / (float)DIM + epsilon);
            for (uint32_t d = 0; d < DIM; d++) {
                uint32_t pair = d < HALF ? d + HALF : d - HALF;
                float c = bf16_value(rcos[row * HALF + (d % HALF)]);
                float s = bf16_value(rsin[row * HALF + (d % HALF)]);
                float k0 = bf16_value(kin[k_base + d]) * ki * bf16_value(kw[d]);
                float k1 = bf16_value(kin[k_base + pair]) * ki *
                           bf16_value(kw[pair]);
                float rotated = d < HALF ? k0 * c - k1 * s : k0 * c + k1 * s;
                expected_k[k_base + d] = bf16_bits(rotated);
            }
        }
    }
    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, qin, SEQ * Q_HEADS * DIM);
    h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, kin, SEQ * KV_HEADS * DIM);
    h3_gpu_tensor *wq = h3_gpu_tensor_from_bf16(gpu, qw, DIM);
    h3_gpu_tensor *wk = h3_gpu_tensor_from_bf16(gpu, kw, DIM);
    h3_gpu_tensor *c = h3_gpu_tensor_from_bf16(gpu, rcos, SEQ * HALF);
    h3_gpu_tensor *s = h3_gpu_tensor_from_bf16(gpu, rsin, SEQ * HALF);
    h3_gpu_tensor *oq = h3_gpu_tensor_new_bf16(gpu, SEQ * Q_HEADS * DIM);
    h3_gpu_tensor *ok = h3_gpu_tensor_new_bf16(gpu, SEQ * KV_HEADS * DIM);
    CHECK(tq && tk && wq && wk && c && s && oq && ok);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_text_qk_rope_bf16(gpu, oq, ok, tq, tk, wq, wk, c, s,
                                       SEQ, Q_HEADS, KV_HEADS, DIM,
                                       epsilon) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[SEQ * Q_HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(oq, got, SEQ * Q_HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(got, expected_q, SEQ * Q_HEADS * DIM, 2,
                             "text_qk_q"));
        uint16_t gotk[SEQ * KV_HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(ok, gotk, SEQ * KV_HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(gotk, expected_k, SEQ * KV_HEADS * DIM, 2,
                             "text_qk_k"));
    }
    h3_gpu_tensor_free(tq); h3_gpu_tensor_free(tk); h3_gpu_tensor_free(wq);
    h3_gpu_tensor_free(wk); h3_gpu_tensor_free(c); h3_gpu_tensor_free(s);
    h3_gpu_tensor_free(oq); h3_gpu_tensor_free(ok);
}

static void test_rope_text_bf16(h3_gpu *gpu) {
    enum { SEQ = 4, Q_HEADS = 2, KV_HEADS = 1, DIM = 8, HALF = 4 };
    uint16_t q[SEQ * Q_HEADS * DIM], k[SEQ * KV_HEADS * DIM];
    float rcos[SEQ * HALF], rsin[SEQ * HALF];
    uint16_t expected_q[SEQ * Q_HEADS * DIM], expected_k[SEQ * KV_HEADS * DIM];
    for (size_t index = 0; index < SEQ * Q_HEADS * DIM; index++)
        q[index] = bf16_bits((float)(sin((double)index * 0.29) * 1.1));
    for (size_t index = 0; index < SEQ * KV_HEADS * DIM; index++)
        k[index] = bf16_bits((float)(cos((double)index * 0.31) * 1.1));
    for (size_t index = 0; index < SEQ * HALF; index++) {
        rcos[index] = cosf((float)index * 0.17f);
        rsin[index] = sinf((float)index * 0.17f);
    }
    memcpy(expected_q, q, sizeof(expected_q));
    memcpy(expected_k, k, sizeof(expected_k));
    for (uint32_t row = 0; row < SEQ; row++) {
        for (uint32_t head = 0; head < Q_HEADS; head++) {
            uint32_t base = (row * Q_HEADS + head) * DIM;
            for (uint32_t d = 0; d < HALF; d++) {
                float first = bf16_value(expected_q[base + d]);
                float second = bf16_value(expected_q[base + HALF + d]);
                float c = rcos[row * HALF + d];
                float s = rsin[row * HALF + d];
                expected_q[base + d] = bf16_bits(first * c - second * s);
                expected_q[base + HALF + d] = bf16_bits(second * c + first * s);
            }
        }
        for (uint32_t head = 0; head < KV_HEADS; head++) {
            uint32_t base = (row * KV_HEADS + head) * DIM;
            for (uint32_t d = 0; d < HALF; d++) {
                float first = bf16_value(expected_k[base + d]);
                float second = bf16_value(expected_k[base + HALF + d]);
                float c = rcos[row * HALF + d];
                float s = rsin[row * HALF + d];
                expected_k[base + d] = bf16_bits(first * c - second * s);
                expected_k[base + HALF + d] = bf16_bits(second * c + first * s);
            }
        }
    }
    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, SEQ * Q_HEADS * DIM);
    h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, SEQ * KV_HEADS * DIM);
    h3_gpu_tensor *c = h3_gpu_tensor_from_f32(gpu, rcos, SEQ * HALF);
    h3_gpu_tensor *s = h3_gpu_tensor_from_f32(gpu, rsin, SEQ * HALF);
    CHECK(tq && tk && c && s);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_rope_text_bf16(gpu, tq, tk, c, s, SEQ, Q_HEADS, KV_HEADS,
                                    DIM) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[SEQ * Q_HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(tq, got, SEQ * Q_HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(got, expected_q, SEQ * Q_HEADS * DIM, 2,
                             "rope_text_q"));
        uint16_t gotk[SEQ * KV_HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(tk, gotk, SEQ * KV_HEADS * DIM) == 0);
        CHECK(check_bf16_ulp(gotk, expected_k, SEQ * KV_HEADS * DIM, 2,
                             "rope_text_k"));
    }
    h3_gpu_tensor_free(tq); h3_gpu_tensor_free(tk);
    h3_gpu_tensor_free(c); h3_gpu_tensor_free(s);
}

static void test_gqa_causal_bf16(h3_gpu *gpu) {
    enum { SEQ = 5, Q_HEADS = 4, KV_HEADS = 2, DIM = 16 };
    uint16_t q[SEQ * Q_HEADS * DIM], k[SEQ * KV_HEADS * DIM],
             v[SEQ * KV_HEADS * DIM];
    uint16_t expected[SEQ * Q_HEADS * DIM];
    float scale = 0.5f;
    for (size_t index = 0; index < SEQ * Q_HEADS * DIM; index++)
        q[index] = bf16_bits((float)(sin((double)index * 0.53) * 1.3));
    for (size_t index = 0; index < SEQ * KV_HEADS * DIM; index++) {
        k[index] = bf16_bits((float)(cos((double)index * 0.47) * 1.3));
        v[index] = bf16_bits((float)(sin((double)index * 0.61) * 0.9));
    }
    for (uint32_t row = 0; row < SEQ; row++) {
        for (uint32_t head = 0; head < Q_HEADS; head++) {
            uint32_t q_base = (row * Q_HEADS + head) * DIM;
            uint32_t kv_head = head / (Q_HEADS / KV_HEADS);
            uint32_t key_count = row + 1;
            float scaled_q[DIM];
            for (uint32_t d = 0; d < DIM; d++)
                scaled_q[d] = bf16_value(bf16_bits(bf16_value(q[q_base + d]) *
                                                   scale));
            float scores[SEQ];
            float maxv = -3.402823466e+38f;
            for (uint32_t kr = 0; kr < key_count; kr++) {
                uint32_t k_base = (kr * KV_HEADS + kv_head) * DIM;
                float dot = 0.0f;
                for (uint32_t d = 0; d < DIM; d++)
                    dot = fmaf(scaled_q[d], bf16_value(k[k_base + d]), dot);
                scores[kr] = dot;
                if (dot > maxv) maxv = dot;
            }
            float sum = 0.0f;
            for (uint32_t kr = 0; kr < key_count; kr++) {
                scores[kr] = expf(scores[kr] - maxv);
                sum += scores[kr];
            }
            float inverse_sum = 1.0f / sum;
            for (uint32_t d = 0; d < DIM; d++) {
                float acc = 0.0f;
                for (uint32_t kr = 0; kr < key_count; kr++) {
                    uint32_t v_index = (kr * KV_HEADS + kv_head) * DIM + d;
                    acc = fmaf(scores[kr] * inverse_sum,
                               bf16_value(v[v_index]), acc);
                }
                expected[q_base + d] = bf16_bits(acc);
            }
        }
    }
    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, SEQ * Q_HEADS * DIM);
    h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, SEQ * KV_HEADS * DIM);
    h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, v, SEQ * KV_HEADS * DIM);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, SEQ * Q_HEADS * DIM);
    CHECK(tq && tk && tv && out);
    if (!failed) {
        h3_gpu_begin(gpu);
        CHECK(h3_gpu_gqa_causal_bf16(gpu, out, tq, tk, tv, SEQ, Q_HEADS,
                                     KV_HEADS, DIM, scale) == 0);
        CHECK(h3_gpu_submit(gpu) == 0);
        uint16_t got[SEQ * Q_HEADS * DIM];
        CHECK(h3_gpu_tensor_read_bf16(out, got, SEQ * Q_HEADS * DIM) == 0);
        CHECK(check_bf16_ulp_abs(got, expected, SEQ * Q_HEADS * DIM, 8,
                                 2e-7f, "gqa_causal"));
    }
    h3_gpu_tensor_free(tq); h3_gpu_tensor_free(tk);
    h3_gpu_tensor_free(tv); h3_gpu_tensor_free(out);
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
    test_linear_bf16(gpu);
    test_linear_f32(gpu);
    test_scale_add_f32(gpu);
    test_swiglu_bf16(gpu);
    test_copy(gpu);
    test_mlp_bf16(gpu);
    test_adaln_bf16(gpu);
    test_adaln_offset(gpu);
    test_gate_bf16(gpu);
    test_gate_adaln_bf16(gpu);
    test_adaln_linear_bf16(gpu);
    test_head_rms_norm_bf16(gpu);
    test_grouped_qkv_rope_bf16(gpu);
    test_qkv_rope_plain_bf16(gpu);
    test_sdpa_bf16(gpu);
    test_sdpa_flash_bf16(gpu);
    test_sdpa_flash_bench(gpu);
    test_patch_linear_bf16(gpu);
    test_patch_linear_offset_bf16(gpu);
    test_patch_linear_map_bf16(gpu);
    test_token_pool_expand(gpu);
    test_token_pool_adaln(gpu);
    test_device_local_load(gpu);
    test_text_qk_rope_bf16(gpu);
    test_rope_text_bf16(gpu);
    test_gqa_causal_bf16(gpu);
    test_continue_chain(gpu);
    h3_gpu_free(gpu);
    if (failed) {
        fprintf(stderr, "FAILED: %d of %d checks\n", failed, tests_run);
        return 1;
    }
    printf("ok: %d checks (Vulkan backend)\n", tests_run);
    return 0;
}
