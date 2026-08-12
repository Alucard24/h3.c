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
    test_mlp_bf16(gpu);
    test_adaln_bf16(gpu);
    test_adaln_offset(gpu);
    test_gate_bf16(gpu);
    test_gate_adaln_bf16(gpu);
    test_adaln_linear_bf16(gpu);
    test_head_rms_norm_bf16(gpu);
    test_continue_chain(gpu);
    h3_gpu_free(gpu);
    if (failed) {
        fprintf(stderr, "FAILED: %d of %d checks\n", failed, tests_run);
        return 1;
    }
    printf("ok: %d checks (Vulkan backend)\n", tests_run);
    return 0;
}
