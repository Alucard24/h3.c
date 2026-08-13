#include <fcntl.h>
#include <unistd.h>
/* Portable C tokenizer test: byte-level BPE, pre-tokenization, added
 * tokens, NFC normalization, and decoding, against a synthetic
 * tokenizer.json with a known vocabulary. */
#include "h3_tokenizer.h"

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

static int ids_equal(const uint32_t *got, size_t got_count,
                     const uint32_t *expected, size_t expected_count) {
    if (got_count != expected_count) return 0;
    return memcmp(got, expected, got_count * sizeof(*got)) == 0;
}

static char *write_tokenizer_json(void) {
    static const char json[] =
        "{\n"
        "  \"normalizer\": {\"type\": \"NFC\"},\n"
        "  \"model\": {\n"
        "    \"type\": \"BPE\",\n"
        "    \"unk_token\": null,\n"
        "    \"vocab\": {\n"
        "      \"h\": 0, \"e\": 1, \"l\": 2, \"o\": 3,\n"
        "      \"he\": 4, \"hel\": 5, \"hell\": 6, \"hello\": 7,\n"
        "      \"\u0120\": 8, \"\u0120h\": 9, \"\u0120he\": 10,\n"
        "      \"\u0120hel\": 11, \"\u0120hell\": 12,\n"
        "      \"\u0120hello\": 13,\n"
        "      \"w\": 14, \"r\": 15, \"d\": 16, \"wo\": 17, \"wor\": 18,\n"
        "      \"worl\": 19, \"world\": 20, \"x\": 21, \"y\": 22,\n"
        "      \"c\": 23, \"a\": 24, \"f\": 25, \"\u00c3\u00a9\": 26,\n"
        "      \"ca\": 27, \"caf\": 28, \"caf\u00c3\u00a9\": 29\n"
        "    },\n"
        "    \"merges\": [\n"
        "      \"\u0120 h\", \"h e\", \"he l\", \"hel l\",\n"
        "      \"hell o\", \"\u0120h e\", \"\u0120he l\",\n"
        "      \"\u0120hel l\", \"\u0120hell o\",\n"
        "      \"w o\", \"wo r\", \"wor l\", \"worl d\",\n"
        "      \"c a\", \"ca f\", \"\u00c3 \u00a9\",\n"
        "      \"caf \u00c3\u00a9\"\n"
        "    ]\n"
        "  },\n"
        "  \"added_tokens\": [\n"
        "    {\"content\": \"SUPER\", \"id\": 100, \"single_word\": false,\n"
        "     \"lstrip\": false, \"rstrip\": false, \"normalized\": false}\n"
        "  ]\n"
        "}\n";
    char path[] = "/tmp/h3-tokenizer-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return NULL;
    write(fd, json, sizeof(json) - 1);
    close(fd);
    return strdup(path);
}

int main(void) {
    char *path = write_tokenizer_json();
    CHECK(path != NULL);
    if (!path) return 1;
    char error[512] = {0};
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, sizeof(error));
    CHECK(tokenizer != NULL);
    if (!tokenizer) {
        fprintf(stderr, "load error: %s\n", error);
        return 1;
    }

    struct { const char *text; const uint32_t *ids; size_t count; } cases[] = {
        {"hello", (uint32_t[]){7}, 1},
        {" hello", (uint32_t[]){13}, 1},
        {"hello world", (uint32_t[]){7, 8, 20}, 3},
        {"xSUPERy", (uint32_t[]){21, 100, 22}, 3},
        {"SUPER", (uint32_t[]){100}, 1},
        {"caf\u00e9", (uint32_t[]){29}, 1},
        /* e + combining acute normalizes to \u00e9 (NFC). */
        {"cafe\u0301", (uint32_t[]){29}, 1},
        {"", (uint32_t[]){151643}, 1}
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        uint32_t *ids = NULL;
        size_t count = 0;
        int ok = h3_tokenizer_encode(tokenizer, cases[index].text, 1,
                                     &ids, &count, error, sizeof(error));
        CHECK(ok);
        CHECK(ids_equal(ids, count, cases[index].ids, cases[index].count));
        if (!ids_equal(ids, count, cases[index].ids, cases[index].count)) {
            fprintf(stderr, "  encode(\"%s\"): got", cases[index].text);
            for (size_t item = 0; item < count; item++)
                fprintf(stderr, " %u", ids[item]);
            fprintf(stderr, " expected");
            for (size_t item = 0; item < cases[index].count; item++)
                fprintf(stderr, " %u", cases[index].ids[item]);
            fprintf(stderr, "\n");
        }
        h3_tokenizer_ids_free(ids);
    }

    /* Decode checks. */
    struct { const uint32_t *ids; size_t count; const char *text; } decodes[] = {
        {(uint32_t[]){7}, 1, "hello"},
        {(uint32_t[]){13}, 1, " hello"},
        {(uint32_t[]){8, 7, 8, 20}, 4, " hello world"},
        {(uint32_t[]){21, 100, 22}, 3, "xSUPERy"},
        {(uint32_t[]){29}, 1, "caf\u00e9"}
    };
    for (size_t index = 0; index < sizeof(decodes) / sizeof(decodes[0]);
         index++) {
        char *decoded = h3_tokenizer_decode(tokenizer, decodes[index].ids,
                                            decodes[index].count, error,
                                            sizeof(error));
        CHECK(decoded != NULL);
        CHECK(strcmp(decoded, decodes[index].text) == 0);
        if (!decoded || strcmp(decoded, decodes[index].text) != 0)
            fprintf(stderr, "  decode: got \"%s\" expected \"%s\"\n",
                    decoded ? decoded : "(null)", decodes[index].text);
        free(decoded);
    }

    /* Round trip: decode(encode(text)) == text. */
    const char *phrase = " hello caf\u00e9 world";
    uint32_t *ids = NULL;
    size_t count = 0;
    CHECK(h3_tokenizer_encode(tokenizer, phrase, 0, &ids, &count, error,
                              sizeof(error)));
    if (ids) {
        char *decoded = h3_tokenizer_decode(tokenizer, ids, count, error,
                                            sizeof(error));
        CHECK(decoded && strcmp(decoded, phrase) == 0);
        if (!decoded || strcmp(decoded, phrase) != 0)
            fprintf(stderr, "  roundtrip: got \"%s\" expected \"%s\"\n",
                    decoded ? decoded : "(null)", phrase);
        free(decoded);
        h3_tokenizer_ids_free(ids);
    }

    /* Regression: the pre-tokenizer must grow beyond its initial 16-piece
     * array without writing past the allocation. */
    enum { LONG_WORDS = 128 };
    char long_text[LONG_WORDS * 6];
    size_t long_length = 0;
    for (size_t index = 0; index < LONG_WORDS; index++) {
        if (index) long_text[long_length++] = ' ';
        memcpy(long_text + long_length, "hello", 5);
        long_length += 5;
    }
    long_text[long_length] = '\0';
    uint32_t *long_ids = NULL;
    size_t long_count = 0;
    CHECK(h3_tokenizer_encode(tokenizer, long_text, 0, &long_ids,
                              &long_count, error, sizeof(error)));
    CHECK(long_count == LONG_WORDS);
    int long_values_match = long_ids && long_count == LONG_WORDS &&
                            long_ids[0] == 7;
    for (size_t index = 1; index < long_count && long_values_match; index++)
        long_values_match = long_ids[index] == 13;
    CHECK(long_values_match);
    char *long_decoded = h3_tokenizer_decode(
        tokenizer, long_ids, long_count, error, sizeof(error));
    CHECK(long_decoded && strcmp(long_decoded, long_text) == 0);
    free(long_decoded);
    h3_tokenizer_ids_free(long_ids);

    h3_tokenizer_free(tokenizer);
    unlink(path);
    free(path);

    if (failed) {
        fprintf(stderr, "FAILED: %d of %d checks\n", failed, tests_run);
        return 1;
    }
    printf("ok: %d checks (portable tokenizer)\n", tests_run);
    return 0;
}
