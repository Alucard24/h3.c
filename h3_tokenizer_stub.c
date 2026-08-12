/* Host placeholder for the Qwen tokenizer. The full BPE tokenizer is an
 * Objective-C implementation (h3_tokenizer.m, macOS only). On platforms
 * without Foundation this placeholder reports that the model text pipeline
 * is unavailable, while keeping the host library link-complete. */
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json,
                                char *error, size_t error_size) {
    if (error && error_size) {
        snprintf(error, error_size,
                 "tokenizer unavailable: this build has no text pipeline "
                 "(Foundation-based h3_tokenizer.m is macOS-only; a C port "
                 "is pending with the Vulkan/CUDA backend work)");
    }
    return NULL;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    free(tokenizer);
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    (void)tokenizer; (void)utf8; (void)pad_empty;
    if (ids) *ids = NULL;
    if (count) *count = 0;
    if (error && error_size) {
        snprintf(error, error_size,
                 "tokenizer unavailable: this build has no text pipeline");
    }
    return -1;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    (void)tokenizer; (void)ids; (void)count;
    if (error && error_size) {
        snprintf(error, error_size,
                 "tokenizer unavailable: this build has no text pipeline");
    }
    return NULL;
}
