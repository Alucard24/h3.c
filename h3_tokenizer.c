/* Qwen2 BPE tokenizer in portable C.
 *
 * Port of h3_tokenizer.m with identical semantics: the GPT-2-style byte
 * encoder, the manual Qwen2 pre-tokenizer scan (letters/numbers/space via
 * ICU categories, contractions, newline handling), the leftmost-minimal
 * BPE merge loop, added-token longest-leftmost matching, NFC
 * normalization, and byte-level decoding with replacement characters.
 * Uses ICU (libicuuc), the same engine as macOS's -licucore, so Unicode
 * category and NFC behavior match the released tokenizer.
 */
#include "h3_tokenizer.h"

#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ maps */

typedef struct {
    char *key;
    uint32_t value;
} h3_map_entry;

typedef struct {
    h3_map_entry *entries;
    size_t capacity;
    size_t count;
} h3_map;

static uint64_t h3_map_hash(const char *key, size_t length) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < length; index++) {
        hash ^= (unsigned char)key[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void h3_map_init(h3_map *map, size_t capacity) {
    map->capacity = capacity ? capacity : 64;
    map->count = 0;
    map->entries = calloc(map->capacity, sizeof(*map->entries));
}

static void h3_map_free(h3_map *map) {
    for (size_t index = 0; index < map->capacity; index++)
        free(map->entries[index].key);
    free(map->entries);
    map->entries = NULL;
    map->capacity = 0;
    map->count = 0;
}

static int h3_map_grow(h3_map *map) {
    size_t new_capacity = map->capacity * 2;
    h3_map_entry *entries = calloc(new_capacity, sizeof(*entries));
    if (!entries) return 0;
    for (size_t index = 0; index < map->capacity; index++) {
        h3_map_entry entry = map->entries[index];
        if (!entry.key) continue;
        size_t length = strlen(entry.key);
        size_t slot = (size_t)(h3_map_hash(entry.key, length) &
                               (new_capacity - 1));
        while (entries[slot].key) slot = (slot + 1) & (new_capacity - 1);
        entries[slot] = entry;
    }
    free(map->entries);
    map->entries = entries;
    map->capacity = new_capacity;
    return 1;
}

static int h3_map_put(h3_map *map, const char *key, size_t length,
                      uint32_t value) {
    if ((map->count + 1) * 2 > map->capacity && !h3_map_grow(map)) return 0;
    size_t slot = (size_t)(h3_map_hash(key, length) & (map->capacity - 1));
    while (map->entries[slot].key) {
        if (strlen(map->entries[slot].key) == length &&
            memcmp(map->entries[slot].key, key, length) == 0) {
            map->entries[slot].value = value;
            return 1;
        }
        slot = (slot + 1) & (map->capacity - 1);
    }
    char *copy = malloc(length + 1);
    if (!copy) return 0;
    memcpy(copy, key, length);
    copy[length] = '\0';
    map->entries[slot].key = copy;
    map->entries[slot].value = value;
    map->count++;
    return 1;
}

static int h3_map_get(const h3_map *map, const char *key, size_t length,
                      uint32_t *value) {
    if (!map->entries) return 0;
    size_t slot = (size_t)(h3_map_hash(key, length) & (map->capacity - 1));
    for (size_t probe = 0; probe < map->capacity; probe++) {
        h3_map_entry entry = map->entries[slot];
        if (!entry.key) return 0;
        if (strlen(entry.key) == length &&
            memcmp(entry.key, key, length) == 0) {
            *value = entry.value;
            return 1;
        }
        slot = (slot + 1) & (map->capacity - 1);
    }
    return 0;
}

/* --------------------------------------------------------------- token */

struct h3_tokenizer {
    h3_map vocab;              /* symbol -> id */
    char **inverse_vocab;      /* id -> symbol (owned) */
    size_t inverse_count;
    h3_map merge_ranks;        /* "left<FFFF>right" -> rank */
    h3_map added_tokens;       /* content -> id */
    char **inverse_added;      /* id -> content (owned) */
    size_t inverse_added_count;
    char **added_alternatives; /* sorted longest-first */
    size_t added_count;
    uint32_t byte_encoder[256];  /* byte -> codepoint */
    int16_t byte_decoder[324];   /* codepoint -> byte */
};

static void h3_tok_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
}

/* --------------------------------------------------------------- JSON */

typedef struct {
    const char *cursor;
    const char *end;
} h3_json;

static void h3_json_skip_ws(h3_json *json) {
    while (json->cursor < json->end) {
        char c = *json->cursor;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            json->cursor++;
        else
            break;
    }
}

static int h3_json_hex4(const char *text, uint32_t *value) {
    uint32_t result = 0;
    for (int index = 0; index < 4; index++) {
        char c = text[index];
        uint32_t digit;
        if (c >= '0' && c <= '9') digit = (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') digit = (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') digit = (uint32_t)(c - 'A' + 10);
        else return 0;
        result = result * 16 + digit;
    }
    *value = result;
    return 1;
}

/* Parse a JSON string with escapes into a freshly allocated UTF-8 buffer
 * (no surrounding quotes). Returns length. */
static int h3_json_string(h3_json *json, char **out) {
    h3_json_skip_ws(json);
    if (json->cursor >= json->end || *json->cursor != '"') return -1;
    json->cursor++;
    size_t capacity = 64, length = 0;
    char *buffer = malloc(capacity);
    if (!buffer) return -1;
    while (json->cursor < json->end && *json->cursor != '"') {
        unsigned char c = (unsigned char)*json->cursor;
        uint32_t codepoint = c;
        size_t units = 1;
        if (c == '\\') {
            json->cursor++;
            if (json->cursor >= json->end) goto fail;
            char escape = *json->cursor;
            switch (escape) {
                case '"': codepoint = '"'; break;
                case '\\': codepoint = '\\'; break;
                case '/': codepoint = '/'; break;
                case 'b': codepoint = '\b'; break;
                case 'f': codepoint = '\f'; break;
                case 'n': codepoint = '\n'; break;
                case 'r': codepoint = '\r'; break;
                case 't': codepoint = '\t'; break;
                case 'u': {
                    if (json->cursor + 4 >= json->end ||
                        !h3_json_hex4(json->cursor + 1, &codepoint))
                        goto fail;
                    json->cursor += 4;
                    /* surrogate pair */
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff &&
                        json->cursor + 6 < json->end &&
                        json->cursor[1] == '\\' &&
                        json->cursor[2] == 'u') {
                        uint32_t low;
                        if (h3_json_hex4(json->cursor + 3, &low) &&
                            low >= 0xdc00 && low <= 0xdfff) {
                            codepoint = UINT32_C(0x10000) +
                                ((codepoint - 0xd800) << 10) +
                                (low - 0xdc00);
                            json->cursor += 6;
                        }
                    }
                    break;
                }
                default: goto fail;
            }
            /* The trailing json->cursor += units advances past the escape
             * byte; \u already advanced through its hex digits. */
            units = 1;
        } else {
            /* raw UTF-8 continuation handling */
            if (c >= 0x80) {
                int extra = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : 1;
                if ((size_t)(json->end - json->cursor) <= (size_t)extra)
                    goto fail;
                codepoint = c & (0x7f >> extra);
                for (int index = 1; index <= extra; index++) {
                    unsigned char cont = (unsigned char)json->cursor[index];
                    if ((cont & 0xc0) != 0x80) goto fail;
                    codepoint = (codepoint << 6) | (cont & 0x3f);
                }
                units = (size_t)extra + 1;
                /* re-encode below; raw bytes pass through unchanged */
                if (length + units + 1 > capacity) {
                    capacity *= 2;
                    char *grown = realloc(buffer, capacity);
                    if (!grown) goto fail;
                    buffer = grown;
                }
                memcpy(buffer + length, json->cursor, units);
                length += units;
                json->cursor += units;
                continue;
            }
        }
        if (length + 4 + 1 > capacity) {
            capacity *= 2;
            char *grown = realloc(buffer, capacity);
            if (!grown) goto fail;
            buffer = grown;
        }
        if (codepoint < 0x80) {
            buffer[length++] = (char)codepoint;
        } else if (codepoint < 0x800) {
            buffer[length++] = (char)(0xc0 | (codepoint >> 6));
            buffer[length++] = (char)(0x80 | (codepoint & 0x3f));
        } else if (codepoint < 0x10000) {
            buffer[length++] = (char)(0xe0 | (codepoint >> 12));
            buffer[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
            buffer[length++] = (char)(0x80 | (codepoint & 0x3f));
        } else {
            buffer[length++] = (char)(0xf0 | (codepoint >> 18));
            buffer[length++] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
            buffer[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
            buffer[length++] = (char)(0x80 | (codepoint & 0x3f));
        }
        json->cursor += units;
    }
    if (json->cursor >= json->end) goto fail;
    json->cursor++; /* closing quote */
    buffer[length] = '\0';
    *out = buffer;
    return (int)length;
fail:
    free(buffer);
    return -1;
}

/* Skip one JSON value (used to advance past members). */
static int h3_json_skip_value(h3_json *json) {
    h3_json_skip_ws(json);
    if (json->cursor >= json->end) return 0;
    char c = *json->cursor;
    if (c == '{' || c == '[') {
        json->cursor++;
        int depth = 1;
        while (json->cursor < json->end && depth > 0) {
            char current = *json->cursor;
            if (current == '"') {
                char *ignored = NULL;
                if (h3_json_string(json, &ignored) < 0) return 0;
                free(ignored);
                continue;
            }
            if (current == '{' || current == '[') depth++;
            if (current == '}' || current == ']') depth--;
            json->cursor++;
        }
        return depth == 0;
    }
    if (c == '"') {
        char *ignored = NULL;
        int result = h3_json_string(json, &ignored) >= 0;
        free(ignored);
        return result;
    }
    while (json->cursor < json->end && *json->cursor != ',' &&
           *json->cursor != '}' && *json->cursor != ']')
        json->cursor++;
    return 1;
}

/* Skip a JSON boolean; returns 1 for true, 0 for false, -1 on garbage. */
static int h3_json_boolean(h3_json *json) {
    h3_json_skip_ws(json);
    if (json->cursor + 4 <= json->end &&
        memcmp(json->cursor, "true", 4) == 0) {
        json->cursor += 4;
        return 1;
    }
    if (json->cursor + 5 <= json->end &&
        memcmp(json->cursor, "false", 5) == 0) {
        json->cursor += 5;
        return 0;
    }
    return -1;
}

/* Enter an object; iterate members with h3_json_member. */
static int h3_json_enter_object(h3_json *json) {
    h3_json_skip_ws(json);
    if (json->cursor >= json->end || *json->cursor != '{') return 0;
    json->cursor++;
    return 1;
}

static int h3_json_enter_array(h3_json *json) {
    h3_json_skip_ws(json);
    if (json->cursor >= json->end || *json->cursor != '[') return 0;
    json->cursor++;
    return 1;
}

/* Next member key; returns 1 with a malloc'd key, 0 at the end. */
static int h3_json_member(h3_json *json, char **key, size_t *key_length) {
    h3_json_skip_ws(json);
    if (json->cursor >= json->end) return 0;
    if (*json->cursor == '}') {
        json->cursor++;
        return 0;
    }
    if (*json->cursor == ',') json->cursor++;
    h3_json_skip_ws(json);
    int length = h3_json_string(json, key);
    if (length < 0) return 0;
    *key_length = (size_t)length;
    h3_json_skip_ws(json);
    if (json->cursor >= json->end || *json->cursor != ':') {
        free(*key);
        *key = NULL;
        return 0;
    }
    json->cursor++;
    return 1;
}

/* ------------------------------------------------------------- helpers */

static int h3_letter(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_UPPERCASE_LETTER || category == U_LOWERCASE_LETTER ||
           category == U_TITLECASE_LETTER || category == U_MODIFIER_LETTER ||
           category == U_OTHER_LETTER;
}

static int h3_number(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
           category == U_OTHER_NUMBER;
}

static int h3_space(uint32_t value) {
    return u_isUWhiteSpace((UChar32)value) ||
           (value >= 0x1c && value <= 0x1f);
}

typedef struct {
    uint32_t value;
    size_t byte_start;
    size_t byte_end;
} h3_codepoint;

/* Decode a UTF-8 buffer into codepoints with byte spans. */
static size_t h3_codepoints(const char *text, size_t length,
                            h3_codepoint **out) {
    h3_codepoint *points = malloc(length * sizeof(*points));
    if (!points) return 0;
    size_t count = 0, index = 0;
    while (index < length) {
        unsigned char c = (unsigned char)text[index];
        uint32_t value = c;
        size_t units = 1;
        if (c >= 0xf0) {
            if (index + 3 < length && (text[index + 1] & 0xc0) == 0x80 &&
                (text[index + 2] & 0xc0) == 0x80 &&
                (text[index + 3] & 0xc0) == 0x80) {
                value = ((uint32_t)(c & 0x07) << 18) |
                        ((uint32_t)(text[index + 1] & 0x3f) << 12) |
                        ((uint32_t)(text[index + 2] & 0x3f) << 6) |
                        (uint32_t)(text[index + 3] & 0x3f);
                units = 4;
            }
        } else if (c >= 0xe0) {
            if (index + 2 < length && (text[index + 1] & 0xc0) == 0x80 &&
                (text[index + 2] & 0xc0) == 0x80) {
                value = ((uint32_t)(c & 0x0f) << 12) |
                        ((uint32_t)(text[index + 1] & 0x3f) << 6) |
                        (uint32_t)(text[index + 2] & 0x3f);
                units = 3;
            }
        } else if (c >= 0xc0) {
            if (index + 1 < length && (text[index + 1] & 0xc0) == 0x80) {
                value = ((uint32_t)(c & 0x1f) << 6) |
                        (uint32_t)(text[index + 1] & 0x3f);
                units = 2;
            }
        }
        points[count++] = (h3_codepoint){value, index, index + units};
        index += units;
    }
    *out = points;
    return count;
}

static char *h3_slice(const char *text, const h3_codepoint *points,
                      size_t start, size_t stop) {
    size_t begin = points[start].byte_start;
    size_t end = points[stop - 1].byte_end;
    char *result = malloc(end - begin + 1);
    if (!result) return NULL;
    memcpy(result, text + begin, end - begin);
    result[end - begin] = '\0';
    return result;
}

static int h3_append_slice(char ***pieces, size_t *count, size_t *capacity,
                           const char *text, const h3_codepoint *points,
                           size_t start, size_t stop) {
    char *piece = h3_slice(text, points, start, stop);
    if (!piece) return 0;
    if (*count == *capacity) {
        if (*capacity > SIZE_MAX / 2 ||
            *capacity * 2 > SIZE_MAX / sizeof(**pieces)) {
            free(piece);
            return 0;
        }
        size_t grown_capacity = *capacity * 2;
        char **grown = realloc(*pieces, grown_capacity * sizeof(**pieces));
        if (!grown) {
            free(piece);
            return 0;
        }
        *pieces = grown;
        *capacity = grown_capacity;
    }
    (*pieces)[(*count)++] = piece;
    return 1;
}

static size_t h3_contraction(const h3_codepoint *points, size_t count,
                             size_t index) {
    static const char *values[] = {"'s", "'t", "'re", "'ve", "'m", "'ll",
                                   "'d"};
    if (points[index].value != '\'') return 0;
    for (size_t item = 0; item < sizeof(values) / sizeof(values[0]); item++) {
        size_t length = strlen(values[item]);
        if (index + length > count) continue;
        int matches = 1;
        for (size_t offset = 1; offset < length; offset++) {
            uint32_t got = points[index + offset].value;
            if (got >= 'A' && got <= 'Z') got += 'a' - 'A';
            if (got != (unsigned char)values[item][offset]) matches = 0;
        }
        if (matches) return length;
    }
    return 0;
}

/* -------------------------------------------------------- pre-tokenize */

/* Split the normalized text into Qwen2 pieces. Returns an array of
 * malloc'd UTF-8 strings; *count receives the length. */
static char **h3_pretokenize(const char *text, size_t length, size_t *count,
                             char *error, size_t error_size) {
    h3_codepoint *points = NULL;
    size_t point_count = h3_codepoints(text, length, &points);
    if (!points || !point_count) {
        h3_tok_error(error, error_size, "unable to decode input");
        return NULL;
    }
    char **pieces = NULL;
    size_t piece_count = 0, piece_capacity = 16;
    pieces = malloc(piece_capacity * sizeof(*pieces));
    if (!pieces) {
        free(points);
        h3_tok_error(error, error_size, "out of memory");
        return NULL;
    }
    size_t index = 0;
    int failed = 0;
    while (index < point_count) {
        size_t contraction = h3_contraction(points, point_count, index);
        if (contraction) {
            if (!h3_append_slice(&pieces, &piece_count, &piece_capacity,
                                 text, points, index, index + contraction)) {
                failed = 1;
                break;
            }
            index += contraction;
            continue;
        }
        uint32_t value = points[index].value;
        ptrdiff_t letter_start = (ptrdiff_t)index;
        if (h3_letter(value)) {
            /* already at the first letter */
        } else if (value != '\r' && value != '\n' && !h3_number(value) &&
                   index + 1 < point_count &&
                   h3_letter(points[index + 1].value)) {
            letter_start++;
        } else {
            letter_start = -1;
        }
        if (letter_start >= 0) {
            size_t stop = (size_t)letter_start;
            while (stop < point_count && h3_letter(points[stop].value)) stop++;
            if (!h3_append_slice(&pieces, &piece_count, &piece_capacity,
                                 text, points, index, stop)) {
                failed = 1;
                break;
            }
            index = stop;
            continue;
        }
        if (h3_number(value)) {
            if (!h3_append_slice(&pieces, &piece_count, &piece_capacity,
                                 text, points, index, index + 1)) {
                failed = 1;
                break;
            }
            index++;
            continue;
        }
        size_t punct_start = index +
            (value == ' ' && index + 1 < point_count &&
             !h3_space(points[index + 1].value) &&
             !h3_letter(points[index + 1].value) &&
             !h3_number(points[index + 1].value));
        size_t stop = punct_start;
        while (stop < point_count && !h3_space(points[stop].value) &&
               !h3_letter(points[stop].value) &&
               !h3_number(points[stop].value)) stop++;
        if (stop > punct_start) {
            while (stop < point_count &&
                   (points[stop].value == '\r' ||
                    points[stop].value == '\n')) stop++;
            if (!h3_append_slice(&pieces, &piece_count, &piece_capacity,
                                 text, points, index, stop)) {
                failed = 1;
                break;
            }
            index = stop;
            continue;
        }
        if (h3_space(value)) {
            size_t whitespace_end = index + 1;
            while (whitespace_end < point_count &&
                   h3_space(points[whitespace_end].value)) whitespace_end++;
            ptrdiff_t newline_end = -1;
            for (size_t cursor = index; cursor < whitespace_end; cursor++) {
                if (points[cursor].value == '\r' ||
                    points[cursor].value == '\n') {
                    newline_end = (ptrdiff_t)cursor + 1;
                }
            }
            size_t piece_end;
            if (newline_end >= 0) piece_end = (size_t)newline_end;
            else if (whitespace_end == point_count)
                piece_end = whitespace_end;
            else if (whitespace_end - index > 1)
                piece_end = whitespace_end - 1;
            else
                piece_end = index + 1;
            if (!h3_append_slice(&pieces, &piece_count, &piece_capacity,
                                 text, points, index, piece_end)) {
                failed = 1;
                break;
            }
            index = piece_end;
            continue;
        }
        /* Unreachable in the Qwen2 grammar; emit the codepoint alone. */
        if (!h3_append_slice(&pieces, &piece_count, &piece_capacity,
                             text, points, index, index + 1)) {
            failed = 1;
            break;
        }
        index++;
    }
    free(points);
    if (failed) {
        for (size_t item = 0; item < piece_count; item++)
            free(pieces[item]);
        free(pieces);
        h3_tok_error(error, error_size, "out of memory");
        return NULL;
    }
    *count = piece_count;
    return pieces;
}

/* ---------------------------------------------------------------- BPE */

/* The merge key is left + U+FFFF + right, as in the reference. */
static char *h3_pair_key(const char *left, const char *right) {
    size_t left_length = strlen(left);
    size_t right_length = strlen(right);
    char *key = malloc(left_length + 3 + right_length + 1);
    if (!key) return NULL;
    memcpy(key, left, left_length);
    memcpy(key + left_length, "\xef\xbf\xbf", 3);
    memcpy(key + left_length + 3, right, right_length);
    key[left_length + 3 + right_length] = '\0';
    return key;
}

static int h3_utf8_codepoint(uint32_t value, char *out) {
    if (value < 0x80) {
        out[0] = (char)value;
        return 1;
    }
    if (value < 0x800) {
        out[0] = (char)(0xc0 | (value >> 6));
        out[1] = (char)(0x80 | (value & 0x3f));
        return 2;
    }
    if (value < 0x10000) {
        out[0] = (char)(0xe0 | (value >> 12));
        out[1] = (char)(0x80 | ((value >> 6) & 0x3f));
        out[2] = (char)(0x80 | (value & 0x3f));
        return 3;
    }
    out[0] = (char)(0xf0 | (value >> 18));
    out[1] = (char)(0x80 | ((value >> 12) & 0x3f));
    out[2] = (char)(0x80 | ((value >> 6) & 0x3f));
    out[3] = (char)(0x80 | (value & 0x3f));
    return 4;
}

static int h3_bpe(const h3_tokenizer *tokenizer, const char *piece,
                  size_t piece_length, uint32_t **out_ids, size_t *out_count,
                  char *error, size_t error_size) {
    /* Byte-encode into codepoint strings. */
    size_t capacity = piece_length * 2 + 16;
    char **symbols = malloc(capacity * sizeof(*symbols));
    if (!symbols) {
        h3_tok_error(error, error_size, "out of memory");
        return 0;
    }
    size_t symbol_count = 0;
    for (size_t index = 0; index < piece_length; index++) {
        unsigned byte = (unsigned char)piece[index];
        char buffer[4];
        int length = h3_utf8_codepoint(tokenizer->byte_encoder[byte], buffer);
        symbols[symbol_count] = malloc((size_t)length + 1);
        if (!symbols[symbol_count]) {
            for (size_t item = 0; item < symbol_count; item++)
                free(symbols[item]);
            free(symbols);
            h3_tok_error(error, error_size, "out of memory");
            return 0;
        }
        memcpy(symbols[symbol_count], buffer, (size_t)length);
        symbols[symbol_count][length] = '\0';
        symbol_count++;
    }
    while (symbol_count > 1) {
        uint32_t best_rank = 0;
        size_t best = 0;
        int found = 0;
        for (size_t index = 0; index + 1 < symbol_count; index++) {
            char *key = h3_pair_key(symbols[index], symbols[index + 1]);
            if (!key) {
                for (size_t item = 0; item < symbol_count; item++)
                    free(symbols[item]);
                free(symbols);
                h3_tok_error(error, error_size, "out of memory");
                return 0;
            }
            uint32_t rank;
            if (h3_map_get(&tokenizer->merge_ranks, key, strlen(key),
                           &rank) &&
                (!found || rank < best_rank)) {
                best_rank = rank;
                best = index;
                found = 1;
            }
            free(key);
        }
        if (!found) break;
        const char *left = symbols[best];
        const char *right = symbols[best + 1];
        size_t left_length = strlen(left);
        size_t right_length = strlen(right);
        char **merged = malloc(symbol_count * sizeof(*merged));
        if (!merged) {
            for (size_t item = 0; item < symbol_count; item++)
                free(symbols[item]);
            free(symbols);
            h3_tok_error(error, error_size, "out of memory");
            return 0;
        }
        size_t merged_count = 0;
        char **dropped = malloc(symbol_count * sizeof(*dropped));
        size_t drops = 0;
        if (!dropped) {
            for (size_t item = 0; item < symbol_count; item++)
                free(symbols[item]);
            free(symbols);
            free(merged);
            h3_tok_error(error, error_size, "out of memory");
            return 0;
        }
        for (size_t index = 0; index < symbol_count;) {
            if (index + 1 < symbol_count &&
                strcmp(symbols[index], left) == 0 &&
                strcmp(symbols[index + 1], right) == 0) {
                merged[merged_count] = malloc(left_length + right_length + 1);
                if (!merged[merged_count]) {
                    for (size_t item = 0; item < merged_count; item++)
                        free(merged[item]);
                    free(merged);
                    for (size_t item = 0; item < drops; item++)
                        free(dropped[item]);
                    free(dropped);
                    for (size_t item = 0; item < symbol_count; item++)
                        free(symbols[item]);
                    free(symbols);
                    h3_tok_error(error, error_size, "out of memory");
                    return 0;
                }
                memcpy(merged[merged_count], left, left_length);
                memcpy(merged[merged_count] + left_length, right,
                       right_length);
                merged[merged_count][left_length + right_length] = '\0';
                merged_count++;
                /* The replaced pair is still owned by the symbols array;
                 * record it here and free after the pass. */
                dropped[drops++] = symbols[index];
                dropped[drops++] = symbols[index + 1];
                index += 2;
            } else {
                merged[merged_count++] = symbols[index++];
            }
        }
        for (size_t item = 0; item < drops; item++) free(dropped[item]);
        free(dropped);
        free(symbols);
        symbols = merged;
        symbol_count = merged_count;
    }
    uint32_t *ids = malloc(symbol_count * sizeof(*ids));
    if (!ids) {
        for (size_t item = 0; item < symbol_count; item++) free(symbols[item]);
        free(symbols);
        h3_tok_error(error, error_size, "out of memory");
        return 0;
    }
    size_t id_count = 0;
    for (size_t index = 0; index < symbol_count; index++) {
        uint32_t identifier;
        if (!h3_map_get(&tokenizer->vocab, symbols[index],
                        strlen(symbols[index]), &identifier)) {
            h3_tok_error(error, error_size, "BPE symbol is absent from "
                                            "vocabulary");
            free(ids);
            for (size_t item = 0; item < symbol_count; item++)
                free(symbols[item]);
            free(symbols);
            return 0;
        }
        ids[id_count++] = identifier;
    }
    for (size_t item = 0; item < symbol_count; item++) free(symbols[item]);
    free(symbols);
    *out_ids = ids;
    *out_count = id_count;
    return 1;
}

/* ---------------------------------------------------------------- load */

static int h3_load_byte_encoder(h3_tokenizer *tokenizer) {
    unsigned extra = 0;
    for (size_t index = 0; index < 324; index++) tokenizer->byte_decoder[index] = -1;
    for (unsigned byte = 0; byte < 256; byte++) {
        int visible = (byte >= '!' && byte <= '~') ||
                      (byte >= 0xa1 && byte <= 0xac) ||
                      (byte >= 0xae && byte <= 0xff);
        uint32_t codepoint = visible ? byte : 256 + extra++;
        tokenizer->byte_encoder[byte] = codepoint;
        if (codepoint < 324) tokenizer->byte_decoder[codepoint] = (int16_t)byte;
    }
    return 1;
}

h3_tokenizer *h3_tokenizer_load(const char *path, char *error,
                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) {
        h3_tok_error(error, error_size, "tokenizer path is required");
        return NULL;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        char message[256];
        snprintf(message, sizeof(message), "cannot read tokenizer: %s", path);
        h3_tok_error(error, error_size, message);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (file_size <= 0) {
        fclose(file);
        h3_tok_error(error, error_size, "empty tokenizer file");
        return NULL;
    }
    char *data = malloc((size_t)file_size + 1);
    if (!data) {
        fclose(file);
        h3_tok_error(error, error_size, "out of memory");
        return NULL;
    }
    if (fread(data, 1, (size_t)file_size, file) != (size_t)file_size) {
        fclose(file);
        free(data);
        h3_tok_error(error, error_size, "cannot read tokenizer");
        return NULL;
    }
    fclose(file);
    data[file_size] = '\0';

    h3_json json = { data, data + file_size };
    if (!h3_json_enter_object(&json)) {
        free(data);
        h3_tok_error(error, error_size, "invalid tokenizer JSON");
        return NULL;
    }
    h3_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer) {
        free(data);
        h3_tok_error(error, error_size, "out of memory");
        return NULL;
    }
    int failed = 0;
    int fail_phase = 0;
    char *key = NULL;
    size_t key_length = 0;
    while (h3_json_member(&json, &key, &key_length)) {
        if (key_length == 5 && memcmp(key, "model", 5) == 0) {
            h3_json *model = &json;
            if (!h3_json_enter_object(model)) {
                failed = 1, fail_phase = __LINE__;
            } else {
                char *model_key = NULL;
                size_t model_key_length = 0;
                while (h3_json_member(model, &model_key, &model_key_length)) {
                    if (model_key_length == 4 &&
                        memcmp(model_key, "type", 4) == 0) {
                        char *type = NULL;
                        if (h3_json_string(model, &type) < 0 ||
                            strcmp(type, "BPE") != 0) failed = 1, fail_phase = __LINE__;
                        free(type);
                    } else if (model_key_length == 9 &&
                               memcmp(model_key, "unk_token", 9) == 0) {
                        h3_json_skip_ws(model);
                        if (model->cursor + 4 <= model->end &&
                            memcmp(model->cursor, "null", 4) != 0) failed = 1, fail_phase = __LINE__;
                        model->cursor += 4;
                    } else if (model_key_length == 5 &&
                               memcmp(model_key, "vocab", 5) == 0) {
                        h3_json *vocab = model;
                        if (!h3_json_enter_object(vocab)) {
                            failed = 1, fail_phase = __LINE__;
                        } else {
                            h3_map_init(&tokenizer->vocab, 1 << 18);
                            char *symbol = NULL;
                            size_t symbol_length = 0;
                            while (h3_json_member(vocab, &symbol,
                                                  &symbol_length)) {
                                h3_json_skip_ws(vocab);
                                char *number_end = NULL;
                                long identifier = strtol(vocab->cursor,
                                                         &number_end, 10);
                                if (number_end > vocab->cursor)
                                    vocab->cursor = number_end;
                                if (identifier < 0 ||
                                    !h3_map_put(&tokenizer->vocab, symbol,
                                                symbol_length,
                                                (uint32_t)identifier)) {
                                    failed = 1, fail_phase = __LINE__;
                                    free(symbol);
                                    break;
                                }
                                free(symbol);
                                symbol = NULL;
                            }
                            free(symbol);
                            if (!failed && tokenizer->vocab.count > 0 &&
                                !h3_map_grow(&tokenizer->vocab)) failed = 1, fail_phase = __LINE__;
                        }
                    } else if (model_key_length == 6 &&
                               memcmp(model_key, "merges", 6) == 0) {
                        h3_json *merges = model;
                        if (!h3_json_enter_array(merges)) {
                            failed = 1, fail_phase = __LINE__;
                        } else {
                            h3_map_init(&tokenizer->merge_ranks, 1 << 18);
                            uint32_t rank = 0;
                            h3_json_skip_ws(merges);
                            while (merges->cursor < merges->end &&
                                   *merges->cursor != ']') {
                                char *entry = NULL;
                                int length = h3_json_string(merges, &entry);
                                if (length < 0) {
                                    failed = 1, fail_phase = __LINE__;
                                    break;
                                }
                                /* Entry is "left right" or [left, right]. */
                                const char *space = memchr(entry, ' ', (size_t)length);
                                char *left = NULL, *right = NULL;
                                if (space) {
                                    left = malloc((size_t)(space - entry) + 1);
                                    right = malloc((size_t)(entry + length - space - 1) + 1);
                                    if (left && right) {
                                        memcpy(left, entry, (size_t)(space - entry));
                                        left[space - entry] = '\0';
                                        memcpy(right, space + 1, (size_t)(entry + length - space - 1));
                                        right[entry + length - space - 1] = '\0';
                                    }
                                } else {
                                    /* array form: peek the next string */
                                    char *second = NULL;
                                    h3_json_skip_ws(merges);
                                    if (merges->cursor < merges->end &&
                                        *merges->cursor == ',') {
                                        merges->cursor++;
                                        if (h3_json_string(merges, &second) >= 0) {
                                            left = malloc((size_t)length + 1);
                                            if (left) {
                                                memcpy(left, entry, (size_t)length);
                                                left[length] = '\0';
                                            }
                                            right = second;
                                        }
                                    }
                                }
                                free(entry);
                                if (!left || !right) {
                                    failed = 1, fail_phase = __LINE__;
                                    free(left);
                                    free(right);
                                    break;
                                }
                                char *pair_key = h3_pair_key(left, right);
                                free(left);
                                free(right);
                                if (!pair_key ||
                                    !h3_map_put(&tokenizer->merge_ranks,
                                                pair_key, strlen(pair_key),
                                                rank++)) {
                                    failed = 1, fail_phase = __LINE__;
                                    free(pair_key);
                                    break;
                                }
                                free(pair_key);
                                h3_json_skip_ws(merges);
                                if (merges->cursor < merges->end &&
                                    *merges->cursor == ',') merges->cursor++;
                                h3_json_skip_ws(merges);
                            }
                            if (merges->cursor < merges->end &&
                                *merges->cursor == ']') merges->cursor++;
                        }
                    } else {
                        /* Unhandled model keys (dropout, suffix flags, ...)
                         * carry values that must be consumed. */
                        h3_json_skip_value(model);
                    }
                    free(model_key);
                    model_key = NULL;
                    if (failed) break;
                }
                free(model_key);
            }
        } else if (key_length == 10 && memcmp(key, "normalizer", 10) == 0) {
            h3_json *normalizer = &json;
            if (!h3_json_enter_object(normalizer)) {
                failed = 1, fail_phase = __LINE__;
            } else {
                char *normalizer_key = NULL;
                size_t normalizer_key_length = 0;
                while (h3_json_member(normalizer, &normalizer_key,
                                      &normalizer_key_length)) {
                    if (normalizer_key_length == 4 &&
                        memcmp(normalizer_key, "type", 4) == 0) {
                        char *type = NULL;
                        int type_length = h3_json_string(normalizer, &type);
                        if (type_length < 0 ||
                            strcmp(type, "NFC") != 0) failed = 1, fail_phase = __LINE__;
                        free(type);
                    }
                    free(normalizer_key);
                    normalizer_key = NULL;
                }
                free(normalizer_key);
            }
        } else if (key_length == 12 && memcmp(key, "added_tokens", 12) == 0) {
            h3_json *added = &json;
            if (!h3_json_enter_array(added)) {
                failed = 1, fail_phase = __LINE__;
            } else {
                h3_map_init(&tokenizer->added_tokens, 64);
                h3_json_skip_ws(added);
                while (added->cursor < added->end && *added->cursor != ']') {
                    h3_json *entry = added;
                    if (!h3_json_enter_object(entry)) {
                        failed = 1, fail_phase = __LINE__;
                        break;
                    }
                    char *content = NULL, *entry_key = NULL;
                    size_t entry_key_length = 0;
                    long identifier = -1;
                    while (h3_json_member(entry, &entry_key,
                                          &entry_key_length)) {
                        if (entry_key_length == 7 &&
                            memcmp(entry_key, "content", 7) == 0) {
                            if (h3_json_string(entry, &content) < 0)
                                failed = 1, fail_phase = __LINE__;
                        } else if (entry_key_length == 2 &&
                                   memcmp(entry_key, "id", 2) == 0) {
                            h3_json_skip_ws(entry);
                            char *number_end = NULL;
                            identifier = strtol(entry->cursor,
                                                &number_end, 10);
                            if (number_end > entry->cursor)
                                entry->cursor = number_end;
                        } else if (entry_key_length == 10 &&
                                   memcmp(entry_key, "single_word", 10) == 0) {
                            h3_json_skip_ws(entry);
                            if (entry->cursor + 4 <= entry->end &&
                                memcmp(entry->cursor, "true", 4) == 0)
                                failed = 1, fail_phase = __LINE__;
                            if (h3_json_boolean(entry) < 0)
                                failed = 1, fail_phase = __LINE__;
                            entry->cursor = entry->cursor;
                        } else if (entry_key_length == 6 &&
                                   memcmp(entry_key, "lstrip", 6) == 0) {
                            h3_json_skip_ws(entry);
                            if (entry->cursor + 4 <= entry->end &&
                                memcmp(entry->cursor, "true", 4) == 0)
                                failed = 1, fail_phase = __LINE__;
                            if (h3_json_boolean(entry) < 0)
                                failed = 1, fail_phase = __LINE__;
                            entry->cursor = entry->cursor;
                        } else if (entry_key_length == 6 &&
                                   memcmp(entry_key, "rstrip", 6) == 0) {
                            h3_json_skip_ws(entry);
                            if (entry->cursor + 4 <= entry->end &&
                                memcmp(entry->cursor, "true", 4) == 0)
                                failed = 1, fail_phase = __LINE__;
                            if (h3_json_boolean(entry) < 0)
                                failed = 1, fail_phase = __LINE__;
                            entry->cursor = entry->cursor;
                        } else if (entry_key_length == 10 &&
                                   memcmp(entry_key, "normalized", 10) == 0) {
                            h3_json_skip_ws(entry);
                            if (entry->cursor + 4 <= entry->end &&
                                memcmp(entry->cursor, "true", 4) == 0)
                                failed = 1, fail_phase = __LINE__;
                            if (h3_json_boolean(entry) < 0)
                                failed = 1, fail_phase = __LINE__;
                            entry->cursor = entry->cursor;
                        } else {
                            h3_json_skip_value(entry);
                        }
                        free(entry_key);
                        entry_key = NULL;
                    }
                    free(entry_key);
                    if (content && identifier >= 0) {
                        if (!h3_map_put(&tokenizer->added_tokens, content,
                                        strlen(content),
                                        (uint32_t)identifier))
                            failed = 1, fail_phase = __LINE__;
                    }
                    free(content);
                    h3_json_skip_ws(added);
                    if (added->cursor < added->end && *added->cursor == ',')
                        added->cursor++;
                    h3_json_skip_ws(added);
                    if (failed) break;
                }
                if (added->cursor < added->end && *added->cursor == ']')
                    added->cursor++;
            }
        } else {
            h3_json_skip_value(&json);
        }
        free(key);
        key = NULL;
        if (failed) break;
    }
    free(key);
    free(data);
    if (failed || !tokenizer->vocab.entries ||
        !tokenizer->merge_ranks.entries) {
        if (!failed)
            {
                char spec_message[128];
                snprintf(spec_message, sizeof(spec_message),
                         "unexpected tokenizer specification (line %d)",
                         fail_phase);
                h3_tok_error(error, error_size, spec_message);
            }
        h3_tokenizer_free(tokenizer);
        return NULL;
    }

    /* Inverse vocabularies and added-token alternatives. */
    uint32_t maximum_id = 0;
    for (size_t index = 0; index < tokenizer->vocab.capacity; index++) {
        if (tokenizer->vocab.entries[index].key &&
            tokenizer->vocab.entries[index].value > maximum_id)
            maximum_id = tokenizer->vocab.entries[index].value;
    }
    for (size_t index = 0; index < tokenizer->added_tokens.capacity; index++) {
        if (tokenizer->added_tokens.entries[index].key &&
            tokenizer->added_tokens.entries[index].value > maximum_id)
            maximum_id = tokenizer->added_tokens.entries[index].value;
    }
    tokenizer->inverse_count = (size_t)maximum_id + 1;
    tokenizer->inverse_vocab = calloc(tokenizer->inverse_count,
                                      sizeof(*tokenizer->inverse_vocab));
    tokenizer->inverse_added_count = tokenizer->inverse_count;
    tokenizer->inverse_added = calloc(tokenizer->inverse_added_count,
                                      sizeof(*tokenizer->inverse_added));
    if (!tokenizer->inverse_vocab || !tokenizer->inverse_added) {
        h3_tok_error(error, error_size, "out of memory");
        h3_tokenizer_free(tokenizer);
        return NULL;
    }
    for (size_t index = 0; index < tokenizer->vocab.capacity; index++) {
        h3_map_entry entry = tokenizer->vocab.entries[index];
        if (entry.key && entry.value < tokenizer->inverse_count)
            tokenizer->inverse_vocab[entry.value] = entry.key;
    }
    tokenizer->added_count = tokenizer->added_tokens.count;
    if (tokenizer->added_count) {
        tokenizer->added_alternatives = malloc(tokenizer->added_count *
                                              sizeof(*tokenizer->added_alternatives));
        if (!tokenizer->added_alternatives) {
            h3_tok_error(error, error_size, "out of memory");
            h3_tokenizer_free(tokenizer);
            return NULL;
        }
        size_t used = 0;
        for (size_t index = 0; index < tokenizer->added_tokens.capacity;
             index++) {
            h3_map_entry entry = tokenizer->added_tokens.entries[index];
            if (entry.key) {
                tokenizer->added_alternatives[used++] = entry.key;
                if (entry.value < tokenizer->inverse_added_count)
                    tokenizer->inverse_added[entry.value] = entry.key;
            }
        }
        /* Longest first; ties by byte order. */
        for (size_t a = 1; a < used; a++) {
            char *value = tokenizer->added_alternatives[a];
            size_t b = a;
            while (b > 0) {
                char *previous = tokenizer->added_alternatives[b - 1];
                size_t previous_length = strlen(previous);
                size_t value_length = strlen(value);
                int after = value_length < previous_length ||
                            (value_length == previous_length &&
                             strcmp(value, previous) > 0);
                if (!after) break;
                tokenizer->added_alternatives[b] = previous;
                b--;
            }
            tokenizer->added_alternatives[b] = value;
        }
    }
    h3_load_byte_encoder(tokenizer);
    return tokenizer;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    h3_map_free(&tokenizer->vocab);
    h3_map_free(&tokenizer->merge_ranks);
    h3_map_free(&tokenizer->added_tokens);
    free(tokenizer->inverse_vocab);
    free(tokenizer->inverse_added);
    free(tokenizer->added_alternatives);
    free(tokenizer);
}

/* --------------------------------------------------------------- encode */

static int h3_encode_plain(const h3_tokenizer *tokenizer, const char *text,
                           size_t length, uint32_t **output, size_t *count,
                           char *error, size_t error_size) {
    size_t piece_count = 0;
    char **pieces = h3_pretokenize(text, length, &piece_count, error,
                                   error_size);
    if (!pieces) return 0;
    size_t capacity = 64, total = 0;
    uint32_t *ids = malloc(capacity * sizeof(*ids));
    if (!ids) {
        for (size_t item = 0; item < piece_count; item++) free(pieces[item]);
        free(pieces);
        h3_tok_error(error, error_size, "out of memory");
        return 0;
    }
    for (size_t item = 0; item < piece_count; item++) {
        uint32_t *piece_ids = NULL;
        size_t piece_id_count = 0;
        if (!h3_bpe(tokenizer, pieces[item], strlen(pieces[item]),
                    &piece_ids, &piece_id_count, error, error_size)) {
            free(ids);
            for (size_t rest = item; rest < piece_count; rest++)
                free(pieces[rest]);
            free(pieces);
            return 0;
        }
        if (total + piece_id_count > capacity) {
            while (total + piece_id_count > capacity) capacity *= 2;
            uint32_t *grown = realloc(ids, capacity * sizeof(*ids));
            if (!grown) {
                free(piece_ids);
                free(ids);
                for (size_t rest = item; rest < piece_count; rest++)
                    free(pieces[rest]);
                free(pieces);
                h3_tok_error(error, error_size, "out of memory");
                return 0;
            }
            ids = grown;
        }
        memcpy(ids + total, piece_ids, piece_id_count * sizeof(*ids));
        total += piece_id_count;
        free(piece_ids);
        free(pieces[item]);
    }
    free(pieces);
    *output = ids;
    *count = total;
    return 1;
}

/* Normalize to NFC (the tokenizer spec requires it). */
static char *h3_normalize_nfc(const char *utf8, size_t length,
                              size_t *out_length) {
    UErrorCode status = U_ZERO_ERROR;
    const UNormalizer2 *normalizer = unorm2_getNFCInstance(&status);
    if (!normalizer || U_FAILURE(status)) return NULL;
    int32_t utf16_length = 0;
    u_strFromUTF8(NULL, 0, &utf16_length, utf8, (int32_t)length, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) return NULL;
    status = U_ZERO_ERROR;
    UChar *utf16 = malloc(((size_t)utf16_length + 1) * sizeof(*utf16));
    if (!utf16) return NULL;
    u_strFromUTF8(utf16, utf16_length + 1, NULL, utf8, (int32_t)length,
                  &status);
    if (U_FAILURE(status)) {
        free(utf16);
        return NULL;
    }
    int32_t normalized_length = unorm2_normalize(
        normalizer, utf16, utf16_length, NULL, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        free(utf16);
        return NULL;
    }
    status = U_ZERO_ERROR;
    UChar *normalized = malloc(((size_t)normalized_length + 1) *
                               sizeof(*normalized));
    if (!normalized) {
        free(utf16);
        return NULL;
    }
    normalized_length = unorm2_normalize(
        normalizer, utf16, utf16_length, normalized,
        normalized_length + 1, &status);
    free(utf16);
    if (U_FAILURE(status)) {
        free(normalized);
        return NULL;
    }
    int32_t utf8_length = 0;
    u_strToUTF8(NULL, 0, &utf8_length, normalized, normalized_length,
                &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        free(normalized);
        return NULL;
    }
    status = U_ZERO_ERROR;
    char *result = malloc((size_t)utf8_length + 1);
    if (!result) {
        free(normalized);
        return NULL;
    }
    u_strToUTF8(result, utf8_length + 1, NULL, normalized,
                normalized_length, &status);
    free(normalized);
    if (U_FAILURE(status)) {
        free(result);
        return NULL;
    }
    result[utf8_length] = '\0';
    *out_length = (size_t)utf8_length;
    return result;
}

static int h3_added_match(const h3_tokenizer *tokenizer, const char *text,
                          size_t length, size_t search_start, size_t *match,
                          size_t *match_length, const char **token) {
    int found = 0;
    for (size_t item = 0; item < tokenizer->added_count; item++) {
        const char *candidate = tokenizer->added_alternatives[item];
        size_t candidate_length = strlen(candidate);
        /* leftmost occurrence at or after search_start */
        size_t cursor = search_start;
        while (cursor + candidate_length <= length) {
            if (memcmp(text + cursor, candidate, candidate_length) == 0) break;
            cursor++;
        }
        if (cursor + candidate_length > length) continue;
        if (!found || cursor < *match ||
            (cursor == *match && candidate_length > *match_length)) {
            *match = cursor;
            *match_length = candidate_length;
            *token = candidate;
            found = 1;
        }
    }
    return found;
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || !utf8 || !ids || !count) return 0;
    *ids = NULL;
    *count = 0;
    size_t input_length = strlen(utf8);
    size_t normalized_length = 0;
    char *normalized = h3_normalize_nfc(utf8, input_length,
                                        &normalized_length);
    if (!normalized) {
        h3_tok_error(error, error_size, "prompt is not valid UTF-8");
        return 0;
    }
    size_t capacity = 64, total = 0;
    uint32_t *output = malloc(capacity * sizeof(*output));
    if (!output) {
        free(normalized);
        h3_tok_error(error, error_size, "out of memory encoding prompt");
        return 0;
    }
    size_t start = 0;
    int failed = 0;
    while (start < normalized_length) {
        size_t match, match_length;
        const char *added = NULL;
        if (!h3_added_match(tokenizer, normalized, normalized_length, start,
                            &match, &match_length, &added))
            break;
        if (match > start) {
            uint32_t *plain = NULL;
            size_t plain_count = 0;
            if (!h3_encode_plain(tokenizer, normalized + start,
                                 match - start, &plain, &plain_count, error,
                                 error_size)) {
                failed = 1;
                break;
            }
            if (total + plain_count > capacity) {
                while (total + plain_count > capacity) capacity *= 2;
                uint32_t *grown = realloc(output,
                                          capacity * sizeof(*output));
                if (!grown) {
                    free(plain);
                    failed = 1;
                    break;
                }
                output = grown;
            }
            memcpy(output + total, plain, plain_count * sizeof(*output));
            total += plain_count;
            free(plain);
        }
        uint32_t added_id;
        if (!h3_map_get(&tokenizer->added_tokens, added, match_length,
                        &added_id)) {
            h3_tok_error(error, error_size, "added token lookup failed");
            failed = 1;
            break;
        }
        if (total + 1 > capacity) {
            capacity *= 2;
            uint32_t *grown = realloc(output, capacity * sizeof(*output));
            if (!grown) {
                failed = 1;
                break;
            }
            output = grown;
        }
        output[total++] = added_id;
        start = match + match_length;
    }
    if (!failed && start < normalized_length) {
        uint32_t *plain = NULL;
        size_t plain_count = 0;
        if (!h3_encode_plain(tokenizer, normalized + start,
                             normalized_length - start, &plain, &plain_count,
                             error, error_size)) {
            failed = 1;
        } else {
            if (total + plain_count > capacity) {
                while (total + plain_count > capacity) capacity *= 2;
                uint32_t *grown = realloc(output,
                                          capacity * sizeof(*output));
                if (!grown) {
                    free(plain);
                    failed = 1;
                } else {
                    output = grown;
                }
            }
            if (!failed) {
                memcpy(output + total, plain,
                       plain_count * sizeof(*output));
                total += plain_count;
            }
            free(plain);
        }
    }
    free(normalized);
    if (failed) {
        free(output);
        return 0;
    }
    if (total == 0 && pad_empty) output[total++] = H3_PAD_TOKEN_ID;
    if (total) *ids = output;
    *count = total;
    return 1;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

/* --------------------------------------------------------------- decode */

/* Validate and copy UTF-8; replace invalid sequences with U+FFFD. */
static void h3_utf8_sanitize(const char *bytes, size_t length,
                             char *output, size_t *out_length) {
    static const char replacement[] = "\xef\xbf\xbd";
    size_t written = 0;
    size_t index = 0;
    while (index < length) {
        unsigned char c = (unsigned char)bytes[index];
        size_t units = 1;
        uint32_t value = c;
        if (c >= 0xf0) {
            if (index + 3 < length &&
                (bytes[index + 1] & 0xc0) == 0x80 &&
                (bytes[index + 2] & 0xc0) == 0x80 &&
                (bytes[index + 3] & 0xc0) == 0x80) {
                value = ((uint32_t)(c & 0x07) << 18) |
                        ((uint32_t)(bytes[index + 1] & 0x3f) << 12) |
                        ((uint32_t)(bytes[index + 2] & 0x3f) << 6) |
                        (uint32_t)(bytes[index + 3] & 0x3f);
                units = 4;
            } else {
                value = 0;
            }
        } else if (c >= 0xe0) {
            if (index + 2 < length &&
                (bytes[index + 1] & 0xc0) == 0x80 &&
                (bytes[index + 2] & 0xc0) == 0x80) {
                value = ((uint32_t)(c & 0x0f) << 12) |
                        ((uint32_t)(bytes[index + 1] & 0x3f) << 6) |
                        (uint32_t)(bytes[index + 2] & 0x3f);
                units = 3;
            } else {
                value = 0;
            }
        } else if (c >= 0xc0) {
            if (index + 1 < length &&
                (bytes[index + 1] & 0xc0) == 0x80) {
                value = ((uint32_t)(c & 0x1f) << 6) |
                        (uint32_t)(bytes[index + 1] & 0x3f);
                units = 2;
            } else {
                value = 0;
            }
        } else if (c < 0x80) {
            value = c;
        } else {
            value = 0;
        }
        if (value == 0) {
            memcpy(output + written, replacement, 3);
            written += 3;
            index++;
            continue;
        }
        /* Reject overlong encodings and surrogates. */
        int valid = 1;
        if (units == 2 && value < 0x80) valid = 0;
        if (units == 3 && value < 0x800) valid = 0;
        if (units == 4 && value < 0x10000) valid = 0;
        if (value >= 0xd800 && value <= 0xdfff) valid = 0;
        if (value > 0x10ffff) valid = 0;
        if (!valid) {
            memcpy(output + written, replacement, 3);
            written += 3;
            index++;
            continue;
        }
        memcpy(output + written, bytes + index, units);
        written += units;
        index += units;
    }
    *out_length = written;
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || (!ids && count)) return NULL;
    size_t capacity = 256, length = 0;
    char *result = malloc(capacity);
    if (!result) {
        h3_tok_error(error, error_size, "out of memory decoding tokens");
        return NULL;
    }
    for (size_t index = 0; index < count; index++) {
        uint32_t identifier = ids[index];
        if (identifier >= tokenizer->inverse_added_count) {
            h3_tok_error(error, error_size, "token ID is out of range");
            free(result);
            return NULL;
        }
        const char *added = tokenizer->inverse_added[identifier];
        if (added) {
            size_t added_length = strlen(added);
            if (length + added_length + 1 > capacity) {
                while (length + added_length + 1 > capacity) capacity *= 2;
                char *grown = realloc(result, capacity);
                if (!grown) {
                    h3_tok_error(error, error_size, "out of memory");
                    free(result);
                    return NULL;
                }
                result = grown;
            }
            memcpy(result + length, added, added_length);
            length += added_length;
            continue;
        }
        const char *symbol = identifier < tokenizer->inverse_count ?
            tokenizer->inverse_vocab[identifier] : NULL;
        if (!symbol) {
            h3_tok_error(error, error_size, "unknown token ID");
            free(result);
            return NULL;
        }
        size_t symbol_length = strlen(symbol);
        /* byte-decode the symbol into a temp buffer, then sanitize */
        char *bytes = malloc(symbol_length + 1);
        if (!bytes) {
            h3_tok_error(error, error_size, "out of memory");
            free(result);
            return NULL;
        }
        size_t byte_count = 0;
        size_t offset = 0;
        int invalid = 0;
        while (offset < symbol_length) {
            /* decode one UTF-8 codepoint from the symbol */
            unsigned char c = (unsigned char)symbol[offset];
            size_t units = 1;
            uint32_t value = c;
            if (c >= 0xf0 && offset + 3 < symbol_length) {
                value = ((uint32_t)(c & 0x07) << 18) |
                        ((uint32_t)(symbol[offset + 1] & 0x3f) << 12) |
                        ((uint32_t)(symbol[offset + 2] & 0x3f) << 6) |
                        (uint32_t)(symbol[offset + 3] & 0x3f);
                units = 4;
            } else if (c >= 0xe0 && offset + 2 < symbol_length) {
                value = ((uint32_t)(c & 0x0f) << 12) |
                        ((uint32_t)(symbol[offset + 1] & 0x3f) << 6) |
                        (uint32_t)(symbol[offset + 2] & 0x3f);
                units = 3;
            } else if (c >= 0xc0 && offset + 1 < symbol_length) {
                value = ((uint32_t)(c & 0x1f) << 6) |
                        (uint32_t)(symbol[offset + 1] & 0x3f);
                units = 2;
            }
            if (value >= 324 || tokenizer->byte_decoder[value] < 0) {
                invalid = 1;
                break;
            }
            bytes[byte_count++] = (char)tokenizer->byte_decoder[value];
            offset += units;
        }
        if (invalid) {
            free(bytes);
            h3_tok_error(error, error_size, "invalid byte-level token");
            free(result);
            return NULL;
        }
        char *sanitized = malloc(symbol_length * 3 + 1);
        if (!sanitized) {
            free(bytes);
            h3_tok_error(error, error_size, "out of memory");
            free(result);
            return NULL;
        }
        size_t sanitized_length = 0;
        h3_utf8_sanitize(bytes, byte_count, sanitized, &sanitized_length);
        free(bytes);
        if (length + sanitized_length + 1 > capacity) {
            while (length + sanitized_length + 1 > capacity) capacity *= 2;
            char *grown = realloc(result, capacity);
            if (!grown) {
                free(sanitized);
                h3_tok_error(error, error_size, "out of memory");
                free(result);
                return NULL;
            }
            result = grown;
        }
        memcpy(result + length, sanitized, sanitized_length);
        length += sanitized_length;
        free(sanitized);
    }
    result[length] = '\0';
    return result;
}
