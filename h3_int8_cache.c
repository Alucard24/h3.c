#include "h3_int8_cache.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define H3_INT8_CACHE_VERSION 1u

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint64_t source_size;
    uint64_t source_mtime_seconds;
    uint64_t source_mtime_nanoseconds;
    uint64_t source_offset;
    uint64_t rows;
    uint64_t columns;
    uint64_t weight_elements;
    uint64_t scale_elements;
} h3_int8_cache_header;

_Static_assert(sizeof(h3_int8_cache_header) == 80,
               "int8 cache header must remain stable");

static void cache_fail(char *error, size_t error_size, const char *format,
                       ...) {
    if (!error || !error_size)
        return;
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static uint64_t cache_mtime_nanoseconds(const struct stat *status) {
#if defined(__APPLE__)
    return (uint64_t)status->st_mtimespec.tv_nsec;
#else
    return (uint64_t)status->st_mtim.tv_nsec;
#endif
}

static int cache_source_header(h3_int8_cache_header *header,
                               const char *source_path,
                               uint64_t source_offset, uint64_t rows,
                               uint64_t columns, char *error,
                               size_t error_size) {
    struct stat status;
    if (!header || !source_path || stat(source_path, &status) != 0) {
        cache_fail(error, error_size, "cannot stat int8 cache source %s: %s",
                   source_path ? source_path : "(null)", strerror(errno));
        return 0;
    }
    if (rows && columns > UINT64_MAX / rows) {
        cache_fail(error, error_size, "int8 cache tensor shape overflows");
        return 0;
    }
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, "H3I8C01", sizeof(header->magic));
    header->version = H3_INT8_CACHE_VERSION;
    header->header_size = (uint32_t)sizeof(*header);
    header->source_size = (uint64_t)status.st_size;
    header->source_mtime_seconds = (uint64_t)status.st_mtime;
    header->source_mtime_nanoseconds = cache_mtime_nanoseconds(&status);
    header->source_offset = source_offset;
    header->rows = rows;
    header->columns = columns;
    header->weight_elements = rows * columns;
    header->scale_elements = rows;
    return 1;
}

static char *cache_entry_path(const char *directory, const char *key,
                              const char *suffix) {
    if (!directory || !*directory || !key || !*key || !suffix)
        return NULL;
    size_t directory_length = strlen(directory);
    size_t key_length = strlen(key);
    size_t suffix_length = strlen(suffix);
    if (directory_length > SIZE_MAX - key_length - suffix_length - 2u)
        return NULL;
    size_t bytes = directory_length + key_length + suffix_length + 2u;
    char *path = malloc(bytes);
    if (!path)
        return NULL;
    size_t cursor = 0;
    memcpy(path + cursor, directory, directory_length);
    cursor += directory_length;
    path[cursor++] = '/';
    for (size_t index = 0; index < key_length; index++) {
        unsigned char value = (unsigned char)key[index];
        int safe = (value >= (unsigned char)'a' && value <= (unsigned char)'z') ||
                   (value >= (unsigned char)'A' && value <= (unsigned char)'Z') ||
                   (value >= (unsigned char)'0' && value <= (unsigned char)'9') ||
                   value == (unsigned char)'-' || value == (unsigned char)'_';
        path[cursor++] = safe ? (char)value : '_';
    }
    memcpy(path + cursor, suffix, suffix_length + 1u);
    return path;
}

static int cache_mkdir_p(const char *directory, char *error,
                         size_t error_size) {
    char *path = strdup(directory ? directory : "");
    if (!path || !*path) {
        free(path);
        cache_fail(error, error_size, "invalid int8 cache directory");
        return 0;
    }
    for (char *cursor = path + 1; *cursor; cursor++) {
        if (*cursor != '/')
            continue;
        *cursor = '\0';
        if (mkdir(path, 0755) != 0 && errno != EEXIST) {
            cache_fail(error, error_size, "cannot create %s: %s", path,
                       strerror(errno));
            free(path);
            return 0;
        }
        *cursor = '/';
    }
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        cache_fail(error, error_size, "cannot create %s: %s", path,
                   strerror(errno));
        free(path);
        return 0;
    }
    free(path);
    return 1;
}

static int cache_read_all(int descriptor, void *buffer, size_t bytes,
                          uint64_t offset) {
    unsigned char *cursor = buffer;
    while (bytes) {
        ssize_t count = pread(descriptor, cursor, bytes, (off_t)offset);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return 0;
        cursor += (size_t)count;
        bytes -= (size_t)count;
        offset += (uint64_t)count;
    }
    return 1;
}

static int cache_write_all(int descriptor, const void *buffer, size_t bytes) {
    const unsigned char *cursor = buffer;
    while (bytes) {
        ssize_t count = write(descriptor, cursor, bytes);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return 0;
        cursor += (size_t)count;
        bytes -= (size_t)count;
    }
    return 1;
}

static int cache_headers_equal(const h3_int8_cache_header *left,
                               const h3_int8_cache_header *right) {
    return memcmp(left, right, sizeof(*left)) == 0;
}

int h3_int8_cache_resolve(const char *directory, const char *key,
                          const char *source_path, uint64_t source_offset,
                          uint64_t rows, uint64_t columns, char **entry_path,
                          uint64_t *weight_offset, uint64_t *scale_offset,
                          int *hit, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (entry_path) *entry_path = NULL;
    if (weight_offset) *weight_offset = 0;
    if (scale_offset) *scale_offset = 0;
    if (hit) *hit = 0;
    if (!entry_path || !weight_offset || !scale_offset || !hit) {
        cache_fail(error, error_size, "invalid int8 cache resolve request");
        return 0;
    }
    h3_int8_cache_header expected;
    if (!cache_source_header(&expected, source_path, source_offset, rows,
                             columns, error, error_size))
        return 0;
    char *path = cache_entry_path(directory, key, ".h3i8");
    if (!path) {
        cache_fail(error, error_size, "cannot allocate int8 cache path");
        return 0;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        int missing = errno == ENOENT;
        if (!missing)
            cache_fail(error, error_size, "cannot open int8 cache: %s",
                       strerror(errno));
        free(path);
        return missing;
    }
    struct stat status;
    h3_int8_cache_header found;
    uint64_t expected_bytes = sizeof(found) + expected.weight_elements +
                              expected.scale_elements * sizeof(float);
    int valid = fstat(descriptor, &status) == 0 &&
                cache_read_all(descriptor, &found, sizeof(found), 0) &&
                cache_headers_equal(&found, &expected) &&
                (uint64_t)status.st_size == expected_bytes;
    close(descriptor);
    if (!valid) {
        (void)unlink(path);
        free(path);
        return 1;
    }
    *entry_path = path;
    *weight_offset = sizeof(found);
    *scale_offset = sizeof(found) + expected.weight_elements;
    *hit = 1;
    return 1;
}

int h3_int8_cache_load(h3_gpu *gpu, const char *directory, const char *key,
                       const char *source_path, uint64_t source_offset,
                       uint64_t rows, uint64_t columns,
                       h3_gpu_tensor **weight, h3_gpu_tensor **scales,
                       int *hit, uint64_t *loaded_bytes,
                       char *error, size_t error_size) {
    if (error && error_size)
        error[0] = '\0';
    if (weight)
        *weight = NULL;
    if (scales)
        *scales = NULL;
    if (hit)
        *hit = 0;
    if (loaded_bytes)
        *loaded_bytes = 0;
    if (!gpu || !weight || !scales || !hit) {
        cache_fail(error, error_size, "invalid int8 cache load request");
        return 0;
    }
    h3_int8_cache_header expected;
    if (!cache_source_header(&expected, source_path, source_offset, rows,
                             columns, error, error_size))
        return 0;
    char *path = cache_entry_path(directory, key, ".h3i8");
    if (!path) {
        cache_fail(error, error_size, "cannot allocate int8 cache path");
        return 0;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        free(path);
        if (errno == ENOENT)
            return 1;
        cache_fail(error, error_size, "cannot open int8 cache: %s",
                   strerror(errno));
        return 0;
    }
    struct stat status;
    h3_int8_cache_header found;
    uint64_t expected_bytes = sizeof(found) + expected.weight_elements +
                              expected.scale_elements * sizeof(float);
    int valid = fstat(descriptor, &status) == 0 &&
                cache_read_all(descriptor, &found, sizeof(found), 0) &&
                cache_headers_equal(&found, &expected) &&
                (uint64_t)status.st_size == expected_bytes;
    close(descriptor);
    if (!valid) {
        (void)unlink(path);
        free(path);
        return 1;
    }
    if (expected.weight_elements > SIZE_MAX ||
        expected.scale_elements > SIZE_MAX) {
        free(path);
        cache_fail(error, error_size, "int8 cache exceeds process limits");
        return 0;
    }
    *weight = h3_gpu_tensor_load_i8(
        gpu, path, sizeof(found), (size_t)expected.weight_elements);
    *scales = *weight ? h3_gpu_tensor_load_f32(
                            gpu, path,
                            sizeof(found) + expected.weight_elements,
                            (size_t)expected.scale_elements)
                      : NULL;
    if (!*weight || !*scales) {
        h3_gpu_tensor_free(*weight);
        h3_gpu_tensor_free(*scales);
        *weight = NULL;
        *scales = NULL;
        cache_fail(error, error_size, "cannot upload int8 cache %s: %s", path,
                   h3_gpu_error(gpu));
        free(path);
        return 0;
    }
    free(path);
    *hit = 1;
    if (loaded_bytes)
        *loaded_bytes = expected.weight_elements +
                        expected.scale_elements * sizeof(float);
    return 1;
}

int h3_int8_cache_save(const char *directory, const char *key,
                       const char *source_path, uint64_t source_offset,
                       uint64_t rows, uint64_t columns,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *scales,
                       uint64_t *stored_bytes,
                       char *error, size_t error_size) {
    if (error && error_size)
        error[0] = '\0';
    if (stored_bytes)
        *stored_bytes = 0;
    h3_int8_cache_header header;
    if (!weight || !scales ||
        !cache_source_header(&header, source_path, source_offset, rows,
                             columns, error, error_size))
        return 0;
    if (header.weight_elements > SIZE_MAX || header.scale_elements > SIZE_MAX ||
        h3_gpu_tensor_dtype(weight) != H3_GPU_I8 ||
        h3_gpu_tensor_dtype(scales) != H3_GPU_F32 ||
        h3_gpu_tensor_elements(weight) < (size_t)header.weight_elements ||
        h3_gpu_tensor_elements(scales) < (size_t)header.scale_elements) {
        cache_fail(error, error_size, "invalid tensors for int8 cache save");
        return 0;
    }
    if (!cache_mkdir_p(directory, error, error_size))
        return 0;
    char *path = cache_entry_path(directory, key, ".h3i8");
    char suffix[64];
    (void)snprintf(suffix, sizeof(suffix), ".tmp.%ld", (long)getpid());
    char *temporary = cache_entry_path(directory, key, suffix);
    if (!path || !temporary) {
        free(path);
        free(temporary);
        cache_fail(error, error_size, "cannot allocate int8 cache paths");
        return 0;
    }
    int descriptor = open(temporary, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                          0644);
    if (descriptor < 0) {
        cache_fail(error, error_size, "cannot create int8 cache %s: %s",
                   temporary, strerror(errno));
        free(path);
        free(temporary);
        return 0;
    }
    int8_t *weight_values = malloc((size_t)header.weight_elements);
    float *scale_values = malloc((size_t)header.scale_elements * sizeof(float));
    int ok = weight_values && scale_values &&
             h3_gpu_tensor_read_i8(weight, weight_values,
                                   (size_t)header.weight_elements) &&
             h3_gpu_tensor_read_f32(scales, scale_values,
                                    (size_t)header.scale_elements) &&
             cache_write_all(descriptor, &header, sizeof(header)) &&
             cache_write_all(descriptor, weight_values,
                             (size_t)header.weight_elements) &&
             cache_write_all(descriptor, scale_values,
                             (size_t)header.scale_elements * sizeof(float));
    int saved_errno = errno;
    if (close(descriptor) != 0)
        ok = 0;
    if (ok && rename(temporary, path) != 0) {
        saved_errno = errno;
        ok = 0;
    }
    if (!ok) {
        (void)unlink(temporary);
        cache_fail(error, error_size, "cannot write int8 cache %s: %s", path,
                   strerror(saved_errno ? saved_errno : EIO));
    } else if (stored_bytes) {
        *stored_bytes = header.weight_elements +
                        header.scale_elements * sizeof(float);
    }
    free(weight_values);
    free(scale_values);
    free(path);
    free(temporary);
    return ok ? 1 : 0;
}
