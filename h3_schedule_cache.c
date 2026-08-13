#include "h3_schedule_cache.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define H3_SCHEDULE_CACHE_VERSION 1u

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint64_t weight_size;
    uint64_t weight_mtime_seconds;
    uint64_t weight_mtime_nanoseconds;
    uint64_t weight_offset;
    uint64_t bias_size;
    uint64_t bias_mtime_seconds;
    uint64_t bias_mtime_nanoseconds;
    uint64_t bias_offset;
    uint64_t time_hash;
    uint64_t rows;
    uint64_t columns;
    uint64_t elements;
} h3_schedule_cache_header;

_Static_assert(sizeof(h3_schedule_cache_header) == 112,
               "schedule cache header must remain stable");

static void schedule_fail(char *error, size_t error_size, const char *format,
                          ...) {
    if (!error || !error_size)
        return;
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static uint64_t schedule_mtime_nanoseconds(const struct stat *status) {
#if defined(__APPLE__)
    return (uint64_t)status->st_mtimespec.tv_nsec;
#else
    return (uint64_t)status->st_mtim.tv_nsec;
#endif
}

static int schedule_source(const char *path, uint64_t *size,
                           uint64_t *seconds, uint64_t *nanoseconds,
                           char *error, size_t error_size) {
    struct stat status;
    if (!path || stat(path, &status) != 0) {
        schedule_fail(error, error_size, "cannot stat schedule source %s: %s",
                      path ? path : "(null)", strerror(errno));
        return 0;
    }
    *size = (uint64_t)status.st_size;
    *seconds = (uint64_t)status.st_mtime;
    *nanoseconds = schedule_mtime_nanoseconds(&status);
    return 1;
}

static int schedule_header(h3_schedule_cache_header *header,
                           const char *weight_path, uint64_t weight_offset,
                           const char *bias_path, uint64_t bias_offset,
                           uint64_t time_hash, uint64_t rows, uint64_t columns,
                           char *error, size_t error_size) {
    if (!header || (rows && columns > UINT64_MAX / rows)) {
        schedule_fail(error, error_size, "schedule cache shape overflows");
        return 0;
    }
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, "H3ADLC01", sizeof(header->magic));
    header->version = H3_SCHEDULE_CACHE_VERSION;
    header->header_size = (uint32_t)sizeof(*header);
    if (!schedule_source(weight_path, &header->weight_size,
                         &header->weight_mtime_seconds,
                         &header->weight_mtime_nanoseconds, error,
                         error_size) ||
        !schedule_source(bias_path, &header->bias_size,
                         &header->bias_mtime_seconds,
                         &header->bias_mtime_nanoseconds, error, error_size))
        return 0;
    header->weight_offset = weight_offset;
    header->bias_offset = bias_offset;
    header->time_hash = time_hash;
    header->rows = rows;
    header->columns = columns;
    header->elements = rows * columns;
    return 1;
}

static char *schedule_path(const char *directory, const char *key,
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

static int schedule_mkdir_p(const char *directory, char *error,
                            size_t error_size) {
    char *path = strdup(directory ? directory : "");
    if (!path || !*path) {
        free(path);
        schedule_fail(error, error_size, "invalid schedule cache directory");
        return 0;
    }
    for (char *cursor = path + 1; *cursor; cursor++) {
        if (*cursor != '/')
            continue;
        *cursor = '\0';
        if (mkdir(path, 0755) != 0 && errno != EEXIST) {
            schedule_fail(error, error_size, "cannot create %s: %s", path,
                          strerror(errno));
            free(path);
            return 0;
        }
        *cursor = '/';
    }
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        schedule_fail(error, error_size, "cannot create %s: %s", path,
                      strerror(errno));
        free(path);
        return 0;
    }
    free(path);
    return 1;
}

static int schedule_read_all(int descriptor, void *buffer, size_t bytes,
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

static int schedule_write_all(int descriptor, const void *buffer,
                              size_t bytes) {
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

int h3_schedule_cache_load(
    h3_gpu *gpu, const char *directory, const char *key,
    const char *weight_path, uint64_t weight_offset,
    const char *bias_path, uint64_t bias_offset, uint64_t time_hash,
    uint64_t rows, uint64_t columns, h3_gpu_tensor **output, int *hit,
    uint64_t *loaded_bytes, char *error, size_t error_size) {
    if (error && error_size)
        error[0] = '\0';
    if (output)
        *output = NULL;
    if (hit)
        *hit = 0;
    if (loaded_bytes)
        *loaded_bytes = 0;
    if (!gpu || !output || !hit) {
        schedule_fail(error, error_size, "invalid schedule cache request");
        return 0;
    }
    h3_schedule_cache_header expected;
    if (!schedule_header(&expected, weight_path, weight_offset, bias_path,
                         bias_offset, time_hash, rows, columns, error,
                         error_size))
        return 0;
    char *path = schedule_path(directory, key, ".h3adaln");
    if (!path) {
        schedule_fail(error, error_size, "cannot allocate schedule cache path");
        return 0;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        free(path);
        if (errno == ENOENT)
            return 1;
        schedule_fail(error, error_size, "cannot open schedule cache: %s",
                      strerror(errno));
        return 0;
    }
    struct stat status;
    h3_schedule_cache_header found;
    uint64_t bytes = sizeof(found) + expected.elements * sizeof(uint16_t);
    int valid = fstat(descriptor, &status) == 0 &&
                schedule_read_all(descriptor, &found, sizeof(found), 0) &&
                memcmp(&found, &expected, sizeof(found)) == 0 &&
                (uint64_t)status.st_size == bytes;
    close(descriptor);
    if (!valid) {
        (void)unlink(path);
        free(path);
        return 1;
    }
    if (expected.elements > SIZE_MAX) {
        free(path);
        schedule_fail(error, error_size, "schedule cache exceeds limits");
        return 0;
    }
    *output = h3_gpu_tensor_load_bf16(gpu, path, sizeof(found),
                                      (size_t)expected.elements);
    if (!*output) {
        schedule_fail(error, error_size, "cannot upload schedule cache %s: %s",
                      path, h3_gpu_error(gpu));
        free(path);
        return 0;
    }
    free(path);
    *hit = 1;
    if (loaded_bytes)
        *loaded_bytes = expected.elements * sizeof(uint16_t);
    return 1;
}

int h3_schedule_cache_save(
    const char *directory, const char *key,
    const char *weight_path, uint64_t weight_offset,
    const char *bias_path, uint64_t bias_offset, uint64_t time_hash,
    uint64_t rows, uint64_t columns, const h3_gpu_tensor *output,
    uint64_t *stored_bytes, char *error, size_t error_size) {
    if (error && error_size)
        error[0] = '\0';
    if (stored_bytes)
        *stored_bytes = 0;
    h3_schedule_cache_header header;
    if (!output || !schedule_header(&header, weight_path, weight_offset,
                                    bias_path, bias_offset, time_hash, rows,
                                    columns, error, error_size))
        return 0;
    if (header.elements > SIZE_MAX ||
        h3_gpu_tensor_dtype(output) != H3_GPU_BF16 ||
        h3_gpu_tensor_elements(output) < (size_t)header.elements) {
        schedule_fail(error, error_size, "invalid schedule cache tensor");
        return 0;
    }
    if (!schedule_mkdir_p(directory, error, error_size))
        return 0;
    char *path = schedule_path(directory, key, ".h3adaln");
    char suffix[64];
    (void)snprintf(suffix, sizeof(suffix), ".tmp.%ld", (long)getpid());
    char *temporary = schedule_path(directory, key, suffix);
    if (!path || !temporary) {
        free(path);
        free(temporary);
        schedule_fail(error, error_size, "cannot allocate schedule paths");
        return 0;
    }
    int descriptor = open(temporary, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                          0644);
    uint16_t *values = header.elements <= SIZE_MAX / sizeof(uint16_t)
                           ? malloc((size_t)header.elements * sizeof(uint16_t))
                           : NULL;
    int ok = descriptor >= 0 && values &&
             h3_gpu_tensor_read_bf16(output, values,
                                     (size_t)header.elements) &&
             schedule_write_all(descriptor, &header, sizeof(header)) &&
             schedule_write_all(descriptor, values,
                                (size_t)header.elements * sizeof(uint16_t));
    int saved_errno = errno;
    if (descriptor >= 0 && close(descriptor) != 0)
        ok = 0;
    if (ok && rename(temporary, path) != 0) {
        saved_errno = errno;
        ok = 0;
    }
    if (!ok) {
        (void)unlink(temporary);
        schedule_fail(error, error_size, "cannot write schedule cache %s: %s",
                      path, strerror(saved_errno ? saved_errno : EIO));
    } else if (stored_bytes) {
        *stored_bytes = header.elements * sizeof(uint16_t);
    }
    free(values);
    free(path);
    free(temporary);
    return ok ? 1 : 0;
}
