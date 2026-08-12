/* Vulkan GPU backend for the h3 inference engine.
 *
 * Implements the h3_gpu.h contract on Linux/NVIDIA (any Vulkan 1.3 device
 * with shaderInt16). Shaders are compiled at runtime from
 * h3_vulkan_shaders.comp through libshaderc, mirroring the Metal backend's
 * runtime-compilation model.
 *
 * Phase 1: tensors live in host-visible coherent buffers and every dispatch
 * takes a full memory barrier. This is correct for the current unit-test
 * workloads; the DiT-scale residency and barrier strategy arrives with the
 * linear/int8 kernels.
 */
#include "h3.h"
#include "h3_gpu.h"

#include <shaderc/shaderc.h>
#include <vulkan/vulkan.h>

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define H3_VK_BINDINGS 11
#define H3_VK_THREADS 256u
#define H3_VK_POOL_SETS 2048
/* Kernel args are written by the host before submission but read by the GPU
 * during execution, so each dispatch gets its own slot in the shared args
 * buffer. One slot per dispatch; large graphs must call h3_gpu_continue. */
#define H3_VK_ARGS_SLOTS 1024u

/* Layout shared by every kernel; matches the H3Args block in
 * h3_vulkan_shaders.comp (binding 7). */
typedef struct {
    uint32_t count;
    uint32_t elements;
    uint32_t sample_offset;
    uint32_t tokens;
    uint32_t width;
    uint32_t rows;
    uint32_t vocab_size;
    uint32_t approximate;
    float left_scale;
    float right_scale;
    float minimum;
    float maximum;
    float delta;
    float ratio;
    float epsilon;
    uint32_t has_bias;
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    uint32_t gate_slot;
    uint32_t grouped;
    /* Conv3d / VAE-pad / group-norm parameters (Video VAE family). */
    uint32_t conv_batch;
    uint32_t conv_depth;
    uint32_t conv_height;
    uint32_t conv_width;
    uint32_t conv_in_channels;
    uint32_t conv_out_channels;
    uint32_t conv_kernel_depth;
    uint32_t conv_kernel_height;
    uint32_t conv_kernel_width;
    uint32_t conv_stride_depth;
    uint32_t conv_stride_height;
    uint32_t conv_stride_width;
    uint32_t pad_depth_front;
    uint32_t pad_height_before;
    uint32_t pad_height_after;
    uint32_t pad_width_before;
    uint32_t pad_width_after;
} h3_gpu_vulkan_args;

typedef enum {
    H3_VK_KERNEL_CAST_F32_TO_BF16,
    H3_VK_KERNEL_CAST_BF16_TO_F32,
    H3_VK_KERNEL_SILU_F32,
    H3_VK_KERNEL_SILU_BF16,
    H3_VK_KERNEL_SILU_MUL_BF16,
    H3_VK_KERNEL_CLIP_F32,
    H3_VK_KERNEL_GELU_BF16,
    H3_VK_KERNEL_ADD_BF16,
    H3_VK_KERNEL_SUB_BF16,
    H3_VK_KERNEL_ADD_SCALED_F32,
    H3_VK_KERNEL_GEGLU_F32,
    H3_VK_KERNEL_EULER_BF16,
    H3_VK_KERNEL_EMBEDDING_BF16,
    H3_VK_KERNEL_RMS_NORM_F32,
    H3_VK_KERNEL_LAYER_NORM_F32,
    H3_VK_KERNEL_RMS_NORM_BF16,
    H3_VK_KERNEL_LAYER_NORM_BF16,
    H3_VK_KERNEL_LINEAR_BF16,
    H3_VK_KERNEL_SWIGLU_HALVES_BF16,
    H3_VK_KERNEL_ADALN_BF16,
    H3_VK_KERNEL_GATE_BF16,
    H3_VK_KERNEL_GATE_ADALN_BF16,
    H3_VK_KERNEL_RMS_INVERSE_BF16,
    H3_VK_KERNEL_ADALN_LINEAR_BF16,
    H3_VK_KERNEL_HEAD_RMS_NORM_BF16,
    H3_VK_KERNEL_GROUPED_QKV_ROPE_BF16,
    H3_VK_KERNEL_SDPA_BF16,
    H3_VK_KERNEL_SDPA_FLASH_BF16,
    H3_VK_KERNEL_PATCH_LINEAR_BF16,
    H3_VK_KERNEL_PATCH_LINEAR_BF16_MAP,
    H3_VK_KERNEL_TOKEN_POOL_BF16,
    H3_VK_KERNEL_TOKEN_POOL_ADALN_BF16,
    H3_VK_KERNEL_TOKEN_EXPAND_DELTA_BF16,
    H3_VK_KERNEL_TOKEN_EXPAND_ADALN_BF16,
    H3_VK_KERNEL_TEXT_QK_ROPE_BF16,
    H3_VK_KERNEL_ROPE_TEXT_BF16,
    H3_VK_KERNEL_GQA_CAUSAL_BF16,
    H3_VK_KERNEL_LINEAR_F32,
    H3_VK_KERNEL_SCALE_ADD_F32,
    H3_VK_KERNEL_SWIGLU_F32,
    H3_VK_KERNEL_VIDEO_QKV_ROPE_F32,
    H3_VK_KERNEL_VAE_ENCODER_PAD_F32,
    H3_VK_KERNEL_VAE_ENCODER_GROUP_NORM_SILU_F32,
    H3_VK_KERNEL_SDPA_F32,
    H3_VK_KERNEL_CONV3D_F32,
    H3_VK_KERNEL_WEIGHT_NORM_F32,
    H3_VK_KERNEL_SNAKE1D_F32,
    H3_VK_KERNEL_ALIAS_FREE_SNAKE_F32,
    H3_VK_KERNEL_AUDIO_QKV_SPLIT_F32,
    H3_VK_KERNEL_AUDIO_ATTENTION_POOL_F32,
    H3_VK_KERNEL_CONV1D_STRIDE_F32,
    H3_VK_KERNEL_CONV_TRANSPOSE1D_F32,
    H3_VK_KERNEL_SDPA_CAUSAL_F32,
    H3_VK_KERNEL_VISION_QKV_ROPE_BF16,
    H3_VK_KERNEL_COUNT
} h3_vk_kernel;

static const char *const h3_vk_kernel_names[H3_VK_KERNEL_COUNT] = {
    "main_cast_f32_to_bf16", "main_cast_bf16_to_f32",
    "main_silu_f32", "main_silu_bf16", "main_silu_mul_bf16",
    "main_clip_f32", "main_gelu_bf16", "main_add_bf16", "main_sub_bf16",
    "main_add_scaled_f32", "main_geglu_f32", "main_euler_bf16",
    "main_embedding_bf16", "main_rms_norm_f32", "main_layer_norm_f32",
    "main_rms_norm_bf16", "main_layer_norm_bf16",
    "main_linear_bf16", "main_swiglu_halves_bf16",
    "main_adaln_bf16", "main_gate_bf16", "main_gate_adaln_bf16",
    "main_rms_inverse_bf16", "main_adaln_linear_bf16",
    "main_head_rms_norm_bf16",
    "main_grouped_qkv_rope_bf16", "main_sdpa_bf16",
    "main_sdpa_flash_bf16",
    "main_patch_linear_bf16", "main_patch_linear_bf16_map",
    "main_token_pool_bf16", "main_token_pool_adaln_bf16",
    "main_token_expand_delta_bf16", "main_token_expand_adaln_bf16",
    "main_text_qk_rope_bf16", "main_rope_text_bf16",
    "main_gqa_causal_bf16",
    "main_linear_f32", "main_scale_add_f32",
    "main_swiglu_f32", "main_video_qkv_rope_f32",
    "main_vae_encoder_pad_f32", "main_vae_encoder_group_norm_silu_f32",
    "main_sdpa_f32", "main_conv3d_f32",
    "main_weight_norm_f32", "main_snake1d_f32",
    "main_alias_free_snake_f32", "main_audio_qkv_split_f32",
    "main_audio_attention_pool_f32", "main_conv1d_stride_f32",
    "main_conv_transpose1d_f32", "main_sdpa_causal_f32",
    "main_vision_qkv_rope_bf16"
};

/* Storage-buffer bindings consumed by each kernel (0..n-1 plus binding 7
 * for the args buffer). */
/* Highest storage-buffer binding used by each kernel plus one (bindings
 * 0..n-1 carry tensors, binding 7 always carries the args buffer). */
static const uint32_t h3_vk_kernel_bindings[H3_VK_KERNEL_COUNT] = {
    2, 2, 2, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 4, 2,
    5, 5, 8, 2, 8, 2, 8, 4, 4, 4, 5, 6, 10, 6, 10, 8, 4, 4, 4, 4, 2, 6, 2, 4, 4, 4, 3, 3, 6, 7, 2, 4, 4, 4, 6
};

/* Workgroup layout per kernel: 0 = 256 threads, 1 = 16x16 tiles,
 * 2 = 128 threads (SDPA flash). */
static const int h3_vk_kernel_layout[H3_VK_KERNEL_COUNT] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 1, 0, 0, 1, 1, 0, 0, 2, 1, 1, 1, 0, 1, 0, 0, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1
};

struct h3_gpu {
    char error[512];
    h3_gpu_stats stats;
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    VkCommandPool command_pool;
    VkCommandBuffer command;
    int command_active;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipelines[H3_VK_KERNEL_COUNT];
    VkDescriptorPool desc_pool;
    VkDescriptorPool desc_pools[32];
    uint32_t desc_pool_count;
    VkBuffer args_buffer;
    VkDeviceMemory args_memory;
    h3_gpu_vulkan_args *args;
    h3_gpu_vulkan_args *args_base;
    uint32_t args_slot;
    uint32_t args_dispatches;
    shaderc_compiler_t shaderc;
    double encode_start;
    double wait_start;
    /* Tensors freed while their dispatches are still recorded or in flight.
     * Metal keeps these alive through ARC; Vulkan must defer destruction. */
    h3_gpu_tensor *pending[64];
    uint32_t pending_count;
};

struct h3_gpu_tensor {
    h3_gpu *gpu;
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    size_t elements;
    h3_gpu_dtype dtype;
    size_t byte_size;
    /* Device-local buffers (weights) have no host mapping; transfers go
     * through staging copies. */
    int device_local;
};

static double h3_vk_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void h3_vk_drain_pending(h3_gpu *gpu);

static void h3_vk_set_error(h3_gpu *gpu, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(gpu->error, sizeof(gpu->error), format, args);
    va_end(args);
}

static size_t h3_vk_dtype_size(h3_gpu_dtype dtype) {
    switch (dtype) {
        case H3_GPU_BF16: return 2;
        case H3_GPU_I8: return 1;
        case H3_GPU_F32:
        case H3_GPU_U32: return 4;
    }
    return 0;
}

static int h3_vk_require_command(h3_gpu *gpu) {
    if (!gpu->command_active) {
        h3_vk_set_error(gpu, "no active command buffer (call h3_gpu_begin)");
        return 0;
    }
    return 1;
}

static int h3_vk_check_tensors(h3_gpu *gpu, size_t count, ...) {
    va_list args;
    va_start(args, count);
    for (size_t index = 0; index < count; index++) {
        const void *tensor = va_arg(args, const void *);
        if (!tensor) {
            h3_vk_set_error(gpu, "null tensor argument");
            va_end(args);
            return 0;
        }
    }
    va_end(args);
    return 1;
}

/* ---------------------------------------------------------------- device */

static int h3_vk_create_instance(h3_gpu *gpu, char *error, size_t error_size) {
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "h3-metal",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "h3",
        .apiVersion = VK_API_VERSION_1_3
    };
    const char *layers[1] = { NULL };
    uint32_t layer_count = 0;
    if (getenv("H3_VK_VALIDATION")) {
        uint32_t available = 0;
        vkEnumerateInstanceLayerProperties(&available, NULL);
        VkLayerProperties *props = calloc(available, sizeof(*props));
        if (props) {
            vkEnumerateInstanceLayerProperties(&available, props);
            for (uint32_t index = 0; index < available; index++) {
                if (strcmp(props[index].layerName,
                           "VK_LAYER_KHRONOS_validation") == 0) {
                    layers[0] = "VK_LAYER_KHRONOS_validation";
                    layer_count = 1;
                    break;
                }
            }
            free(props);
        }
    }
    VkInstanceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledLayerCount = layer_count,
        .ppEnabledLayerNames = layers
    };
    VkResult result = vkCreateInstance(&info, NULL, &gpu->instance);
    if (result != VK_SUCCESS) {
        if (error && error_size) {
            snprintf(error, error_size,
                     "vkCreateInstance failed: %d (is a Vulkan driver installed?)",
                     (int)result);
        }
        return 0;
    }
    return 1;
}

static int h3_vk_pick_device(h3_gpu *gpu, char *error, size_t error_size) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(gpu->instance, &count, NULL) !=
            VK_SUCCESS || count == 0) {
        if (error && error_size)
            snprintf(error, error_size, "no Vulkan physical device found");
        return 0;
    }
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    if (!devices) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory");
        return 0;
    }
    vkEnumeratePhysicalDevices(gpu->instance, &count, devices);
    int best_score = -1;
    for (uint32_t index = 0; index < count; index++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[index], &props);
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(devices[index], &features);
        if (!features.shaderInt16) continue;
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index],
                                                 &family_count, NULL);
        VkQueueFamilyProperties *families =
            calloc(family_count, sizeof(*families));
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index],
                                                 &family_count, families);
        int has_compute = 0;
        for (uint32_t family = 0; family < family_count; family++) {
            if (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                has_compute = 1;
                break;
            }
        }
        free(families);
        if (!has_compute) continue;
        int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ?
                    2 : 1;
        if (score > best_score) {
            best_score = score;
            gpu->physical = devices[index];
        }
    }
    free(devices);
    if (best_score < 0) {
        if (error && error_size)
            snprintf(error, error_size,
                     "no Vulkan 1.3 device with shaderInt16 found");
        return 0;
    }
    return 1;
}

static int h3_vk_create_device(h3_gpu *gpu, char *error, size_t error_size) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu->physical, &family_count,
                                             NULL);
    VkQueueFamilyProperties *families =
        calloc(family_count, sizeof(*families));
    vkGetPhysicalDeviceQueueFamilyProperties(gpu->physical, &family_count,
                                             families);
    uint32_t compute_family = UINT32_MAX;
    for (uint32_t family = 0; family < family_count; family++) {
        if (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_family = family;
            break;
        }
    }
    free(families);
    if (compute_family == UINT32_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "device has no compute queue");
        return 0;
    }
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = compute_family,
        .queueCount = 1,
        .pQueuePriorities = &priority
    };
    VkPhysicalDeviceFeatures features = {0};
    features.shaderInt16 = VK_TRUE;
    VkDeviceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .pEnabledFeatures = &features
    };
    VkResult result = vkCreateDevice(gpu->physical, &info, NULL,
                                     &gpu->device);
    if (result != VK_SUCCESS) {
        if (error && error_size)
            snprintf(error, error_size, "vkCreateDevice failed: %d",
                     (int)result);
        return 0;
    }
    vkGetDeviceQueue(gpu->device, compute_family, 0, &gpu->queue);
    return 1;
}

/* ---------------------------------------------------------------- shaders */

static VkShaderModule h3_vk_compile(h3_gpu *gpu, const char *source,
                                    size_t source_size,
                                    const char *entry_point, int layout) {
    shaderc_compilation_result_t result = NULL;
    if (gpu->shaderc) {
        shaderc_compile_options_t options =
            shaderc_compile_options_initialize();
        shaderc_compile_options_set_target_env(
            options, shaderc_target_env_vulkan,
            shaderc_env_version_vulkan_1_3);
        shaderc_compile_options_set_optimization_level(
            options, shaderc_optimization_level_zero);
        shaderc_compile_options_add_macro_definition(
            options, "H3_ENTRY", strlen("H3_ENTRY"), entry_point,
            strlen(entry_point));
        if (layout == 1)
            shaderc_compile_options_add_macro_definition(
                options, "H3_LS16", strlen("H3_LS16"), "1", 1);
        else if (layout == 2)
            shaderc_compile_options_add_macro_definition(
                options, "H3_LS128", strlen("H3_LS128"), "1", 1);
        result = shaderc_compile_into_spv(
            gpu->shaderc, source, source_size,
            shaderc_compute_shader, "h3_vulkan_shaders.comp",
            "main", options);
        shaderc_compile_options_release(options);
    }
    if (!result ||
        shaderc_result_get_compilation_status(result) !=
            shaderc_compilation_status_success) {
        h3_vk_set_error(gpu, "shader %s failed: %s", entry_point,
                        result ? shaderc_result_get_error_message(result)
                               : "shaderc unavailable");
        if (result) shaderc_result_release(result);
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shaderc_result_get_length(result),
        .pCode = (const uint32_t *)shaderc_result_get_bytes(result)
    };
    VkShaderModule module = VK_NULL_HANDLE;
    vkCreateShaderModule(gpu->device, &info, NULL, &module);
    shaderc_result_release(result);
    return module;
}

static int h3_vk_create_pipelines(h3_gpu *gpu, const char *source,
                                  size_t source_size,
                                  char *error, size_t error_size) {
    VkDescriptorSetLayoutBinding bindings[H3_VK_BINDINGS];
    for (uint32_t index = 0; index < H3_VK_BINDINGS; index++) {
        bindings[index] = (VkDescriptorSetLayoutBinding){
            .binding = index,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = H3_VK_BINDINGS,
        .pBindings = bindings
    };
    if (vkCreateDescriptorSetLayout(gpu->device, &layout_info, NULL,
                                    &gpu->set_layout) != VK_SUCCESS) {
        if (error && error_size)
            snprintf(error, error_size, "vkCreateDescriptorSetLayout failed");
        return 0;
    }
    VkPipelineLayoutCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &gpu->set_layout
    };
    if (vkCreatePipelineLayout(gpu->device, &pipeline_info, NULL,
                               &gpu->pipeline_layout) != VK_SUCCESS) {
        if (error && error_size)
            snprintf(error, error_size, "vkCreatePipelineLayout failed");
        return 0;
    }
    for (int kernel = 0; kernel < H3_VK_KERNEL_COUNT; kernel++) {
        VkShaderModule module = h3_vk_compile(gpu, source, source_size,
                                              h3_vk_kernel_names[kernel],
                                              h3_vk_kernel_layout[kernel]);
        if (module == VK_NULL_HANDLE) {
            if (error && error_size)
                snprintf(error, error_size, "%s", gpu->error);
            return 0;
        }
        VkPipelineShaderStageCreateInfo stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module,
            .pName = "main"
        };
        VkComputePipelineCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = stage,
            .layout = gpu->pipeline_layout
        };
        VkResult result = vkCreateComputePipelines(
            gpu->device, VK_NULL_HANDLE, 1, &info, NULL,
            &gpu->pipelines[kernel]);
        vkDestroyShaderModule(gpu->device, module, NULL);
        if (result != VK_SUCCESS) {
            if (error && error_size)
                snprintf(error, error_size,
                         "vkCreateComputePipelines(%s) failed: %d",
                         h3_vk_kernel_names[kernel], (int)result);
            return 0;
        }
    }
    return 1;
}

/* ------------------------------------------------------------ descriptors */

static VkDescriptorSet h3_vk_alloc_set(h3_gpu *gpu, uint32_t binding_count) {
    (void)binding_count;
    if (!gpu->desc_pool) {
        VkDescriptorPoolSize size = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = H3_VK_POOL_SETS * H3_VK_BINDINGS
        };
        VkDescriptorPoolCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = H3_VK_POOL_SETS,
            .poolSizeCount = 1,
            .pPoolSizes = &size
        };
        if (vkCreateDescriptorPool(gpu->device, &info, NULL,
                                   &gpu->desc_pool) != VK_SUCCESS) {
            h3_vk_set_error(gpu, "vkCreateDescriptorPool failed");
            return VK_NULL_HANDLE;
        }
        if (gpu->desc_pool_count < 32)
            gpu->desc_pools[gpu->desc_pool_count++] = gpu->desc_pool;
    }
    VkDescriptorSetLayout layouts[1] = { gpu->set_layout };
    VkDescriptorSetAllocateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = gpu->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = layouts
    };
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(gpu->device, &info, &set) != VK_SUCCESS) {
        h3_vk_set_error(gpu, "vkAllocateDescriptorSets failed");
        return VK_NULL_HANDLE;
    }
    return set;
}

static void h3_vk_set_buffer_offset(h3_gpu *gpu, VkDescriptorSet set,
                                    uint32_t binding, VkBuffer buffer,
                                    VkDeviceSize offset) {
    (void)gpu;
    VkDescriptorBufferInfo info = {
        .buffer = buffer,
        .offset = offset,
        .range = VK_WHOLE_SIZE
    };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &info
    };
    vkUpdateDescriptorSets(gpu->device, 1, &write, 0, NULL);
}

/* -------------------------------------------------------------- command */

static int h3_vk_begin_command(h3_gpu *gpu) {
    if (gpu->command_active) return 1;
    VkCommandBufferAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = gpu->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    if (vkAllocateCommandBuffers(gpu->device, &alloc,
                                 &gpu->command) != VK_SUCCESS) {
        h3_vk_set_error(gpu, "vkAllocateCommandBuffers failed");
        return 0;
    }
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    if (vkBeginCommandBuffer(gpu->command, &begin) != VK_SUCCESS) {
        h3_vk_set_error(gpu, "vkBeginCommandBuffer failed");
        return 0;
    }
    gpu->command_active = 1;
    return 1;
}

static int h3_vk_submit_command(h3_gpu *gpu, int wait) {
    if (!gpu->command_active) return 1;
    if (vkEndCommandBuffer(gpu->command) != VK_SUCCESS) {
        h3_vk_set_error(gpu, "vkEndCommandBuffer failed");
        return 0;
    }
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &gpu->command
    };
    gpu->wait_start = h3_vk_now();
    if (vkQueueSubmit(gpu->queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) {
        h3_vk_set_error(gpu, "vkQueueSubmit failed");
        return 0;
    }
    gpu->stats.submissions++;
    gpu->command_active = 0;
    if (wait) {
        if (vkDeviceWaitIdle(gpu->device) != VK_SUCCESS) {
            h3_vk_set_error(gpu, "vkDeviceWaitIdle failed");
            return 0;
        }
        gpu->stats.command_wait_seconds += h3_vk_now() - gpu->wait_start;
        gpu->stats.gpu_seconds += h3_vk_now() - gpu->wait_start;
        h3_vk_drain_pending(gpu);
        vkResetCommandPool(gpu->device, gpu->command_pool, 0);
        for (uint32_t index = 0; index < gpu->desc_pool_count; index++)
            vkDestroyDescriptorPool(gpu->device, gpu->desc_pools[index], NULL);
        gpu->desc_pool_count = 0;
        gpu->desc_pool = VK_NULL_HANDLE;
    }
    return 1;
}

/* ---------------------------------------------------------------- public */

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    h3_gpu *gpu = calloc(1, sizeof(*gpu));
    if (!gpu) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory");
        return NULL;
    }
    if (!shader_source_path) {
        if (error && error_size)
            snprintf(error, error_size, "no shader source path");
        free(gpu);
        return NULL;
    }
    struct stat status;
    if (stat(shader_source_path, &status) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot open shader source %s",
                     shader_source_path);
        free(gpu);
        return NULL;
    }
    size_t source_size = (size_t)status.st_size;
    char *source = malloc(source_size + 1);
    if (!source) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory");
        free(gpu);
        return NULL;
    }
    int fd = open(shader_source_path, O_RDONLY);
    if (fd < 0 || read(fd, source, source_size) != (ssize_t)source_size) {
        if (error && error_size)
            snprintf(error, error_size, "cannot read shader source %s",
                     shader_source_path);
        if (fd >= 0) close(fd);
        free(source);
        free(gpu);
        return NULL;
    }
    close(fd);
    source[source_size] = '\0';

    if (!h3_vk_create_instance(gpu, error, error_size) ||
        !h3_vk_pick_device(gpu, error, error_size) ||
        !h3_vk_create_device(gpu, error, error_size)) {
        h3_gpu_free(gpu);
        free(source);
        return NULL;
    }
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };
    if (vkCreateCommandPool(gpu->device, &pool_info, NULL,
                            &gpu->command_pool) != VK_SUCCESS) {
        if (error && error_size)
            snprintf(error, error_size, "vkCreateCommandPool failed");
        h3_gpu_free(gpu);
        free(source);
        return NULL;
    }
    gpu->shaderc = shaderc_compiler_initialize();
    if (!gpu->shaderc) {
        if (error && error_size)
            snprintf(error, error_size, "shaderc unavailable");
        h3_gpu_free(gpu);
        free(source);
        return NULL;
    }
    if (!h3_vk_create_pipelines(gpu, source, source_size,
                                error, error_size)) {
        h3_gpu_free(gpu);
        free(source);
        return NULL;
    }
    free(source);

    /* Per-kernel args buffer (binding 7). */
    VkBufferCreateInfo args_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sizeof(h3_gpu_vulkan_args) * H3_VK_ARGS_SLOTS,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    };
    if (vkCreateBuffer(gpu->device, &args_info, NULL,
                       &gpu->args_buffer) != VK_SUCCESS) {
        if (error && error_size)
            snprintf(error, error_size, "vkCreateBuffer(args) failed");
        h3_gpu_free(gpu);
        return NULL;
    }
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(gpu->device, gpu->args_buffer,
                                  &requirements);
    VkMemoryAllocateInfo memory_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size
    };
    VkPhysicalDeviceMemoryProperties properties;
    vkGetPhysicalDeviceMemoryProperties(gpu->physical, &properties);
    for (uint32_t index = 0; index < properties.memoryTypeCount; index++) {
        if ((requirements.memoryTypeBits & (1u << index)) &&
            (properties.memoryTypes[index].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memory_info.memoryTypeIndex = index;
            break;
        }
    }
    if (vkAllocateMemory(gpu->device, &memory_info, NULL,
                         &gpu->args_memory) != VK_SUCCESS ||
        vkBindBufferMemory(gpu->device, gpu->args_buffer, gpu->args_memory,
                           0) != VK_SUCCESS ||
        vkMapMemory(gpu->device, gpu->args_memory, 0, VK_WHOLE_SIZE, 0,
                    (void **)&gpu->args_base) != VK_SUCCESS) {
        if (error && error_size)
            snprintf(error, error_size, "args buffer setup failed");
        h3_gpu_free(gpu);
        return NULL;
    }
    gpu->args = gpu->args_base;
    return gpu;
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    if (gpu->device) {
        h3_vk_drain_pending(gpu);
        if (gpu->args_memory) {
            vkUnmapMemory(gpu->device, gpu->args_memory);
            vkFreeMemory(gpu->device, gpu->args_memory, NULL);
        }
        if (gpu->args_buffer)
            vkDestroyBuffer(gpu->device, gpu->args_buffer, NULL);
        for (uint32_t index = 0; index < gpu->desc_pool_count; index++)
            vkDestroyDescriptorPool(gpu->device, gpu->desc_pools[index], NULL);
        if (gpu->pipeline_layout)
            vkDestroyPipelineLayout(gpu->device, gpu->pipeline_layout, NULL);
        if (gpu->set_layout)
            vkDestroyDescriptorSetLayout(gpu->device, gpu->set_layout, NULL);
        for (int kernel = 0; kernel < H3_VK_KERNEL_COUNT; kernel++) {
            if (gpu->pipelines[kernel])
                vkDestroyPipeline(gpu->device, gpu->pipelines[kernel], NULL);
        }
        if (gpu->command_pool)
            vkDestroyCommandPool(gpu->device, gpu->command_pool, NULL);
        vkDestroyDevice(gpu->device, NULL);
    }
    if (gpu->instance) vkDestroyInstance(gpu->instance, NULL);
    if (gpu->shaderc) shaderc_compiler_release(gpu->shaderc);
    free(gpu);
}

int h3_gpu_is_m5(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

int h3_gpu_has_nax_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu ? gpu->error : "null gpu";
}

/* ---------------------------------------------------------------- tensors */

static h3_gpu_tensor *h3_vk_tensor_new_mem(h3_gpu *gpu, size_t elements,
                                           h3_gpu_dtype dtype,
                                           int device_local);

static h3_gpu_tensor *h3_vk_tensor_new(h3_gpu *gpu, size_t elements,
                                       h3_gpu_dtype dtype) {
    return h3_vk_tensor_new_mem(gpu, elements, dtype, 0);
}

static h3_gpu_tensor *h3_vk_tensor_new_mem(h3_gpu *gpu, size_t elements,
                                           h3_gpu_dtype dtype,
                                           int device_local) {
    size_t dtype_size = h3_vk_dtype_size(dtype);
    size_t byte_size = dtype_size ? elements * dtype_size : 0;
    if (elements && !byte_size) {
        h3_vk_set_error(gpu, "bad tensor size");
        return NULL;
    }
    h3_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3_vk_set_error(gpu, "out of memory");
        return NULL;
    }
    VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = byte_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    };
    if (byte_size > 0 &&
        vkCreateBuffer(gpu->device, &info, NULL, &tensor->buffer) !=
            VK_SUCCESS) {
        h3_vk_set_error(gpu, "vkCreateBuffer failed");
        free(tensor);
        return NULL;
    }
    if (byte_size > 0) {
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(gpu->device, tensor->buffer,
                                      &requirements);
        VkMemoryAllocateInfo memory_info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size
        };
        VkPhysicalDeviceMemoryProperties properties;
        vkGetPhysicalDeviceMemoryProperties(gpu->physical, &properties);
        VkMemoryPropertyFlags wanted = device_local ?
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT :
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        int found = 0;
        for (uint32_t index = 0; index < properties.memoryTypeCount; index++) {
            if ((requirements.memoryTypeBits & (1u << index)) &&
                (properties.memoryTypes[index].propertyFlags & wanted) ==
                    wanted) {
                memory_info.memoryTypeIndex = index;
                found = 1;
                break;
            }
        }
        if (!found ||
            vkAllocateMemory(gpu->device, &memory_info, NULL,
                             &tensor->memory) != VK_SUCCESS ||
            vkBindBufferMemory(gpu->device, tensor->buffer, tensor->memory,
                               0) != VK_SUCCESS ||
            (!device_local &&
             vkMapMemory(gpu->device, tensor->memory, 0, VK_WHOLE_SIZE, 0,
                         &tensor->mapped) != VK_SUCCESS)) {
            h3_vk_set_error(gpu, "tensor memory setup failed");
            if (tensor->buffer)
                vkDestroyBuffer(gpu->device, tensor->buffer, NULL);
            free(tensor);
            return NULL;
        }
    }
    tensor->elements = elements;
    tensor->dtype = dtype;
    tensor->byte_size = byte_size;
    tensor->gpu = gpu;
    tensor->device_local = device_local;
    gpu->stats.tensor_allocations++;
    gpu->stats.allocated_bytes += byte_size;
    gpu->stats.live_bytes += byte_size;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    return tensor;
}

/* One-shot staging transfer through a dedicated command buffer. */
static int h3_vk_staging_copy(h3_gpu *gpu, VkBuffer destination,
                              VkBuffer source, VkDeviceSize bytes) {
    VkCommandBufferAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = gpu->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VkCommandBuffer command;
    if (vkAllocateCommandBuffers(gpu->device, &alloc, &command) !=
        VK_SUCCESS) {
        h3_vk_set_error(gpu, "staging: vkAllocateCommandBuffers failed");
        return -1;
    }
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = bytes };
    vkBeginCommandBuffer(command, &begin);
    vkCmdCopyBuffer(command, source, destination, 1, &region);
    vkEndCommandBuffer(command);
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command
    };
    int ok = vkQueueSubmit(gpu->queue, 1, &submit, VK_NULL_HANDLE) ==
                 VK_SUCCESS &&
             vkDeviceWaitIdle(gpu->device) == VK_SUCCESS;
    vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &command);
    if (!ok) {
        h3_vk_set_error(gpu, "staging copy failed");
        return -1;
    }
    return 0;
}

/* Host-visible staging buffer covering `bytes`. */
static int h3_vk_staging_alloc(h3_gpu *gpu, VkDeviceSize bytes,
                               VkBuffer *buffer, VkDeviceMemory *memory,
                               void **mapped) {
    VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT
    };
    if (vkCreateBuffer(gpu->device, &info, NULL, buffer) != VK_SUCCESS) {
        h3_vk_set_error(gpu, "staging: vkCreateBuffer failed");
        return -1;
    }
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(gpu->device, *buffer, &requirements);
    VkMemoryAllocateInfo memory_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size
    };
    VkPhysicalDeviceMemoryProperties properties;
    vkGetPhysicalDeviceMemoryProperties(gpu->physical, &properties);
    int found = 0;
    for (uint32_t index = 0; index < properties.memoryTypeCount; index++) {
        if ((requirements.memoryTypeBits & (1u << index)) &&
            (properties.memoryTypes[index].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memory_info.memoryTypeIndex = index;
            found = 1;
            break;
        }
    }
    if (!found ||
        vkAllocateMemory(gpu->device, &memory_info, NULL, memory) !=
            VK_SUCCESS ||
        vkBindBufferMemory(gpu->device, *buffer, *memory, 0) != VK_SUCCESS ||
        vkMapMemory(gpu->device, *memory, 0, VK_WHOLE_SIZE, 0, mapped) !=
            VK_SUCCESS) {
        h3_vk_set_error(gpu, "staging memory setup failed");
        if (*buffer) vkDestroyBuffer(gpu->device, *buffer, NULL);
        return -1;
    }
    return 0;
}

static void h3_vk_staging_free(h3_gpu *gpu, VkBuffer buffer,
                               VkDeviceMemory memory, void *mapped) {
    if (mapped) vkUnmapMemory(gpu->device, memory);
    if (memory) vkFreeMemory(gpu->device, memory, NULL);
    if (buffer) vkDestroyBuffer(gpu->device, buffer, NULL);
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_vk_tensor_new(gpu, elements, H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_vk_tensor_new(gpu, elements, H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_vk_tensor_new(gpu, elements, H3_GPU_I8);
}

static h3_gpu_tensor *h3_vk_tensor_from(h3_gpu *gpu, const void *values,
                                        size_t elements,
                                        h3_gpu_dtype dtype) {
    h3_gpu_tensor *tensor = h3_vk_tensor_new(gpu, elements, dtype);
    if (!tensor || !elements) return tensor;
    memcpy(tensor->mapped, values, tensor->byte_size);
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_vk_tensor_from(gpu, values, elements, H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return h3_vk_tensor_from(gpu, values, elements, H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return h3_vk_tensor_from(gpu, values, elements, H3_GPU_U32);
}

static void h3_vk_tensor_destroy(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (gpu && gpu->device && tensor->memory) {
        if (tensor->mapped)
            vkUnmapMemory(gpu->device, tensor->memory);
        vkFreeMemory(gpu->device, tensor->memory, NULL);
    }
    if (gpu && gpu->device && tensor->buffer)
        vkDestroyBuffer(gpu->device, tensor->buffer, NULL);
    if (gpu && gpu->stats.live_bytes >= tensor->byte_size)
        gpu->stats.live_bytes -= tensor->byte_size;
    free(tensor);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    h3_gpu *gpu = tensor->gpu;
    /* The Metal backend keeps buffers alive through ARC; here, a tensor
     * freed while commands referencing it are recorded (or already
     * submitted without a wait) must survive until the final submit's
     * device wait. Defer those frees. */
    if (gpu && gpu->device && (gpu->command_active || gpu->pending_count)) {
        if (gpu->pending_count < 64) {
            gpu->pending[gpu->pending_count++] = tensor;
            return;
        }
    }
    h3_vk_tensor_destroy(gpu, tensor);
}

static void h3_vk_drain_pending(h3_gpu *gpu) {
    for (uint32_t index = 0; index < gpu->pending_count; index++)
        h3_vk_tensor_destroy(gpu, gpu->pending[index]);
    gpu->pending_count = 0;
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->elements : 0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->dtype : H3_GPU_F32;
}

static int h3_vk_tensor_read(h3_gpu *gpu, const h3_gpu_tensor *tensor,
                             size_t source_offset, void *values,
                             size_t elements) {
    if (!tensor || !values) return -1;
    if (source_offset > tensor->elements ||
        elements > tensor->elements - source_offset) return -1;
    if (!elements) return 0;
    size_t bytes = elements * h3_vk_dtype_size(tensor->dtype);
    size_t byte_offset = source_offset * h3_vk_dtype_size(tensor->dtype);
    if (!tensor->device_local) {
        memcpy(values, (const uint8_t *)tensor->mapped + byte_offset, bytes);
        return 0;
    }
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void *mapped = NULL;
    if (h3_vk_staging_alloc(gpu, bytes, &staging, &staging_memory,
                            &mapped) != 0)
        return -1;
    /* Download through a temporary command buffer. */
    VkCommandBufferAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = gpu->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VkCommandBuffer command;
    int ok = vkAllocateCommandBuffers(gpu->device, &alloc, &command) ==
                 VK_SUCCESS;
    if (ok) {
        VkCommandBufferBeginInfo begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        VkBufferCopy region = {
            .srcOffset = byte_offset, .dstOffset = 0, .size = bytes
        };
        vkBeginCommandBuffer(command, &begin);
        vkCmdCopyBuffer(command, tensor->buffer, staging, 1, &region);
        vkEndCommandBuffer(command);
        VkSubmitInfo submit = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command
        };
        ok = vkQueueSubmit(gpu->queue, 1, &submit, VK_NULL_HANDLE) ==
                 VK_SUCCESS &&
             vkDeviceWaitIdle(gpu->device) == VK_SUCCESS;
        vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &command);
    }
    if (ok)
        memcpy(values, mapped, bytes);
    else
        h3_vk_set_error(gpu, "device-local readback failed");
    h3_vk_staging_free(gpu, staging, staging_memory, mapped);
    return ok ? 0 : -1;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_vk_tensor_read(tensor->gpu, tensor, 0, values, elements);
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    return h3_vk_tensor_read(tensor->gpu, tensor, source_offset, values,
                             elements);
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    return h3_vk_tensor_read(tensor->gpu, tensor, 0, values, elements);
}

static int h3_vk_tensor_write(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              size_t destination_offset, const void *values,
                              size_t elements) {
    if (!tensor || !values) return -1;
    if (destination_offset > tensor->elements ||
        elements > tensor->elements - destination_offset) return -1;
    if (!elements) return 0;
    size_t bytes = elements * h3_vk_dtype_size(tensor->dtype);
    size_t byte_offset = destination_offset * h3_vk_dtype_size(tensor->dtype);
    if (!tensor->device_local) {
        memcpy((uint8_t *)tensor->mapped + byte_offset, values, bytes);
        return 0;
    }
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void *mapped = NULL;
    if (h3_vk_staging_alloc(gpu, bytes, &staging, &staging_memory,
                            &mapped) != 0)
        return -1;
    memcpy(mapped, values, bytes);
    int ok = h3_vk_staging_copy(gpu, tensor->buffer, staging, bytes) == 0;
    if (!ok) h3_vk_set_error(gpu, "device-local upload failed");
    h3_vk_staging_free(gpu, staging, staging_memory, mapped);
    return ok ? 0 : -1;
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_vk_tensor_write(tensor->gpu, tensor, 0, values, elements);
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    return h3_vk_tensor_write(tensor->gpu, tensor, destination_offset,
                              values, elements);
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_vk_tensor_write(tensor->gpu, tensor, 0, values, elements);
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    return h3_vk_tensor_write(tensor->gpu, tensor, destination_offset,
                              values, elements);
}

/* ---------------------------------------------------------------- load */

static h3_gpu_tensor *h3_vk_tensor_load(h3_gpu *gpu, const char *path,
                                        uint64_t file_offset,
                                        size_t elements,
                                        h3_gpu_dtype dtype) {
    /* Checkpoint weights are resident in device-local memory; the payload
     * is pread into a host staging buffer and copied once. */
    h3_gpu_tensor *tensor = h3_vk_tensor_new_mem(gpu, elements, dtype, 1);
    if (!tensor || !elements) return tensor;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        h3_vk_set_error(gpu, "cannot open %s", path);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    size_t byte_size = tensor->byte_size;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void *mapped = NULL;
    if (h3_vk_staging_alloc(gpu, byte_size, &staging, &staging_memory,
                            &mapped) != 0) {
        close(fd);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    uint8_t *cursor = mapped;
    while (byte_size > 0) {
        ssize_t got = pread(fd, cursor, byte_size, (off_t)file_offset);
        if (got <= 0) {
            h3_vk_set_error(gpu, "short read from %s", path);
            close(fd);
            h3_vk_staging_free(gpu, staging, staging_memory, mapped);
            h3_gpu_tensor_free(tensor);
            return NULL;
        }
        cursor += (size_t)got;
        file_offset += (uint64_t)got;
        byte_size -= (size_t)got;
    }
    close(fd);
    if (h3_vk_staging_copy(gpu, tensor->buffer, staging,
                           tensor->byte_size) != 0) {
        h3_vk_staging_free(gpu, staging, staging_memory, mapped);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    h3_vk_staging_free(gpu, staging, staging_memory, mapped);
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset,
                                       size_t elements) {
    return h3_vk_tensor_load(gpu, path, file_offset, elements, H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset,
                                      size_t elements) {
    return h3_vk_tensor_load(gpu, path, file_offset, elements, H3_GPU_F32);
}

static int h3_vk_tensor_read_file(h3_gpu *gpu, h3_gpu_tensor *tensor,
                                  const char *path, uint64_t file_offset,
                                  size_t elements, char *error,
                                  size_t error_size) {
    if (!tensor || !elements) return 0;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot open %s", path);
        return -1;
    }
    size_t byte_size = elements * h3_vk_dtype_size(tensor->dtype);
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void *mapped = NULL;
    int failed = 0;
    if (!tensor->device_local) {
        mapped = tensor->mapped;
    } else if (h3_vk_staging_alloc(gpu, byte_size, &staging,
                                   &staging_memory, &mapped) != 0) {
        failed = 1;
    }
    uint8_t *cursor = mapped;
    while (!failed && byte_size > 0) {
        ssize_t got = pread(fd, cursor, byte_size, (off_t)file_offset);
        if (got <= 0) {
            if (error && error_size)
                snprintf(error, error_size, "short read from %s", path);
            failed = 1;
            break;
        }
        cursor += (size_t)got;
        file_offset += (uint64_t)got;
        byte_size -= (size_t)got;
    }
    close(fd);
    if (!failed && tensor->device_local)
        failed = h3_vk_staging_copy(gpu, tensor->buffer, staging,
                                    elements *
                                        h3_vk_dtype_size(tensor->dtype)) != 0;
    if (tensor->device_local)
        h3_vk_staging_free(gpu, staging, staging_memory,
                           tensor->device_local ? mapped : NULL);
    if (failed && error && error_size && !error[0])
        snprintf(error, error_size, "device-local file load failed");
    return failed ? -1 : 0;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_vk_tensor_read_file(tensor->gpu, tensor, path, file_offset,
                                  elements, error, error_size);
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    /* Host-visible tensors need no cache hint; the coherent mapping is the
     * destination copy. Streaming optimizations arrive with device-local
     * residency. */
    return h3_gpu_tensor_read_file_bf16(tensor, path, file_offset, elements,
                                        error, error_size);
}

/* ------------------------------------------------------------ commands */

int h3_gpu_begin(h3_gpu *gpu) {
    if (!gpu) return -1;
    gpu->args_dispatches = 0;
    gpu->encode_start = h3_vk_now();
    return h3_vk_begin_command(gpu) ? 0 : -1;
}

int h3_gpu_continue(h3_gpu *gpu) {
    if (!gpu) return -1;
    if (!h3_vk_submit_command(gpu, 0)) return -1;
    gpu->stats.command_encode_seconds += h3_vk_now() - gpu->encode_start;
    gpu->encode_start = h3_vk_now();
    gpu->args_dispatches = 0;
    return h3_vk_begin_command(gpu) ? 0 : -1;
}

int h3_gpu_submit(h3_gpu *gpu) {
    if (!gpu) return -1;
    if (!h3_vk_submit_command(gpu, 1)) return -1;
    gpu->stats.command_encode_seconds += h3_vk_now() - gpu->encode_start;
    return 0;
}

int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats) return -1;
    *stats = gpu->stats;
    return 0;
}

/* -------------------------------------------------------------- kernels */

static int h3_vk_dispatch(h3_gpu *gpu, h3_vk_kernel kernel,
                          VkDescriptorSet set, uint32_t groups_x,
                          uint32_t groups_y, uint32_t groups_z) {
    if (!h3_vk_require_command(gpu)) return -1;
    if (groups_x == 0) groups_x = 1;
    if (groups_y == 0) groups_y = 1;
    if (groups_z == 0) groups_z = 1;
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                         VK_ACCESS_MEMORY_WRITE_BIT |
                         VK_ACCESS_HOST_READ_BIT
    };
    vkCmdPipelineBarrier(gpu->command,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
    vkCmdBindPipeline(gpu->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      gpu->pipelines[kernel]);
    vkCmdBindDescriptorSets(gpu->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            gpu->pipeline_layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(gpu->command, groups_x, groups_y, groups_z);
    gpu->stats.direct_dispatches++;
    return 0;
}

static VkDescriptorSet h3_vk_prepare_off(h3_gpu *gpu, h3_vk_kernel kernel,
                                         const h3_gpu_tensor *const *tensors,
                                         uint32_t tensor_count,
                                         VkDeviceSize first_offset,
                                         VkDeviceSize last_offset) {
    uint32_t bindings = h3_vk_kernel_bindings[kernel];
    if (tensor_count != bindings) {
        h3_vk_set_error(gpu, "kernel %s expects %u bindings, got %u",
                        h3_vk_kernel_names[kernel], bindings, tensor_count);
        return VK_NULL_HANDLE;
    }
    if (gpu->args_dispatches >= H3_VK_ARGS_SLOTS) {
        h3_vk_set_error(gpu, "too many dispatches between submits "
                             "(call h3_gpu_continue)");
        return VK_NULL_HANDLE;
    }
    VkDescriptorSet set = h3_vk_alloc_set(gpu, bindings + 1);
    if (set == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    for (uint32_t index = 0; index < tensor_count; index++) {
        VkDeviceSize offset = 0;
        if (index == 0) offset = first_offset;
        if (index == tensor_count - 1) offset = last_offset;
        h3_vk_set_buffer_offset(gpu, set, index, tensors[index]->buffer,
                                offset);
    }
    h3_vk_set_buffer_offset(gpu, set, 10, gpu->args_buffer,
                            (VkDeviceSize)gpu->args_slot *
                                sizeof(h3_gpu_vulkan_args));
    gpu->args_slot = (gpu->args_slot + 1) % H3_VK_ARGS_SLOTS;
    gpu->args = gpu->args_base + gpu->args_slot;
    gpu->args_dispatches++;
    return set;
}

static VkDescriptorSet h3_vk_prepare(h3_gpu *gpu, h3_vk_kernel kernel,
                                     const h3_gpu_tensor *const *tensors,
                                     uint32_t tensor_count) {
    return h3_vk_prepare_off(gpu, kernel, tensors, tensor_count, 0, 0);
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            uint32_t elements) {
    if (!h3_vk_check_tensors(gpu, 2, output, input)) return -1;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[2] = { input, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_CAST_F32_TO_BF16,
                                        tensors, 2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_CAST_F32_TO_BF16, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input,
                            uint32_t elements) {
    if (!h3_vk_check_tensors(gpu, 2, output, input)) return -1;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[2] = { input, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_CAST_BF16_TO_F32,
                                        tensors, 2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_CAST_BF16_TO_F32, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_vk_check_tensors(gpu, 2, output, input)) return -1;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[2] = { input, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SILU_F32, tensors,
                                        2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SILU_F32, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_vk_check_tensors(gpu, 2, output, input)) return -1;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[2] = { input, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SILU_BF16, tensors,
                                        2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SILU_BF16, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    if (!h3_vk_check_tensors(gpu, 3, output, gate, up)) return -1;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[3] = { gate, up, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SILU_MUL_BF16,
                                        tensors, 3);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SILU_MUL_BF16, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    if (!h3_vk_check_tensors(gpu, 2, output, input)) return -1;
    gpu->args->elements = elements;
    gpu->args->minimum = minimum;
    gpu->args->maximum = maximum;
    const h3_gpu_tensor *tensors[2] = { input, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_CLIP_F32, tensors,
                                        2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_CLIP_F32, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    if (!h3_vk_check_tensors(gpu, 2, output, input)) return -1;
    gpu->args->elements = elements;
    gpu->args->approximate = approximate ? 1u : 0u;
    const h3_gpu_tensor *tensors[2] = { input, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_GELU_BF16, tensors,
                                        2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_GELU_BF16, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!h3_vk_check_tensors(gpu, 3, output, left, right)) return -1;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[3] = { left, right, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_ADD_BF16, tensors,
                                        3);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_ADD_BF16, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!h3_vk_check_tensors(gpu, 3, output, left, right)) return -1;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[3] = { left, right, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SUB_BF16, tensors,
                                        3);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SUB_BF16, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    if (!h3_vk_check_tensors(gpu, 3, output, left, right)) return -1;
    gpu->args->elements = elements;
    gpu->args->left_scale = left_scale;
    gpu->args->right_scale = right_scale;
    const h3_gpu_tensor *tensors[3] = { left, right, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_ADD_SCALED_F32,
                                        tensors, 3);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_ADD_SCALED_F32, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    if (!h3_vk_check_tensors(gpu, 3, output, gate, linear)) return -1;
    gpu->args->count = elements;
    const h3_gpu_tensor *tensors[3] = { gate, linear, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_GEGLU_F32, tensors,
                                        3);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_GEGLU_F32, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    if (!h3_vk_check_tensors(gpu, 3, sample, last, previous)) return -1;
    gpu->args->sample_offset = (uint32_t)sample_offset;
    gpu->args->elements = elements;
    gpu->args->delta = delta;
    gpu->args->ratio = ratio;
    const h3_gpu_tensor *tensors[3] = { sample, last, previous };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_EULER_BF16,
                                        tensors, 3);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_EULER_BF16, set,
                          (elements + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    if (!h3_vk_check_tensors(gpu, 3, output, weight, token_ids)) return -1;
    gpu->args->tokens = tokens;
    gpu->args->width = width;
    gpu->args->vocab_size = vocab_size;
    const h3_gpu_tensor *tensors[3] = { weight, token_ids, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_EMBEDDING_BF16,
                                        tensors, 3);
    if (set == VK_NULL_HANDLE) return -1;
    uint64_t elements = (uint64_t)tokens * (uint64_t)width;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_EMBEDDING_BF16, set,
                          (uint32_t)((elements + H3_VK_THREADS - 1) /
                                     H3_VK_THREADS),
                          1, 1);
}

static int h3_vk_norm(h3_gpu *gpu, h3_vk_kernel kernel,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias,
                      h3_gpu_tensor *output, uint32_t rows,
                      uint32_t width, float epsilon) {
    if (!h3_vk_check_tensors(gpu, 3, output, input, weight)) return -1;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->epsilon = epsilon;
    uint32_t count = 3;
    const h3_gpu_tensor *tensors[4] = { input, weight, output, bias };
    if (bias) count = 4;
    VkDescriptorSet set = h3_vk_prepare(gpu, kernel, tensors, count);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, kernel, set, rows, 1, 1);
}

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    return h3_vk_norm(gpu, H3_VK_KERNEL_RMS_NORM_F32, input, weight, NULL,
                      output, rows, width, epsilon);
}

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    return h3_vk_norm(gpu, H3_VK_KERNEL_LAYER_NORM_F32, input, weight, bias,
                      output, rows, width, epsilon);
}

int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    return h3_vk_norm(gpu, H3_VK_KERNEL_RMS_NORM_BF16, input, weight, NULL,
                      output, rows, width, epsilon);
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    return h3_vk_norm(gpu, H3_VK_KERNEL_LAYER_NORM_BF16, input, weight, bias,
                      output, rows, width, epsilon);
}


/* ------------------------------------------------------------- linear/mlp */

static int h3_vk_linear(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight,
                        const h3_gpu_tensor *bias, uint32_t rows,
                        uint32_t input_dim, uint32_t output_dim) {
    if (!h3_vk_check_tensors(gpu, 3, output, input, weight)) return -1;
    if ((size_t)rows * input_dim > h3_gpu_tensor_elements(input) ||
        (size_t)output_dim * input_dim > h3_gpu_tensor_elements(weight) ||
        (size_t)rows * output_dim > h3_gpu_tensor_elements(output) ||
        (bias && output_dim > h3_gpu_tensor_elements(bias))) {
        h3_vk_set_error(gpu, "linear tensor size mismatch");
        return -1;
    }
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = { input, weight, bias_buffer, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_LINEAR_BF16,
                                        tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_LINEAR_BF16, set,
                          (output_dim + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    return h3_vk_linear(gpu, output, input, weight, bias, rows, input_dim,
                        output_dim);
}

/* Fused fc1 -> SwiGLU -> fc2. The FC1 weight is [2*hidden, input_dim] with
 * gate in the first half and up in the second, matching the checkpoint
 * layout. Each boundary rounds to BF16 exactly once (fc1, silu*gate, fc2),
 * the same numerical contract as the Metal silu_mul kernel. */
int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    if (!h3_vk_check_tensors(gpu, 4, output, input, fc1_weight, fc2_weight))
        return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(gpu,
                                                (size_t)rows * hidden_dim * 2);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(gpu,
                                                      (size_t)rows * hidden_dim);
    if (!fc1 || !activated) {
        h3_gpu_tensor_free(fc1);
        h3_gpu_tensor_free(activated);
        h3_vk_set_error(gpu, "MLP scratch allocation failed");
        return -1;
    }
    int ok = 0;
    if (h3_vk_linear(gpu, fc1, input, fc1_weight, NULL, rows, input_dim,
                     hidden_dim * 2) == 0) {
        gpu->args->elements = rows * hidden_dim;
        gpu->args->width = hidden_dim;
        const h3_gpu_tensor *tensors[2] = { fc1, activated };
        VkDescriptorSet set = h3_vk_prepare(gpu,
                                            H3_VK_KERNEL_SWIGLU_HALVES_BF16,
                                            tensors, 2);
        if (set != VK_NULL_HANDLE &&
            h3_vk_dispatch(gpu, H3_VK_KERNEL_SWIGLU_HALVES_BF16, set,
                           (uint32_t)(((uint64_t)rows * hidden_dim +
                                       H3_VK_THREADS - 1) / H3_VK_THREADS),
                           1, 1) == 0 &&
            h3_vk_linear(gpu, output, activated, fc2_weight, NULL, rows,
                         hidden_dim, output_dim) == 0) {
            ok = 1;
        }
    }
    h3_gpu_tensor_free(fc1);
    h3_gpu_tensor_free(activated);
    return ok ? 0 : -1;
}

/* ------------------------------------------------------------ adaln/gate */

int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    if (!h3_vk_check_tensors(gpu, 5, output, input, norm_weight, modulation,
                             row_map))
        return -1;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->slots = slots;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[5] = { input, norm_weight, modulation,
                                        row_map, output };
    VkDescriptorSet set = h3_vk_prepare_off(gpu, H3_VK_KERNEL_ADALN_BF16,
                                            tensors, 5,
                                            (VkDeviceSize)input_offset * 2,
                                            0);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_ADALN_BF16, set, rows, 1, 1);
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    return h3_gpu_adaln_bf16_offset(gpu, output, input, 0, norm_weight,
                                    modulation, row_map, rows, width, slots,
                                    shift_slot, scale_slot, epsilon);
}

int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    if (!h3_vk_check_tensors(gpu, 5, output, residual, branch, modulation,
                             row_map))
        return -1;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->slots = slots;
    gpu->args->gate_slot = gate_slot;
    const h3_gpu_tensor *tensors[5] = { residual, branch, modulation,
                                        row_map, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_GATE_BF16, tensors,
                                        5);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_GATE_BF16, set,
                          (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_gate_adaln_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon) {
    if (!h3_vk_check_tensors(gpu, 8, gated_residual, output, residual,
                             branch, norm_weight, gate_modulation,
                             norm_modulation, row_map))
        return -1;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->slots = slots;
    gpu->args->gate_slot = gate_slot;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[8] = { residual, branch, gate_modulation,
                                        row_map, norm_weight, norm_modulation,
                                        gated_residual, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_GATE_ADALN_BF16,
                                        tensors, 8);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_GATE_ADALN_BF16, set, rows, 1, 1);
}

int h3_gpu_adaln_linear_bf16(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon) {
    if (!h3_vk_check_tensors(gpu, 8, output, inverse, input, norm_weight,
                             modulation, row_map, weight, bias))
        return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    /* Inverse RMS scalars first (rows F32). */
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *rms_tensors[2] = { input, inverse };
    VkDescriptorSet rms_set = h3_vk_prepare_off(
        gpu, H3_VK_KERNEL_RMS_INVERSE_BF16, rms_tensors, 2,
        (VkDeviceSize)input_offset * 2, 0);
    if (rms_set == VK_NULL_HANDLE ||
        h3_vk_dispatch(gpu, H3_VK_KERNEL_RMS_INVERSE_BF16, rms_set, rows, 1,
                       1) != 0)
        return -1;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->output_dim = output_dim;
    gpu->args->slots = slots;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[8] = { input, inverse, norm_weight,
                                        modulation, row_map, weight,
                                        bias_buffer, output };
    VkDescriptorSet set = h3_vk_prepare_off(gpu,
                                            H3_VK_KERNEL_ADALN_LINEAR_BF16,
                                            tensors, 8,
                                            (VkDeviceSize)input_offset * 2,
                                            0);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_ADALN_LINEAR_BF16, set,
                          (output_dim + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) {
    if (!h3_vk_check_tensors(gpu, 2, tensor, weight)) return -1;
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[2] = { tensor, weight };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_HEAD_RMS_NORM_BF16,
                                        tensors, 2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_HEAD_RMS_NORM_BF16, set,
                          (sequence + 15) / 16, (heads + 15) / 16, 1);
}


/* ------------------------------------------------------------ qkv/sdpa */


static int h3_vk_patch_linear(h3_gpu *gpu, h3_gpu_tensor *output,
                              size_t output_offset,
                              const h3_gpu_tensor *input,
                              size_t input_offset,
                              const h3_gpu_tensor *weight,
                              const h3_gpu_tensor *bias, uint32_t rows,
                              uint32_t input_dim, uint32_t output_dim) {
    if (!h3_vk_check_tensors(gpu, 3, output, input, weight)) return -1;
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = { input, weight, bias_buffer, output };
    VkDescriptorSet set = h3_vk_prepare_off(gpu, H3_VK_KERNEL_PATCH_LINEAR_BF16,
                                            tensors, 4,
                                            (VkDeviceSize)input_offset * 4,
                                            (VkDeviceSize)output_offset * 2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_PATCH_LINEAR_BF16, set,
                          (output_dim + 15) / 16, (rows + 15) / 16, 1);
}

static int h3_vk_qkv_rope(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key, h3_gpu_tensor *value,
                          const h3_gpu_tensor *qkv,
                          const h3_gpu_tensor *q_norm,
                          const h3_gpu_tensor *k_norm,
                          const h3_gpu_tensor *rope_cos,
                          const h3_gpu_tensor *rope_sin,
                          uint32_t sequence, uint32_t heads,
                          uint32_t head_dim, uint32_t rope_half,
                          float epsilon, int grouped) {
    if (!h3_vk_check_tensors(gpu, 8, query, key, value, qkv, q_norm, k_norm,
                             rope_cos, rope_sin))
        return -1;
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->output_dim = rope_half;
    gpu->args->epsilon = epsilon;
    gpu->args->grouped = grouped ? 1u : 0u;
    const h3_gpu_tensor *tensors[8] = { qkv, q_norm, k_norm, rope_cos,
                                        rope_sin, query, key, value };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_GROUPED_QKV_ROPE_BF16,
                                        tensors, 8);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_GROUPED_QKV_ROPE_BF16, set,
                          (head_dim + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          heads, sequence);
}

static int h3_vk_sdpa(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                      const h3_gpu_tensor *value, uint32_t sequence,
                      uint32_t heads, uint32_t head_dim, float scale,
                      int head_major_output) {
    if (!h3_vk_check_tensors(gpu, 4, output, query, key, value)) return -1;
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->left_scale = scale;
    gpu->args->grouped = head_major_output ? 1u : 0u;
    /* Long sequences use the 128-thread flash kernel; short ones keep the
     * one-thread-per-output naive kernel, whose bit-exact reference order
     * is what the parity tests compare against. */
    int flash = sequence >= 128 && head_dim <= 128;
    h3_vk_kernel kernel = flash ? H3_VK_KERNEL_SDPA_FLASH_BF16
                                : H3_VK_KERNEL_SDPA_BF16;
    const h3_gpu_tensor *tensors[4] = { query, key, value, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, kernel, tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, kernel, set, heads, sequence, 1);
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) {
    return h3_vk_qkv_rope(gpu, query, key, value, qkv, q_norm, k_norm,
                          rope_cos, rope_sin, sequence, heads, head_dim,
                          rope_half, epsilon, 1);
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) {
    return h3_vk_qkv_rope(gpu, query, key, value, qkv, q_norm, k_norm,
                          rope_cos, rope_sin, sequence, heads, head_dim,
                          rope_half, epsilon, 0);
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_vk_sdpa(gpu, output, query, key, value, sequence, heads,
                      head_dim, scale, 0);
}

int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_vk_sdpa(gpu, output, query, key, value, sequence, heads,
                      head_dim, scale, 1);
}



/* ------------------------------------------------------------ patch */

int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_vk_patch_linear(gpu, output, output_offset, input, input_offset,
                              weight, bias, rows, input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(gpu, output, 0, input, 0, weight,
                                           bias, rows, input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    if (!h3_vk_check_tensors(gpu, 4, output, input, weight, row_map))
        return -1;
    if (bias && !h3_vk_check_tensors(gpu, 1, bias)) return -1;
    (void)output_rows;
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[5] = { input, weight, bias_buffer, output,
                                        row_map };
    VkDescriptorSet set = h3_vk_prepare(gpu,
                                        H3_VK_KERNEL_PATCH_LINEAR_BF16_MAP,
                                        tensors, 5);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_PATCH_LINEAR_BF16_MAP, set,
                          (output_dim + 15) / 16, (rows + 15) / 16, 1);
}


/* ---------------------------------------------------------- token pool */

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    if (!h3_vk_check_tensors(gpu, 6, output, input, original, baseline,
                             baseline_indices, pairs))
        return -1;
    (void)input_rows;
    (void)baseline_rows;
    gpu->args->sample_offset = (uint32_t)input_offset;
    gpu->args->tokens = (uint32_t)original_offset;
    gpu->args->vocab_size = (uint32_t)baseline_offset;
    gpu->args->rows = rows;
    gpu->args->width = width;
    const h3_gpu_tensor *tensors[6] = { input, pairs, output, baseline,
                                        baseline_indices, original };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_TOKEN_POOL_BF16,
                                        tensors, 6);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_TOKEN_POOL_BF16, set,
                          (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    if (!h3_vk_check_tensors(gpu, 10, residual, output, input, original,
                             baseline, baseline_indices, pairs, norm_weight,
                             modulation, row_map))
        return -1;
    (void)input_rows;
    (void)baseline_rows;
    gpu->args->sample_offset = (uint32_t)input_offset;
    gpu->args->tokens = (uint32_t)original_offset;
    gpu->args->vocab_size = (uint32_t)baseline_offset;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->slots = slots;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[10] = {
        input, pairs, residual, baseline, baseline_indices, original,
        norm_weight, modulation, row_map, output
    };
    VkDescriptorSet set = h3_vk_prepare(
        gpu, H3_VK_KERNEL_TOKEN_POOL_ADALN_BF16, tensors, 10);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_TOKEN_POOL_ADALN_BF16, set,
                          rows, 1, 1);
}

int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width, uint32_t exact_prefix_rows,
                           float update_scale) {
    if (!h3_vk_check_tensors(gpu, 6, output, original, reduced, baseline,
                             baseline_indices, parents))
        return -1;
    (void)reduced_rows;
    (void)baseline_rows;
    gpu->args->tokens = (uint32_t)original_offset;
    gpu->args->vocab_size = (uint32_t)baseline_offset;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->elements = exact_prefix_rows;
    gpu->args->left_scale = update_scale;
    const h3_gpu_tensor *tensors[6] = { original, reduced, baseline,
                                        baseline_indices, parents, output };
    VkDescriptorSet set = h3_vk_prepare(
        gpu, H3_VK_KERNEL_TOKEN_EXPAND_DELTA_BF16, tensors, 6);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_TOKEN_EXPAND_DELTA_BF16, set,
                          (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width, uint32_t exact_prefix_rows,
                           float update_scale, uint32_t slots,
                           uint32_t shift_slot, uint32_t scale_slot,
                           float epsilon) {
    if (!h3_vk_check_tensors(gpu, 10, residual, output, original, reduced,
                             baseline, baseline_indices, parents, norm_weight,
                             modulation, row_map))
        return -1;
    (void)reduced_rows;
    (void)baseline_rows;
    gpu->args->tokens = (uint32_t)original_offset;
    gpu->args->vocab_size = (uint32_t)baseline_offset;
    gpu->args->rows = rows;
    gpu->args->width = width;
    gpu->args->elements = exact_prefix_rows;
    gpu->args->left_scale = update_scale;
    gpu->args->slots = slots;
    gpu->args->shift_slot = shift_slot;
    gpu->args->scale_slot = scale_slot;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[10] = {
        original, reduced, baseline, baseline_indices, parents, residual,
        norm_weight, modulation, row_map, output
    };
    VkDescriptorSet set = h3_vk_prepare(
        gpu, H3_VK_KERNEL_TOKEN_EXPAND_ADALN_BF16, tensors, 10);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_TOKEN_EXPAND_ADALN_BF16, set,
                          rows, 1, 1);
}


/* ------------------------------------------------------------ qwen text */

int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_weight,
                             const h3_gpu_tensor *k_weight,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) {
    if (!h3_vk_check_tensors(gpu, 8, query_output, key_output, query_input,
                             key_input, q_weight, k_weight, rope_cos,
                             rope_sin))
        return -1;
    gpu->args->rows = sequence;
    gpu->args->width = query_heads;
    gpu->args->input_dim = kv_heads;
    gpu->args->output_dim = head_dim;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[8] = { query_input, key_input, q_weight,
                                        k_weight, rope_cos, rope_sin,
                                        query_output, key_output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_TEXT_QK_ROPE_BF16,
                                        tensors, 8);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_TEXT_QK_ROPE_BF16, set,
                          (head_dim + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          query_heads, sequence);
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) {
    if (!h3_vk_check_tensors(gpu, 4, query, key, rope_cos_f32,
                             rope_sin_f32))
        return -1;
    gpu->args->rows = sequence;
    gpu->args->width = query_heads;
    gpu->args->input_dim = kv_heads;
    gpu->args->output_dim = head_dim;
    uint32_t maximum_heads = query_heads > kv_heads ? query_heads : kv_heads;
    const h3_gpu_tensor *tensors[4] = { query, key, rope_cos_f32,
                                        rope_sin_f32 };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_ROPE_TEXT_BF16,
                                        tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_ROPE_TEXT_BF16, set,
                          (sequence + 15) / 16, (maximum_heads + 15) / 16, 1);
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale) {
    if (!h3_vk_check_tensors(gpu, 4, output, query, key, value)) return -1;
    if (sequence > 4096) {
        h3_vk_set_error(gpu, "GQA sequence exceeds the 4096-row score "
                             "threadgroup buffer");
        return -1;
    }
    gpu->args->rows = sequence;
    gpu->args->width = query_heads;
    gpu->args->input_dim = kv_heads;
    gpu->args->output_dim = head_dim;
    gpu->args->left_scale = scale;
    const h3_gpu_tensor *tensors[4] = { query, key, value, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_GQA_CAUSAL_BF16,
                                        tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_GQA_CAUSAL_BF16, set, sequence,
                          query_heads, 1);
}

void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    (void)gpu;
    (void)label;
}

void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    (void)gpu;
    (void)phase;
}

/* ------------------------------------------------------------------ */
/* Kernels not yet ported to Vulkan. They fail cleanly with a
 * descriptive error; the callers' existing error paths handle it. */
static int h3_vk_not_ported(h3_gpu *gpu, const char *name) {
    h3_vk_set_error(gpu, "kernel %s is not ported to the Vulkan backend yet",
                    name);
    return -1;
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    if (!h3_vk_check_tensors(gpu, 3, output, input, weight)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    if ((size_t)rows * input_dim > h3_gpu_tensor_elements(input) ||
        (size_t)output_dim * input_dim > h3_gpu_tensor_elements(weight) ||
        (size_t)rows * output_dim > h3_gpu_tensor_elements(output) ||
        (bias && output_dim > h3_gpu_tensor_elements(bias))) {
        h3_vk_set_error(gpu, "linear f32 tensor size mismatch");
        return -1;
    }
    gpu->args->rows = rows;
    gpu->args->input_dim = input_dim;
    gpu->args->output_dim = output_dim;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = { input, weight, bias_buffer, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_LINEAR_F32, tensors,
                                        4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_LINEAR_F32, set,
                          (output_dim + 15) / 16, (rows + 15) / 16, 1);
}

static int h3_vk_copy(h3_gpu *gpu, h3_gpu_tensor *destination,
                      size_t destination_offset,
                      const h3_gpu_tensor *source, size_t source_offset,
                      size_t elements, h3_gpu_dtype dtype) {
    if (!h3_vk_check_tensors(gpu, 2, destination, source)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    if (h3_gpu_tensor_dtype(destination) != dtype ||
        h3_gpu_tensor_dtype(source) != dtype) {
        h3_vk_set_error(gpu, "copy dtype mismatch");
        return -1;
    }
    size_t dtype_size = h3_vk_dtype_size(dtype);
    if (source_offset > h3_gpu_tensor_elements(source) ||
        elements > h3_gpu_tensor_elements(source) - source_offset ||
        destination_offset > h3_gpu_tensor_elements(destination) ||
        elements > h3_gpu_tensor_elements(destination) - destination_offset) {
        h3_vk_set_error(gpu, "invalid blit range");
        return -1;
    }
    if (elements) {
        VkBufferCopy region = {
            .srcOffset = (VkDeviceSize)source_offset * dtype_size,
            .dstOffset = (VkDeviceSize)destination_offset * dtype_size,
            .size = (VkDeviceSize)elements * dtype_size
        };
        vkCmdCopyBuffer(gpu->command, source->buffer, destination->buffer, 1,
                        &region);
    }
    gpu->stats.blit_copies++;
    return 0;
}

int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) {
    return h3_vk_copy(gpu, destination, destination_offset, source,
                      source_offset, elements, H3_GPU_BF16);
}

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements) {
    return h3_vk_copy(gpu, destination, destination_offset, source,
                      source_offset, elements, H3_GPU_F32);
}

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
(void)gpu; (void)output; (void)input; (void)norm_weight; (void)modulation; (void)row_map; (void)rows; (void)width; (void)slots; (void)shift_slot; (void)scale_slot; (void)epsilon;
    return h3_vk_not_ported(gpu, "h3_gpu_adaln_f32");
}

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
(void)gpu; (void)output; (void)residual; (void)branch; (void)modulation; (void)row_map; (void)rows; (void)width; (void)slots; (void)gate_slot;
    return h3_vk_not_ported(gpu, "h3_gpu_gate_f32");
}

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
(void)gpu; (void)query; (void)key; (void)value; (void)qkv; (void)q_norm; (void)k_norm; (void)rope_cos; (void)rope_sin; (void)sequence; (void)heads; (void)head_dim; (void)rope_half; (void)epsilon;
    return h3_vk_not_ported(gpu, "h3_gpu_qkv_rope_f32");
}

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    if (!h3_vk_check_tensors(gpu, 4, output, query, key, value)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->left_scale = scale;
    const h3_gpu_tensor *tensors[4] = { query, key, value, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SDPA_F32, tensors,
                                        4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SDPA_F32, set, heads, sequence,
                          1);
}

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    if (!h3_vk_check_tensors(gpu, 2, output, fused)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    if ((size_t)rows * width * 2 > h3_gpu_tensor_elements(fused) ||
        (size_t)rows * width > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "swiglu f32 tensor size mismatch");
        return -1;
    }
    gpu->args->rows = rows;
    gpu->args->width = width;
    const h3_gpu_tensor *tensors[2] = { fused, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SWIGLU_F32, tensors,
                                        2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SWIGLU_F32, set,
                          (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    if (!h3_vk_check_tensors(gpu, 4, output, residual, branch, scale))
        return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    size_t count = (size_t)rows * width;
    if (count > h3_gpu_tensor_elements(residual) ||
        count > h3_gpu_tensor_elements(branch) ||
        width > h3_gpu_tensor_elements(scale) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "scale-add tensor size mismatch");
        return -1;
    }
    gpu->args->rows = rows;
    gpu->args->width = width;
    const h3_gpu_tensor *tensors[4] = { residual, branch, scale, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SCALE_ADD_F32,
                                        tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SCALE_ADD_F32, set,
                          (width + 15) / 16, (rows + 15) / 16, 1);
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    if (!h3_vk_check_tensors(gpu, 6, query, key, value, qkv, rope_cos,
                             rope_sin))
        return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    size_t inner = (size_t)heads * head_dim;
    if ((size_t)sequence * inner * 3 > h3_gpu_tensor_elements(qkv) ||
        (size_t)sequence * rope_half > h3_gpu_tensor_elements(rope_cos) ||
        (size_t)sequence * rope_half > h3_gpu_tensor_elements(rope_sin) ||
        (size_t)sequence * inner > h3_gpu_tensor_elements(query) ||
        (size_t)sequence * inner > h3_gpu_tensor_elements(key) ||
        (size_t)sequence * inner > h3_gpu_tensor_elements(value)) {
        h3_vk_set_error(gpu, "video qkv rope tensor size mismatch");
        return -1;
    }
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->output_dim = rope_half;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[6] = { qkv, rope_cos, rope_sin, query,
                                        key, value };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_VIDEO_QKV_ROPE_F32,
                                        tensors, 6);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_VIDEO_QKV_ROPE_F32, set,
                          (head_dim + 15) / 16, (heads + 15) / 16,
                          sequence);
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(gpu, output, input, weight, bias, batch,
                                    length, input_channels, output_channels,
                                    kernel, 1, padding, dilation);
}

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation) {
    if (!h3_vk_check_tensors(gpu, 3, output, input, weight)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || !dilation || (uint64_t)length + 2 * padding < effective) {
        h3_vk_set_error(gpu, "conv1d invalid dimensions");
        return -1;
    }
    uint32_t output_length = (uint32_t)(((uint64_t)length + 2 * padding -
                                         effective) / stride + 1);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (input_count > h3_gpu_tensor_elements(input) ||
        weight_count > h3_gpu_tensor_elements(weight) ||
        output_count > h3_gpu_tensor_elements(output) ||
        (bias && output_channels > h3_gpu_tensor_elements(bias))) {
        h3_vk_set_error(gpu, "conv1d tensor size mismatch");
        return -1;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = length;
    gpu->args->conv_in_channels = input_channels;
    gpu->args->conv_out_channels = output_channels;
    gpu->args->conv_kernel_width = kernel;
    gpu->args->conv_stride_width = stride;
    gpu->args->conv_stride_height = padding;
    gpu->args->conv_stride_depth = dilation;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = { input, weight, bias_buffer, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_CONV1D_STRIDE_F32,
                                        tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_CONV1D_STRIDE_F32, set,
                          (output_channels + 15) / 16,
                          (output_length + 15) / 16, batch);
}

int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding) {
    if (!h3_vk_check_tensors(gpu, 3, output, input, weight)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || (uint64_t)(length - 1) * stride + kernel < 2 * padding) {
        h3_vk_set_error(gpu, "conv-transpose1d invalid dimensions");
        return -1;
    }
    uint32_t output_length = (uint32_t)((uint64_t)(length - 1) * stride +
                                        kernel - 2 * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (input_count > h3_gpu_tensor_elements(input) ||
        weight_count > h3_gpu_tensor_elements(weight) ||
        output_count > h3_gpu_tensor_elements(output) ||
        (bias && output_channels > h3_gpu_tensor_elements(bias))) {
        h3_vk_set_error(gpu, "conv-transpose1d tensor size mismatch");
        return -1;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = length;
    gpu->args->conv_in_channels = input_channels;
    gpu->args->conv_out_channels = output_channels;
    gpu->args->conv_kernel_width = kernel;
    gpu->args->conv_stride_width = stride;
    gpu->args->conv_stride_height = padding;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = { input, weight, bias_buffer, output };
    VkDescriptorSet set = h3_vk_prepare(gpu,
                                        H3_VK_KERNEL_CONV_TRANSPOSE1D_F32,
                                        tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_CONV_TRANSPOSE1D_F32, set,
                          (output_channels + 15) / 16,
                          (output_length + 15) / 16, batch);
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    if (!h3_vk_check_tensors(gpu, 3, output, vector, magnitude)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    if ((size_t)outer * inner > h3_gpu_tensor_elements(vector) ||
        outer > h3_gpu_tensor_elements(magnitude) ||
        (size_t)outer * inner > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "weight-norm tensor size mismatch");
        return -1;
    }
    gpu->args->rows = outer;
    gpu->args->width = inner;
    const h3_gpu_tensor *tensors[3] = { vector, magnitude, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_WEIGHT_NORM_F32,
                                        tensors, 3);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_WEIGHT_NORM_F32, set,
                          (outer + H3_VK_THREADS - 1) / H3_VK_THREADS,
                          1, 1);
}

int h3_gpu_alias_free_snake_f32(
                          h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels) {
    if (!h3_vk_check_tensors(gpu, 6, output, input, alpha_log, beta_log,
                             upsample_filter, downsample_filter))
        return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    size_t count = (size_t)batch * length * channels;
    if (count > h3_gpu_tensor_elements(input) ||
        channels > h3_gpu_tensor_elements(alpha_log) ||
        channels > h3_gpu_tensor_elements(beta_log) ||
        12 > h3_gpu_tensor_elements(upsample_filter) ||
        12 > h3_gpu_tensor_elements(downsample_filter) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "alias-free snake tensor size mismatch");
        return -1;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = length;
    gpu->args->conv_in_channels = channels;
    const h3_gpu_tensor *tensors[6] = { input, alpha_log, beta_log,
                                        upsample_filter, downsample_filter,
                                        output };
    VkDescriptorSet set = h3_vk_prepare(gpu,
                                        H3_VK_KERNEL_ALIAS_FREE_SNAKE_F32,
                                        tensors, 6);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_ALIAS_FREE_SNAKE_F32, set,
                          (channels + 15) / 16, (length + 15) / 16, batch);
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    if (!h3_vk_check_tensors(gpu, 3, output, input, alpha)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    size_t count = (size_t)batch * length * channels;
    if (count > h3_gpu_tensor_elements(input) ||
        channels > h3_gpu_tensor_elements(alpha) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "snake1d tensor size mismatch");
        return -1;
    }
    gpu->args->elements = (uint32_t)count;
    gpu->args->conv_in_channels = channels;
    const h3_gpu_tensor *tensors[3] = { input, alpha, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SNAKE1D_F32,
                                        tensors, 3);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SNAKE1D_F32, set,
                          (uint32_t)((count + H3_VK_THREADS - 1) /
                                     H3_VK_THREADS), 1, 1);
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) {
    if (!h3_vk_check_tensors(gpu, 7, query, key, value, qkv, q_bias, k_bias,
                             v_bias))
        return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (count * 3 > h3_gpu_tensor_elements(qkv) ||
        width > h3_gpu_tensor_elements(q_bias) ||
        width > h3_gpu_tensor_elements(k_bias) ||
        width > h3_gpu_tensor_elements(v_bias) ||
        count > h3_gpu_tensor_elements(query) ||
        count > h3_gpu_tensor_elements(key) ||
        count > h3_gpu_tensor_elements(value)) {
        h3_vk_set_error(gpu, "audio qkv split tensor size mismatch");
        return -1;
    }
    gpu->args->elements = (uint32_t)count;
    gpu->args->conv_width = heads;
    gpu->args->conv_height = head_dim;
    const h3_gpu_tensor *tensors[7] = { qkv, q_bias, k_bias, v_bias,
                                        query, key, value };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_AUDIO_QKV_SPLIT_F32,
                                        tensors, 7);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_AUDIO_QKV_SPLIT_F32, set,
                          (uint32_t)((count + H3_VK_THREADS - 1) /
                                     H3_VK_THREADS), 1, 1);
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale) {
    if (!h3_vk_check_tensors(gpu, 4, output, query, key, value)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    size_t count = (size_t)batch * sequence * heads * head_dim;
    if (count > h3_gpu_tensor_elements(query) ||
        count > h3_gpu_tensor_elements(key) ||
        count > h3_gpu_tensor_elements(value) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "causal sdpa tensor size mismatch");
        return -1;
    }
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->left_scale = scale;
    gpu->args->conv_batch = batch;
    const h3_gpu_tensor *tensors[4] = { query, key, value, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SDPA_CAUSAL_F32,
                                        tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SDPA_CAUSAL_F32, set, heads,
                          sequence, batch);
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) {
    if (!h3_vk_check_tensors(gpu, 2, output, attended)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    if (!output_dim || head_dim % output_dim) {
        h3_vk_set_error(gpu, "audio pool invalid dimensions");
        return -1;
    }
    size_t count = (size_t)batch * length * output_dim;
    if ((size_t)batch * length * heads * head_dim >
            h3_gpu_tensor_elements(attended) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "audio pool tensor size mismatch");
        return -1;
    }
    gpu->args->elements = (uint32_t)count;
    gpu->args->conv_width = heads;
    gpu->args->conv_height = head_dim;
    gpu->args->conv_in_channels = output_dim;
    const h3_gpu_tensor *tensors[2] = { attended, output };
    VkDescriptorSet set = h3_vk_prepare(
        gpu, H3_VK_KERNEL_AUDIO_ATTENTION_POOL_F32, tensors, 2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_AUDIO_ATTENTION_POOL_F32, set,
                          (uint32_t)((count + H3_VK_THREADS - 1) /
                                     H3_VK_THREADS), 1, 1);
}

int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) {
    if (!h3_vk_check_tensors(gpu, 2, output, input)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * (depth + depth_front) *
                          (height + height_before + height_after) *
                          (width + width_before + width_after) * channels;
    if (input_count > h3_gpu_tensor_elements(input) ||
        output_count > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "vae pad tensor size mismatch");
        return -1;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = depth;
    gpu->args->conv_height = height;
    gpu->args->conv_width = width;
    gpu->args->conv_in_channels = channels;
    gpu->args->pad_depth_front = depth_front;
    gpu->args->pad_height_before = height_before;
    gpu->args->pad_height_after = height_after;
    gpu->args->pad_width_before = width_before;
    gpu->args->pad_width_after = width_after;
    const h3_gpu_tensor *tensors[2] = { input, output };
    VkDescriptorSet set = h3_vk_prepare(gpu,
                                        H3_VK_KERNEL_VAE_ENCODER_PAD_F32,
                                        tensors, 2);
    if (set == VK_NULL_HANDLE) return -1;
    uint32_t out_height = height + height_before + height_after;
    uint32_t out_width = width + width_before + width_after;
    uint32_t out_depth = depth + depth_front;
    uint32_t planes = batch * out_depth * out_height;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_VAE_ENCODER_PAD_F32, set,
                          (channels + 15) / 16, (out_width + 15) / 16,
                          planes);
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) {
    if (!h3_vk_check_tensors(gpu, 3, output, input, weight)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    if (!batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height ||
        !kernel_width || !stride_depth || !stride_height || !stride_width ||
        depth < kernel_depth || height < kernel_height ||
        width < kernel_width) {
        h3_vk_set_error(gpu, "conv3d invalid dimensions");
        return -1;
    }
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    size_t input_count = (size_t)batch * depth * height * width *
                         input_channels;
    size_t weight_count = (size_t)output_channels * input_channels *
                          kernel_depth * kernel_height * kernel_width;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * output_channels;
    if (input_count > h3_gpu_tensor_elements(input) ||
        weight_count > h3_gpu_tensor_elements(weight) ||
        output_count > h3_gpu_tensor_elements(output) ||
        (bias && output_channels > h3_gpu_tensor_elements(bias))) {
        h3_vk_set_error(gpu, "conv3d tensor size mismatch");
        return -1;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = depth;
    gpu->args->conv_height = height;
    gpu->args->conv_width = width;
    gpu->args->conv_in_channels = input_channels;
    gpu->args->conv_out_channels = output_channels;
    gpu->args->conv_kernel_depth = kernel_depth;
    gpu->args->conv_kernel_height = kernel_height;
    gpu->args->conv_kernel_width = kernel_width;
    gpu->args->conv_stride_depth = stride_depth;
    gpu->args->conv_stride_height = stride_height;
    gpu->args->conv_stride_width = stride_width;
    gpu->args->has_bias = bias ? 1u : 0u;
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    const h3_gpu_tensor *tensors[4] = { input, weight, bias_buffer, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_CONV3D_F32,
                                        tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    uint32_t planes = batch * output_depth * output_height;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_CONV3D_F32, set,
                          (output_width + 15) / 16, (output_height + 15) / 16,
                          planes);
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) {
    if (!h3_vk_check_tensors(gpu, 4, output, input, weight, bias)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    size_t count = (size_t)batch * depth * height * width * channels;
    if (count > h3_gpu_tensor_elements(input) ||
        channels > h3_gpu_tensor_elements(weight) ||
        channels > h3_gpu_tensor_elements(bias) ||
        count > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "vae group norm tensor size mismatch");
        return -1;
    }
    gpu->args->conv_batch = batch;
    gpu->args->conv_depth = depth;
    gpu->args->conv_height = height;
    gpu->args->conv_width = width;
    gpu->args->conv_in_channels = channels;
    gpu->args->conv_out_channels = groups;
    gpu->args->epsilon = epsilon;
    const h3_gpu_tensor *tensors[4] = { input, weight, bias, output };
    VkDescriptorSet set = h3_vk_prepare(
        gpu, H3_VK_KERNEL_VAE_ENCODER_GROUP_NORM_SILU_F32, tensors, 4);
    if (set == VK_NULL_HANDLE) return -1;
    uint32_t rows = batch * depth * groups;
    return h3_vk_dispatch(gpu,
                          H3_VK_KERNEL_VAE_ENCODER_GROUP_NORM_SILU_F32, set,
                          rows, 1, 1);
}

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    /* The Metal NAX path is a tensor-ops matmul (Apple-only); Vulkan uses
     * the same fused fc1 -> SwiGLU -> fc2 contract as h3_gpu_mlp_bf16,
     * writing through the caller-provided activated scratch when usable. */
    if (!h3_vk_check_tensors(gpu, 4, output, input, fc1_weight, fc2_weight))
        return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(gpu,
                                                (size_t)rows * hidden_dim * 2);
    h3_gpu_tensor *scratch = activated;
    int own_scratch = 0;
    if (!scratch || h3_gpu_tensor_elements(scratch) <
                        (size_t)rows * hidden_dim) {
        scratch = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim);
        own_scratch = 1;
    }
    if (!fc1 || !scratch) {
        h3_gpu_tensor_free(fc1);
        if (own_scratch) h3_gpu_tensor_free(scratch);
        h3_vk_set_error(gpu, "NAX MLP scratch allocation failed");
        return -1;
    }
    int ok = -1;
    if (h3_vk_linear(gpu, fc1, input, fc1_weight, NULL, rows, input_dim,
                     hidden_dim * 2) == 0) {
        gpu->args->elements = rows * hidden_dim;
        gpu->args->width = hidden_dim;
        const h3_gpu_tensor *tensors[2] = { fc1, scratch };
        VkDescriptorSet set = h3_vk_prepare(gpu,
                                            H3_VK_KERNEL_SWIGLU_HALVES_BF16,
                                            tensors, 2);
        if (set != VK_NULL_HANDLE &&
            h3_vk_dispatch(gpu, H3_VK_KERNEL_SWIGLU_HALVES_BF16, set,
                           (uint32_t)(((uint64_t)rows * hidden_dim +
                                       H3_VK_THREADS - 1) / H3_VK_THREADS),
                           1, 1) == 0 &&
            h3_vk_linear(gpu, output, scratch, fc2_weight, NULL, rows,
                         hidden_dim, output_dim) == 0)
            ok = 0;
    }
    h3_gpu_tensor_free(fc1);
    if (own_scratch) h3_gpu_tensor_free(scratch);
    return ok;
}

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
(void)gpu; (void)output; (void)scales; (void)input; (void)rows; (void)columns;
    return h3_vk_not_ported(gpu, "h3_gpu_quantize_weight_int8");
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
(void)gpu; (void)output; (void)quantized_input; (void)input_scales; (void)input; (void)weight; (void)weight_scales; (void)rows; (void)input_dim; (void)output_dim; (void)use_slower_uncached_int8_scales;
    return h3_vk_not_ported(gpu, "h3_gpu_linear_int8_bf16");
}

int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim) {
(void)gpu; (void)output; (void)quantized_input; (void)input_scales; (void)input; (void)weight; (void)weight_scales; (void)rows; (void)heads; (void)head_dim; (void)output_dim;
    return h3_vk_not_ported(gpu, "h3_gpu_linear_int8_head_major_bf16");
}

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized) {
(void)gpu; (void)output; (void)activated; (void)quantized_activation; (void)activation_scales; (void)input; (void)fc1_weight; (void)fc1_scales; (void)fc2_weight; (void)fc2_scales; (void)fc1_bf16; (void)fc2_bf16; (void)rows; (void)input_dim; (void)hidden_dim; (void)output_dim; (void)use_slower_grouped_quantizer; (void)use_slower_dynamic_fc1_k; (void)use_int8_row_fc2; (void)input_is_quantized;
    return h3_vk_not_ported(gpu, "h3_gpu_mlp_int8_bf16");
}

int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) {
    if (!h3_vk_check_tensors(gpu, 6, query, key, value, qkv, rope_cos,
                             rope_sin))
        return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (count * 3 > h3_gpu_tensor_elements(qkv) ||
        rope_count > h3_gpu_tensor_elements(rope_cos) ||
        rope_count > h3_gpu_tensor_elements(rope_sin) ||
        count > h3_gpu_tensor_elements(query) ||
        count > h3_gpu_tensor_elements(key) ||
        count > h3_gpu_tensor_elements(value)) {
        h3_vk_set_error(gpu, "vision qkv rope tensor size mismatch");
        return -1;
    }
    gpu->args->rows = sequence;
    gpu->args->width = heads;
    gpu->args->input_dim = head_dim;
    gpu->args->output_dim = rope_half;
    const h3_gpu_tensor *tensors[6] = { qkv, rope_cos, rope_sin, query,
                                        key, value };
    VkDescriptorSet set = h3_vk_prepare(gpu,
                                        H3_VK_KERNEL_VISION_QKV_ROPE_BF16,
                                        tensors, 6);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_VISION_QKV_ROPE_BF16, set,
                          (head_dim + 15) / 16, (heads + 15) / 16,
                          sequence);
}

int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
(void)gpu; (void)gated_residual; (void)quantized_output; (void)quantized_scales; (void)residual; (void)branch; (void)norm_weight; (void)gate_modulation; (void)norm_modulation; (void)row_map; (void)rows; (void)padded_rows; (void)width; (void)slots; (void)gate_slot; (void)shift_slot; (void)scale_slot; (void)epsilon;
    return h3_vk_not_ported(gpu, "h3_gpu_gate_adaln_quantize_int8");
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon) {
    /* Vulkan has no tensor-ops matmul; use the Metal fallback path:
     * plain BF16 projection followed by the grouped norm/RoPE kernel. */
    uint32_t inner = heads * head_dim;
    if (h3_gpu_linear_bf16(gpu, qkv, input, weight, NULL, rows, input_dim,
                           inner * 3) != 0)
        return -1;
    return h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm,
                                        k_norm, rope_cos, rope_sin, rows,
                                        heads, head_dim, rope_half, epsilon);
}

int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales) {
(void)gpu; (void)query; (void)key; (void)value; (void)quantized_input; (void)input_scales; (void)input; (void)weight; (void)weight_scales; (void)q_norm; (void)k_norm; (void)rope_cos; (void)rope_sin; (void)rows; (void)input_dim; (void)heads; (void)head_dim; (void)rope_half; (void)epsilon; (void)input_is_quantized; (void)use_slower_unfused_qkv_rope; (void)use_slower_scalar_qkv_rms; (void)use_slower_uncached_int8_scales;
    return h3_vk_not_ported(gpu, "h3_gpu_grouped_qkv_linear_rope_int8");
}

int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    if (!h3_vk_check_tensors(gpu, 2, output, fused)) return -1;
    if (!h3_vk_require_command(gpu)) return -1;
    if ((size_t)rows * width * 2 > h3_gpu_tensor_elements(fused) ||
        (size_t)rows * width > h3_gpu_tensor_elements(output)) {
        h3_vk_set_error(gpu, "swiglu bf16 tensor size mismatch");
        return -1;
    }
    gpu->args->elements = rows * width;
    gpu->args->rows = rows;
    gpu->args->width = width;
    const h3_gpu_tensor *tensors[2] = { fused, output };
    VkDescriptorSet set = h3_vk_prepare(gpu, H3_VK_KERNEL_SWIGLU_HALVES_BF16,
                                        tensors, 2);
    if (set == VK_NULL_HANDLE) return -1;
    return h3_vk_dispatch(gpu, H3_VK_KERNEL_SWIGLU_HALVES_BF16, set,
                          (uint32_t)(((uint64_t)rows * width +
                                       H3_VK_THREADS - 1) / H3_VK_THREADS),
                          1, 1);
}


/* ------------------------------------------------------------------ probe */

/* Device probe used by h3_metal_probe() on non-Apple platforms. Reports the
 * selected Vulkan device without building any shader pipelines. */
int h3_vulkan_probe(h3_device_info *info, char *error, size_t error_size) {
    if (!info) return 0;
    memset(info, 0, sizeof(*info));
    h3_gpu *gpu = calloc(1, sizeof(*gpu));
    if (!gpu) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory");
        return 0;
    }
    if (!h3_vk_create_instance(gpu, error, error_size) ||
        !h3_vk_pick_device(gpu, error, error_size)) {
        h3_gpu_free(gpu);
        return 0;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu->physical, &props);
    snprintf(info->name, sizeof(info->name), "%s", props.deviceName);
    snprintf(info->architecture, sizeof(info->architecture), "Vulkan");
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(gpu->physical, &memory);
    uint64_t total = 0;
    for (uint32_t index = 0; index < memory.memoryHeapCount; index++)
        if (memory.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total += memory.memoryHeaps[index].size;
    info->physical_memory = total;
    info->recommended_working_set = total;
    info->max_buffer_length = props.limits.maxStorageBufferRange;
    info->unified_memory = 0;
    h3_gpu_free(gpu);
    return 1;
}
