CC := clang
AR := ar
UNAME_S := $(shell uname -s)
GPU ?= auto
NVCC ?= $(shell command -v nvcc 2>/dev/null || \
	{ test -x /opt/cuda/bin/nvcc && echo /opt/cuda/bin/nvcc; })
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(NVCC))
CUDA_ARCH ?= native
NVCCFLAGS ?= -std=c++17 -O3 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
CUDNN ?= $(shell { test -f /usr/include/cudnn.h || \
	test -f "$(CUDA_HOME)/include/cudnn.h"; } && \
	{ ldconfig -p 2>/dev/null | grep -q 'libcudnn\.so' || \
	test -f "$(CUDA_HOME)/lib64/libcudnn.so"; } && echo 1)

# Feature-test macros expose POSIX functions (strdup, setenv, mkdtemp) under
# strict -std=c11 on both platforms.
ifeq ($(UNAME_S),Darwin)
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE \
	-DH3_SHADER_SOURCE=\"h3_shaders.metal\"
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm
else
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L
LDLIBS := -lm
endif

LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_int8_cache.c \
	h3_schedule_cache.c h3_text_encoder.c h3_dit_schedule.c h3_dit.c

LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c \
	h3_terminal.c h3_vision_encoder.c h3_multimodal.c
# The Metal runtime and the Foundation tokenizer are macOS-only. Linux uses
# Vulkan by default when available; request the independent native CUDA backend
# with `make GPU=cuda`. Keeping selection explicit makes Vulkan builds stable on
# developer machines that happen to have nvcc installed.
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m
GPU_EXTRA_OBJ :=
ifeq ($(UNAME_S),Darwin)
GPU_STUB :=
SHADER_SOURCE := h3_shaders.metal
else
VULKAN := $(shell pkg-config --exists vulkan shaderc && echo 1)
ifeq ($(GPU),cuda)
ifeq ($(strip $(NVCC)),)
$(error GPU=cuda requested but nvcc was not found)
endif
GPU_STUB := h3_gpu_cuda.c h3_metal_stub.c
GPU_EXTRA_OBJ := h3_cuda_kernels.o h3_cuda_accel.o
SHADER_SOURCE := h3_cuda_kernels.cu
CFLAGS += -I$(CUDA_HOME)/include -DH3_HAVE_CUDA
LDLIBS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 \
	-lcublasLt -lcudart -lstdc++
ifeq ($(CUDNN),1)
CFLAGS += -DH3_HAVE_CUDNN
NVCCFLAGS += -DH3_HAVE_CUDNN
LDLIBS += -lcudnn
endif
else ifeq ($(GPU),stub)
GPU_STUB := h3_gpu_stub.c h3_metal_stub.c
SHADER_SOURCE := h3_shaders.metal
else ifeq ($(VULKAN),1)
GPU_STUB := h3_gpu_vulkan.c h3_metal_stub.c
SHADER_SOURCE := h3_vulkan_shaders.comp
LDLIBS += $(shell pkg-config --libs vulkan shaderc)
CFLAGS += -DH3_HAVE_VULKAN
else
GPU_STUB := h3_gpu_stub.c h3_metal_stub.c
SHADER_SOURCE := h3_shaders.metal
endif
CFLAGS += -DH3_SHADER_SOURCE=\"$(SHADER_SOURCE)\"
# The Qwen BPE tokenizer is a portable C port of h3_tokenizer.m; it uses
# ICU (libicuuc) on non-Darwin platforms, matching macOS's -licucore.
GPU_STUB += h3_tokenizer.c
LDLIBS += -licuuc
LIB_M :=
endif
LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o) $(GPU_STUB:.c=.o) $(GPU_EXTRA_OBJ)
CLI_OBJ := main.o h3_cli.o linenoise.o
BUILD_CONFIG := .build-config-$(UNAME_S)-$(GPU)-$(CUDNN)

.PHONY: all test parity real-parity clean

all: h3 libh3.a

h3: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

libh3.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

h3_tests: tests/test_h3.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_metal_tests: tests/test_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_bf16_tests: tests/test_bf16.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tokenizer_tests: tests/test_tokenizer.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_text_tests: tests/test_text_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_audio_gpu_tests: tests/test_audio_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_vulkan_kernels_test: tests/test_vulkan_kernels.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_vulkan_dit_block_test: tests/test_vulkan_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_cuda_kernels_test: tests/test_vulkan_kernels.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_cuda_dit_block_test: tests/test_vulkan_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tokenizer_c_test: tests/test_tokenizer_c.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_vae_test: tests/test_real_audio_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_encoder_test: tests/test_real_audio_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_av_mux_test: tests/test_av_mux.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_video_encoder_test: tests/test_real_video_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_qwen_vision_test: tests/test_real_qwen_vision.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_multimodal_text_test: tests/test_real_multimodal_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ref_video_text_test: tests/test_real_ref_video_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_prompt_test: tests/test_real_prompt.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_block_test: tests/test_real_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_schedule_test: tests/test_real_dit_schedule.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_test: tests/test_real_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_dit_test: tests/test_semantic_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench: tests/bench_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench_864: tests/bench_dit_864.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_dit_864.o: tests/bench_dit.c $(BUILD_CONFIG)
	$(CC) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@

h3_real_video_vae_test: tests/test_real_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_vae_test: tests/test_semantic_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

ifeq ($(UNAME_S),Linux)
ifeq ($(GPU),cuda)
LINUX_KERNEL_TEST := h3_cuda_kernels_test
LINUX_DIT_TEST := h3_cuda_dit_block_test
else
LINUX_KERNEL_TEST := h3_vulkan_kernels_test
LINUX_DIT_TEST := h3_vulkan_dit_block_test
endif

test: h3_tests $(LINUX_KERNEL_TEST) $(LINUX_DIT_TEST) h3_tokenizer_c_test
	./h3_tests
	./$(LINUX_KERNEL_TEST)
	./$(LINUX_DIT_TEST)
	./h3_tokenizer_c_test

parity:
	@echo "parity (Metal/MLX fixtures) requires macOS; on Linux run 'make real-parity'"

real-parity:
	@test -f MiniMax-H3/FL2VA/transformer/config.json || \
		{ echo "model missing: run 'make model' first (scripts/download_model.sh)"; exit 1; }
	@for f in misc/fixtures/h3_real_prompt_bf16.safetensors \
		misc/fixtures/h3_real_dit_block0_bf16.safetensors \
		misc/fixtures/h3_real_dit_step0_bf16.safetensors \
		misc/fixtures/h3_real_dit_denoise20_bf16.safetensors; do \
		if [ ! -f "$$f" ]; then \
			echo "warning: fixture $$f missing (real-parity needs the MLX fixtures)"; \
		fi; \
	done
	H3_SHADER_SOURCE="$(SHADER_SOURCE)" ./h3_real_prompt_test MiniMax-H3
	H3_SHADER_SOURCE="$(SHADER_SOURCE)" ./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors
	H3_SHADER_SOURCE="$(SHADER_SOURCE)" ./h3_real_dit_schedule_test MiniMax-H3
	H3_SHADER_SOURCE="$(SHADER_SOURCE)" ./h3_real_dit_test MiniMax-H3

# Checkpoint download: FL2VA (~37 GiB), optional Ref2VA (~62 GiB).
model:
	scripts/download_model.sh MiniMax-H3

model-ref2va:
	scripts/download_model.sh MiniMax-H3 --ref2va

# Fast end-to-end generation on the selected Linux GPU backend.
smoke:
	@test -f MiniMax-H3/FL2VA/transformer/config.json || \
		{ echo "model missing: run 'make model' first (scripts/download_model.sh)"; exit 1; }
	rm -f /tmp/h3-smoke.mp4
	./h3 -d MiniMax-H3 -p "A red fox walking through snow" \
		-o /tmp/h3-smoke.mp4 --width 256 --height 256 --frames 8 \
		--steps 5 --reuse 3 --layers 35
	@if command -v ffprobe >/dev/null 2>&1; then \
		ffprobe -v error -show_entries format=duration /tmp/h3-smoke.mp4; \
	else \
		echo "smoke output written to /tmp/h3-smoke.mp4"; \
	fi
else
test: h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests \
	h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
	h3_av_mux_test \
	h3_real_video_encoder_test h3_real_qwen_vision_test \
	h3_real_multimodal_text_test h3_real_ref_video_text_test

	./h3_tests
	@if test -f misc/fixtures/h3_dit.safetensors && \
	         test -f misc/fixtures/h3_dit_bf16.safetensors; then \
		./h3_metal_tests misc/fixtures/h3_dit.safetensors; \
		./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors; \
	else \
		echo "skip: MLX toy-block fixtures are not installed"; \
	fi
	@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
		./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
	else \
		echo "skip: released tokenizer is not installed"; \
	fi
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		./h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	./h3_audio_gpu_tests
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_vae_37.safetensors; then \
		./h3_real_audio_vae_test; \
	else \
		echo "skip: released AudioVAE weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_encoder_64000.safetensors; then \
		./h3_real_audio_encoder_test; \
	else \
		echo "skip: released audio encoder weights/fixture are not installed"; \
	fi
	@if command -v ffmpeg >/dev/null 2>&1; then \
		./h3_av_mux_test; \
	else \
		echo "skip: FFmpeg is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_256.safetensors; then \
		./h3_real_video_encoder_test; \
	else \
		echo "skip: released visual encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; then \
		./h3_real_video_encoder_test MiniMax-H3 \
			misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; \
	else \
		echo "skip: released reference-video encoder fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_64.safetensors; then \
		./h3_real_qwen_vision_test; \
	else \
		echo "skip: released Qwen vision weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; then \
		./h3_real_qwen_vision_test MiniMax-H3 \
			misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; \
	else \
		echo "skip: released Qwen video-pair fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_multimodal_text_64.safetensors; then \
		./h3_real_multimodal_text_test; \
	else \
		echo "skip: released multimodal Qwen weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_ref_video_text_64.safetensors; then \
		./h3_real_ref_video_text_test; \
	else \
		echo "skip: Ref2VA video presentation fixture is not installed"; \
	fi

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	./h3_metal_tests misc/fixtures/h3_dit.safetensors
	./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	./h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	./h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors
endif

$(BUILD_CONFIG):
	rm -f .build-config-*
	touch $@

h3_cuda_kernels.cu: h3_vulkan_shaders.comp h3_gpu_vulkan.c \
		scripts/generate_cuda_kernels.py
	scripts/generate_cuda_kernels.py

%.o: %.c $(BUILD_CONFIG)
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m $(BUILD_CONFIG)
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

%.o: %.cu h3_cuda_kernels.h $(BUILD_CONFIG)
	$(NVCC) $(NVCCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c $(BUILD_CONFIG)
	$(CC) $(CFLAGS) -I. -c $< -o $@

# Vendored from Iris. Keep the main project strict without rewriting this small
# terminal editor for conversion diagnostics unrelated to H3.
linenoise.o: CFLAGS += -Wno-conversion -Wno-variadic-macro-arguments-omitted

-include $(wildcard *.d tests/*.d)

clean:
	rm -f h3 h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests \
		h3_text_tests h3_real_prompt_test h3_real_dit_block_test \
		h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
		h3_av_mux_test h3_vulkan_kernels_test h3_vulkan_dit_block_test \
		h3_cuda_kernels_test h3_cuda_dit_block_test \
		h3_real_video_encoder_test h3_real_qwen_vision_test \
		h3_real_multimodal_text_test h3_real_ref_video_text_test \
		h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
		h3_real_video_vae_test h3_semantic_vae_test \
	h3_dit_bench h3_dit_bench_864 \
	libh3.a *.o *.d tests/*.o tests/*.d .build-config-*
