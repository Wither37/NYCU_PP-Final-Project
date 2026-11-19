# Kernel K-means framework + strategies.
#
#   make                      bin/kkm and bin/test_kkm for the RTX A6000 (sm_86)
#   make test                 build and run the tests (needs a GPU)
#   make test-cpu             CPU-only tests (no GPU needed)
#   make NVCC=.cuda/bin/nvcc  use the project-local compiler (tools/fetch_cuda.sh)
#   make ARCH="86 89"         also target the RTX 4070 Ti (sm_89) used for testing
#   make DEBUG=1              -G -lineinfo, synchronous launch checks

NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda-12.4/bin/nvcc)
ARCH ?= 86
GENCODE := $(foreach a,$(ARCH),-gencode arch=compute_$(a),code=sm_$(a))

CXXSTD   := -std=c++17
HOSTOPT  := -O3 -fopenmp -Wall -Wextra
NVFLAGS  := $(CXXSTD) $(GENCODE) -Xcompiler "$(HOSTOPT)" -Isrc
ifeq ($(DEBUG),1)
  NVFLAGS += -G -lineinfo -DKKM_DEBUG_SYNC
else
  NVFLAGS += -O3 -lineinfo
endif
LDFLAGS  := -Xcompiler -fopenmp

BUILD := build
LIB_SRCS := src/budget.cu src/data.cpp src/cpu_ref.cpp src/tile_kernels.cu \
            src/assign.cu src/strategy.cu src/s_inmem.cu src/s1_symtiles.cu \
            src/s2_outofcore.cu src/driver.cu
LIB_OBJS := $(patsubst src/%,$(BUILD)/%.o,$(LIB_SRCS))

all: bin/kkm bin/test_kkm

$(BUILD)/%.o: src/%
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/main.cu.o: src/main.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/test_kkm.cu.o: tests/test_kkm.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -MMD -MP -c $< -o $@

bin/kkm: $(LIB_OBJS) $(BUILD)/main.cu.o
	@mkdir -p bin
	$(NVCC) $(GENCODE) $^ $(LDFLAGS) -o $@

bin/test_kkm: $(LIB_OBJS) $(BUILD)/test_kkm.cu.o
	@mkdir -p bin
	$(NVCC) $(GENCODE) $^ $(LDFLAGS) -o $@

test: bin/test_kkm
	./bin/test_kkm

test-cpu: bin/test_kkm
	./bin/test_kkm --cpu

clean:
	rm -rf $(BUILD) bin

-include $(wildcard $(BUILD)/*.d)

.PHONY: all test test-cpu clean
