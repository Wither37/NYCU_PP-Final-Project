# Microbenchmarks for the hardware report. RTX A6000 = sm_86.
NVCC  ?= nvcc
ARCH  ?= sm_86
FLAGS ?= -O3 -arch=$(ARCH)

BINS = devinfo bw_pcie bw_gpu

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) -o $@ $<

run: all
	./run.sh

clean:
	rm -f $(BINS)

.PHONY: all run clean
