# Makefile for CUDA Probe Experiment

NVCC = nvcc
NVCC_FLAGS = -O3

all: gpu_probe

gpu_probe: src/gpu_probe.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: gpu_probe
	./gpu_probe

clean:
	rm -f gpu_probe

.PHONY: all run clean
