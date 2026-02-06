# Makefile for Banded Sparsity Experiment

NVCC = nvcc
NVCC_FLAGS = -O3 -lcublas -lm

all: train train_banded train_banded_fc2 generate generate_banded

train: src/train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

train_banded: src/train_gpt2_fp32_banded.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

train_banded_fc2: src/train_gpt2_fp32_banded_fc2.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

generate: src/generate.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

generate_banded: src/generate_banded.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f train train_banded train_banded_fc2 generate generate_banded

# Setup: create venv and prepare data
setup: venv data

venv:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install tiktoken numpy requests tqdm torch

data: venv
	./venv/bin/python prepare_data.py

model:
	./venv/bin/python create_model.py

# Run experiment: compare banded vs dense
experiment: train_banded data
	@echo "=== Running Banded Sparsity Experiment ==="
	@echo ""
	@echo "Training BANDED (bandwidth=256, ~44% density)..."
	./train_banded -e model.bin -c checkpoint_banded.bin -w 256 -n 2000 -b 16 -t 256 -v 200 -s 500 -o log_banded.txt
	@echo ""
	@echo "Training DENSE (100% density)..."
	./train_banded -e model.bin -c checkpoint_dense.bin -d 1 -n 2000 -b 16 -t 256 -v 200 -s 500 -o log_dense.txt
	@echo ""
	@echo "=== Experiment Complete ==="
	@echo "Compare log_banded.txt vs log_dense.txt"

.PHONY: all clean setup venv data model experiment
