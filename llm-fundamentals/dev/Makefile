# Makefile for GPT-2 Training & Inference
# Supports optional banded sparsity via -1 and -2 flags

NVCC = nvcc
NVCC_FLAGS = -O3 -lcublas -lm

all: train generate

train: src/train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

generate: src/generate.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

test_gpt2_fp32: src/test_gpt2_fp32.cu src/train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

test_banded_sparsity: src/test_banded_sparsity.cu src/train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

test_task_position: src/test_task_position.cu src/train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

test_checkpoint: src/test_checkpoint.cu src/train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

test_generate: src/test_generate.cu src/generate.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run test: requires gpt2_124M.bin and gpt2_124M_debug_state.bin
# Generate with: ./venv/bin/python train_gpt2.py --model gpt2 --write_tensors 1 --num_iterations 0
test: test_gpt2_fp32
	./test_gpt2_fp32

test_all: test_gpt2_fp32 test_banded_sparsity test_task_position test_checkpoint test_generate
	./test_gpt2_fp32
	./test_banded_sparsity
	./test_task_position
	./test_checkpoint
	./test_generate

clean:
	rm -f train generate test_gpt2_fp32 test_banded_sparsity test_task_position test_checkpoint test_generate

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
# Usage: make experiment
#   -1 <int>  FC1 bandwidth (0 = dense)
#   -2 <int>  FC2 bandwidth (0 = dense)
experiment: train data
	@echo "=== Running Banded Sparsity Experiment ==="
	@echo ""
	@echo "Training BANDED FC1+FC2 (bandwidth=256)..."
	./train -e model.bin -c checkpoint_banded.bin -1 256 -2 256 -n 2000 -b 16 -t 256 -v 200 -s 500 -o log_banded_fc1_fc2.txt
	@echo ""
	@echo "Training DENSE (no sparsity)..."
	./train -e model.bin -c checkpoint_dense.bin -1 0 -2 0 -n 2000 -b 16 -t 256 -v 200 -s 500 -o log_dense.txt
	@echo ""
	@echo "=== Experiment Complete ==="
	@echo "Compare log_banded_fc1_fc2.txt vs log_dense.txt"

# Additional experiments with different configurations
experiment_fc1_only: train data
	@echo "Training BANDED FC1 only (bandwidth=256)..."
	./train -e model.bin -c checkpoint_fc1.bin -1 256 -2 0 -n 2000 -b 16 -t 256 -v 200 -s 500 -o log_banded_fc1.txt

experiment_fc2_only: train data
	@echo "Training BANDED FC2 only (bandwidth=256)..."
	./train -e model.bin -c checkpoint_fc2.bin -1 0 -2 256 -n 2000 -b 16 -t 256 -v 200 -s 500 -o log_banded_fc2.txt

experiment_asymmetric: train data
	@echo "Training BANDED FC1=128 FC2=512..."
	./train -e model.bin -c checkpoint_asym.bin -1 128 -2 512 -n 2000 -b 16 -t 256 -v 200 -s 500 -o log_banded_asym.txt

.PHONY: all clean setup venv data model test test_all experiment experiment_fc1_only experiment_fc2_only experiment_asymmetric
