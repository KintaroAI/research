# Makefile for Delayed Generalization Experiment

NVCC = nvcc
NVCC_FLAGS = -O3 -lcublas -lm

all: train

train: src/train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f train

# Setup: create venv and prepare data
setup: venv data model

venv:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install numpy torch

data: venv
	./venv/bin/python generate_dataset.py

model: venv
	./venv/bin/python create_model.py

# Run the main experiment (with weight decay = 1.0)
experiment: train data model
	@echo "=== Delayed Generalization Experiment ==="
	@echo "Training with weight decay = 1.0 (expect delayed generalization)..."
	./train -e model.bin -n 100000 -b 512 -t 5 -v 100 -l 0.001 -w 1.0 -s 0 -o log_wd1.0.txt -c checkpoint_wd1.0.bin -k 10000
	@echo ""
	@echo "=== Experiment Complete ==="
	@echo "See log_wd1.0.txt for train/val loss curves"

# Ablation: no weight decay (expect memorization only, no generalization)
experiment_no_wd: train data model
	@echo "Training WITHOUT weight decay (expect no generalization)..."
	./train -e model.bin -n 100000 -b 512 -t 5 -v 100 -l 0.001 -w 0.0 -s 0 -o log_wd0.0.txt

# Ablation: vary weight decay
experiment_ablation: train data model
	@echo "=== Weight Decay Ablation ==="
	@echo "Training with wd=0.0..."
	./train -e model.bin -n 100000 -b 512 -t 5 -v 100 -l 0.001 -w 0.0 -s 0 -o log_wd0.0.txt
	@echo "Training with wd=0.1..."
	./train -e model.bin -n 100000 -b 512 -t 5 -v 100 -l 0.001 -w 0.1 -s 0 -o log_wd0.1.txt
	@echo "Training with wd=0.5..."
	./train -e model.bin -n 100000 -b 512 -t 5 -v 100 -l 0.001 -w 0.5 -s 0 -o log_wd0.5.txt
	@echo "Training with wd=1.0..."
	./train -e model.bin -n 100000 -b 512 -t 5 -v 100 -l 0.001 -w 1.0 -s 0 -o log_wd1.0.txt
	@echo "=== Ablation Complete ==="

.PHONY: all clean setup venv data model experiment experiment_no_wd experiment_ablation
