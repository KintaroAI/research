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

# Run the main experiment (with weight decay = 1.0, answer-only loss masking)
# NOTE: B=4688 for full-batch training (paper uses all training equations per step).
# 4688 is the largest multiple of 16 that fits: B*T+1 <= train_tokens.
# -p 4 = only compute loss/gradient at position 4 (the answer token c in "BOS a OP b EQ c EOS EOS")
experiment: train data model
	@echo "=== Delayed Generalization Experiment (full-batch, answer-only loss) ==="
	@echo "Training with weight decay = 1.0, B=4688, loss masked to position 4..."
	./train -e model.bin -i data/train.bin -j data/val.bin -n 100000 -b 4688 -t 8 -v 100 -l 0.001 -w 1.0 -a 0.98 -s 0 -p 4 -o log_wd1.0.txt -c checkpoint_wd1.0.bin -k 10000
	@echo ""
	@echo "=== Experiment Complete ==="
	@echo "See log_wd1.0.txt for train/val loss curves"

# Ablation: no weight decay (expect memorization only, no generalization)
experiment_no_wd: train data model
	@echo "Training WITHOUT weight decay (expect no generalization)..."
	./train -e model.bin -i data/train.bin -j data/val.bin -n 100000 -b 4688 -t 8 -v 100 -l 0.001 -w 0.0 -a 0.98 -s 0 -p 4 -o log_wd0.0.txt

# Ablation: vary weight decay
experiment_ablation: train data model
	@echo "=== Weight Decay Ablation ==="
	@echo "Training with wd=0.0..."
	./train -e model.bin -i data/train.bin -j data/val.bin -n 100000 -b 4688 -t 8 -v 100 -l 0.001 -w 0.0 -a 0.98 -s 0 -p 4 -o log_wd0.0.txt
	@echo "Training with wd=0.1..."
	./train -e model.bin -i data/train.bin -j data/val.bin -n 100000 -b 4688 -t 8 -v 100 -l 0.001 -w 0.1 -a 0.98 -s 0 -p 4 -o log_wd0.1.txt
	@echo "Training with wd=0.5..."
	./train -e model.bin -i data/train.bin -j data/val.bin -n 100000 -b 4688 -t 8 -v 100 -l 0.001 -w 0.5 -a 0.98 -s 0 -p 4 -o log_wd0.5.txt
	@echo "Training with wd=1.0..."
	./train -e model.bin -i data/train.bin -j data/val.bin -n 100000 -b 4688 -t 8 -v 100 -l 0.001 -w 1.0 -a 0.98 -s 0 -p 4 -o log_wd1.0.txt
	@echo "=== Ablation Complete ==="

.PHONY: all clean setup venv data model experiment experiment_no_wd experiment_ablation
