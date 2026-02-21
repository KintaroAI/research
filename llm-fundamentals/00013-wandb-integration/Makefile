# Reproduction Makefile for Experiment 00013: W&B Integration
#
# Prerequisites: CUDA toolkit, Python 3
# One-time: wandb login (paste API key from https://wandb.ai/authorize)
#
# Usage:
#   make all       # Full pipeline: setup → data → model → run
#   make setup     # Install wandb in dev venv
#   make data      # Generate modular arithmetic dataset
#   make model     # Create grokking model
#   make run       # Run training through wandb wrapper

DEV = ../dev

all: setup data model run

setup:
	cd $(DEV) && ./venv/bin/pip install wandb

data:
	cd $(DEV) && ./venv/bin/python gen_modular_data.py \
		--prime 97 --op add --train-frac 0.5 --val-frac 0.5 \
		--seed 42 --output-dir data/modular

model:
	cd $(DEV) && ./venv/bin/python create_model.py --preset grokking --prime 97 -o model_grok.bin

run: $(DEV)/train
	cd $(DEV) && ./venv/bin/python wandb_train.py \
		--project gpt2-cuda --name "wandb-test" -- \
		./train -e model_grok.bin \
		-i data/modular/train.bin -j data/modular/val.bin -x data/modular/test.bin \
		-t 8 -b 4703 -n 5000 -v 500 -s 0 \
		-l 0.001 -w 1.0 -a 0.98 -q 1337

$(DEV)/train:
	$(MAKE) -C $(DEV) train

.PHONY: all setup data model run
