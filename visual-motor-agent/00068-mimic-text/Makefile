# Makefile for experiment vm-00068-mimic-text
#
# Reproduces the runs from the source tagged exp/vm-00068 (dev/ at that tag): half 2 learns from half 1 and says the missing words out loud
# (task 00078, reference/mimic.py, a PyTorch prototype). Needs the CUDA PyTorch venv (~/venvs/vm-gpu), the MPS daemon and the corpus
# (../../llm-fundamentals/dev: prepare_tinyshakespeare.py). Outputs go to ~/data/research/visual-motor-agent/exp_00068/.
#
# Usage:
#   make runs — seeds 1-3: Z, N, A, B, S, E at q = 0.3; A, B, S, E at q = 0.8; 4 at a time on the RTX 4090

DEV = ../dev
TPY = $(HOME)/venvs/vm-gpu/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00068
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe CUDA_VISIBLE_DEVICES=1

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in 1 2 3; do for a in Z N A B S E; do echo "$$a 0.3 $$s"; done; for a in A B S E; do echo "$$a 0.8 $$s"; done; done; } | \
	  xargs -P 4 -L 1 sh -c '$(G) $(TPY) reference/mimic.py --arm $$0 --q $$1 --seed $$2 -o $(OUT)/$${0}_q$${1}_s$$2 > $(OUT)/$${0}_q$${1}_s$$2.log 2>&1'
