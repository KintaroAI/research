# Makefile for experiment vm-00112-dreamer-short-horizon
#
# Reproduces the runs of task 00125: vm-00108's DreamerV3 (PyTorch port NM512/dreamerv3-torch, commit 6ef8646, in
# ~/src/dreamerv3-torch; ~/venvs/dreamer) in the food world (configs/U_agent.json's world), train ratio 512, its defaults
# except discount = .95 (default .997; our label's gamma). Arm DG, seeds 1-3, 60k ticks (the 2,500-tick prefill included).
# On the RTX 4090 (GPU 1 under CUDA_DEVICE_ORDER=PCI_BUS_ID), 3 at once, under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00112/.
#
# Usage:
#   make runs — DG, seeds 1-3
#   Options: MPS= (no MPS daemon), TR=1024 (Dreamer's train ratio)

DEV = ../dev
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00112
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 $(MPS)
DPY = $(HOME)/venvs/dreamer/bin/python -W ignore
TR ?= 512
SEEDS = 1 2 3

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in $(SEEDS); do \
	    env $(G) $(DPY) baselines/run_dreamer.py --config configs/U_agent.json --seed $$s --steps 60000 --train-ratio $(TR) \
	      --dset discount=0.95 --out $(OUT)/DG_s$$s > $(OUT)/DG_s$$s.log 2>&1 & \
	  done; wait; }
