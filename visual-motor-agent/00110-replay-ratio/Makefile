# Makefile for experiment vm-00110-replay-ratio
#
# Reproduces the runs of task 00122: the default agent in Breakout stage 1 (configs/U_breakout.json, as vm-00103's BD)
# with Dreamer's training budget per tick: online.batch = 56 (56 windows of 8 steps, the newest and 55 replayed, 448
# steps per update; BD: 8 windows, 64 steps; vm-00108's Dreamer: 512 replayed steps per tick). Arm RB, seeds 1-12, 60k ticks.
# Seeds 1-6 on the RTX 3090 (GPU 0 under CUDA_DEVICE_ORDER=PCI_BUS_ID), 7-12 on the RTX 4090 (GPU 1), under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00110/.
#
# Usage:
#   make runs — RB, seeds 1-12
#   make frozen — (after the runs) per checkpoint (30k, 60k) and life: 2000 ticks with learning off (ppo.lr = 0) in new
#                 courts (seed 100 + s, as vm-00103)
#   Options: MPS= (no MPS daemon), T="10000 60000" (frozen's checkpoints)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00110
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
T ?= 30000 60000
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
SET = --config configs/U_breakout.json --set online.batch=56
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000
SEEDS = 1 2 3 4 5 6 7 8 9 10 11 12
CARD = $$([ $$s -le 6 ] && echo 0 || echo 1)

.PHONY: all runs frozen

all: runs frozen

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) --seed $$s --set name=RB -o $(OUT)/RB_s$$s > $(OUT)/RB_s$$s.log 2>&1 & \
	  done; wait; }

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(FROZEN) --init $(OUT)/RB_s$$s/ckpt_$$(printf %08d $$t) --seed 10$$s \
	      -o $$o/RB_s$$s > $$o/RB_s$$s.log 2>&1 & \
	  done; wait; \
	done
