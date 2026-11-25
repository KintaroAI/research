# Makefile for experiment vm-00111-long-horizon-bricks
#
# Reproduces the runs of task 00124: vm-00104's bricks-only court (configs/U_breakout2.json: stage 2, a ball at 2.25 px a
# tick, no food for a paddle hit, 24 bricks of 3 looks feeding .1, .25 and .5) with vm-00104's BS (online.meal_size = .5)
# and a long label horizon, online.label_gamma = .997 (default .95). Arm BH, seeds 1-3, 60k ticks; the look shift
# env.brick_shift = seed mod 3, as vm-00104. On the RTX 3090 (GPU 0 under CUDA_DEVICE_ORDER=PCI_BUS_ID), under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00111/.
#
# Usage:
#   make runs — BH, seeds 1-3
#   make frozen — (after the runs) per checkpoint (30k, 60k) and life: 2000 ticks with learning off (ppo.lr = 0) in new
#                 courts (seed 100 + s, as vm-00104)
#   Options: MPS= (no MPS daemon), T="10000 60000" (frozen's checkpoints)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00111
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
T ?= 30000 60000
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $(MPS)
SET = --config configs/U_breakout2.json --set online.meal_size=0.5 --set online.label_gamma=0.997
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000
SEEDS = 1 2 3
SHIFT = --set env.brick_shift=$$((s % 3))

.PHONY: all runs frozen

all: runs frozen

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in $(SEEDS); do \
	    env $(G) $(PY) lifetime.py $(SET) $(SHIFT) --seed $$s --set name=BH -o $(OUT)/BH_s$$s > $(OUT)/BH_s$$s.log 2>&1 & \
	  done; wait; }

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; \
	  for s in $(SEEDS); do \
	    env $(G) $(PY) lifetime.py $(SET) $(SHIFT) $(FROZEN) --init $(OUT)/BH_s$$s/ckpt_$$(printf %08d $$t) --seed 10$$s \
	      -o $$o/BH_s$$s > $$o/BH_s$$s.log 2>&1 & \
	  done; wait; \
	done
