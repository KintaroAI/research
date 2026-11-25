# Makefile for experiment vm-00113-replay-ratio-temp
#
# Reproduces the runs of task 00126: vm-00110's RB (Breakout stage 1, configs/U_breakout.json, online.batch = 56: Dreamer's
# replayed steps per tick) with the instinct's candidates drawn at temperature 4 (online.instinct_temp; the default 2).
# Arm RT, seeds 1-6, 60k ticks, on the RTX 3090 (GPU 0 under CUDA_DEVICE_ORDER=PCI_BUS_ID; seeds 7-12 would go to GPU 1),
# under MPS. Outputs go to ~/data/research/visual-motor-agent/exp_00113/.
#
# Usage:
#   make runs — RT, seeds 1-6
#   make frozen — (after the runs) per checkpoint (30k, 60k) and life: 2000 ticks with learning off (ppo.lr = 0) in new
#                 courts (seed 100 + s, as vm-00110)
#   Options: MPS= (no MPS daemon), T="10000 60000" (frozen's checkpoints)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00113
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
T ?= 30000 60000
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
SET = --config configs/U_breakout.json --set online.batch=56 --set online.instinct_temp=4
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000
SEEDS = 1 2 3 4 5 6
CARD = $$([ $$s -le 6 ] && echo 0 || echo 1)

.PHONY: all runs frozen

all: runs frozen

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) --seed $$s --set name=RT -o $(OUT)/RT_s$$s > $(OUT)/RT_s$$s.log 2>&1 & \
	  done; wait; }

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(FROZEN) --init $(OUT)/RT_s$$s/ckpt_$$(printf %08d $$t) --seed 10$$s \
	      -o $$o/RT_s$$s > $$o/RT_s$$s.log 2>&1 & \
	  done; wait; \
	done
