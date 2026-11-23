# Makefile for experiment vm-00103-breakout
#
# Reproduces the runs of task 00111: the default agent (configs/U_agent.json, vm-00101's TT) in a Breakout court, the
# model unchanged (configs/U_breakout.json: U_agent.json with the env block replaced). One arm, BD, 12 seeds, 60k ticks:
# seeds 1-6 on the RTX 3090 (GPU 0 under CUDA_DEVICE_ORDER=PCI_BUS_ID), 7-12 on the RTX 4090 (GPU 1), under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00103/.
#
# Usage:
#   make runs — BD, seeds 1-12, 6 lives on each card
#   make walkers — random, still and tracker paddles, seeds 1-12, 20k ticks each (no model)
#   make frozen — (after the runs; 6 lives at a time per card) per checkpoint (10k ... 60k) and life: 2000 ticks with
#                 learning off (ppo.lr = 0) in new worlds (seed 100 + s)
#   make video — frames_video.py of seeds 1-3 at 60k, frozen, 600 ticks each
#   Options: MPS= (no MPS daemon), T="10000 30000" (frozen's checkpoints)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00103
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
T ?= 10000 20000 30000 40000 50000 60000
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
SET = --config configs/U_breakout.json
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000

.PHONY: all runs walkers frozen video

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    g=$$([ $$s -le 6 ] && echo 0 || echo 1); \
	    env $(G) CUDA_VISIBLE_DEVICES=$$g $(PY) lifetime.py $(SET) --seed $$s --set name=BD -o $(OUT)/BD_s$$s > $(OUT)/BD_s$$s.log 2>&1 & \
	  done; wait; }

walkers:
	mkdir -p $(OUT)
	cd $(DEV) && $(PY) breakout_walkers.py $(SET) --seeds 1 2 3 4 5 6 7 8 9 10 11 12 --steps 20000 --out $(OUT)/walkers.json \
	  > $(OUT)/walkers.txt 2>&1

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    g=$$([ $$s -le 6 ] && echo 0 || echo 1); \
	    env $(G) CUDA_VISIBLE_DEVICES=$$g $(PY) lifetime.py $(SET) $(FROZEN) --init $(OUT)/BD_s$$s/ckpt_$$(printf %08d $$t) --seed 10$$s \
	      -o $$o/BD_s$$s > $$o/BD_s$$s.log 2>&1 & \
	  done; wait; \
	done

video:
	mkdir -p $(D)/videos/vm103
	cd $(DEV) && for s in 1 2 3; do \
	  env $(G) CUDA_VISIBLE_DEVICES=1 $(PY) lifetime.py $(SET) $(FROZEN) --set online.steps=600 --set online.frame_record=600 \
	    --init $(OUT)/BD_s$$s/ckpt_00060000 --seed 20$$s -o $(OUT)/video_BD_s$$s > $(OUT)/video_BD_s$$s.log 2>&1 & \
	done; wait; \
	$(PY) frames_video.py $(OUT)/video_BD_s1:"BD seed 1, 60k" $(OUT)/video_BD_s2:"BD seed 2, 60k" $(OUT)/video_BD_s3:"BD seed 3, 60k" \
	  -o $(D)/videos/vm103/play_60k.mp4 --fps 15
