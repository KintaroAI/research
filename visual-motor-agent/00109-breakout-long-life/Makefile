# Makefile for experiment vm-00109-breakout-long-life
#
# Reproduces the run of task 00121: one long life in vm-00107's court (configs/U_breakout2p.json: stage 2, a ball at
# 2.25 px a tick, 24 bricks of 3 looks feeding .1, .25 and .5, a paddle hit .25): vm-00107's PS (online.meal_size = .5),
# seed 1 (env.brick_shift = 1), 1M ticks, checkpoints every 50k. On the RTX 4090 (GPU 1 under
# CUDA_DEVICE_ORDER=PCI_BUS_ID), under MPS. Outputs go to ~/data/research/visual-motor-agent/exp_00109/.
#
# Usage:
#   make run — the life (PS_s1)
#   make frozen — (after the run) per checkpoint: 2000 ticks with learning off (ppo.lr = 0) in 6 new courts (seeds 101-106)
#   make video — frames_video.py of the life at 1M beside vm-00107's PS seed 1 at 60k, frozen, 900 ticks, the same court
#   Options: MPS= (no MPS daemon), T="100000 1000000" (frozen's checkpoints)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00109
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
T ?= 100000 250000 500000 750000 1000000
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $(MPS)
SET = --config configs/U_breakout2p.json --set env.brick_shift=1 --set online.meal_size=0.5
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000
COURTS = 1 2 3 4 5 6

.PHONY: all run frozen video

all: run frozen video

run:
	mkdir -p $(OUT)
	cd $(DEV) && env $(G) $(PY) lifetime.py $(SET) --set online.steps=1000000 --set online.save_every=50000 --seed 1 --set name=PS \
	  -o $(OUT)/PS_s1 > $(OUT)/PS_s1.log 2>&1

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; \
	  for c in $(COURTS); do \
	    env $(G) $(PY) lifetime.py $(SET) $(FROZEN) --init $(OUT)/PS_s1/ckpt_$$(printf %08d $$t) --seed 10$$c \
	      -o $$o/PS_s1_c$$c > $$o/PS_s1_c$$c.log 2>&1 & \
	  done; wait; \
	done

video:
	mkdir -p $(D)/videos/vm109
	cd $(DEV) && env $(G) $(PY) lifetime.py $(SET) $(FROZEN) --set online.steps=900 --set online.frame_record=900 \
	    --init $(OUT)/PS_s1/ckpt_01000000 --seed 201 -o $(OUT)/video_PS_s1 > $(OUT)/video_PS_s1.log 2>&1 && \
	  $(PY) frames_video.py $(D)/exp_00107/video_PS_s1:"PS seed 1, 60k (vm-00107)" $(OUT)/video_PS_s1:"PS seed 1, 1M" \
	    -o $(D)/videos/vm109/play_1M.mp4 --fps 15
