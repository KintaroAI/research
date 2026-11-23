# Makefile for experiment vm-00104-breakout-bricks
#
# Reproduces the runs of task 00117: Breakout stage 2 (configs/U_breakout2.json: a ball at 2.25 px a tick, no food for a
# paddle hit, 24 bricks of 3 looks feeding .1, .25 and .5, refilled with no slot keeping its type). Two arms, 12 seeds,
# 60k ticks: BK the default agent unchanged, BS the same with online.meal_size = .5 (a meal worth its size). Which look
# feeds how much rotates with the seed: env.brick_shift = seed mod 3, the same in every test of that seed.
# Seeds 1-4 (both arms) on the RTX 3090 (GPU 0 under CUDA_DEVICE_ORDER=PCI_BUS_ID), 5-12 on the RTX 4090 (GPU 1), under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00104/.
#
# Usage:
#   make runs — BK and BS, seeds 1-12, 8 lives on the 3090 and 16 on the 4090
#   make walkers — random, still, tracker and aimer paddles, seeds 1-12, 20k ticks each (no model)
#   make frozen — (after the runs) per checkpoint (10k ... 60k) and life: 2000 ticks with learning off (ppo.lr = 0) in new
#                 worlds (seed 100 + s)
#   make video — frames_video.py of BK and BS seed 1 at 60k, frozen, 900 ticks each, from the same court
#   Options: MPS= (no MPS daemon), T="10000 30000" (frozen's checkpoints)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00104
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
T ?= 10000 20000 30000 40000 50000 60000
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
SET = --config configs/U_breakout2.json
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000
SEEDS = 1 2 3 4 5 6 7 8 9 10 11 12
# per seed s: the card, the look shift; per arm: its setting
CARD = $$([ $$s -le 4 ] && echo 0 || echo 1)
SHIFT = --set env.brick_shift=$$((s % 3))
ARM = $$([ $$a = BS ] && echo --set online.meal_size=0.5)

.PHONY: all runs walkers frozen video

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in $(SEEDS); do for a in BK BS; do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(SHIFT) $(ARM) --seed $$s --set name=$$a -o $(OUT)/$${a}_s$$s \
	      > $(OUT)/$${a}_s$$s.log 2>&1 & \
	  done; done; wait; }

walkers:
	mkdir -p $(OUT)
	cd $(DEV) && $(PY) breakout_walkers.py $(SET) --seeds $(SEEDS) --steps 20000 --out $(OUT)/walkers.json > $(OUT)/walkers.txt 2>&1

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; \
	  for s in $(SEEDS); do for a in BK BS; do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(SHIFT) $(ARM) $(FROZEN) \
	      --init $(OUT)/$${a}_s$$s/ckpt_$$(printf %08d $$t) --seed 10$$s -o $$o/$${a}_s$$s > $$o/$${a}_s$$s.log 2>&1 & \
	  done; done; wait; \
	done

video:
	mkdir -p $(D)/videos/vm104
	cd $(DEV) && s=1; for a in BK BS; do \
	  env $(G) CUDA_VISIBLE_DEVICES=1 $(PY) lifetime.py $(SET) $(SHIFT) $(ARM) $(FROZEN) --set online.steps=900 --set online.frame_record=900 \
	    --init $(OUT)/$${a}_s1/ckpt_00060000 --seed 201 -o $(OUT)/video_$${a}_s1 > $(OUT)/video_$${a}_s1.log 2>&1 & \
	done; wait; \
	$(PY) frames_video.py $(OUT)/video_BK_s1:"BK seed 1, 60k" $(OUT)/video_BS_s1:"BS seed 1, 60k" -o $(D)/videos/vm104/play_60k.mp4 --fps 15
