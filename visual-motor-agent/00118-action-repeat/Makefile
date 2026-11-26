# Makefile for experiment vm-00118-action-repeat
#
# Task 00131: action repeat (online.act_every = 4) for the default agent in Breakout stage 1: a move is decided every
# 4 ticks and made again on the 3 between; the model still sees, predicts and learns every tick. Arm BA, seeds 1-12,
# 60k ticks, against BD (vm-00103), paired by seed. It follows vm-00116's autopsy: BD's label knows the court as a state
# but orders two candidate moves correctly only .60-.62 of the time, and its paddle travels enough but not to the ball.
# GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS. As run: all 12 on the RTX 4090
# (BD's seeds 1-6 ran on the 3090, 7-12 on the 4090).
# Outputs go to ~/data/research/visual-motor-agent/exp_00118/.
#
# Usage:
#   make runs CARD=1 SEEDS="1 2 3 4 5 6"   — the lives
#   make frozen CARD=1 SEEDS="..."         — BA's 60k weights, learning off (ppo.lr = 0), 6000 ticks in new courts
#                                            (seed 100 + s), acting as they lived (vm-00116's ring arm is BD's)
#   make autopsy CARD=1                    — vm-00116's miss autopsy of BA's 60k weights (16 courts x 1500 ticks a seed)
#   make read                              — its tables (the label's order is scored by one tick of each candidate, as
#                                            for BD, and by where the hold leaves the paddle)
#   make table                             — the README's tables of the lives and the learning-off tests
#                                            (dev/analyze_lives.py repeat; no GPU)
#   Options: MPS= (no MPS daemon), K=4 (act_every)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00118
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
SEEDS ?= 1 2 3 4 5 6 7 8 9 10 11 12
K ?= 4
BA = --config configs/U_breakout.json --set online.act_every=$(K)
FROZEN = --set ppo.lr=0 --set online.steps=6000 --set online.log_every=100 --set online.save_every=100000

.PHONY: runs frozen autopsy read table

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in $(SEEDS); do \
	  env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(BA) --seed $$s --set name=BA \
	    -o $(OUT)/BA_s$$s > $(OUT)/BA_s$$s.log 2>&1 & \
	done; wait; }

frozen:
	mkdir -p $(OUT)/frozen60k
	cd $(DEV) && { for s in $(SEEDS); do \
	  env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(BA) $(FROZEN) --init $(OUT)/BA_s$$s/ckpt_00060000 --seed 10$$s \
	    -o $(OUT)/frozen60k/s$$s > $(OUT)/frozen60k/s$$s.log 2>&1 & \
	done; wait; }; for s in $(SEEDS); do rm -f $(OUT)/frozen60k/s$$s/ckpt_*.bin; done

autopsy:
	mkdir -p $(OUT)/autopsy
	cd $(DEV) && for part in "1 2 3 4" "5 6 7 8" "9 10 11 12"; do \
	  for s in $$part; do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) analyze_misses.py record $(OUT)/autopsy/s$$s.npz $(OUT)/BA_s$$s --step 60000 \
	      --ticks 1500 --bodies 16 --seed 10$$s > $(OUT)/autopsy/s$$s.log 2>&1 & \
	  done; wait; \
	done

read:
	cd $(DEV) && $(PY) analyze_misses.py read $(OUT)/autopsy/s*.npz

table:
	cd $(DEV) && $(PY) analyze_lives.py repeat --seeds $(SEEDS)
