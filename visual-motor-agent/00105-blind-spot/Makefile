# Makefile for experiment vm-00105-blind-spot
#
# Reproduces the runs of task 00118, the first experiment: the default agent (configs/U_agent.json) with a blind spot,
# a 12 x 12 px box of its 64 x 64 frame (x 38-49, y 16-27) black in everything the model sees, the same for all four
# transformers; the world, memory and targets whole (online.blind). One arm, BL, 12 seeds, 60k ticks; the control is
# vm-00101's TT (the default agent, 12 seeds), not rerun. Seeds 1-4 on the RTX 3090 (GPU 0 under
# CUDA_DEVICE_ORDER=PCI_BUS_ID), 5-12 on the RTX 4090 (GPU 1), under MPS. Outputs go to
# ~/data/research/visual-motor-agent/exp_00105/.
#
# Usage:
#   make runs — BL, seeds 1-12
#   make frozen — (after the runs) per checkpoint (20k, 40k, 60k) and seed, 2000 ticks with learning off (ppo.lr = 0) in
#                 new worlds (seed 100 + s), recording the predicted frame (online.pred_record): BL with the gap (BL_gap) and
#                 without (BL_open), TT with the gap (TT_gap) and without (TT_open)
#   make measure — eval_blind.py over every frozen life (exp_00105/blind.jsonl)
#   make video — eval_blind.py --video: BL_gap and TT_gap seed 1 at 60k, 600 ticks (what it sees, its prediction, the truth)
#   Options: MPS= (no MPS daemon), T="20000 60000" (frozen's checkpoints)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00105
TT = $(D)/exp_00101
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
T ?= 20000 40000 60000
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
SET = --config configs/U_agent.json
GAP = 38 16 12 12
BLIND = --set "online.blind=[38,16,12,12]"
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000 --set online.pred_record=2000
SEEDS = 1 2 3 4 5 6 7 8 9 10 11 12
CARD = $$([ $$s -le 4 ] && echo 0 || echo 1)

.PHONY: all runs frozen measure video

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(BLIND) --seed $$s --set name=BL -o $(OUT)/BL_s$$s > $(OUT)/BL_s$$s.log 2>&1 & \
	  done; wait; }

# per checkpoint two batches of 24 lives: BL with and without the gap, then TT with and without it
frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; c=$$(printf %08d $$t); \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(FROZEN) $(BLIND) --init $(OUT)/BL_s$$s/ckpt_$$c --seed 10$$s -o $$o/BL_gap_s$$s > $$o/BL_gap_s$$s.log 2>&1 & \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(FROZEN) --init $(OUT)/BL_s$$s/ckpt_$$c --seed 10$$s -o $$o/BL_open_s$$s > $$o/BL_open_s$$s.log 2>&1 & \
	  done; wait; \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(FROZEN) $(BLIND) --init $(TT)/TT_s$$s/ckpt_$$c --seed 10$$s -o $$o/TT_gap_s$$s > $$o/TT_gap_s$$s.log 2>&1 & \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(FROZEN) --init $(TT)/TT_s$$s/ckpt_$$c --seed 10$$s -o $$o/TT_open_s$$s > $$o/TT_open_s$$s.log 2>&1 & \
	  done; wait; \
	done

measure:
	rm -f $(OUT)/blind.jsonl
	cd $(DEV) && for t in $(T); do for c in BL_gap BL_open TT_gap TT_open; do \
	  $(PY) eval_blind.py $(OUT)/frozen$$((t / 1000))k/$${c}_s* --gap $(GAP) --label $${c}_$$((t / 1000))k --out $(OUT)/blind.jsonl > /dev/null; \
	done; done

video:
	mkdir -p $(D)/videos/vm105
	cd $(DEV) && for c in BL_gap TT_gap; do \
	  $(PY) eval_blind.py --video $(D)/videos/vm105/$${c}_s1_60k.mp4 $(OUT)/frozen60k/$${c}_s1 --gap $(GAP) --ticks 600; \
	done
