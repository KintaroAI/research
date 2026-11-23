# Makefile for experiment vm-00106-blind-spot-teacher
#
# Reproduces the runs of task 00118, the second experiment: the blind spot of vm-00105 (a 12 x 12 px box, x 38-49,
# y 16-27, black in everything the model sees) now in the targets too, as the eye never sees it. Two arms, 12 seeds,
# 60k ticks, the default agent (configs/U_agent.json):
#   G0 — online.blind_target = black: the targets have the gap as the model sees it (the null);
#   GB — online.blind_target = motion: each step's gap filled from a canvas of the world painted only with what was
#        seen outside the gap (GapTeacher, the body's position from the world as a training signal).
# Seeds 1-4 of both arms on the RTX 3090 (GPU 0 under CUDA_DEVICE_ORDER=PCI_BUS_ID), 5-12 on the RTX 4090 (GPU 1),
# under MPS. vm-00105's BL (whole targets) and TT (no gap) are the comparisons, not rerun.
# Outputs go to ~/data/research/visual-motor-agent/exp_00106/.
#
# Usage:
#   make runs — G0 and GB, seeds 1-12
#   make frozen — (after the runs) per checkpoint (20k, 40k, 60k) and life: 2000 ticks with learning off in new worlds
#                 (seed 100 + s), the gap in, recording the predicted frame (G0_gap, GB_gap; as vm-00105's BL_gap)
#   make measure — eval_blind.py over every frozen life (exp_00106/blind.jsonl)
#   make video — eval_blind.py --video: GB_gap and G0_gap seed 1 at 60k, 600 ticks
#   Options: MPS= (no MPS daemon), T="20000 60000"

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00106
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
T ?= 20000 40000 60000
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
SET = --config configs/U_agent.json
GAP = 38 16 12 12
BLIND = --set "online.blind=[38,16,12,12]"
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000 --set online.pred_record=2000
SEEDS = 1 2 3 4 5 6 7 8 9 10 11 12
CARD = $$([ $$s -le 4 ] && echo 0 || echo 1)
TARGET = --set online.blind_target=$$([ $$a = GB ] && echo motion || echo black)

.PHONY: all runs frozen measure video

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in $(SEEDS); do for a in G0 GB; do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(BLIND) $(TARGET) --seed $$s --set name=$$a -o $(OUT)/$${a}_s$$s \
	      > $(OUT)/$${a}_s$$s.log 2>&1 & \
	  done; done; wait; }

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; c=$$(printf %08d $$t); \
	  for s in $(SEEDS); do for a in G0 GB; do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(SET) $(FROZEN) $(BLIND) --init $(OUT)/$${a}_s$$s/ckpt_$$c --seed 10$$s \
	      -o $$o/$${a}_gap_s$$s > $$o/$${a}_gap_s$$s.log 2>&1 & \
	  done; done; wait; \
	done

measure:
	rm -f $(OUT)/blind.jsonl
	cd $(DEV) && for t in $(T); do for c in G0_gap GB_gap; do \
	  $(PY) eval_blind.py $(OUT)/frozen$$((t / 1000))k/$${c}_s*/ --gap $(GAP) --label $${c}_$$((t / 1000))k --out $(OUT)/blind.jsonl > /dev/null; \
	done; done

video:
	mkdir -p $(D)/videos/vm106
	cd $(DEV) && for c in GB_gap G0_gap; do \
	  $(PY) eval_blind.py --video $(D)/videos/vm106/$${c}_s1_60k.mp4 $(OUT)/frozen60k/$${c}_s1 --gap $(GAP) --ticks 600; \
	done
