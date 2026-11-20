# Makefile for experiment vm-00100-relabeled-label
#
# Reproduces the runs of task 00112: vm-00099's SD (the default agent C8 with the stored meal label as a body value and the
# full dose of poison, in the world with inertia 4 from birth) with online.label_refresh = 1: the label is not stored once
# but re-labeled at every update, each transformer from its current prediction for the next move (the label channel), by
# the stored label's rule. Twelve seeds on the RTX 3090 (GPU 0 under CUDA_DEVICE_ORDER=PCI_BUS_ID), under MPS, 60k ticks.
# Controls linked: SD (vm-00099), SB (vm-00098), MB (vm-00097), IB (vm-00095).
# Outputs go to ~/data/research/visual-motor-agent/exp_00100/.
#
# Usage:
#   make runs — RD, 12 lives at once on GPU 0
#   make frozen — eval_hunger.py chain (1,500 ticks), value (2,000), reach (1,500) and move (1,500), without MPS, two at a time
#   make candidates — eval_candidates.py on RD (300 ticks x 16 bodies)
#   make videos — compare_video.py of seeds 5 and 3: IB, MB, SD and RD

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00100
SD = $(HOME)/data/research/visual-motor-agent/exp_00099
M97 = $(HOME)/data/research/visual-motor-agent/exp_00097
IB = $(HOME)/data/research/visual-motor-agent/exp_00095
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe CUDA_VISIBLE_DEVICES=0
RUN = $(PY) lifetime.py --config configs/U_view.json --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set model.body_label=1 --set online.label=meal --set online.label_gamma=0.95 --set env.inertia=4 --set online.full_dose=1 \
  --set online.label_refresh=1

.PHONY: all runs frozen candidates videos

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    $(G) $(RUN) --seed $$s --set name=RD -o $(OUT)/RD_s$$s > $(OUT)/RD_s$$s.log 2>&1 & \
	  done; wait; }

A = $$(for s in 1 2 3 4 5 6; do echo $(OUT)/RD_s$$s; done | tr "\n" " ")
B = $$(for s in 7 8 9 10 11 12; do echo $(OUT)/RD_s$$s; done | tr "\n" " ")
EVAL = cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 &&

frozen:
	$(EVAL) a="$(A)" && b="$(B)" && { \
	  $(PY) eval_hunger.py chain $$a > $(OUT)/chain_a.jsonl & $(PY) eval_hunger.py chain $$b > $(OUT)/chain_b.jsonl & wait; \
	  $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & wait; \
	  $(PY) eval_hunger.py reach $$a > $(OUT)/reach_a.jsonl & $(PY) eval_hunger.py reach $$b > $(OUT)/reach_b.jsonl & wait; \
	  $(PY) eval_hunger.py move $$a > $(OUT)/move_a.jsonl & $(PY) eval_hunger.py move $$b > $(OUT)/move_b.jsonl & wait; }

candidates:
	$(EVAL) $(PY) eval_candidates.py $$(for s in 1 2 3 4 5 6 7 8 9 10 11 12; do echo $(OUT)/RD_s$$s; done) > $(OUT)/candidates.jsonl

videos:
	$(EVAL) for s in 5 3; do \
	  $(PY) compare_video.py $(IB)/IB_s$$s:"IB (hunger value)" $(M97)/MB_s$$s:"MB (meal label, re-labeled)" \
	    $(SD)/SD_s$$s:"SD (stored once, full dose)" $(OUT)/RD_s$$s:"RD (re-labeled, own channel, full dose)" \
	    --steps 1000 --worlds 8 --cols 2 -o $(HOME)/data/research/visual-motor-agent/videos/vm100_s$$s; \
	done
