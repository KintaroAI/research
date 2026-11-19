# Makefile for experiment vm-00098-stored-label
#
# Reproduces the runs of task 00109: the default agent C8 (configs/U_view.json, 4 quarter transformers of 8 layers, tolerance .002,
# context 8, the ring) with the owner's stored meal label as a body value (model.body_label = 1, online.label = meal,
# label_gamma 0.95). The body carries hunger, restlessness and the label, each its own channel; the label's input is blank (0.5)
# and the move token predicts it. When acting each transformer scores the move made; one tick later step t's label is written to
# memory once, never re-labeled: a meal 1, else 0.95 x that transformer's prediction for the next step's move, less the hunger a
# poison added. Hunger is plain hunger (no future_k, no value label). S0 without inertia (against vm-00097's M0 and vm-00094's
# C8r), SB with inertia from birth (against vm-00097's MB and vm-00095's IB). Twelve seeds per arm, seeds 1-$(SPLIT) on the RTX
# 4090, the rest on the RTX 3090, under MPS, 60k ticks. Each life logs the label's predictions and the stored labels against what
# happened, by ticks to the next meal (metrics "meal", "meal_stored"). Outputs go to ~/data/research/visual-motor-agent/exp_00098/.
#
# Usage:
#   make runs — S0 and SB, 24 lives at once (SPLIT = the last seed on GPU 1, the 4090; CUDA_DEVICE_ORDER=PCI_BUS_ID)
#   make frozen — eval_hunger.py chain (1,500 ticks), value (2,000, with the meal label's profile by ticks to the next meal),
#                 reach (1,500) and move (1,500), without MPS, two per card (frozen_a = seeds 1-6 on GPU 1, frozen_b = 7-12 on GPU 0)
#   make videos — compare_video.py of seeds 5 and 12: C8r, M0 and S0 without inertia; IB, MB and SB with it
#   make video_stuck — IB, MB and SB of seed 3, where SB ate almost nothing all life (GPU 1)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00098
M97 = $(HOME)/data/research/visual-motor-agent/exp_00097
IB = $(HOME)/data/research/visual-motor-agent/exp_00095
C8R = $(HOME)/data/research/visual-motor-agent/exp_00094
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_view.json --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set model.body_label=1 --set online.label=meal --set online.label_gamma=0.95
INERTIA = --set env.inertia=4
SPLIT ?= 8   # vm-00097's split for 24 lives of this agent (about 3.8 h)

.PHONY: all runs frozen frozen_a frozen_b videos video_stuck

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le $(SPLIT) ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --seed $$s --set name=S0 -o $(OUT)/S0_s$$s > $(OUT)/S0_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(INERTIA) --seed $$s --set name=SB -o $(OUT)/SB_s$$s > $(OUT)/SB_s$$s.log 2>&1 & \
	  done; wait; }

A = $$(for s in 1 2 3 4 5 6; do echo $(OUT)/S0_s$$s $(OUT)/SB_s$$s; done | tr "\n" " ")
B = $$(for s in 7 8 9 10 11 12; do echo $(OUT)/S0_s$$s $(OUT)/SB_s$$s; done | tr "\n" " ")
EVAL = cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID &&

frozen:
	$(MAKE) frozen_a & $(MAKE) frozen_b & wait

frozen_a:
	$(EVAL) a="$(A)" && { \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1500 > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$a --ticks 1500 > $(OUT)/reach_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py move $$a --ticks 1500 > $(OUT)/move_a.jsonl & \
	  wait; }

frozen_b:
	$(EVAL) b="$(B)" && { \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b --ticks 1500 > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$b --ticks 1500 > $(OUT)/reach_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py move $$b --ticks 1500 > $(OUT)/move_b.jsonl & \
	  wait; }

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { for s in 5 12; do \
	  if [ $$s = 5 ]; then d=1; else d=0; fi; \
	  { CUDA_VISIBLE_DEVICES=$$d $(PY) compare_video.py $(C8R)/C8r_s$$s:"C8r (hunger value)" $(M97)/M0_s$$s:"M0 (meal label, re-labeled)" \
	      $(OUT)/S0_s$$s:"S0 (meal label, stored, in the body)" \
	      --steps 1000 --worlds 8 --cols 2 -o $(HOME)/data/research/visual-motor-agent/videos/vm98_plain_s$$s; \
	    CUDA_VISIBLE_DEVICES=$$d $(PY) compare_video.py $(IB)/IB_s$$s:"IB (hunger value)" $(M97)/MB_s$$s:"MB (meal label, re-labeled)" \
	      $(OUT)/SB_s$$s:"SB (meal label, stored, in the body)" \
	      --steps 1000 --worlds 8 --cols 2 -o $(HOME)/data/research/visual-motor-agent/videos/vm98_inertia_s$$s; } & \
	done; wait; }

video_stuck:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
	  CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py $(IB)/IB_s3:"IB (hunger value)" $(M97)/MB_s3:"MB (meal label, re-labeled)" \
	    $(OUT)/SB_s3:"SB (meal label, stored, in the body)" \
	    --steps 1000 --worlds 8 --cols 2 -o $(HOME)/data/research/visual-motor-agent/videos/vm98_inertia_s3
