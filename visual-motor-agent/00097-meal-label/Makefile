# Makefile for experiment vm-00097-meal-label
#
# Reproduces the runs of task 00108: the default agent C8 (configs/U_view.json, 4 quarter transformers of 8 layers, tolerance .002,
# context 8, the ring) with the meal label in place of the hunger value. A meal (hunger falls) is worth 1; every other tick costs the
# hunger it adds (a poison's 0.5, the drift's 0.001) plus 0.95 x the transformer's own fresh prediction for the next tick (one step:
# online.value_target = meal, value_gamma 0.95, value_n 1, value_refresh 1). In the world with inertia from birth (MB, against
# vm-00095's IB) and without (M0, against vm-00094's C8r); then (runs2) the same with the label weighted by hunger when choosing
# (online.need = 1: the instinct's tolerance over max(hunger, .05)), MBN and M0N. Twelve seeds per arm, seeds 1-$(SPLIT) on the
# RTX 4090, the rest on the RTX 3090, under MPS, 60k ticks. Each life logs the label's predictions against what happened, by ticks
# to the next meal (metrics "meal"). Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to
# ~/data/research/visual-motor-agent/exp_00097/.
#
# Usage:
#   make runs — MB and M0, 24 lives at once (SPLIT = the last seed on GPU 1, the 4090; CUDA_DEVICE_ORDER=PCI_BUS_ID)
#   make runs2 — MBN and M0N, 24 lives at once
#   make frozen — eval_hunger.py chain (1,500 ticks), value (2,000, with the meal label's profile by ticks to the next meal),
#                 reach (1,500) and move (1,500) on MB and M0, without MPS, two per card (frozen_a = seeds 1-6 on GPU 1,
#                 frozen_b = seeds 7-12 on GPU 0)
#   make frozen2 — the same on MBN and M0N (files chain2_a.jsonl and so on)
#   make videos — compare_video.py of seeds 5 and 12: IB and MB in the world with inertia, C8r and M0 without

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00097
IB = $(HOME)/data/research/visual-motor-agent/exp_00095
C8R = $(HOME)/data/research/visual-motor-agent/exp_00094
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set online.value_gamma=0.95 --set online.value_n=1 --set online.value_refresh=1 --set online.value_target=meal
INERTIA = --set env.inertia=4
NEED = --set online.need=1
SPLIT ?= 8   # vm-00095's split for 24 lives of this agent (4.4 ticks/s per life, 3.8 h)
ARMS ?= MB M0
SUF ?=

.PHONY: all runs runs2 frozen frozen2 frozen_a frozen_b videos

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le $(SPLIT) ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(INERTIA) --seed $$s --set name=MB -o $(OUT)/MB_s$$s > $(OUT)/MB_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --seed $$s --set name=M0 -o $(OUT)/M0_s$$s > $(OUT)/M0_s$$s.log 2>&1 & \
	  done; wait; }

runs2:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le $(SPLIT) ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(NEED) $(INERTIA) --seed $$s --set name=MBN -o $(OUT)/MBN_s$$s > $(OUT)/MBN_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(NEED) --seed $$s --set name=M0N -o $(OUT)/M0N_s$$s > $(OUT)/M0N_s$$s.log 2>&1 & \
	  done; wait; }

A = $$(for s in 1 2 3 4 5 6; do for a in $(ARMS); do echo $(OUT)/$${a}_s$$s; done; done | tr "\n" " ")
B = $$(for s in 7 8 9 10 11 12; do for a in $(ARMS); do echo $(OUT)/$${a}_s$$s; done; done | tr "\n" " ")
EVAL = cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID &&

frozen:
	$(MAKE) frozen_a & $(MAKE) frozen_b & wait

frozen2:
	$(MAKE) frozen ARMS="MBN M0N" SUF=2

frozen_a:
	$(EVAL) a="$(A)" && { \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1500 > $(OUT)/chain$(SUF)_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value$(SUF)_a.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$a --ticks 1500 > $(OUT)/reach$(SUF)_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py move $$a --ticks 1500 > $(OUT)/move$(SUF)_a.jsonl & \
	  wait; }

frozen_b:
	$(EVAL) b="$(B)" && { \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b --ticks 1500 > $(OUT)/chain$(SUF)_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value$(SUF)_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$b --ticks 1500 > $(OUT)/reach$(SUF)_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py move $$b --ticks 1500 > $(OUT)/move$(SUF)_b.jsonl & \
	  wait; }

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { for s in 5 12; do \
	  if [ $$s = 5 ]; then d=1; else d=0; fi; \
	  { CUDA_VISIBLE_DEVICES=$$d $(PY) compare_video.py $(IB)/IB_s$$s:"IB (hunger value)" $(OUT)/MB_s$$s:"MB (meal label)" \
	      --steps 1000 --worlds 8 --cols 2 -o $(HOME)/data/research/visual-motor-agent/videos/vm97_inertia_s$$s; \
	    CUDA_VISIBLE_DEVICES=$$d $(PY) compare_video.py $(C8R)/C8r_s$$s:"C8r (hunger value)" $(OUT)/M0_s$$s:"M0 (meal label)" \
	      --steps 1000 --worlds 8 --cols 2 -o $(HOME)/data/research/visual-motor-agent/videos/vm97_plain_s$$s; } & \
	done; wait; }
