# Makefile for experiment vm-00094-flexible-layer
#
# Reproduces the runs of task 00105: vm-00091's value agent (VW) in the same world (configs/U_view.json: 128 x 128, 2 foods, a 64 x 64
# window), with the flexible layer W = S + F (online.fast = 0.99, online.fast_k = 1): F learns from the newest window's gradient, a
# clipped Adam-sized step, and decays by .99 a tick; S learns as before. Memory is measured in life (online.mem_score = 1: the aim at
# unseen food by ticks since the last sight, against a blind world stepped with the same moves), so the control is vm-00091's VW rerun
# with the same measure (C8r). Twelve seeds per arm, seeds 1-$(SPLIT) of both arms on the RTX 4090, the rest on the RTX 3090, under MPS.
# 60k ticks. Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to
# ~/data/research/visual-motor-agent/exp_00094/.
#
# Usage:
#   make runs — the 24 lives at once (SPLIT = the last seed on GPU 1, the 4090; CUDA_DEVICE_ORDER=PCI_BUS_ID)
#   make frozen — eval_hunger.py chain (1,500 ticks), value and reach on F and C8r, without MPS, two per card (frozen_a = seeds 1-6
#                 on GPU 1, frozen_b = seeds 7-12 on GPU 0). F is frozen at its value at 60k, so these do not show F's memory.
#   make show — F against C8r, paired by seed (frozen)
#   make videos — compare_video.py, C8r and F of seeds 5 and 12, each view with the whole world beside it

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00094
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set online.value_gamma=0.9 --set online.value_refresh=1 --set online.mem_score=1
SPLIT ?= 8   # timing (600-tick lives, 2026-11-16): 5.15 ticks/s per life with 14 on the 4090, 3.97 with 10 on the 3090

.PHONY: all runs frozen frozen_a frozen_b show videos

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le $(SPLIT) ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --seed $$s --set name=C8r -o $(OUT)/C8r_s$$s > $(OUT)/C8r_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --set online.fast=0.99 --set online.fast_k=1 --seed $$s --set name=F -o $(OUT)/F_s$$s > $(OUT)/F_s$$s.log 2>&1 & \
	  done; wait; }

A = $$(for s in 1 2 3 4 5 6; do echo $(OUT)/F_s$$s $(OUT)/C8r_s$$s; done | tr "\n" " ")
B = $$(for s in 7 8 9 10 11 12; do echo $(OUT)/F_s$$s $(OUT)/C8r_s$$s; done | tr "\n" " ")
EVAL = cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID &&

frozen:
	$(MAKE) frozen_a & $(MAKE) frozen_b & wait

frozen_a:
	$(EVAL) a="$(A)" && { \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1500 > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$a --ticks 1500 > $(OUT)/reach_a.jsonl; }

frozen_b:
	$(EVAL) b="$(B)" && { \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b --ticks 1500 > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$b --ticks 1500 > $(OUT)/reach_b.jsonl; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/chain_a.jsonl $(OUT)/chain_b.jsonl $(OUT)/value_a.jsonl $(OUT)/value_b.jsonl --base C8r
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/reach_a.jsonl $(OUT)/reach_b.jsonl

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && for s in 5 12; do \
	  CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py $(OUT)/C8r_s$$s:"C8r (no F)" $(OUT)/F_s$$s:"F (flexible layer)" --steps 1000 --worlds 8 \
	    --cols 2 -o $(HOME)/data/research/visual-motor-agent/videos/vm94_s$$s; done
