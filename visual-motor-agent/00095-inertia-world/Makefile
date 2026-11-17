# Makefile for experiment vm-00095-inertia-world
#
# Reproduces the runs of task 00106: vm-00091's value agent C8 (configs/U_view.json: 128 x 128, 2 foods, a 64 x 64 window; value
# gamma .9 with n 4 and a fresh tail, 4 quarter transformers of 8 layers, tolerance .002, context 8), unchanged, in a world with
# inertia (env.inertia = 4: the move pushes a body that keeps its velocity, 4 ticks of a full push from rest to full speed, 8 to
# reverse; with no push it keeps moving). IB: inertia from birth. IS: no inertia until tick 30k, then inertia 4 (online.env_sched).
# The control is vm-00094's C8r (the same agent and code in the world without inertia), read from exp_00094. Twelve seeds per arm,
# seeds 1-$(SPLIT) of both arms on the RTX 4090, the rest on the RTX 3090, under MPS. 60k ticks. Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00095/.
#
# Usage:
#   make runs — the 24 lives at once (SPLIT = the last seed on GPU 1, the 4090; CUDA_DEVICE_ORDER=PCI_BUS_ID)
#   make frozen — eval_hunger.py chain (1,500 ticks), value and reach on IB and IS, each in the world its life ended in (inertia 4),
#                 and chain on vm-00094's C8r in the inertia world (--env inertia=4, rows C8r-inertia4_sN: a model raised without
#                 inertia, not learning), then eval_hunger.py move (how the body moves: speed, braking, speed and aim by the gap to
#                 food and poison) on IB, IS, C8r in its own world and C8r in the inertia world; without MPS, two per card
#                 (frozen_a = seeds 1-6 on GPU 1, frozen_b = seeds 7-12 on GPU 0)
#   make frozen2 — added during the analysis: IS at tick 30k frozen in the inertia world (chain, move), and the mean hunger of every
#                  arm (eval_hunger.py poison); one process per card
#   make show — IS and C8r-inertia4 against IB, paired by seed (frozen)
#   make videos — compare_video.py, C8r (no inertia), IB and IS of seeds 5 and 12, each view with the whole world beside it
#   make videos2 — what learning with inertia changed: C8r in its own world and frozen in the inertia world, IS at 30k and at 60k

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00095
C8R = $(HOME)/data/research/visual-motor-agent/exp_00094
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set online.value_gamma=0.9 --set online.value_refresh=1
SCHED = {"inertia": [[30000, 4]]}
SPLIT ?= 8   # vm-00094's split (24 lives of the same agent, with F and the memory measure: 4.4 h)

.PHONY: all runs frozen frozen_a frozen_b frozen2 show videos videos2

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le $(SPLIT) ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --set env.inertia=4 --seed $$s --set name=IB -o $(OUT)/IB_s$$s > $(OUT)/IB_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --set 'online.env_sched=$(SCHED)' --seed $$s --set name=IS -o $(OUT)/IS_s$$s > $(OUT)/IS_s$$s.log 2>&1 & \
	  done; wait; }

A = $$(for s in 1 2 3 4 5 6; do echo $(OUT)/IS_s$$s $(OUT)/IB_s$$s; done | tr "\n" " ")
B = $$(for s in 7 8 9 10 11 12; do echo $(OUT)/IS_s$$s $(OUT)/IB_s$$s; done | tr "\n" " ")
CA = $$(for s in 1 2 3 4 5 6; do echo $(C8R)/C8r_s$$s; done | tr "\n" " ")
CB = $$(for s in 7 8 9 10 11 12; do echo $(C8R)/C8r_s$$s; done | tr "\n" " ")
EVAL = cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID &&

frozen:
	$(MAKE) frozen_a & $(MAKE) frozen_b & wait

frozen_a:
	$(EVAL) a="$(A)" && c="$(CA)" && { \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1500 > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$a --ticks 1500 > $(OUT)/reach_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$c --ticks 1500 --env inertia=4 > $(OUT)/chain_c_a.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py move $$a --ticks 1500 > $(OUT)/move_a.jsonl & \
	  { CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py move $$c --ticks 1500 > $(OUT)/move_c0_a.jsonl; \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py move $$c --ticks 1500 --env inertia=4 > $(OUT)/move_c_a.jsonl; } & \
	  wait; }

frozen_b:
	$(EVAL) b="$(B)" && c="$(CB)" && { \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b --ticks 1500 > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$b --ticks 1500 > $(OUT)/reach_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$c --ticks 1500 --env inertia=4 > $(OUT)/chain_c_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py move $$b --ticks 1500 > $(OUT)/move_b.jsonl & \
	  { CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py move $$c --ticks 1500 > $(OUT)/move_c0_b.jsonl; \
	    CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py move $$c --ticks 1500 --env inertia=4 > $(OUT)/move_c_b.jsonl; } & \
	  wait; }

# Added during the analysis (task 00106, descriptive): IS's checkpoint at tick 30k (the model at the switch, before any learning with
# inertia) frozen in the inertia world, rows IS-step30000_sN, the control for what IS's 30k ticks of learning with inertia did; and
# every arm's mean hunger, the agent's own objective (eval_hunger.py poison): IB, IS, IS at 30k, C8r in the inertia world and in its own.
I = $$(for s in 1 2 3 4 5 6 7 8 9 10 11 12; do echo $(OUT)/IS_s$$s; done | tr "\n" " ")

frozen2:
	$(EVAL) i="$(I)" && ab="$(A) $(B)" && c="$(CA) $(CB)" && { \
	  { CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$i --ticks 1500 --step 30000 > $(OUT)/chain_s30.jsonl; \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py poison $$ab --ticks 1500 > $(OUT)/poison_ab.jsonl; \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py poison $$c --ticks 1500 > $(OUT)/poison_c0.jsonl; } & \
	  { CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py move $$i --ticks 1500 --step 30000 > $(OUT)/move_s30.jsonl; \
	    CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py poison $$i --ticks 1500 --step 30000 > $(OUT)/poison_s30.jsonl; \
	    CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py poison $$c --ticks 1500 --env inertia=4 > $(OUT)/poison_c.jsonl; } & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/chain_a.jsonl $(OUT)/chain_b.jsonl $(OUT)/chain_c_a.jsonl $(OUT)/chain_c_b.jsonl \
	  $(OUT)/value_a.jsonl $(OUT)/value_b.jsonl --base IB
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/reach_a.jsonl $(OUT)/reach_b.jsonl

# What learning with inertia changed (added after the write-up): C8r in its own world, C8r frozen in the inertia world, IS's
# checkpoint at 30k (the model at the switch) and IS at 60k, both in the inertia world; seed 5 on GPU 1, seed 12 on GPU 0
videos2:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { for s in 5 12; do \
	  if [ $$s = 5 ]; then d=1; else d=0; fi; \
	  CUDA_VISIBLE_DEVICES=$$d $(PY) compare_video.py $(C8R)/C8r_s$$s:"C8r (no inertia)" $(C8R)/C8r_s$$s+inertia=4:"C8r-in-inertia (raised without)" \
	    $(OUT)/IS_s$$s@30000:"IS-30k (at the switch)" $(OUT)/IS_s$$s:"IS-60k (after learning)" --steps 1000 --worlds 8 --cols 2 \
	    -o $(HOME)/data/research/visual-motor-agent/videos/vm95_adapt_s$$s & \
	done; wait; }

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && for s in 5 12; do \
	  CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py $(C8R)/C8r_s$$s:"C8r (no inertia)" $(OUT)/IB_s$$s:"IB (inertia from birth)" \
	    $(OUT)/IS_s$$s:"IS (inertia from 30k)" --steps 1000 --worlds 8 --cols 2 \
	    -o $(HOME)/data/research/visual-motor-agent/videos/vm95_s$$s; done
