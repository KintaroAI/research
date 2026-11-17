# Makefile for experiment vm-00096-instinct-horizon
#
# Reproduces the runs of task 00107: the default agent C8 (configs/U_view.json, value gamma .9 with n 4 and a fresh tail, 4 quarter
# transformers of 8 layers, tolerance .002, context 8) in the world with inertia from birth (env.inertia = 4), acting without the
# ring (online.ring = 0). IR: the instinct as before, one move ahead. IH: the instinct's horizon, each of its 16 candidate moves held
# for 4 ticks, the 3 ticks after the first imagined by the model (online.instinct_h = 4, one Jacobi pass a step with pair decoding),
# scored by the predicted label after the last. vm-00095's IB (the same agent with the ring) is linked, not rerun. Twelve seeds per
# arm, seeds 1-$(SPLIT) of both on the RTX 4090, the rest on the RTX 3090, under MPS. 60k ticks. Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00096/.
#
# Usage:
#   make runs — the 24 lives at once (SPLIT = the last seed on GPU 1, the 4090; CUDA_DEVICE_ORDER=PCI_BUS_ID)
#   make frozen — eval_hunger.py chain and move (1,000 ticks x 16 bodies) on IR and IH, value and reach on IR only, each agent
#                 acting as it lived (full windows, its own horizon), without MPS, two per card (frozen_a = seeds 1-6 on GPU 1,
#                 frozen_b = seeds 7-12 on GPU 0)
#   make show — IH against IR, paired by seed (frozen chain); IR's value and reach
#   make videos — compare_video.py, IB (vm-00095), IR and IH of seeds 5 and 12, each view with the whole world beside it

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00096
IB = $(HOME)/data/research/visual-motor-agent/exp_00095
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set online.value_gamma=0.9 --set online.value_refresh=1 --set env.inertia=4 --set online.ring=0
H4 = --set online.instinct_h=4 --set online.instinct_jacobi=1 --set online.instinct_decode=pair
SPLIT ?= 8   # timing (300-tick lives, 24 at once, 2026-11-17): IR 2.1 ticks/s per life on the 4090, 2.5 on the 3090; IH (J = 2) 1.4 and 1.6

.PHONY: all runs frozen frozen_a frozen_b show videos

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le $(SPLIT) ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --seed $$s --set name=IR -o $(OUT)/IR_s$$s > $(OUT)/IR_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(H4) --seed $$s --set name=IH -o $(OUT)/IH_s$$s > $(OUT)/IH_s$$s.log 2>&1 & \
	  done; wait; }

A = $$(for s in 1 2 3 4 5 6; do echo $(OUT)/IH_s$$s $(OUT)/IR_s$$s; done | tr "\n" " ")
B = $$(for s in 7 8 9 10 11 12; do echo $(OUT)/IH_s$$s $(OUT)/IR_s$$s; done | tr "\n" " ")
RA = $$(for s in 1 2 3 4 5 6; do echo $(OUT)/IR_s$$s; done | tr "\n" " ")
RB = $$(for s in 7 8 9 10 11 12; do echo $(OUT)/IR_s$$s; done | tr "\n" " ")
EVAL = cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID &&

frozen:
	$(MAKE) frozen_a & $(MAKE) frozen_b & wait

frozen_a:
	$(EVAL) a="$(A)" && r="$(RA)" && { \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1000 > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py move $$a --ticks 1000 > $(OUT)/move_a.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$r --ticks 1000 > $(OUT)/value_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$r --ticks 1000 > $(OUT)/reach_a.jsonl & \
	  wait; }

frozen_b:
	$(EVAL) b="$(B)" && r="$(RB)" && { \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b --ticks 1000 > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py move $$b --ticks 1000 > $(OUT)/move_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$r --ticks 1000 > $(OUT)/value_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$r --ticks 1000 > $(OUT)/reach_b.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/chain_a.jsonl $(OUT)/chain_b.jsonl --base IR
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/value_a.jsonl $(OUT)/value_b.jsonl $(OUT)/reach_a.jsonl $(OUT)/reach_b.jsonl

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { for s in 5 12; do \
	  if [ $$s = 5 ]; then d=1; else d=0; fi; \
	  CUDA_VISIBLE_DEVICES=$$d $(PY) compare_video.py $(IB)/IB_s$$s:"IB (ring, one move)" $(OUT)/IR_s$$s:"IR (no ring, one move)" \
	    $(OUT)/IH_s$$s:"IH (no ring, 4-tick courses)" --steps 1000 --worlds 8 --cols 2 \
	    -o $(HOME)/data/research/visual-motor-agent/videos/vm96_s$$s & \
	done; wait; }
