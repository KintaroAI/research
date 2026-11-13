# Makefile for experiment vm-00090-big-world-depth
#
# Reproduces the runs of task 00101: vm-00089's K4D8T (KB) and V90RD8T (VB) in vm-00072's big world (configs/U_big.json: 128 x 128, 4 foods, 8 poisons,
# 32 dots): four quarter transformers of 8 layers, online.instinct_tol = 0.002, K4's label or the fresh value (gamma 0.9). Six seeds, each seed's two
# lives on one card: seeds 1-4 on the RTX 4090, 5-6 on the RTX 3090 (the 4090 is about twice as fast). 60k ticks. Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00090/.
#
# Usage:
#   make runs — KB and VB seeds 1-4 on GPU 1 (CUDA_DEVICE_ORDER=PCI_BUS_ID: the 4090), seeds 5-6 on GPU 0 (the 3090); 12 lives at once
#   make frozen — eval_hunger.py chain, value and reach on every run, without MPS, two per card; chain and reach at 1,500 ticks, as vm-00072
#   make show — KB and VB paired by seed, with vm-00072's 2-layer K4 (tolerance 0.005) beside them

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00090
O72 = $(HOME)/data/research/visual-motor-agent/exp_00072
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_big.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002
KB =
VB = --set online.value_gamma=0.9 --set online.value_refresh=1

.PHONY: all runs frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6; do \
	    if [ $$s -le 4 ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(VB) --seed $$s --set name=VB -o $(OUT)/VB_s$$s > $(OUT)/VB_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(KB) --seed $$s --set name=KB -o $(OUT)/KB_s$$s > $(OUT)/KB_s$$s.log 2>&1 & \
	  done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  a="$$(for s in 1 2 3 4; do echo $(OUT)/VB_s$$s $(OUT)/KB_s$$s; done | tr "\n" " ")"; \
	  b="$$(for s in 5 6; do echo $(OUT)/VB_s$$s $(OUT)/KB_s$$s; done | tr "\n" " ")"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1500 > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b --ticks 1500 > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$a --ticks 1500 > $(OUT)/reach_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$b --ticks 1500 > $(OUT)/reach_b.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(O72)/hunger_chain.jsonl $(OUT)/chain_a.jsonl $(OUT)/value_a.jsonl $(OUT)/chain_b.jsonl $(OUT)/value_b.jsonl --base KB
	cd $(DEV) && $(PY) eval_hunger.py show $(O72)/hunger_reach.jsonl $(OUT)/reach_a.jsonl $(OUT)/reach_b.jsonl
