# Makefile for experiment vm-00088-value-refresh-depth
#
# Reproduces the runs of task 00099: four quadrant transformers of 8 layers (model.n_layer=8) with vm-00087's fresh value, gamma 0.9, its tail the mean
# over the four transformers recomputed at every update (V90RD8), against K4's label at 8 layers (K4D8). Twelve seeds; each seed pair on one card:
# seeds 1-6 on the RTX 4090 (K4D8 1-6 are vm-00086's, not rerun), seeds 7-12 of both arms on the RTX 3090. 60k ticks. Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00088/.
#
# Usage:
#   make runs — V90RD8 seeds 1-6 on GPU 1 (CUDA_DEVICE_ORDER=PCI_BUS_ID: the 4090); V90RD8 and K4D8 seeds 7-12 on GPU 0 (the 3090); 18 lives at once
#   make frozen — eval_hunger.py chain and value on every new run, without MPS, two per card
#   make show — with vm-00086's files: K4D8 and V90RD8 over 12 seeds, against K4D8 paired by seed; then with vm-00086/87: 6 seeds against V90R and V90D8

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00088
O86 = $(HOME)/data/research/visual-motor-agent/exp_00086
O87 = $(HOME)/data/research/visual-motor-agent/exp_00087
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4 --set model.n_layer=8
K4D8 =
V90RD8 = --set online.value_gamma=0.9 --set online.value_refresh=1

.PHONY: all runs frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6; do \
	    $(G) CUDA_VISIBLE_DEVICES=1 $(RUN) $(V90RD8) --seed $$s --set name=V90RD8 -o $(OUT)/V90RD8_s$$s > $(OUT)/V90RD8_s$$s.log 2>&1 & \
	  done; \
	  for s in 7 8 9 10 11 12; do \
	    $(G) CUDA_VISIBLE_DEVICES=0 $(RUN) $(V90RD8) --seed $$s --set name=V90RD8 -o $(OUT)/V90RD8_s$$s > $(OUT)/V90RD8_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=0 $(RUN) $(K4D8) --seed $$s --set name=K4D8 -o $(OUT)/K4D8_s$$s > $(OUT)/K4D8_s$$s.log 2>&1 & \
	  done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  a="$$(for s in 1 2 3 4 5 6; do echo $(OUT)/V90RD8_s$$s; done | tr "\n" " ")"; \
	  b="$$(for s in 7 8 9 10 11 12; do echo $(OUT)/V90RD8_s$$s $(OUT)/K4D8_s$$s; done | tr "\n" " ")"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(O86)/chain.jsonl $(O86)/value.jsonl $(OUT)/chain_a.jsonl $(OUT)/value_a.jsonl $(OUT)/chain_b.jsonl $(OUT)/value_b.jsonl --base K4D8
	cd $(DEV) && $(PY) eval_hunger.py show $(O86)/chain.jsonl $(O86)/value.jsonl $(O87)/chain.jsonl $(O87)/value.jsonl $(OUT)/chain_a.jsonl $(OUT)/value_a.jsonl --base V90R
	cd $(DEV) && $(PY) eval_hunger.py show $(O86)/chain.jsonl $(O86)/value.jsonl $(OUT)/chain_a.jsonl $(OUT)/value_a.jsonl --base V90D8
