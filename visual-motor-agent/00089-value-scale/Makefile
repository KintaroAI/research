# Makefile for experiment vm-00089-value-scale
#
# Reproduces the runs of task 00100: the instinct's tolerance at the value's scale, online.instinct_tol = 0.002 (0.005 x (1 - gamma) x k), with K4's
# label (K4D8T) and with vm-00088's fresh value (V90RD8T), four quadrant transformers of 8 layers. Twelve seeds; every seed's lives on one card, as vm-00088:
# seeds 1-6 on the RTX 4090, 7-12 on the RTX 3090. 60k ticks. Controls at 0.005: vm-00086/88's K4D8 and vm-00088's V90RD8, not rerun. Needs the GPU
# library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00089/.
#
# Usage:
#   make precheck — before the runs: eval_hunger.py reach and told (the tolerance at 0) on vm-00086/88's K4D8 and V90RD8, without MPS
#   make runs — K4D8T and V90RD8T seeds 1-6 on GPU 1 (CUDA_DEVICE_ORDER=PCI_BUS_ID: the 4090), seeds 7-12 on GPU 0 (the 3090); 24 lives at once
#   make frozen — eval_hunger.py chain, value and reach on every new run, without MPS, two per card
#   make show — with vm-00086/88's files: the four arms over 12 seeds, against K4D8T and against V90RD8, paired by seed

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00089
O86 = $(HOME)/data/research/visual-motor-agent/exp_00086
O88 = $(HOME)/data/research/visual-motor-agent/exp_00088
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002
K4D8T =
V90RD8T = --set online.value_gamma=0.9 --set online.value_refresh=1

.PHONY: all precheck runs frozen show

all: runs

precheck:
	mkdir -p $(OUT)
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  r="$$(for s in 1 2 3 4 5 6; do echo $(O86)/K4D8_s$$s $(O88)/V90RD8_s$$s; done; for s in 7 8 9 10 11 12; do echo $(O88)/K4D8_s$$s $(O88)/V90RD8_s$$s; done)"; \
	  r="$$(echo $$r)"; \
	  a="$$(echo $$r | cut -d' ' -f1-12)"; b="$$(echo $$r | cut -d' ' -f13-24)"; \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$r > $(OUT)/pre_reach.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py told $$a > $(OUT)/pre_told_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py told $$b > $(OUT)/pre_told_b.jsonl & \
	  wait; }

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le 6 ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(V90RD8T) --seed $$s --set name=V90RD8T -o $(OUT)/V90RD8T_s$$s > $(OUT)/V90RD8T_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(K4D8T) --seed $$s --set name=K4D8T -o $(OUT)/K4D8T_s$$s > $(OUT)/K4D8T_s$$s.log 2>&1 & \
	  done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  a="$$(for s in 1 2 3 4 5 6; do echo $(OUT)/V90RD8T_s$$s $(OUT)/K4D8T_s$$s; done | tr "\n" " ")"; \
	  b="$$(for s in 7 8 9 10 11 12; do echo $(OUT)/V90RD8T_s$$s $(OUT)/K4D8T_s$$s; done | tr "\n" " ")"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$a > $(OUT)/reach_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$b > $(OUT)/reach_b.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(O86)/chain.jsonl $(O86)/value.jsonl $(O88)/chain_a.jsonl $(O88)/value_a.jsonl $(O88)/chain_b.jsonl $(O88)/value_b.jsonl $(OUT)/chain_a.jsonl $(OUT)/value_a.jsonl $(OUT)/chain_b.jsonl $(OUT)/value_b.jsonl --base K4D8T
	cd $(DEV) && $(PY) eval_hunger.py show $(O86)/chain.jsonl $(O86)/value.jsonl $(O88)/chain_a.jsonl $(O88)/value_a.jsonl $(O88)/chain_b.jsonl $(O88)/value_b.jsonl $(OUT)/chain_a.jsonl $(OUT)/value_a.jsonl $(OUT)/chain_b.jsonl $(OUT)/value_b.jsonl --base V90RD8
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/pre_reach.jsonl $(OUT)/pre_told_a.jsonl $(OUT)/pre_told_b.jsonl
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/reach_a.jsonl $(OUT)/reach_b.jsonl
