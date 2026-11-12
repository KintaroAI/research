# Makefile for experiment vm-00086-value-depth
#
# Reproduces the runs of task 00097: the default model (4 quadrant transformers, gain, log summaries, the ring, the egocentric touch world, K4) with
# 8 layers in each transformer (model.n_layer=8), with K4's label (K4D8) and with vm-00085's bootstrapped value, gamma 0.9 (V90D8). The 2-layer arms
# are vm-00085's K4 and V90, not rerun. Six seeds, 60k ticks, all 12 lives on the RTX 4090.
# Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00086/.
#
# Usage:
#   make runs — K4D8, V90D8, seeds 1-6, 12 lives at once on GPU 1 (CUDA_DEVICE_ORDER=PCI_BUS_ID: the 4090)
#   make frozen — eval_hunger.py chain and value on every run, without MPS, both on GPU 1
#   make show — the value at 8 layers (V90D8 against K4D8), and depth (8 layers against vm-00085's 2, paired by seed)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00086
O85 = $(HOME)/data/research/visual-motor-agent/exp_00085
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4 --set model.n_layer=8
ARMS = K4D8 V90D8
SEEDS = 1 2 3 4 5 6
GPU = 1
K4D8 =
V90D8 = --set online.value_gamma=0.9

.PHONY: all runs frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  $(foreach a,$(ARMS),for s in $(SEEDS); do \
	    $(G) CUDA_VISIBLE_DEVICES=$(GPU) $(RUN) $($(a)) --seed $$s --set name=$(a) -o $(OUT)/$(a)_s$$s > $(OUT)/$(a)_s$$s.log 2>&1 & \
	  done;) wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  r="$$(for a in $(ARMS); do for s in $(SEEDS); do echo $(OUT)/$${a}_s$$s; done; done | tr "\n" " ")"; \
	  CUDA_VISIBLE_DEVICES=$(GPU) $(PY) eval_hunger.py chain $$r > $(OUT)/chain.jsonl & \
	  CUDA_VISIBLE_DEVICES=$(GPU) $(PY) eval_hunger.py value $$r --ticks 2000 > $(OUT)/value.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/chain.jsonl $(OUT)/value.jsonl --base K4D8
	cd $(DEV) && $(PY) eval_hunger.py show $(O85)/chain.jsonl $(O85)/value.jsonl $(OUT)/chain.jsonl $(OUT)/value.jsonl --base K4
