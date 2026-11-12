# Makefile for experiment vm-00087-value-refresh
#
# Reproduces the runs of task 00098: the default model (4 quadrant transformers of 2 layers, gain, log summaries, the ring, the egocentric touch world,
# K4) with vm-00085's bootstrapped value, gamma 0.9, its tail made fresh at every update (online.value_refresh=1): the mean over the four transformers of
# the predicted label of the move made n = 4 ticks later (V90R). The controls are vm-00085's K4 and V90, not rerun (the same card). Six seeds, 60k ticks,
# on the RTX 3090. Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00087/.
#
# Usage:
#   make runs — V90R, seeds 1-6, 6 lives at once on GPU 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID: the 3090)
#   make frozen — eval_hunger.py chain and value on every run, without MPS, both on GPU 0
#   make show — V90R with vm-00085's arms, against K4 and against V90, paired by seed

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00087
O85 = $(HOME)/data/research/visual-motor-agent/exp_00085
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4 --set online.value_gamma=0.9
ARMS = V90R
SEEDS = 1 2 3 4 5 6
GPU = 0
V90R = --set online.value_refresh=1

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
	cd $(DEV) && $(PY) eval_hunger.py show $(O85)/chain.jsonl $(O85)/value.jsonl $(OUT)/chain.jsonl $(OUT)/value.jsonl --base K4
	cd $(DEV) && $(PY) eval_hunger.py show $(O85)/chain.jsonl $(O85)/value.jsonl $(OUT)/chain.jsonl $(OUT)/value.jsonl --base V90
