# Makefile for experiment vm-00085-value-bootstrap
#
# Reproduces the runs of task 00096: the default model (4 quadrant transformers, gain, log summaries, the ring, the egocentric touch world) with
# K4's hindsight label (the mean hunger over the next 4 ticks) against a bootstrapped value (online.value_gamma): 4 real ticks, then the model's own
# predicted label of the move taken 4 ticks later, gamma 0.9 (V90) and 0.97 (V97). Six seeds, 60k ticks, all 18 lives on the RTX 3090 (the 4090 was out).
# Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00085/.
#
# Usage:
#   make runs — K4, V90, V97, seeds 1-6, 18 lives at once on GPU 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID: the 3090)
#   make frozen — eval_hunger.py chain and value on every run, without MPS, both on GPU 0
#   make show — tables, every arm against K4 paired by seed

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00085
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4
ARMS = K4 V90 V97
SEEDS = 1 2 3 4 5 6
GPU = 0
K4 =
V90 = --set online.value_gamma=0.9
V97 = --set online.value_gamma=0.97

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
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/chain.jsonl $(OUT)/value.jsonl --base K4
