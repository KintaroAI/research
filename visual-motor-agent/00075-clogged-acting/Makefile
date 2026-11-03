# Makefile for experiment vm-00075-clogged-acting
#
# Reproduces the runs of task 00086: vm-00074's arms with each transformer also acting on its own damaged view. Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00075/. The comparisons are exp_00074 and exp_00073, not rerun.
#
# Usage:
#   make runs — A10, A30, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, six lives on each at once
#   make frozen — chain with damage while acting, chain on clean input, eval_clog.py, without MPS

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00075
C = $(HOME)/data/research/visual-motor-agent/exp_00074
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4 --set online.clog_act=1

.PHONY: all runs frozen

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for a in 10:0.1 30:0.3; do for s in 1 2 3 4 5 6; do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.clog=$${a#*:} --seed $$s --set name=A$${a%:*} -o $(OUT)/A$${a%:*}_s$$s > $(OUT)/A$${a%:*}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  all="$$(for s in 1 2 3 4 5 6; do echo $(OUT)/A10_s$$s $(OUT)/A30_s$$s; done)"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_clog.py score $$all > $(OUT)/clog.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$all --clog .3 > $(OUT)/chain_clog.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$all > $(OUT)/chain_clean.jsonl & \
	  wait; }
