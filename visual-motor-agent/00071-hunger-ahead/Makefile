# Makefile for experiment vm-00071-hunger-ahead
#
# Reproduces the runs of task 00082: the hunger a move is scored on is the mean over the next future_k ticks. Needs the GPU library (cd ../dev && make libvmcuda.so)
# and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00071/.
#
# Usage:
#   make runs — K1, K4, K16, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, nine lives on each at once

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00071
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for k in 1 4 16; do for s in 1 2 3 4 5 6; do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.future_k=$$k --seed $$s --set name=K$$k -o $(OUT)/K$${k}_s$$s > $(OUT)/K$${k}_s$$s.log 2>&1 & \
	done; done; wait; }
