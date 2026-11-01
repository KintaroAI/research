# Makefile for experiment vm-00058-model-size
#
# Reproduces the runs from the source tagged exp/vm-00058 (dev/ at that tag): the default model (configs/U_default.json) at half and
# twice the width (task 00066). Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon; all six lifetimes on the
# RTX 4090 (CUDA_DEVICE_ORDER=PCI_BUS_ID, device 1). Outputs go to ~/data/research/visual-motor-agent/exp_00058/.
#
# Usage:
#   make runs — 60k ticks, seeds 1-3: S (d 64, 2 heads, d_ff 256), B (d 256, 8 heads, d_ff 1024)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00058
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe CUDA_VISIBLE_DEVICES=1
RUN = $(G) $(PY) lifetime.py --config configs/U_default.json
SMALL = --set model.d_model=64 --set model.n_head=2 --set model.d_ff=256
BIG = --set model.d_model=256 --set model.n_head=8 --set model.d_ff=1024

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(RUN) $(SMALL) --seed $$s --set name=S -o $(OUT)/S_s$$s > $(OUT)/S_s$$s.log 2>&1 & \
	    $(RUN) $(BIG) --seed $$s --set name=B -o $(OUT)/B_s$$s > $(OUT)/B_s$$s.log 2>&1 & \
	  done; wait; }
