# Makefile for experiment vm-00069-long-life-pieces
#
# Reproduces the runs from the source tagged exp/vm-00069 (dev/ at that tag): 500k ticks with a memory of the whole life (task 00080). Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00069/.
#
# Usage:
#   make runs-lt — LT, seeds 1-4 on the RTX 4090: 16 pieces, shared tables; about 10 hours
#   make runs-ld — LD, seeds 1-4 on the RTX 3090: the default, 4 pieces; about 7 hours

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00069
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.steps=500000 --set online.replay=500000 --set online.save_every=50000

.PHONY: all runs-lt runs-ld

all: runs-lt runs-ld

runs-lt:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in 1 2 3 4; do \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(RUN) --set model.band_n=16 --set online.tie_tables=inout --seed $$s --set name=LT -o $(OUT)/LT_s$$s > $(OUT)/LT_s$$s.log 2>&1 & \
	done; wait; }

runs-ld:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in 1 2 3 4; do \
	  $(G) CUDA_VISIBLE_DEVICES=0 $(RUN) --seed $$s --set name=LD -o $(OUT)/LD_s$$s > $(OUT)/LD_s$$s.log 2>&1 & \
	done; wait; }
