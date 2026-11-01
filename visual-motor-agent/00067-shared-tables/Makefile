# Makefile for experiment vm-00067-shared-tables
#
# Reproduces the runs from the source tagged exp/vm-00067 (dev/ at that tag): 16 pieces under gain sharing one table (task 00076). Needs
# the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. All seeds on the RTX 4090 (CUDA_DEVICE_ORDER=PCI_BUS_ID).
# Outputs go to ~/data/research/visual-motor-agent/exp_00067/. The controls are vm-00064's and vm-00063's, not rerun.
#
# Usage:
#   make runs — 60k ticks, seeds 1-8: TB (gain, online.tie_tables=inout)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00067
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe CUDA_VISIBLE_DEVICES=1
RUN = $(PY) lifetime.py --config configs/U_default.json --set model.band_n=16 --set online.piece_rule=gain --set online.tie_tables=inout

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8; do \
	    $(G) $(RUN) --seed $$s --set name=TB -o $(OUT)/TB_s$$s > $(OUT)/TB_s$$s.log 2>&1 & \
	  done; wait; }
