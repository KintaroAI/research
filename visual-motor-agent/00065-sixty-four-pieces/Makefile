# Makefile for experiment vm-00065-sixty-four-pieces
#
# Reproduces the runs from the source tagged exp/vm-00065 (dev/ at that tag): the default model split into 64 transformers of 1 patch each (task 00074). Needs
# the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Seeds 1-4 on the RTX 3090, 5-8 on the RTX 4090 (CUDA_DEVICE_ORDER=PCI_BUS_ID): about 2.8 GB per lifetime, so 8 per GPU.
# Outputs go to ~/data/research/visual-motor-agent/exp_00065/. The controls are vm-00064's G16, C16 and vm-00063's NA, CO.
#
# Usage:
#   make runs — 60k ticks, seeds 1-8: G64 (gain), C64 (council), both model.band_n=64

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00065
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set model.band_n=64

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8; do \
	    g=$$(( s > 4 ? 1 : 0 )); \
	    for arm in G64:gain C64:council; do \
	      a=$${arm%%:*}; r=$${arm##*:}; \
	      $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.piece_rule=$$r --seed $$s --set name=$$a -o $(OUT)/$${a}_s$$s > $(OUT)/$${a}_s$$s.log 2>&1 & \
	    done; \
	  done; wait; }
