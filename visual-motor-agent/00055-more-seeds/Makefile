# Makefile for experiment vm-00055-more-seeds
#
# Reproduces the runs from the source tagged exp/vm-00055 (dev/ at that tag): five more seeds of the best arms, to go with
# seeds 1-3 of vm-00052 and vm-00054 (task 00063). Needs the GPU library (cd ../dev && make libvmcuda.so) and the CUDA
# multi-process service:
#   CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/claude-1000/mps_log nvidia-cuda-mps-control -d
# GPU 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID); all fifteen lifetimes on one card.
# Outputs go to ~/data/research/visual-motor-agent/exp_00055/.
#
# Usage:
#   make runs     — seeds 4-8, 60k ticks: QN (gain), QNA (gain, age key, ring), QCA (council, age key, ring)

DEV = ../dev
PY = ./venv/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00055
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(G) $(PY) lifetime.py --config configs/U_touch_ego_c2.json --set online.steps=60000 \
  --set online.save_every=10000 --set model.uni_pyr=3 --set model.sum_thumb=1 --set online.pieces=1 \
  --set model.band_n=4 --set model.band_cols=2 --set env.no_overlap=true
AR = --set model.age_key=1 --set online.ring=1

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 4 5 6 7 8; do \
	    $(RUN) --set online.piece_rule=gain --seed $$s --set name=QN -o $(OUT)/QN_s$$s > $(OUT)/QN_s$$s.log 2>&1 & \
	    $(RUN) $(AR) --set online.piece_rule=gain --seed $$s --set name=QNA -o $(OUT)/QNA_s$$s > $(OUT)/QNA_s$$s.log 2>&1 & \
	    $(RUN) $(AR) --set online.piece_rule=council --seed $$s --set name=QCA -o $(OUT)/QCA_s$$s > $(OUT)/QCA_s$$s.log 2>&1 & \
	  done; wait; }
