# Makefile for experiment vm-00051-pieces-one-body
#
# Reproduces the runs from the source tagged exp/vm-00051 (dev/ at that tag): the egocentric touch world with checker
# poison (configs/U_touch_ego_c2.json) and the pyramid with thumbnails (uni_pyr 3, sum_thumb 1: vm-00045's XFT), the
# view split among separate transformers that share one body (task 00059, online.pieces): each piece its own move,
# the body their mean. Needs the GPU library (cd ../dev && make libvmcuda.so). GPU 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID);
# all twelve lifetimes on one card.
# Outputs go to ~/data/research/visual-motor-agent/exp_00051/.
#
# Usage:
#   make runs     — seeds 1-3, 60k ticks, checkpoints every 10k: F (one model), H (vertical halves: band_n 2,
#                   band_cols 1), Q (quadrants: band_n 4, band_cols 2), QW (quadrants with band_win 1)

DEV = ../dev
PY = ./venv/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00051
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
RUN = $(G) $(PY) lifetime.py --config configs/U_touch_ego_c2.json --set online.steps=60000 \
  --set online.save_every=10000 --set model.uni_pyr=3 --set model.sum_thumb=1
H = --set online.pieces=1 --set model.band_n=2 --set model.band_cols=1
Q = --set online.pieces=1 --set model.band_n=4 --set model.band_cols=2

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(RUN) --seed $$s --set name=F -o $(OUT)/00$${s}_F_s$$s > $(OUT)/00$${s}_F_s$$s.log 2>&1 & \
	    $(RUN) $(H) --seed $$s --set name=H -o $(OUT)/00$$((s + 3))_H_s$$s > $(OUT)/00$$((s + 3))_H_s$$s.log 2>&1 & \
	    $(RUN) $(Q) --seed $$s --set name=Q -o $(OUT)/00$$((s + 6))_Q_s$$s > $(OUT)/00$$((s + 6))_Q_s$$s.log 2>&1 & \
	    $(RUN) $(Q) --set model.band_win=1 --seed $$s --set name=QW \
	      -o $(OUT)/0$$((s + 9))_QW_s$$s > $(OUT)/0$$((s + 9))_QW_s$$s.log 2>&1 & \
	  done; wait; }
