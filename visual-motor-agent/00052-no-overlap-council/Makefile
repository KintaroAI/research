# Makefile for experiment vm-00052-no-overlap-council
#
# Reproduces the runs from the source tagged exp/vm-00052 (dev/ at that tag): vm-00051's QG (four quadrants in one body,
# the egocentric touch world with checker poison, uni_pyr 3, sum_thumb 1) in a world where circles never overlap
# (env.no_overlap) and with a council of pieces (online.piece_rule=council; task 00060). Needs the GPU library
# (cd ../dev && make libvmcuda.so). GPU 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID); all nine lifetimes on one card.
# Outputs go to ~/data/research/visual-motor-agent/exp_00052/.
#
# Usage:
#   make runs     — seeds 1-3, 60k ticks, checkpoints every 10k: QN (gain, no_overlap), QC (council), QCN (council,
#                   no_overlap); the control is vm-00051's QG

DEV = ../dev
PY = ./venv/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00052
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
RUN = $(G) $(PY) lifetime.py --config configs/U_touch_ego_c2.json --set online.steps=60000 \
  --set online.save_every=10000 --set model.uni_pyr=3 --set model.sum_thumb=1 --set online.pieces=1 \
  --set model.band_n=4 --set model.band_cols=2
NO = --set env.no_overlap=true
GAIN = --set online.piece_rule=gain
COUNCIL = --set online.piece_rule=council

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(RUN) $(GAIN) $(NO) --seed $$s --set name=QN -o $(OUT)/00$${s}_QN_s$$s > $(OUT)/00$${s}_QN_s$$s.log 2>&1 & \
	    $(RUN) $(COUNCIL) --seed $$s --set name=QC -o $(OUT)/00$$((s + 3))_QC_s$$s > $(OUT)/00$$((s + 3))_QC_s$$s.log 2>&1 & \
	    $(RUN) $(COUNCIL) $(NO) --seed $$s --set name=QCN -o $(OUT)/00$$((s + 6))_QCN_s$$s > $(OUT)/00$$((s + 6))_QCN_s$$s.log 2>&1 & \
	  done; wait; }
