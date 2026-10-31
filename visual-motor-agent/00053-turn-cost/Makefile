# Makefile for experiment vm-00053-turn-cost
#
# Reproduces the runs from the source tagged exp/vm-00053 (dev/ at that tag): vm-00051's QG (four quadrants in one body,
# the winner by gain, the egocentric touch world with checker poison, uni_pyr 3, sum_thumb 1) where changing the move
# costs hunger (env.turn_cost; task 00061), with online.instinct_tol 0.001. Needs the GPU library
# (cd ../dev && make libvmcuda.so). GPU 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID).
# Outputs go to ~/data/research/visual-motor-agent/exp_00053/.
#
# Usage:
#   make runs     — seeds 1-3, 60k ticks, checkpoints every 10k: QT2 (turn_cost 0.02), QT4 (turn_cost 0.04); the
#                   control is vm-00051's QG

DEV = ../dev
PY = ./venv/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00053
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
RUN = $(G) $(PY) lifetime.py --config configs/U_touch_ego_c2.json --set online.steps=60000 \
  --set online.save_every=10000 --set model.uni_pyr=3 --set model.sum_thumb=1 --set online.pieces=1 \
  --set model.band_n=4 --set model.band_cols=2 --set online.piece_rule=gain --set online.instinct_tol=0.001

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(RUN) --set env.turn_cost=0.02 --seed $$s --set name=QT2 -o $(OUT)/00$${s}_QT2_s$$s > $(OUT)/00$${s}_QT2_s$$s.log 2>&1 & \
	    $(RUN) --set env.turn_cost=0.04 --seed $$s --set name=QT4 -o $(OUT)/00$$((s + 3))_QT4_s$$s > $(OUT)/00$$((s + 3))_QT4_s$$s.log 2>&1 & \
	  done; wait; }
