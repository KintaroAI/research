# Makefile for experiment vm-00056-thought-world
#
# Reproduces the runs from the source tagged exp/vm-00056 (dev/ at that tag): the no-overlap egocentric touch world beside a
# thought field, one wide frame, two transformers (configs/U_thought.json; task 00064). Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the CUDA multi-process service. GPU 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID).
# Outputs go to ~/data/research/visual-motor-agent/exp_00056/.
#
# Usage:
#   make runs     — seeds 1-3, 60k ticks: T (thought: the right piece's dream, and waves), TF (thought frozen: the control)

DEV = ../dev
PY = ./venv/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00056
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(G) $(PY) lifetime.py --config configs/U_thought.json

.PHONY: all runs controls

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(RUN) --seed $$s --set name=T -o $(OUT)/T_s$$s > $(OUT)/T_s$$s.log 2>&1 & \
	    $(RUN) --set env.thought.mode=frozen --seed $$s --set name=TF -o $(OUT)/TF_s$$s > $(OUT)/TF_s$$s.log 2>&1 & \
	  done; wait; }

# make controls — seeds 1-3, 60k ticks: TN (thought a new random field each tick), TL (only the left transformer: no second
#                 piece to compete with; thought frozen)
controls:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(RUN) --set env.thought.mode=noise --seed $$s --set name=TN -o $(OUT)/TN_s$$s > $(OUT)/TN_s$$s.log 2>&1 & \
	    $(RUN) --set env.thought.mode=frozen --set online.pieces=0 --set model.band_k=0 --set online.ring=1 --seed $$s \
	      --set name=TL -o $(OUT)/TL_s$$s > $(OUT)/TL_s$$s.log 2>&1 & \
	  done; wait; }
