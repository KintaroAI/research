# Makefile for experiment vm-00061-slow-tick-car
#
# Reproduces the runs from the source tagged exp/vm-00061 (dev/ at that tag): the MuJoCo car at 5 ticks/s instead of 20 (task 00070). Needs the no-AVX MuJoCo build (see
# ../CLAUDE.md), the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. SD on the RTX 4090, SC on the RTX 3090, each with CUDA and EGL on the same card
# (CUDA_DEVICE_ORDER=PCI_BUS_ID). Outputs go to ~/data/research/visual-motor-agent/exp_00061/.
#
# Usage:
#   make runs   — 60k ticks, seeds 1-3: SD (48 boxes, down to 6 by 40k), SC (6 boxes)
#   make chance — blind drivers among 48, 24, 12 and 6 boxes

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00061
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
D = $(G) CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
C = $(G) CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
SLOW = --set env.frame_skip=40 --set env.reverse=true   # the car could reverse here; since task 00071 it needs saying

.PHONY: all runs chance

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(D) $(PY) lifetime.py --config configs/U_car_dense.json $(SLOW) --seed $$s --set name=SD -o $(OUT)/SD_s$$s > $(OUT)/SD_s$$s.log 2>&1 & \
	    $(C) $(PY) lifetime.py --config configs/U_car.json $(SLOW) --set online.steps=60000 --set online.save_every=10000 --seed $$s --set name=SC \
	      -o $(OUT)/SC_s$$s > $(OUT)/SC_s$$s.log 2>&1 & \
	  done; wait; }

chance:
	mkdir -p $(OUT)
	cd $(DEV) && for n in 24 12 6 3; do echo "== $$((2 * n)) boxes"; MUJOCO_EGL_DEVICE_ID=0 $(PY) car_chance.py --ticks 10000 --frame-skip 40 --reverse --foods $$n --poisons $$n; done | tee $(OUT)/chance.txt
