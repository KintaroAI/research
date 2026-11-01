# Makefile for experiment vm-00060-dense-car
#
# Reproduces the runs from the source tagged exp/vm-00060 (dev/ at that tag): the MuJoCo car with a dense start (task 00068). Needs the no-AVX
# MuJoCo build (see ../CLAUDE.md), the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. D on the RTX 4090, C on the RTX 3090, each
# with CUDA and EGL on the same card (CUDA_DEVICE_ORDER=PCI_BUS_ID). Outputs go to ~/data/research/visual-motor-agent/exp_00060/.
#
# Usage:
#   make runs   — 60k ticks, seeds 1-3: D (48 boxes, down to 6 by 40k, configs/U_car_dense.json), C (6 boxes, configs/U_car.json)
#   make chance — blind drivers among 48, 24, 12 and 6 boxes

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00060
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
D = $(G) CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
C = $(G) CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0

.PHONY: all runs chance

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(D) $(PY) lifetime.py --config configs/U_car_dense.json --seed $$s --set name=D -o $(OUT)/D_s$$s > $(OUT)/D_s$$s.log 2>&1 & \
	    $(C) $(PY) lifetime.py --config configs/U_car.json --set online.steps=60000 --set online.save_every=10000 --seed $$s --set name=C \
	      -o $(OUT)/C_s$$s > $(OUT)/C_s$$s.log 2>&1 & \
	  done; wait; }

chance:
	mkdir -p $(OUT)
	cd $(DEV) && for n in 24 12 6 3; do echo "== $$((2 * n)) boxes"; MUJOCO_EGL_DEVICE_ID=0 $(PY) car_chance.py --ticks 10000 --foods $$n --poisons $$n; done | tee $(OUT)/chance.txt
