# Makefile for experiment vm-00059-mujoco-car
#
# Reproduces the runs from the source tagged exp/vm-00059 (dev/ at that tag): the default model, unchanged, as a small car in MuJoCo (task
# 00067, configs/U_car.json). Needs the no-AVX MuJoCo build (see ../CLAUDE.md), the GPU library (cd ../dev && make libvmcuda.so) and the MPS
# daemon. All on the RTX 3090 (CUDA_DEVICE_ORDER=PCI_BUS_ID, device 0; MUJOCO_EGL_DEVICE_ID=0). Outputs go to
# ~/data/research/visual-motor-agent/exp_00059/.
#
# Usage:
#   make runs   — 20k ticks, seeds 1-3
#   make chance — blind drivers in the same world

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00059
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0

.PHONY: all runs chance

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(G) $(PY) lifetime.py --config configs/U_car.json --seed $$s --set name=C -o $(OUT)/C_s$$s > $(OUT)/C_s$$s.log 2>&1 & \
	  done; wait; }

chance:
	mkdir -p $(OUT)
	cd $(DEV) && MUJOCO_EGL_DEVICE_ID=0 $(PY) car_chance.py | tee $(OUT)/chance.txt
