# Makefile for experiment vm-00023-instinct-lookahead
#
# Reproduces the runs from the source tagged exp/vm-00023 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so, with vmc_forward_uni_step), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID: 0 = RTX 3090,
# 1 = RTX 4090) and the CUDA multi-process service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the
# environment). Outputs go to ~/data/research/visual-motor-agent/exp_00023/00<seed>_H4_s<seed>/.
#
# Usage:
#   make runs     — the default universal model (configs/U_universal.json), 100k ticks, seeds 1-3, the instinct
#                   looking 4 steps ahead (online.instinct_h=4); the 1-step arm is vm-00022's U
#   make frozen   — the forage test (16 worlds) with the 1-step instinct and the 4-step lookahead on vm-00021's six
#                   seeds at 100k and 200k, and the lookahead with imagined pixels decoded by their mode at 200k
#   make analysis — the forage test on this experiment's runs (4 steps at 25k, 50k, 100k; 1 step at 100k) and on
#                   vm-00022's U runs (1 step at 25k, 50k, 100k)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00023
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
F = --eval-device cuda --eval-tests forage --eval-worlds 16

.PHONY: all runs frozen analysis

all: runs frozen analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(PY) lifetime.py --config configs/U_universal.json --set online.steps=100000 \
	    --set online.save_every=25000 --set online.instinct_h=4 --seed $$s --set name=H4 -o $(OUT)/00$${s}_H4_s$$s \
	    > $(OUT)/00$${s}_H4_s$$s.log 2>&1 & \
	done; wait

frozen:
	cd $(DEV) && for s in 1 2 3 4 5 6; do r=$$(ls -d $(D)/exp_00021/00$${s}_GI_s$$s); \
	  for h in 1 4; do $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 200000 $(F) --eval-horizon $$h; done; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 200000 $(F) --eval-horizon 4 --eval-decode mode; \
	done

analysis:
	cd $(DEV) && for r in $(OUT)/00?_H4_s?; do \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 25000 50000 100000 $(F); \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 1; \
	done
	cd $(DEV) && for r in $(D)/exp_00022/00[167]_U_s?; do \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 25000 50000 100000 $(F) --eval-horizon 1; \
	done
