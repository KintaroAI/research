# Makefile for experiment vm-00031-long-life-bin-input
#
# Reproduces the runs from the source tagged exp/vm-00031 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so, with model.in_bins), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID: 0 = RTX 3090,
# 1 = RTX 4090) and the CUDA multi-process service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the
# environment). Outputs go to ~/data/research/visual-motor-agent/exp_00031/RRR_<arm>_s<seed>/.
#
# Usage:
#   make runs     — the default universal model with bin-embedding input (configs/U_universal.json, in_bins = 1 since
#                   task 00037), memory 100k steps, 500k ticks, seeds 1-3, two memories: S (the most surprising
#                   quarter, the oldest replaced; on the 4090) and U (every chunk, the oldest replaced; on the 3090)
#   make analysis — the frozen tests (with the instinct) at 200k and 500k, the policy alone at 500k, the click test on
#                   real windows (task 00027) at 200k and 500k

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00031
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --config configs/U_universal.json --set online.steps=500000 --set online.save_every=50000
ARM_U =
ARM_S = --set online.store_surprise=0.25 --set online.store_base=0

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && i=0; for spec in "1 S 1" "1 S 2" "1 S 3" "0 U 1" "0 U 2" "0 U 3"; do \
	  set -- $$spec; i=$$((i+1)); n=$$(printf "%03d" $$i); \
	  case $$2 in U) X="$(ARM_U)";; S) X="$(ARM_S)";; esac; \
	  $(G)=$$1 $(L) $$X --seed $$3 --set name=$$2 -o $(OUT)/$${n}_$$2_s$$3 > $(OUT)/$${n}_$$2_s$$3.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_s?; do \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 200000 500000 --eval-device cuda; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 500000 --eval-device cuda --eval-instinct 0; \
	  for st in 200000 500000; do $(G)=0 $(PY) analyze_hazards.py click-real $$r --step $$st --device cuda; done; \
	done
