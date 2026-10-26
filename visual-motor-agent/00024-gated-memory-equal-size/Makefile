# Makefile for experiment vm-00024-gated-memory-equal-size
#
# Reproduces the runs from the source tagged exp/vm-00024 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so, with vmc_uni_token_ce), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID: 0 = RTX 3090,
# 1 = RTX 4090) and the CUDA multi-process service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the
# environment). Outputs go to ~/data/research/visual-motor-agent/exp_00024/RRR_<arm>_s<seed>/.
#
# Usage:
#   make runs     — the default universal model (configs/U_universal.json) with a long-term memory of 25k steps
#                   (3,125 chunks), 200k ticks, seeds 1-3, three gates: U (every chunk), S (the most surprising
#                   quarter, task 00028), RS (every chunk that led to relief and the surprising quarter of the rest)
#   make analysis — the frozen tests (with the instinct) and the click test on real windows (task 00027) at 100k and
#                   200k, the policy alone at 200k

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00024
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --config configs/U_universal.json --set online.steps=200000 --set online.save_every=25000 \
    --set online.replay=25000
ARM_U =
ARM_S = --set online.store_surprise=0.25 --set online.store_base=0
ARM_RS = --set online.store_relief=1 --set online.store_surprise=0.25 --set online.store_base=0

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && i=0; for spec in "0 U 1" "0 S 1" "0 RS 1" "0 U 2" "0 S 2" "1 RS 2" "1 U 3" "1 S 3" "1 RS 3"; do \
	  set -- $$spec; i=$$((i+1)); n=$$(printf "%03d" $$i); \
	  case $$2 in U) X="$(ARM_U)";; S) X="$(ARM_S)";; RS) X="$(ARM_RS)";; esac; \
	  $(G)=$$1 $(L) $$X --seed $$3 --set name=$$2 -o $(OUT)/$${n}_$$2_s$$3 > $(OUT)/$${n}_$$2_s$$3.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_s?; do \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 200000 --eval-device cuda; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 200000 --eval-device cuda --eval-instinct 0; \
	  for st in 100000 200000; do $(G)=0 $(PY) analyze_hazards.py click-real $$r --step $$st --device cuda; done; \
	done
