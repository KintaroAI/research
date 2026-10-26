# Makefile for experiment vm-00022-surprise-gated-memory
#
# Reproduces the runs from the source tagged exp/vm-00022 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so, with vmc_uni_token_ce), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID: 0 = RTX 3090,
# 1 = RTX 4090) and the CUDA multi-process service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the
# environment). Outputs go to ~/data/research/visual-motor-agent/exp_00022/RRR_<arm>_s<seed>/.
#
# Usage:
#   make runs     — the default universal model (configs/U_universal.json), 100k ticks, seeds 1-3, three memories:
#                   U (every chunk), S (the most surprising quarter, task 00028), U25 (a random quarter)
#   make analysis — the frozen tests at 25k, 50k, 100k (the policy alone at 100k) and the click test on real windows
#                   (task 00027) at 100k

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00022
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --config configs/U_universal.json --set online.steps=100000 --set online.save_every=25000
ARM_U =
ARM_S = --set online.store_surprise=0.25 --set online.store_base=0
ARM_U25 = --set online.store_base=0.25

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && i=0; for spec in "1 U 1" "1 S 1" "1 U25 1" "1 S 2" "1 U25 2" "0 U 2" "0 U 3" "0 S 3" "0 U25 3"; do \
	  set -- $$spec; i=$$((i+1)); n=$$(printf "%03d" $$i); \
	  case $$2 in U) X="$(ARM_U)";; S) X="$(ARM_S)";; U25) X="$(ARM_U25)";; esac; \
	  $(G)=$$1 $(L) $$X --seed $$3 --set name=$$2 -o $(OUT)/$${n}_$$2_s$$3 > $(OUT)/$${n}_$$2_s$$3.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_s?; do \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 25000 50000 100000 --eval-device cuda; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 --eval-device cuda --eval-instinct 0; \
	done
	cd $(DEV) && for r in $(OUT)/0*_s?; do $(G)=0 $(PY) analyze_hazards.py click-real $$r --step 100000 --device cuda; done
