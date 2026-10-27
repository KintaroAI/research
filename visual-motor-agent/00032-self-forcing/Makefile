# Makefile for experiment vm-00032-self-forcing
#
# Reproduces the runs from the source tagged exp/vm-00032 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so, with vmc_set_uni_inputs), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID) and the CUDA
# multi-process service. The baseline is vm-00030's circle runs (exp_00030/00[4-6]_Uib_s?). Outputs go to
# ~/data/research/visual-motor-agent/exp_00032/.
#
# Usage:
#   make runs     — the default universal lifetime with self-forcing (8 windows, 1-3 dreamed steps, J = 2), 100k ticks,
#                   seeds 1-3
#   make analysis — the dream curve (eval_dream.py) at 25k, 50k, 100k; the frozen tests; the click test; the forage test
#                   (16 worlds) with the 1-step instinct and the 4-step pair lookahead; the same for the baseline

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00032
BASE = $(HOME)/data/research/visual-motor-agent/exp_00030
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --config configs/U_universal.json --set online.steps=100000 --set online.save_every=25000 \
    --set online.sf_windows=8 --set online.sf_steps=3 --set online.sf_jacobi=2
F = --eval-device cuda --eval-tests forage --eval-worlds 16

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for spec in "1 1" "1 2" "0 3"; do set -- $$spec; \
	  $(G)=$$1 $(L) --seed $$2 --set name=SF -o $(OUT)/00$$2_SF_s$$2 > $(OUT)/00$$2_SF_s$$2.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/00?_SF_s? $(BASE)/00[4-6]_Uib_s?; do \
	  $(G)=1 $(PY) eval_dream.py $$r --ckpts 25000 50000 100000; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 1; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode pair; \
	done
	cd $(DEV) && for r in $(OUT)/00?_SF_s?; do \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 25000 50000 100000 --eval-device cuda; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 --eval-device cuda --eval-instinct 0; \
	  $(G)=1 $(PY) analyze_hazards.py click-real $$r --step 100000 --device cuda; \
	done
