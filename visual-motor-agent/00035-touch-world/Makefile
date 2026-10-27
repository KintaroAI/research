# Makefile for experiment vm-00035-touch-world
#
# Reproduces the runs from the source tagged exp/vm-00035 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID) and the CUDA multi-process service.
# Outputs go to ~/data/research/visual-motor-agent/exp_00035/.
#
# Usage:
#   make chance   — blind random walks in the touch world (forage_chance.py)
#   make runs     — TB (no self-forcing) on the 3090, TSF (self-forcing) on the 4090, 100k ticks, seeds 1-3
#   make tsf4     — TSF4: self-forcing with the 4-step pair lookahead in life (seeds 1-2 on the 4090, 3 on the 3090;
#                   run it alone: next to other jobs on one GPU it slows to ~1 tick/s)
#   make analysis — frozen forage tests at 100k (instinct off, 1-step, 4-step mean, 4-step pair) and the dream curve

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00035
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --config configs/U_touch.json --set online.steps=100000 --set online.save_every=25000
SF = --set online.sf_windows=8 --set online.sf_steps=3 --set online.sf_jacobi=2
H4 = --set online.instinct_h=4 --set online.instinct_decode=pair
F = --eval-device cuda --eval-tests forage --eval-worlds 16

.PHONY: all chance runs tsf4 analysis

all: chance runs tsf4 analysis

chance:
	cd $(DEV) && $(PY) forage_chance.py --config configs/U_touch.json

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=0 $(L) --seed $$s --set name=TB -o $(OUT)/00$${s}_TB_s$$s > $(OUT)/00$${s}_TB_s$$s.log 2>&1 & \
	  $(G)=1 $(L) $(SF) --seed $$s --set name=TSF -o $(OUT)/00$$((s+3))_TSF_s$$s > $(OUT)/00$$((s+3))_TSF_s$$s.log 2>&1 & \
	done; wait

tsf4:
	mkdir -p $(OUT)
	cd $(DEV) && for spec in "1 1" "2 1" "3 0"; do set -- $$spec; \
	  $(G)=$$2 $(L) $(SF) $(H4) --seed $$1 --set name=TSF4 -o $(OUT)/00$$(($$1+6))_TSF4_s$$1 \
	    > $(OUT)/00$$(($$1+6))_TSF4_s$$1.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/00?_T*_s?; do \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-instinct 0; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 1; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode mean; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode pair; \
	  $(G)=0 $(PY) eval_dream.py $$r --ckpts 100000; \
	done
