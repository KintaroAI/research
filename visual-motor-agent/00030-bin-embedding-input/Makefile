# Makefile for experiment vm-00030-bin-embedding-input
#
# Reproduces the runs from the source tagged exp/vm-00030 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so, with model.in_bins), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA
# multi-process service and the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py). Outputs go to
# ~/data/research/visual-motor-agent/exp_00030/.
#
# Usage:
#   make runs     — bin-embedding input (model.in_bins=1, task 00036), 100k ticks, seeds 1-3: the text lifetime
#                   (configs/T_text_B.json) and the circle world's default universal lifetime (configs/U_universal.json)
#   make analysis — text: eval_predict.py on the val split every 10k; circle: the frozen tests at 25k, 50k, 100k, the
#                   policy alone and the click test on real windows at 100k (as vm-00022's U)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00030
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --set online.steps=100000 --set model.in_bins=1

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(L) --config configs/T_text_B.json --set online.save_every=10000 --seed $$s --set name=Bib \
	    -o $(OUT)/00$${s}_Bib_s$$s > $(OUT)/00$${s}_Bib_s$$s.log 2>&1 & \
	  $(G)=0 $(L) --config configs/U_universal.json --set online.save_every=25000 --seed $$s --set name=Uib \
	    -o $(OUT)/00$$((s+3))_Uib_s$$s > $(OUT)/00$$((s+3))_Uib_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(PY) eval_predict.py $(OUT)/00$${s}_Bib_s$$s --ckpts 10000 20000 30000 40000 50000 60000 70000 80000 \
	    90000 100000 --env-set split=val --env-set tile=true --device cuda; \
	  r=$(OUT)/00$$((s+3))_Uib_s$$s; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 25000 50000 100000 --eval-device cuda; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 --eval-device cuda --eval-instinct 0; \
	  $(G)=0 $(PY) analyze_hazards.py click-real $$r --step 100000 --device cuda; \
	done
