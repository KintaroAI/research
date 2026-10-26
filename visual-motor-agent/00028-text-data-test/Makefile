# Makefile for experiment vm-00028-text-data-test
#
# Reproduces the runs from the source tagged exp/vm-00028 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service, CUDA
# PyTorch in ~/venvs/vm-gpu and the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py). Outputs go to
# ~/data/research/visual-motor-agent/exp_00028/.
#
# Usage:
#   make runs     — the classical GPT on a lifetime's batches (reference/char_gpt.py --data lifetime) and the universal
#                   model (arm B) trained offline on random windows (train_text_offline.py), 100k updates, seeds 1-3
#   make analysis — eval_predict.py on the val split at every 10k checkpoint of the offline universal model

DEV = ../dev
PY = ./venv/bin/python
GPY = $(HOME)/venvs/vm-gpu/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00028
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=0 $(GPY) reference/char_gpt.py --config configs/T_text_B.json --updates 100000 --eval-every 5000 \
	    --data lifetime --seed $$s -o $(OUT)/00$${s}_GPTlife_s$$s > $(OUT)/00$${s}_GPTlife_s$$s.log 2>&1 & \
	  $(G)=1 $(PY) train_text_offline.py --config configs/T_text_B.json --updates 100000 --seed $$s \
	    -o $(OUT)/00$$((s+3))_Boff_s$$s > $(OUT)/00$$((s+3))_Boff_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(PY) eval_predict.py $(OUT)/00$$((s+3))_Boff_s$$s --ckpts 10000 20000 30000 40000 50000 60000 70000 \
	    80000 90000 100000 --env-set split=val --env-set tile=true --device cuda; \
	done
