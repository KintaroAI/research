# Makefile for experiment vm-00027-one-hot-control
#
# Reproduces the runs from the source tagged exp/vm-00027 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service and the
# corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py). Outputs go to
# ~/data/research/visual-motor-agent/exp_00027/. The arms compared against (B, the classical GPT) are vm-00026's.
#
# Usage:
#   make runs     — arm C (configs/T_text_C.json: one-hot glyphs, one letter per token, bins 2), 100k ticks, seeds 1-3
#   make analysis — eval_predict.py on the val split at every 10k checkpoint (glyph NLL = nats per letter)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00027
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=$$((s % 2)) $(PY) lifetime.py --config configs/T_text_C.json --set online.steps=100000 \
	    --set online.save_every=10000 --seed $$s --set name=C -o $(OUT)/00$${s}_C_s$$s > $(OUT)/00$${s}_C_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(PY) eval_predict.py $(OUT)/00$${s}_C_s$$s --ckpts 10000 20000 30000 40000 50000 60000 70000 80000 \
	    90000 100000 --env-set split=val --env-set tile=true --device cuda; \
	done
