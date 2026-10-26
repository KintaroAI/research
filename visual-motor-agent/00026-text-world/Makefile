# Makefile for experiment vm-00026-text-world
#
# Reproduces the runs from the source tagged exp/vm-00026 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID: 0 = RTX 3090, 1 = RTX 4090), the CUDA
# multi-process service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the environment), CUDA PyTorch
# in ~/venvs/vm-gpu for the reference, and the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py).
# Outputs go to ~/data/research/visual-motor-agent/exp_00026/.
#
# Usage:
#   make runs     — the universal model in the text world (task 00032), 100k ticks, seeds 1-3: arm A (64 letters in
#                   one token, configs/T_text_A.json) and arm B (one letter per token, T_text_B.json); the classical
#                   character-level GPT (reference/char_gpt.py), 100k updates, seeds 1-3
#   make analysis — eval_predict.py on the val split at every 10k checkpoint (nats per letter)

DEV = ../dev
PY = ./venv/bin/python
GPY = $(HOME)/venvs/vm-gpu/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00026
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --set online.steps=100000 --set online.save_every=10000

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=0 $(L) --config configs/T_text_A.json --seed $$s --set name=A -o $(OUT)/00$${s}_A_s$$s > $(OUT)/00$${s}_A_s$$s.log 2>&1 & \
	  $(G)=1 $(L) --config configs/T_text_B.json --seed $$s --set name=B -o $(OUT)/00$$((s+3))_B_s$$s > $(OUT)/00$$((s+3))_B_s$$s.log 2>&1 & \
	  $(G)=0 $(GPY) reference/char_gpt.py --config configs/T_text_B.json --updates 100000 --eval-every 5000 --seed $$s \
	    -o $(OUT)/00$$((s+6))_GPT_s$$s > $(OUT)/00$$((s+6))_GPT_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/00[1-6]_*_s?; do \
	  $(G)=1 $(PY) eval_predict.py $$r --ckpts 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
	    --env-set split=val --env-set tile=true --device cuda; \
	done
