# Makefile for experiment vm-00082-context-length
#
# Reproduces the runs of task 00093: one transformer over the whole view (model.band_n = 0, online.pieces = 0), K4, the default world,
# model.ctx 8, 12, 16, 24 and 32 (C8 ... C32), six seeds. The reference is vm-00073's K4, not rerun.
# Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00082/.
#
# Usage:
#   make runs — C8 to C32, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, 30 lives at once
#   make frozen — eval_hunger.py chain, and eval_hidden.py aim with the food hidden after 4 frames and after 1, without MPS
#   make show — tables

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00082
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4 --set model.band_n=0 --set model.band_cols=0 --set model.band_win=0 --set online.pieces=0
CTXS = 8 12 16 24 32

.PHONY: all runs frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for c in $(CTXS); do for s in 1 2 3 4 5 6; do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set model.ctx=$$c --seed $$s --set name=C$$c -o $(OUT)/C$${c}_s$$s > $(OUT)/C$${c}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  r="$$(for c in $(CTXS); do for s in 1 2 3 4 5 6; do echo $(OUT)/C$${c}_s$$s; done; done)"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$r > $(OUT)/chain.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 sh -c "for v in 4 1; do $(PY) eval_hidden.py aim $$r --show \$$v; done" > $(OUT)/hidden.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/chain.jsonl; $(PY) eval_hidden.py show $(OUT)/hidden.jsonl
