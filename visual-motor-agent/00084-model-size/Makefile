# Makefile for experiment vm-00084-model-size
#
# Reproduces the runs of task 00095: one transformer over the whole view (model.band_n = 0, online.pieces = 0), ctx 8, K4, the default
# world, with more parameters instead of a longer context: D4 and D8 (4 and 8 layers), W2 and W4 (d_model 256 and 512, heads 32 wide,
# d_ff 4 d), six seeds. The reference is vm-00082's C8 (d 128, 2 layers), not rerun.
# Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00084/.
#
# Usage:
#   make runs — D4, D8, W2, W4, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, 24 lives at once
#   make runs SEEDS="4 5 6" — only those seeds (2026-11-04: seeds 4-6 stopped at 17:49 for a reboot, partial outputs in stopped_1749/,
#     rerun from tick 0 on the 3090, the only card afterwards)
#   make frozen — eval_hunger.py chain, and eval_hidden.py aim with the food hidden after 4 frames and after 1, without MPS
#     (make frozen GC=0: both on the 3090)
#   make show — tables, with vm-00082's C8 rows

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00084
REF = $(HOME)/data/research/visual-motor-agent/exp_00082
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4 --set model.band_n=0 --set model.band_cols=0 --set model.band_win=0 --set online.pieces=0
ARMS = D4 D8 W2 W4
SEEDS = 1 2 3 4 5 6
GC = 1
D4 = --set model.n_layer=4
D8 = --set model.n_layer=8
W2 = --set model.d_model=256 --set model.n_head=8 --set model.d_ff=1024
W4 = --set model.d_model=512 --set model.n_head=16 --set model.d_ff=2048

.PHONY: all runs frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  $(foreach a,$(ARMS),for s in $(SEEDS); do \
	    if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) $($(a)) --seed $$s --set name=$(a) -o $(OUT)/$(a)_s$$s > $(OUT)/$(a)_s$$s.log 2>&1 & \
	  done;) wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  r="$$(for a in $(ARMS); do for s in 1 2 3 4 5 6; do echo $(OUT)/$${a}_s$$s; done; done | tr "\n" " ")"; \
	  CUDA_VISIBLE_DEVICES=$(GC) $(PY) eval_hunger.py chain $$r > $(OUT)/chain.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 sh -c "for v in 4 1; do $(PY) eval_hidden.py aim $$r --show \$$v; done" > $(OUT)/hidden.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(REF)/chain.jsonl $(OUT)/chain.jsonl; $(PY) eval_hidden.py show $(REF)/hidden.jsonl $(OUT)/hidden.jsonl
