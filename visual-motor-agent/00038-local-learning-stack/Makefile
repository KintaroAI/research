# Makefile for experiment vm-00038-local-learning-stack
#
# Reproduces the runs from the source tagged exp/vm-00038 (dev/ at that tag). Needs the CUDA PyTorch venv
# (~/venvs/vm-gpu), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service and the corpus
# (../../llm-fundamentals/dev: prepare_tinyshakespeare.py). Outputs go to ~/data/research/visual-motor-agent/exp_00038/.
#
# Usage:
#   make runs    — the five arms (e2e, frozen, local, dtp, dni), seeds 1-3, 30k updates each, all at once
#   make tie     — the same with --tie (the bottom's W_out = the top's W_in transposed)
#   make norm    — dtp with h layer-normalized at the cut, beta 1 and 0.1
#   make summary — the tables (reference/stacked_summary.py)

DEV = ../dev
PY = ./venv/bin/python
TPY = $(HOME)/venvs/vm-gpu/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00038
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
R = $(TPY) reference/stacked_local.py

.PHONY: all runs tie norm summary

all: runs tie norm summary

runs:
	mkdir -p $(OUT)
	cd $(DEV) && i=0; for a in e2e frozen local dtp dni; do for s in 1 2 3; do i=$$((i+1)); \
	  n=$$(printf "%03d_%s_s%d" $$i $$a $$s); $(G)=$$((i % 2)) $(R) --arm $$a --seed $$s -o $(OUT)/$$n \
	    > $(OUT)/$$n.log 2>&1 & done; done; wait

tie:
	mkdir -p $(OUT)
	cd $(DEV) && i=15; for a in e2e frozen local dtp dni; do for s in 1 2 3; do i=$$((i+1)); \
	  n=$$(printf "%03d_%s_tie_s%d" $$i $$a $$s); $(G)=$$((i % 2)) $(R) --arm $$a --tie --seed $$s -o $(OUT)/$$n \
	    > $(OUT)/$$n.log 2>&1 & done; done; wait

norm:
	mkdir -p $(OUT)
	cd $(DEV) && i=30; for b in 1 0.1; do for s in 1 2 3; do i=$$((i+1)); \
	  n=$$(printf "%03d_dtp_norm_b%s_s%d" $$i $$b $$s); $(G)=$$((i % 2)) $(R) --arm dtp --cut-norm --beta $$b \
	    --seed $$s -o $(OUT)/$$n > $(OUT)/$$n.log 2>&1 & done; done; wait

summary:
	cd $(DEV) && $(PY) reference/stacked_summary.py $(OUT)/0??_*
