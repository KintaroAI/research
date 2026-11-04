# Makefile for experiment 00020-fact-memorization
#
# Reproduces task 00007: GPT-2-shaped models trained from scratch on TinyStories with synthetic facts in the stream (dev/facts/train.py),
# 20,000 steps of 32 x 256 tokens, constant learning rate 6e-4 after a 500-step warmup, two seeds per size.
# Stage 1: sizes S (2 layers, d 128) and M (4, 256). Stage 2: L (6, 384), X (8, 512) and G (12, 768, GPT-2 small).
# Runs join the MPS daemon when it is up. Outputs go to ~/data/research/llm-fundamentals/exp_00020/.
#
# Usage:
#   make stage1 — S and M, seeds 1 and 2, all on the RTX 3090 (beside the visual-motor runs, about 2 GB each)
#   make stage2 — L, X and G, seeds 1 and 2: seed 1 on the RTX 4090, seed 2 on the RTX 3090 (after the visual-motor queue)
#   make show — the tables (facts/analyze.py)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/llm-fundamentals/exp_00020
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
STEPS = 20000

.PHONY: stage1 stage2 show

define sizes
	mkdir -p $(OUT)
	cd $(DEV) && { for z in $(1); do for s in 1 2; do \
	  if [ $$s -eq 1 ]; then g=$(2); else g=$(3); fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(PY) facts/train.py --size $$z --steps $(STEPS) --seed $$s --out $(OUT)/$${z}_s$$s > $(OUT)/$${z}_s$$s.log 2>&1 & \
	done; done; wait; }
endef

stage1:
	$(call sizes,S M,0,0)

stage2:
	$(call sizes,L X G,1,0)

show:
	cd $(DEV) && $(PY) facts/analyze.py $$(ls -d $(OUT)/*_s? | sort)
