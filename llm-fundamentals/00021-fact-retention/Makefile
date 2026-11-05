# Makefile for experiment 00021-fact-retention
#
# Reproduces task 00008: what keeps a fact. 00020's stream (dev/facts/train.py) plus one new fact a step (--plus 1), shown 8 times all in
# one step, 10, 100 or 1,000 steps apart, or 100 apart in four wordings; under three learning rates: C6 (6e-4 constant, 00020's), C2 (2e-4
# constant), D (6e-4 falling to 6e-5 along a cosine). Sizes L and G, two seeds, 20,000 steps of 32 x 256 tokens.
# Seed 1 on the RTX 4090, seed 2 on the RTX 3090. A G run peaks at 5.6 GB and an L run at 2.3 GB, so each card runs four lanes:
# two G and two L at a time, then the third G and the third L. Outputs go to ~/data/research/llm-fundamentals/exp_00021/.
#
# Usage:
#   make runs — the 12 runs
#   make show — the tables (facts/analyze.py)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/llm-fundamentals/exp_00021
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe PYTORCH_ALLOC_CONF=expandable_segments:True
STEPS = 20000
C6 = --lr 6e-4
C2 = --lr 2e-4
D = --lr 6e-4 --lr-sched cos --lr-min 6e-5

.PHONY: runs show

# one run: $(call one,size,arm,seed,gpu)
one = $(G) CUDA_VISIBLE_DEVICES=$(4) $(PY) facts/train.py --size $(1) $($(2)) --plus 1 --steps $(STEPS) --seed $(3) --out $(OUT)/$(1)$(2)_s$(3) > $(OUT)/$(1)$(2)_s$(3).log 2>&1

# the four lanes of one card: $(call card,seed,gpu)
define card
	{ $(call one,G,C6,$(1),$(2)); $(call one,G,D,$(1),$(2)); } & \
	{ $(call one,G,C2,$(1),$(2)); $(call one,L,D,$(1),$(2)); } & \
	$(call one,L,C6,$(1),$(2)) & \
	$(call one,L,C2,$(1),$(2)) &
endef

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { $(call card,1,1) $(call card,2,0) wait; }

show:
	cd $(DEV) && $(PY) facts/analyze.py $$(ls -d $(OUT)/*_s? | sort)
