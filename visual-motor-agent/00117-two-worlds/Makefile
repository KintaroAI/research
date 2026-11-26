# Makefile for experiment vm-00117-two-worlds
#
# Task 00130: one set of weights, two worlds. The default agent's two configs differ only in the world (configs/U_agent.json,
# the food world; configs/U_breakout.json, Breakout stage 1), so a life can start in one world from the weights another
# life ended with in the other (lifetime.py --init: the weights only; memory, optimizer and world start new).
#   TB: Breakout from TT's (vm-00101) 60k weights, against BD (vm-00103) from scratch, paired by seed.
#   BF: the food world from BD's 60k weights, against TT from scratch, paired by seed.
# Then the old world, learning off, from the second life's checkpoints (how much is left), and a third life of 30k ticks
# back in the old world (how fast it comes back): TBF, BFB.
# GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS. As run: seeds 1-6 on the 3090 and
# 7-12 on the 4090, BD's split in vm-00103.
# Outputs go to ~/data/research/visual-motor-agent/exp_00117/.
#
# Usage:
#   make runs W=TB CARD=1 SEEDS="7 8 9 10 11 12"    — the second life, 60k ticks (W=TB or BF)
#   make frozen W=TB CARD=1 SEEDS="..."             — W's 10k, 30k and 60k checkpoints, learning off (ppo.lr = 0), in
#                                                     the old world (seed 100 + s): 2000 ticks of the food world for TB
#                                                     (as vm-00114's check/TT_ring), 6000 of Breakout for BF (as
#                                                     vm-00116's ring arm)
#   make floor CARD=1 SEEDS="..."                   — the untrained default model, learning off, in both worlds
#   make back W=TB CARD=1 SEEDS="..."               — the third life: 30k ticks back in the old world from W's 60k
#                                                     weights, learning on (TBF from TB, BFB from BF)
#   make table                                      — the README's tables (dev/analyze_lives.py two-worlds; reads the
#                                                     logged rows, no GPU; SEEDS="..." for a part of the seeds)
#   Options: MPS= (no MPS daemon), T="10000 30000 60000" (frozen's checkpoints)
#
# As run (2026-11-25 and 26), on each card: runs W=TB, runs W=BF, frozen W=TB, frozen W=BF, floor, back W=TB, back W=BF.

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00117
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
W ?= TB
SEEDS ?= 1 2 3 4 5 6 7 8 9 10 11 12
T ?= 10000 30000 60000
FOOD = --config configs/U_agent.json
COURT = --config configs/U_breakout.json
# W's new world, its old world, the first life's runs, the old world's test length and the third life's name
NEW = $(if $(filter TB,$(W)),$(COURT),$(FOOD))
OLD = $(if $(filter TB,$(W)),$(FOOD),$(COURT))
FROM = $(if $(filter TB,$(W)),$(D)/exp_00101/TT,$(D)/exp_00103/BD)
OLDTICKS = $(if $(filter TB,$(W)),2000,6000)
BACK = $(if $(filter TB,$(W)),TBF,BFB)
FROZEN = --set ppo.lr=0 --set online.log_every=100 --set online.save_every=100000

.PHONY: runs frozen floor back table

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in $(SEEDS); do \
	  env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(NEW) --init $(FROM)_s$$s/ckpt_00060000 --seed $$s --set name=$(W) \
	    -o $(OUT)/$(W)_s$$s > $(OUT)/$(W)_s$$s.log 2>&1 & \
	done; wait; }

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/old_$(W)_$$((t / 1000))k; mkdir -p $$o; \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(OLD) $(FROZEN) --set online.steps=$(OLDTICKS) \
	      --init $(OUT)/$(W)_s$$s/ckpt_$$(printf %08d $$t) --seed 10$$s -o $$o/s$$s > $$o/s$$s.log 2>&1 & \
	  done; wait; \
	  for s in $(SEEDS); do rm -f $$o/s$$s/ckpt_*.bin; done; \
	done

floor:
	cd $(DEV) && for w in food court; do \
	  case $$w in food) ws="$(FOOD)"; n=2000;; court) ws="$(COURT)"; n=6000;; esac; \
	  o=$(OUT)/floor_$$w; mkdir -p $$o; \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $$ws $(FROZEN) --set online.steps=$$n \
	      --seed 10$$s -o $$o/s$$s > $$o/s$$s.log 2>&1 & \
	  done; wait; \
	  for s in $(SEEDS); do rm -f $$o/s$$s/ckpt_*.bin; done; \
	done

back:
	cd $(DEV) && { for s in $(SEEDS); do \
	  env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(OLD) --init $(OUT)/$(W)_s$$s/ckpt_00060000 --seed $$s \
	    --set online.steps=30000 --set name=$(BACK) -o $(OUT)/$(BACK)_s$$s > $(OUT)/$(BACK)_s$$s.log 2>&1 & \
	done; wait; }

table:
	cd $(DEV) && $(PY) analyze_lives.py two-worlds --seeds $(SEEDS)
