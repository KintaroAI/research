# Makefile for experiment vm-00009-planner-as-teacher
#
# Reproduces the runs from the source tagged exp/vm-00009 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00009/RRR_<name>_s<seed>/ (explicit -o: parallel runs).
#
# Usage:
#   make runs      — the planner agent (L_gated, memory U: as vm-00007's L_U) whose policy head also learns to
#                    predict the executed moves (ppo.move_coef=1), 4 threads, 30k steps, seeds 1-2:
#                    T (linear head, the move loss reaches the features), TD (linear head, detached),
#                    TDM (detached head with a hidden layer of 128)
#   make own-clip  — follow-up: TD+ and TDM+, as TD and TDM with ppo.pi_own_clip=1 (the heads' gradient is clipped
#                    by its own norm, so a detached head cannot change the world model's update), runs 007-010
#   make analysis  — the head's own moves on the 128 test episodes at 0/10k/20k/30k (no planner);
#                    planner success at 10k/20k/30k
#   make diagnostics — the head's aim and the feature probes at 30k
#
# Controls: vm-00007's L_U runs (the same agent without the move loss), exp_00007/009-010 and 025-026.

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00009
U  = --set online.store_relief=0 --set online.store_surprise=0 --set online.store_base=1
PLAN = cd $(DEV) && nohup $(PY) online.py --config configs/L_gated.json --set online.threads=4 $(U) --set ppo.move_coef=1
DET = --set ppo.pi_detach=1
MLP = --set model.pi_hidden=128
OWN = --set ppo.pi_own_clip=1

.PHONY: all runs own-clip analysis diagnostics

all: runs analysis diagnostics

runs:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/00$$s\_L_T_s$$s; mkdir -p $$d; \
	  $(PLAN) --seed $$s --set name=L_T -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+2))_L_TD_s$$s; mkdir -p $$d; \
	  $(PLAN) --seed $$s --set name=L_TD $(DET) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+4))_L_TDM_s$$s; mkdir -p $$d; \
	  $(PLAN) --seed $$s --set name=L_TDM $(DET) $(MLP) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

own-clip:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/00$$((s+6))_L_TDp_s$$s; mkdir -p $$d; \
	  $(PLAN) --seed $$s --set name=L_TDp $(DET) $(OWN) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+8)))_L_TDMp_s$$s; mkdir -p $$d; \
	  $(PLAN) --seed $$s --set name=L_TDMp $(DET) $(MLP) $(OWN) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_L_*_s*; do $(PY) online.py --eval $$r --ckpts 0 10000 20000 30000; done
	cd $(DEV) && for r in $(OUT)/0*_L_*_s*; do $(PY) plan.py $$r --ckpts 10000 20000 30000 --threads 16; done

diagnostics:
	cd $(DEV) && $(PY) analyze_self_actor.py aim $(OUT)/0*_L_*_s* --step 30000
	cd $(DEV) && $(PY) analyze_self_actor.py probe $(OUT)/0*_L_*_s* --step 30000
