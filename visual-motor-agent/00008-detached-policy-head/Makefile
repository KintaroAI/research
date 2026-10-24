# Makefile for experiment vm-00008-detached-policy-head
#
# Reproduces the runs from the source tagged exp/vm-00008 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00008/RRR_<name>_s<seed>/ (explicit -o: parallel runs).
#
# Usage:
#   make detached  — the basic agent (AHR_gated, as in vm-00007) with the policy head detached (ppo.pi_detach=1:
#                    the move loss trains the head only), 4 threads, 50k steps: D (no inverse dynamics) and DI
#                    (inverse dynamics), memory U and R, seeds 1-2 (8 lifetimes in parallel)
#   make mlp       — stage 2: DMI, a detached policy head with a hidden layer (model.pi_hidden=128) and inverse
#                    dynamics, memory U and R, seeds 1-2
#   make analysis  — own moves on the 128 test episodes every 10k steps
#   make diagnostics — feature probes at 10k/30k/50k; aiming, walls and the command effect at 30k/50k
#
# Controls are vm-00007's runs (the same code, defaults): U, R = exp_00007/001-004; I+U, I+R = exp_00007/017-020.
# Arms: U stores every chunk; R stores relief chunks and 10% of the rest (the first 1,000 steps in full).

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00008
U  = --set online.store_relief=0 --set online.store_surprise=0 --set online.store_base=1
R  = --set online.store_relief=1 --set online.store_surprise=0 --set online.store_base=0.1
BASIC = cd $(DEV) && nohup $(PY) online.py --config configs/AHR_gated.json --set ppo.pi_detach=1
INV = --set model.inv_hidden=128 --set ppo.inv_coef=0.1
MLP = --set model.pi_hidden=128

.PHONY: all runs detached mlp analysis diagnostics

all: runs analysis diagnostics

runs:
	$(MAKE) detached & $(MAKE) mlp & wait

detached:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/00$$s\_AHR_D_U_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_D_U $(U) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+2))_AHR_D_R_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_D_R $(R) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+4))_AHR_DI_U_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_DI_U $(INV) $(U) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+6))_AHR_DI_R_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_DI_R $(INV) $(R) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

mlp:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+8)))_AHR_DMI_U_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_DMI_U $(MLP) $(INV) $(U) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+10)))_AHR_DMI_R_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_DMI_R $(MLP) $(INV) $(R) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_AHR_*_s*; do $(PY) online.py --eval $$r --ckpts 0 10000 20000 30000 40000 50000; done

diagnostics:
	cd $(DEV) && for st in 30000 50000; do $(PY) analyze_self_actor.py aim $(OUT)/0*_AHR_*_s* --step $$st; done
	cd $(DEV) && for st in 10000 30000 50000; do $(PY) analyze_self_actor.py probe $(OUT)/0*_AHR_*_s* --step $$st; done
