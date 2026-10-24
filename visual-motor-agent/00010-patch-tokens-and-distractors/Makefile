# Makefile for experiment vm-00010-patch-tokens-and-distractors
#
# Reproduces the runs from the source tagged exp/vm-00010 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00010/RRR_<name>_s<seed>/ (explicit -o: parallel runs).
#
# Usage:
#   make runs        — the basic agent (AHR_gated, memory U, policy head detached: vm-00008's D), 50k steps, seeds 1-2:
#                      P0 (patch tokens, model.patch=16, 8 threads), PD (patch tokens + 8 moving dots, 8 threads),
#                      CD (the CNN model + 8 moving dots, 4 threads). 6 lifetimes in parallel
#   make hidden      — follow-up after vm-00008's DMI aimed (a detached head with a hidden layer, model.pi_hidden=128):
#                      PMD (patch tokens + dots + that head, 8 threads) and CMDI (vm-00008's DMI + dots: CNN, that head,
#                      inverse dynamics, 4 threads), seeds 1-2, runs 007-010
#   make analysis    — own moves on the 128 test episodes every 10k steps (the dot arms with their dots)
#   make diagnostics — probes (z, and per kind of token in the patch arms) at 0/10k/30k/50k; coupling (patch arms)
#                      and action sensitivity at 10k/30k/50k; aim at 30k/50k
#
# Control without dots: vm-00008's D-U runs (exp_00008/001-002), the same agent with the CNN model.

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00010
U = --set online.store_relief=0 --set online.store_surprise=0 --set online.store_base=1
BASIC = cd $(DEV) && nohup $(PY) online.py --config configs/AHR_gated.json --set ppo.pi_detach=1 $(U)
PATCH = --set model.patch=16 --set online.threads=8
DOTS = --set env.distractors=8
HID = --set model.pi_hidden=128
INV = --set model.inv_hidden=128 --set ppo.inv_coef=0.1

.PHONY: all runs hidden analysis diagnostics

all: runs analysis diagnostics

runs:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/00$$s\_AHR_P0_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_P0 $(PATCH) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+2))_AHR_PD_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_PD $(PATCH) $(DOTS) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+4))_AHR_CD_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_CD $(DOTS) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

hidden:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/00$$((s+6))_AHR_PMD_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_PMD $(PATCH) $(DOTS) $(HID) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+8)))_AHR_CMDI_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_CMDI $(DOTS) $(HID) $(INV) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_AHR_*_s*; do $(PY) online.py --eval $$r --ckpts 0 10000 20000 30000 40000 50000; done

diagnostics:
	cd $(DEV) && for st in 0 10000 30000 50000; do $(PY) analyze_self_actor.py probe $(OUT)/0*_AHR_*_s* --step $$st; \
	  $(PY) analyze_patch.py tokens $(OUT)/0*_AHR_P*_s* --step $$st; done
	cd $(DEV) && for st in 10000 30000 50000; do $(PY) analyze_patch.py coupling $(OUT)/0*_AHR_P*_s* --step $$st; \
	  $(PY) analyze_patch.py sensitivity $(OUT)/0*_AHR_*_s* --step $$st; done
	cd $(DEV) && for st in 30000 50000; do $(PY) analyze_self_actor.py aim $(OUT)/0*_AHR_*_s* --step $$st; done
