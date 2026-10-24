# Makefile for experiment vm-00007-salience-gated-memory
#
# Reproduces the runs from the source tagged exp/vm-00007 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00007/RRR_<name>_s<seed>/ (explicit -o: parallel runs).
#
# Usage:
#   make basic     — the basic agent (AHR_gated: vm-00006 A+H+R with gated memory), 4 threads, 50k steps:
#                    arms U, R, S, RS, seeds 1-2 (8 lifetimes in parallel)
#   make planner   — the planner agent (L_gated: vm-00005 L with gated memory), 4 threads, 30k steps:
#                    arms U, R, RS, seeds 1-2 (6 lifetimes in parallel)
#   make control   — the basic agent's size control U10, seeds 1-2: it stores 10% of the chunks at random,
#                    the base rate of the gated arms, with no relief or surprise rule
#   make inverse   — the basic agent with the inverse-dynamics head (model.inv_hidden=128, ppo.inv_coef=0.1),
#                    arms U and R, seeds 1-2: does a representation that holds the cursor let it aim?
#   make analysis  — basic agent: its own moves on the 128 test episodes every 10k steps;
#                    planner agent: planner success at 10k/20k/30k, imagination tests at 30k
#   make diagnostics — basic agent (dev/analyze_self_actor.py): aiming and walls at 30k/50k, feature probes
#                    at 10k/30k/50k, the aiming signal of hindsight-labelled steps (babbling; R s2, U s1),
#                    momentum at 30k
#
# Arms (online.store_*): U stores every chunk (the control); R stores relief chunks, S the top 10% by
# surprise, RS both; all other chunks with probability 0.1. The first 1,000 steps are stored in full.
# U10 stores every chunk with probability 0.1: as small a memory as the gated arms, uniform in content.

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00007
U  = --set online.store_relief=0 --set online.store_surprise=0 --set online.store_base=1
R  = --set online.store_relief=1 --set online.store_surprise=0 --set online.store_base=0.1
S  = --set online.store_relief=0 --set online.store_surprise=0.1 --set online.store_base=0.1
RS = --set online.store_relief=1 --set online.store_surprise=0.1 --set online.store_base=0.1
U10 = --set online.store_relief=0 --set online.store_surprise=0 --set online.store_base=0.1
BASIC = cd $(DEV) && nohup $(PY) online.py --config configs/AHR_gated.json
INV = --set model.inv_hidden=128 --set ppo.inv_coef=0.1
PLAN = cd $(DEV) && nohup $(PY) online.py --config configs/L_gated.json --set online.threads=4

.PHONY: all runs basic planner control inverse analysis diagnostics

all: runs analysis diagnostics

runs:
	$(MAKE) basic & $(MAKE) planner & $(MAKE) control & $(MAKE) inverse & wait

basic:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/00$$s\_AHR_U_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_U $(U) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+2))_AHR_R_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_R $(R) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+4))_AHR_S_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_S $(S) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+6))_AHR_RS_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_RS $(RS) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

planner:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+8)))_L_U_s$$s; mkdir -p $$d; \
	  $(PLAN) --seed $$s --set name=L_U $(U) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+10)))_L_R_s$$s; mkdir -p $$d; \
	  $(PLAN) --seed $$s --set name=L_R $(R) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+12)))_L_RS_s$$s; mkdir -p $$d; \
	  $(PLAN) --seed $$s --set name=L_RS $(RS) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

control:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+14)))_AHR_U10_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_U10 $(U10) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

inverse:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+16)))_AHR_I_U_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_I_U $(INV) $(U) -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/$$(printf %03d $$((s+18)))_AHR_I_R_s$$s; mkdir -p $$d; \
	  $(BASIC) --seed $$s --set name=AHR_I_R $(INV) $(R) -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_AHR_*_s*; do $(PY) online.py --eval $$r --ckpts 0 10000 20000 30000 40000 50000; done
	cd $(DEV) && for r in $(OUT)/0*_L_*_s*; do $(PY) plan.py $$r --ckpts 10000 20000 30000 --threads 16; \
	  $(PY) analyze_imagination.py $$r --ckpt ckpt_00030000.bin --threads 12; done

diagnostics:
	cd $(DEV) && $(PY) analyze_self_actor.py aim $(OUT)/00[1-8]_AHR_*_s* $(OUT)/01[56]_AHR_U10_s* --step 30000
	cd $(DEV) && $(PY) analyze_self_actor.py aim $(OUT)/0*_AHR_*_s* --step 50000
	cd $(DEV) && for st in 10000 30000 50000; do $(PY) analyze_self_actor.py probe $(OUT)/0*_AHR_*_s* --step $$st; done
	cd $(DEV) && $(PY) analyze_self_actor.py relief
	cd $(DEV) && $(PY) analyze_self_actor.py relief $(OUT)/004_AHR_R_s2 $(OUT)/001_AHR_U_s1 --step 50000
	cd $(DEV) && $(PY) analyze_self_actor.py momentum $(OUT)/001_AHR_U_s1 $(OUT)/003_AHR_R_s1 $(OUT)/005_AHR_S_s1 \
	  $(OUT)/008_AHR_RS_s2 --step 30000
