# Makefile for experiment vm-00006-basic-next-state-predictor
#
# Reproduces the runs from the source tagged exp/vm-00006 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00006/RRR_<name>_s<seed>/ (explicit -o: parallel runs).
#
# Usage:
#   make runs      — 8 independent lifetimes in parallel (4 threads each): A, A+H (eps 0.05, 0.2), A+H+R
#   make analysis  — the agents' own moves on the 128 test episodes at every 10k-step checkpoint
#   make diagnostics — (dev/analyze_self_actor.py, from exp/vm-00007 on) input sensitivity and feature norms,
#                    aiming and walls, and momentum: why the agents do not aim

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00006
RUN = cd $(DEV) && nohup $(PY) online.py --config configs/A_basic.json
H = --set online.goal=1 --set model.use_prev_reward=1
STEPS = 0 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000

.PHONY: all runs analysis diagnostics

all: runs analysis diagnostics

runs:
	mkdir -p $(OUT)
	for s in 1 2; do d=$(OUT)/00$$s\_A_basic_e05_s$$s; mkdir -p $$d; \
	  $(RUN) --seed $$s --set name=A_basic_e05 --set online.explore_eps=0.05 -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+2))_AH_e05_s$$s; mkdir -p $$d; \
	  $(RUN) --seed $$s --set name=AH_e05 $(H) --set online.explore_eps=0.05 -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+4))_AH_e20_s$$s; mkdir -p $$d; \
	  $(RUN) --seed $$s --set name=AH_e20 $(H) --set online.explore_eps=0.2 -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+6))_AHR_e20_s$$s; mkdir -p $$d; \
	  $(RUN) --seed $$s --set name=AHR_e20 $(H) --set online.explore_eps=0.2 --set online.batch=8 \
	  --set online.replay=100000 -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_s*; do $(PY) online.py --eval $$r --ckpts $(STEPS); done

diagnostics:
	cd $(DEV) && for st in 10000 20000 30000 50000 70000 100000; do \
	  $(PY) analyze_self_actor.py inputs $(OUT)/0*_s* --step $$st; done
	cd $(DEV) && $(PY) analyze_self_actor.py aim $(OUT)/005_AH_e20_s1 $(OUT)/006_AH_e20_s2 --step 0
	cd $(DEV) && $(PY) analyze_self_actor.py aim $(OUT)/0*_s* --step 100000
	cd $(DEV) && $(PY) analyze_self_actor.py momentum $(OUT)/005_AH_e20_s1 --step 0
	cd $(DEV) && $(PY) analyze_self_actor.py momentum $(OUT)/002_A_basic_e05_s2 $(OUT)/005_AH_e20_s1 \
	  $(OUT)/007_AHR_e20_s1 $(OUT)/008_AHR_e20_s2 --step 100000
