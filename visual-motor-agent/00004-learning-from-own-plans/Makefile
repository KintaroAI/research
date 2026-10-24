# Makefile for experiment vm-00004-learning-from-own-plans
#
# Reproduces the runs from the source tagged exp/vm-00004 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00004/ (run numbers are assigned automatically;
# --set name=... gives each run the name the analysis globs expect).
# Needs the vm-00002 babblers (005_P_dyn_s1 and 006_P_dyn_p50_s1 at 2.1M steps):
#   make -C ../00002-acting-without-reward babble
#
# Usage:
#   make runs      — S: the model trains on the planner's own experience; B: babbling continues (control);
#                    R: babbling with the hunger head trained on real latents only; SR: S + R;
#                    S006 / B006: S and B from the second babbler (click 0.5)
#   make analysis  — planner success, imagination tests and failure traces at every milestone of every run

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00004
BABBLER = $(firstword $(wildcard $(HOME)/data/research/visual-motor-agent/exp_00002/*_P_dyn_s1/ckpt_02097152.bin))
BABBLER2 = $(firstword $(wildcard $(HOME)/data/research/visual-motor-agent/exp_00002/*_P_dyn_p50_s1/ckpt_02097152.bin))
STEPS = 0 32768 65536 98304 131072 163840 196608 229376 262144
MS = --set 'train.milestones=[0,32768,65536,98304,131072,163840,196608,229376,262144]'
COMMON = --exp 4 --seed 1 --total-steps 262144 --set train.eval_every=0 $(MS)
TRAIN = cd $(DEV) && $(PY) train.py --config configs/P_dyn.json --init $(BABBLER) $(COMMON)
TRAIN2 = cd $(DEV) && $(PY) train.py --config configs/P_dyn_p50.json --init $(BABBLER2) $(COMMON)
RUNS = $(wildcard $(OUT)/0*_s1)

.PHONY: all runs analysis

all: runs analysis

runs:
	@test -n "$(BABBLER)" -a -n "$(BABBLER2)" || { echo "babbler checkpoints not found: see the header"; exit 1; }
	$(TRAIN) --set name=S_self_from005 --set train.actor=planner --set train.threads=64
	$(TRAIN) --set name=B_babble_from005 --set train.threads=32
	$(TRAIN) --set name=R_realhunger_from005 --set ppo.dyn_h_horizon=1 --set train.threads=32
	$(TRAIN) --set name=SR_self_realhunger_from005 --set train.actor=planner --set ppo.dyn_h_horizon=1 --set train.threads=64
	$(TRAIN2) --set name=S_self_from006 --set train.actor=planner --set train.threads=64
	$(TRAIN2) --set name=B_babble_from006 --set train.threads=48

analysis:
	cd $(DEV) && for r in $(RUNS); do $(PY) plan.py $$r --ckpts $(STEPS) --threads 32; \
	  for s in $(STEPS); do ck=$$(printf "ckpt_%08d.bin" $$s); \
	    $(PY) analyze_imagination.py $$r --ckpt $$ck --threads 16; \
	    $(PY) analyze_planner.py $$r --ckpt $$ck; done; done
