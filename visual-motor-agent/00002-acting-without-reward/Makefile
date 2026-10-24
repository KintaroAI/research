# Makefile for experiment vm-00002-acting-without-reward
#
# Reproduces the runs from the source tagged exp/vm-00002 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00002/ (run numbers are assigned automatically;
# --set name=... gives each run the name the analysis globs expect). Runs 001-004 (collapsed features, see the
# log) used code that no longer exists and are not reproduced.
#
# Usage:
#   make babble    — reward-free babbling with latent dynamics: click probability 0.1 (P_dyn) and 0.5 (P_dyn_p50)
#   make plan      — the CEM planner at every milestone of both runs, plus the random baselines and the control
#   make analysis  — imagination accuracy at every milestone; planner settings sweep and failure trace (2.1M)
#   make video     — part 4 of the video (the planner at each checkpoint)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00002
STEPS = 0 65536 131072 262144 393216 524288 786432 1048576 1572864 2097152
MS = --set 'train.milestones=[0,65536,131072,262144,393216,524288,786432,1048576,1572864,2097152]'
TRAIN = cd $(DEV) && $(PY) train.py --exp 2 --seed 1 --total-steps 2097152 --set train.threads=64 $(MS)
RUN1 = $(firstword $(wildcard $(OUT)/*_P_dyn_s1))
RUN2 = $(firstword $(wildcard $(OUT)/*_P_dyn_p50_s1))
CK = ckpt_02097152.bin

.PHONY: all babble plan analysis video

all: babble plan analysis video

babble:
	$(TRAIN) --config configs/P_dyn.json --set name=P_dyn
	$(TRAIN) --config configs/P_dyn_p50.json --set name=P_dyn_p50

plan:
	@test -n "$(RUN1)" -a -n "$(RUN2)" || { echo "runs not found: make babble"; exit 1; }
	cd $(DEV) && $(PY) plan.py $(RUN1) --ckpts $(STEPS) --threads 32
	cd $(DEV) && $(PY) plan.py $(RUN2) --ckpts $(STEPS) --threads 32
	cd $(DEV) && $(PY) plan.py $(RUN1) --policy random --click-p 0.5 -o $(OUT)/plan_random_click0.5.json
	cd $(DEV) && $(PY) plan.py $(RUN1) --policy random --click-p 0.1 -o $(OUT)/plan_random_click0.1.json
	cd $(DEV) && $(PY) plan.py $(RUN1) --ckpt $(CK) --random-cost -o $(OUT)/plan_random_cost.json

analysis:
	cd $(DEV) && for r in $(RUN1) $(RUN2); do for s in $(STEPS); do \
	  $(PY) analyze_imagination.py $$r --ckpt $$(printf "ckpt_%08d.bin" $$s) --threads 16; done; done
	mkdir -p $(OUT)/planner_sweep
	cd $(DEV) && $(PY) plan.py $(RUN1) --ckpt $(CK) --beta 2 -o $(OUT)/planner_sweep/H12_b2_s0.6.json
	cd $(DEV) && $(PY) plan.py $(RUN1) --ckpt $(CK) --beta 2 --sigma0 1.0 -o $(OUT)/planner_sweep/H12_b2_s1.0.json
	cd $(DEV) && $(PY) plan.py $(RUN1) --ckpt $(CK) --beta 3 --sigma0 1.0 -o $(OUT)/planner_sweep/H12_b3_s1.0.json
	cd $(DEV) && $(PY) plan.py $(RUN1) --ckpt $(CK) --sigma0 1.0 -o $(OUT)/planner_sweep/H12_b0_s1.0.json
	cd $(DEV) && $(PY) plan.py $(RUN1) --ckpt $(CK) --horizon 20 -o $(OUT)/planner_sweep/H20_b0_s0.6.json
	cd $(DEV) && $(PY) plan.py $(RUN1) --ckpt $(CK) --horizon 20 --beta 2 --sigma0 1.0 -o $(OUT)/planner_sweep/H20_b2_s1.0.json
	cd $(DEV) && $(PY) analyze_planner.py $(RUN1) --ckpt $(CK)

video:
	cd $(DEV) && $(PY) make_video.py --planner $(RUN1) --only-planner -o $(OUT)/video --threads 32
