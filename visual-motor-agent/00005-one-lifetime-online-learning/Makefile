# Makefile for experiment vm-00005-one-lifetime-online-learning
#
# Reproduces the runs from the source tagged exp/vm-00005 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00005/RRR_<name>_s<seed>/ (explicit -o: the runs start
# in parallel and must not race for run numbers).
#
# Usage:
#   make runs      — 9 independent lifetimes in parallel (8 threads each): L seeds 1-4, P seeds 1-3, B seeds 1-2
#   make analysis  — planner success on the test episodes and imagination tests at lifetime checkpoints
#   make continue  — each L lifetime continues from its 200k weights in a new world (seed 101) for 5k steps:
#                    learning as in the lifetime (fresh optimizer, empty memory), or frozen (lr 0, same loop)
#   make r4        — the frozen planner vs online play: online.py's acting rule with frozen weights and its parts,
#                    another planner seed, fresh episodes (env seed 4242), where the planner fails

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00005
RUN = cd $(DEV) && nohup $(PY) online.py --config configs/L_online.json
STEPS = 10000 20000 30000 50000 100000 150000 200000
RULE = --explore-gain 0.02 --explore-eps 0.05

.PHONY: all runs analysis continue r4

all: runs analysis continue r4

runs:
	mkdir -p $(OUT)
	for s in 1 2 3 4; do d=$(OUT)/00$$s\_L_online_s$$s; mkdir -p $$d; \
	  $(RUN) --seed $$s --set name=L_online -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2 3; do d=$(OUT)/00$$((s+4))_P_pure_online_s$$s; mkdir -p $$d; \
	  $(RUN) --seed $$s --set name=P_pure_online --set online.batch=1 --set online.replay=8 -o $$d > $$d/stdout.log 2>&1 & done; \
	for s in 1 2; do d=$(OUT)/00$$((s+7))_B_online_babble_s$$s; mkdir -p $$d; \
	  $(RUN) --seed $$s --set name=B_online_babble --set online.explore_eps=1.0 -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

analysis:
	cd $(DEV) && for r in $(OUT)/00[1-9]_*; do $(PY) plan.py $$r --ckpts $(STEPS) --threads 16; \
	  for s in 50000 100000 200000; do $(PY) analyze_imagination.py $$r --ckpt $$(printf "ckpt_%08d.bin" $$s) --threads 12; done; done

continue:
	for s in 1 2 3 4; do i=$$((8+2*s)); init=$(OUT)/00$$s\_L_online_s$$s/ckpt_00200000.bin; \
	  d=$(OUT)/0$$i\_L200k_learn_s$$s; mkdir -p $$d; \
	  $(RUN) --seed 101 --init $$init --set name=L200k_learn --set online.steps=5000 -o $$d > $$d/stdout.log 2>&1 & \
	  d=$(OUT)/0$$((i+1))_L200k_frozen_s$$s; mkdir -p $$d; \
	  $(RUN) --seed 101 --init $$init --set name=L200k_frozen --set online.steps=5000 --set ppo.lr=0 \
	  -o $$d > $$d/stdout.log 2>&1 & done; \
	wait

r4:
	for s in 1 2 3 4; do r=$(OUT)/00$$s\_L_online_s$$s; ( cd $(DEV) && \
	  for st in 100000 150000 200000; do ck=$$(printf "ckpt_%08d" $$st); \
	    $(PY) plan.py $$r --ckpt $$ck.bin $(RULE) --threads 16 -o $$r/plan/$${ck}_rule.json; done; \
	  $(PY) plan.py $$r --ckpt ckpt_00200000.bin --env-seed 4242 --threads 16 -o $$r/plan/ckpt_00200000_env4242.json; \
	  $(PY) plan.py $$r --ckpt ckpt_00200000.bin $(RULE) --env-seed 4242 --threads 16 \
	    -o $$r/plan/ckpt_00200000_rule_env4242.json ) & done; \
	wait
	for rs in 002_L_online_s2:200000 003_L_online_s3:100000 003_L_online_s3:200000; do \
	  r=$(OUT)/$${rs%%:*}; ck=$$(printf "ckpt_%08d" $${rs##*:}); ( cd $(DEV) && \
	  $(PY) plan.py $$r --ckpt $$ck.bin --explore-gain 0.02 --threads 16 -o $$r/plan/$${ck}_gate.json; \
	  $(PY) plan.py $$r --ckpt $$ck.bin --explore-eps 0.05 --threads 16 -o $$r/plan/$${ck}_eps.json ) & done; \
	for s in 2 3; do r=$(OUT)/00$$s\_L_online_s$$s; ( cd $(DEV) && \
	  $(PY) plan.py $$r --ckpt ckpt_00200000.bin --seed 1 --threads 16 -o $$r/plan/ckpt_00200000_seed1.json; \
	  $(PY) analyze_planner.py $$r --ckpt ckpt_00200000.bin --threads 16 ) & done; \
	wait
