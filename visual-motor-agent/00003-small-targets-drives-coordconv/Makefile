# Makefile for experiment vm-00003-small-targets-drives-coordconv
#
# Reproduces the runs from the source tagged exp/vm-00003 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00003/ (run numbers are assigned automatically;
# --set name=... gives each run the name the analysis globs expect).
# Needs the vm-00001 babbler checkpoints (runs 012 and 015 at 1M steps): make -C ../00001-pixel-agent-and-babbling babble
#
# Usage:
#   make small     — radius 4-8: PPO from scratch vs PPO initialized from the babbler (transfer architecture)
#   make arms      — the plan's arms B, E, F, D on the skip architecture (radius 12-16)
#   make plain     — plain architecture (no CoordConv), seeds 2 and 3
#   make analysis  — sample-efficiency tables

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00003
OUT1 = $(HOME)/data/research/visual-motor-agent/exp_00001
T64 = --set train.threads=64
ARCH = --set model.coord=1 --set model.pred_residual=1 --set model.pred_skip=1
ARCH8 = $(ARCH) --set model.ctx=8 --set model.inv_hidden=128
BABBLE1 = $(firstword $(wildcard $(OUT1)/*_P_babble_inv_s1/ckpt_01048576.bin))
BABBLE2 = $(firstword $(wildcard $(OUT1)/*_P_babble_inv_s2/ckpt_01048576.bin))
TRAIN = cd $(DEV) && $(PY) train.py --exp 3

.PHONY: all small arms plain analysis

all: small arms plain analysis

small:
	@test -n "$(BABBLE1)" -a -n "$(BABBLE2)" || { echo "babbler checkpoints not found: see the header"; exit 1; }
	$(TRAIN) --config configs/A_small.json --seed 1 --total-steps 2000000 $(ARCH8) $(T64) --set name=A_small_ctx8inv
	$(TRAIN) --config configs/A_small.json --seed 1 --total-steps 2000000 $(ARCH8) $(T64) --set name=T_small_from_babble \
	  --init $(BABBLE1)
	$(TRAIN) --config configs/A_small.json --seed 2 --total-steps 2000000 $(ARCH8) $(T64) --set name=A_small_ctx8inv
	$(TRAIN) --config configs/A_small.json --seed 2 --total-steps 2000000 $(ARCH8) $(T64) --set name=T_small_from_babble \
	  --init $(BABBLE2)

arms:
	for c in B_exploration:B_arch E_drives_only:E_arch F_full_minus_pred:F_arch D_combined_w10:D_arch; do \
	  $(TRAIN) --config configs/$${c%%:*}.json --seed 1 --total-steps 1000000 $(ARCH) $(T64) --set name=$${c##*:}; \
	done

plain:
	for s in 2 3; do $(TRAIN) --config configs/A_baseline.json --seed $$s --total-steps 1000000 $(T64) --set name=A_plain; done

analysis:
	cd $(DEV) && $(PY) summarize_runs.py $(OUT)/*_A_small_ctx8inv_s* $(OUT)/*_T_small_from_babble_s* \
	  --budgets 262144 507904 753664 999424 1245184 1998848
	cd $(DEV) && $(PY) summarize_runs.py $(OUT1)/*_A_arch_s* $(OUT)/*_B_arch_s1 $(OUT)/*_E_arch_s1 $(OUT)/*_F_arch_s1 \
	  $(OUT1)/*_C_arch_s* $(OUT)/*_D_arch_s1 --budgets 180224 344064 507904
	cd $(DEV) && $(PY) summarize_runs.py $(OUT1)/*_A_baseline_s1 $(OUT)/*_A_plain_s* --budgets 180224 344064 507904
