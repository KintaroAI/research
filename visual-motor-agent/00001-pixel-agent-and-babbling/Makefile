# Makefile for experiment vm-00001-pixel-agent-and-babbling
#
# Reproduces the key runs from the source tagged exp/vm-00001 (dev/ at that tag).
# Outputs go to ~/data/research/visual-motor-agent/exp_00001/ (run numbers are assigned automatically).
#
# Usage:
#   make setup      — venv + build + tests
#   make baselines  — random-policy baselines (both radius ranges)
#   make babble     — reward-free babbling: plain decoder (fails), skip decoder, skip + inverse dynamics (2 seeds)
#   make rl         — PPO arms A vs C (skip architecture, seeds 1-3) and the babbling->PPO transfer pairs (seeds 1-2)
#   make small      — radius 4-8: PPO arms A vs C (skip architecture, seed 1)
#   make analysis   — counterfactual dynamics tests over training + comparison figure + sample-efficiency tables

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00001
T64 = --set train.threads=64
ARCH = --set model.coord=1 --set model.pred_residual=1 --set model.pred_skip=1
ARCH8 = $(ARCH) --set model.ctx=8 --set model.inv_hidden=128
# 1M-step checkpoints of the two inverse+forward babblers (expanded when the rl recipe runs)
BABBLE1 = $(firstword $(wildcard $(OUT)/*_P_babble_inv_s1/ckpt_01048576.bin))
BABBLE2 = $(firstword $(wildcard $(OUT)/*_P_babble_inv_s2/ckpt_01048576.bin))

.PHONY: all setup baselines babble rl small analysis

all: setup baselines babble rl small analysis

setup:
	cd $(DEV) && make setup && make && make test && make pytest

baselines:
	cd $(DEV) && $(PY) random_baseline.py --json $(OUT)/random_baseline_r12_16.json
	cd $(DEV) && $(PY) random_baseline.py --radius-min 4 --radius-max 8 --json $(OUT)/random_baseline_r4_8.json

babble:
	cd $(DEV) && $(PY) train.py --config configs/P_babble.json --seed 1 --total-steps 1000000 $(T64)
	cd $(DEV) && $(PY) train.py --config configs/P_babble_skip_ctx8.json --seed 1 --total-steps 2000000 $(T64)
	cd $(DEV) && $(PY) train.py --config configs/P_babble_inv.json --seed 1 --total-steps 2000000 $(T64)
	cd $(DEV) && $(PY) train.py --config configs/P_babble_inv.json --seed 2 --total-steps 1100000 $(T64)

rl:
	cd $(DEV) && $(PY) train.py --config configs/A_baseline.json --seed 1 --total-steps 1000000 $(T64)
	cd $(DEV) && for s in 1 2 3; do \
	  $(PY) train.py --config configs/A_baseline.json --seed $$s --total-steps 1000000 $(ARCH) $(T64) --set name=A_arch; \
	  $(PY) train.py --config configs/C_prediction_w10.json --seed $$s --total-steps 1000000 $(ARCH) $(T64) --set name=C_arch; \
	done
	# transfer: PPO from scratch vs PPO initialized from the 1M babbler of the same seed (needs make babble)
	@test -n "$(BABBLE1)" -a -n "$(BABBLE2)" || { echo "babbler checkpoints not found: run make babble"; exit 1; }
	cd $(DEV) && $(PY) train.py --config configs/A_baseline.json --seed 1 --total-steps 800000 $(ARCH8) $(T64) --set name=A_ctx8inv
	cd $(DEV) && $(PY) train.py --config configs/A_baseline.json --seed 1 --total-steps 800000 $(ARCH8) $(T64) --set name=T_from_babble_inv --init $(BABBLE1)
	cd $(DEV) && $(PY) train.py --config configs/A_baseline.json --seed 2 --total-steps 600000 $(ARCH8) $(T64) --set name=A_ctx8inv
	cd $(DEV) && $(PY) train.py --config configs/A_baseline.json --seed 2 --total-steps 600000 $(ARCH8) $(T64) --set name=T_from_babble_inv --init $(BABBLE2)

small:
	cd $(DEV) && $(PY) train.py --config configs/A_small.json --seed 1 --total-steps 2000000 $(ARCH) $(T64) --set name=A_small_arch
	cd $(DEV) && $(PY) train.py --config configs/C_small_w10.json --seed 1 --total-steps 2000000 $(ARCH) $(T64) --set name=C_small_arch

analysis:
	cd $(DEV) && for r in $(OUT)/*P_babble*; do $(PY) dynamics_over_training.py $$r; done
	cd $(DEV) && $(PY) compare_dynamics.py $(OUT)/*_P_babble_s1 $(OUT)/*_P_babble_skip_ctx8_s1 $(OUT)/*_P_babble_inv_s1 -o $(OUT)/babbling_comparison.png
	cd $(DEV) && $(PY) summarize_runs.py $(OUT)/*_A_baseline_s1 $(OUT)/*_A_arch_s* $(OUT)/*_C_arch_s* --budgets 180224 344064 507904
	cd $(DEV) && $(PY) summarize_runs.py $(OUT)/*_A_ctx8inv_s* $(OUT)/*_T_from_babble_inv_s* --budgets 98304 180224 344064 507904
	cd $(DEV) && $(PY) summarize_runs.py $(OUT)/*_A_small_arch_s1 $(OUT)/*_C_small_arch_s1 --budgets 16384 262144 507904 753664 999424 1245184 1490944 1998848
