# Makefile for experiment vm-00016-target-architecture
#
# Reproduces the runs from the source tagged exp/vm-00016 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the CUDA multi-process service (nvidia-cuda-mps-control -d, the same
# CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00016/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs     — the target architecture (../ARCHITECTURE.md) against its parts, in vm-00015's slow-hunger world
#                   (continual torus, 2 hazards at 0.02, hunger 0.001 per step), 8x8 patch tokens, policy head detached
#                   with a hidden layer, memory U, 60k ticks, seeds 1-3 (lifetime.py):
#                   A (+ the act token), AS (+ the signed command), ASB (+ 32 bodies for 10k ticks, halving every 5k).
#                   Control: vm-00015 S8D (the same agent without these, online.py).
#   make analysis — the frozen tests, hazard contact (with the act token's output as z), token probes, aim
# Exploration is constant (eps 0.2) in every arm, as in the control.

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00016
T = --set env.respawn=true --set env.end_on_success=false --set env.max_steps=1000000000 --set env.wrap=true \
    --set env.hazards=2 --set env.hazard_cost=0.02 --set env.h_rate=0.001
A = cd $(DEV) && $(PY) lifetime.py --config configs/AHR_gated.json --set model.pi_hidden=128 --set online.store_relief=0 \
    --set online.store_surprise=0 --set online.store_base=1 --set online.device=cuda --set online.steps=60000 \
    --set model.patch=8 --set ppo.pi_detach=1 --set model.act_token=1 --set ppo.act_coef=0.5 $(T)
S = --set online.goal_mode=delta
BOD = --set online.bodies=32 --set online.bodies_hold=10000 --set online.bodies_halve=5000

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	for s in 1 2 3; do \
	  $(A) --seed $$s --set name=A -o $(OUT)/00$$s\_A_s$$s > $(OUT)/00$$s\_A.log 2>&1 & \
	  $(A) $(S) --seed $$s --set name=AS -o $(OUT)/00$$((s+3))_AS_s$$s > $(OUT)/00$$((s+3))_AS.log 2>&1 & \
	  $(A) $(S) $(BOD) --seed $$s --set name=ASB -o $(OUT)/00$$((s+6))_ASB_s$$s > $(OUT)/00$$((s+6))_ASB.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_s*; do $(PY) online.py --eval $$r --ckpts 0 10000 20000 40000 60000 --eval-device cuda; done
	cd $(DEV) && for st in 20000 40000 60000; do $(PY) analyze_hazards.py contact $(OUT)/0*_s* --step $$st; done
	cd $(DEV) && $(PY) analyze_patch.py tokens $(OUT)/0*_s* --step 60000 && \
	  $(PY) analyze_self_actor.py aim $(OUT)/0*_s* --step 60000 --device cuda
