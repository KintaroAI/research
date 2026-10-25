# Makefile for experiment vm-00012-planner-continual-world
#
# Reproduces the runs from the source tagged exp/vm-00012 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so); run with the CUDA multi-process service (nvidia-cuda-mps-control -d, the same
# CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00012/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs — the planner agent L (configs/L_gated.json, memory U, online.device=cuda), 40k steps, seeds 1-3, in the
#               continual torus world: LT, LTH (+ 2 hazards), LTM (+ moving circles), LTE (+ egocentric view)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00012
L = cd $(DEV) && $(PY) online.py --config configs/L_gated.json --set online.device=cuda --set online.steps=40000
T = --set env.respawn=true --set env.end_on_success=false --set env.max_steps=1000000000 --set env.wrap=true

.PHONY: all runs strong

all: runs

runs:
	mkdir -p $(OUT)
	for s in 1 2 3; do \
	  $(L) --seed $$s --set name=LT $(T) -o $(OUT)/00$$s\_LT_s$$s > $(OUT)/00$$s\_LT.log 2>&1 & \
	  $(L) --seed $$s --set name=LTH $(T) --set env.hazards=2 -o $(OUT)/00$$((s+3))_LTH_s$$s > $(OUT)/00$$((s+3))_LTH.log 2>&1 & \
	  $(L) --seed $$s --set name=LTM $(T) --set env.target_speed=0.5 -o $(OUT)/00$$((s+6))_LTM_s$$s > $(OUT)/00$$((s+6))_LTM.log 2>&1 & \
	  $(L) --seed $$s --set name=LTE $(T) --set env.egocentric=true -o $(OUT)/$$(printf %03d $$((s+9)))_LTE_s$$s > $(OUT)/$$(printf %03d $$((s+9)))_LTE.log 2>&1 & \
	done; wait

# Follow-up: hazards 5x as costly (0.1 hunger per step inside), the planner agent (LTH10, 40k steps) and vm-00011's
# basic agent DMI (DTH10, 60k steps), seeds 1-3, runs 013-018
DMI = cd $(DEV) && $(PY) online.py --config configs/AHR_gated.json --set ppo.pi_detach=1 --set model.pi_hidden=128 \
      --set model.inv_hidden=128 --set ppo.inv_coef=0.1 --set online.store_relief=0 --set online.store_surprise=0 \
      --set online.store_base=1 --set online.device=cuda --set online.steps=60000
HZ10 = --set env.hazards=2 --set env.hazard_cost=0.1
strong:
	mkdir -p $(OUT)
	for s in 1 2 3; do \
	  $(L) --seed $$s --set name=LTH10 $(T) $(HZ10) -o $(OUT)/$$(printf %03d $$((s+12)))_LTH10_s$$s > $(OUT)/$$(printf %03d $$((s+12)))_LTH10.log 2>&1 & \
	  $(DMI) --seed $$s --set name=DTH10 $(T) $(HZ10) -o $(OUT)/$$(printf %03d $$((s+15)))_DTH10_s$$s > $(OUT)/$$(printf %03d $$((s+15)))_DTH10.log 2>&1 & \
	done; wait
