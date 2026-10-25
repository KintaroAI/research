# Makefile for experiment vm-00013-drive-changed-weight
#
# Reproduces the runs from the source tagged exp/vm-00013 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the CUDA multi-process service (nvidia-cuda-mps-control -d, the same
# CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00013/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs     — ppo.drive_changed_weight=10 in the continual torus with 2 hazards, seeds 1-3:
#                   W10P (planner agent L, hazards 0.1 per step, 40k), W10D (DMI, 0.1, 60k),
#                   W02P (L, 0.02, 40k), W02D (DMI, 0.02, 60k). Controls: vm-00012 LTH10, DTH10, LTH; vm-00011 TH.
#   make strong   — follow-up: W100P (planner, hazards 0.1 per step, drive_changed_weight=100), seeds 1-3, runs 013-015
#   make analysis — analyze_hazards.py contact and hazard at the end

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00013
T = --set env.respawn=true --set env.end_on_success=false --set env.max_steps=1000000000 --set env.wrap=true \
    --set env.hazards=2 --set ppo.drive_changed_weight=10
L = cd $(DEV) && $(PY) online.py --config configs/L_gated.json --set online.device=cuda --set online.steps=40000 $(T)
DMI = cd $(DEV) && $(PY) online.py --config configs/AHR_gated.json --set ppo.pi_detach=1 --set model.pi_hidden=128 \
      --set model.inv_hidden=128 --set ppo.inv_coef=0.1 --set online.store_relief=0 --set online.store_surprise=0 \
      --set online.store_base=1 --set online.device=cuda --set online.steps=60000 $(T)

.PHONY: all runs strong analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	for s in 1 2 3; do \
	  $(L) --seed $$s --set name=W10P --set env.hazard_cost=0.1 -o $(OUT)/00$$s\_W10P_s$$s > $(OUT)/00$$s\_W10P.log 2>&1 & \
	  $(DMI) --seed $$s --set name=W10D --set env.hazard_cost=0.1 -o $(OUT)/00$$((s+3))_W10D_s$$s > $(OUT)/00$$((s+3))_W10D.log 2>&1 & \
	  $(L) --seed $$s --set name=W02P -o $(OUT)/00$$((s+6))_W02P_s$$s > $(OUT)/00$$((s+6))_W02P.log 2>&1 & \
	  $(DMI) --seed $$s --set name=W02D -o $(OUT)/$$(printf %03d $$((s+9)))_W02D_s$$s > $(OUT)/$$(printf %03d $$((s+9)))_W02D.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && $(PY) analyze_hazards.py contact $(OUT)/0*_s* --step 40000 && \
	  $(PY) analyze_hazards.py hazard $(OUT)/00[1-3]_W10P_s* $(OUT)/00[7-9]_W02P_s* --step 40000 --device cuda

# Follow-up: weight 100 (planner, hazards 0.1 per step), runs 013-015
strong:
	mkdir -p $(OUT)
	for s in 1 2 3; do \
	  $(L) --seed $$s --set name=W100P --set env.hazard_cost=0.1 --set ppo.drive_changed_weight=100 \
	    -o $(OUT)/$$(printf %03d $$((s+12)))_W100P_s$$s > $(OUT)/$$(printf %03d $$((s+12)))_W100P.log 2>&1 & \
	done; wait
