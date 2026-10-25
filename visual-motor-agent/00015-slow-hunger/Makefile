# Makefile for experiment vm-00015-slow-hunger
#
# Reproduces the runs from the source tagged exp/vm-00015 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the CUDA multi-process service (nvidia-cuda-mps-control -d, the same
# CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00015/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs     — the continual torus with 2 hazards, hunger rising 0.001 per step and 0.02 more inside a hazard
#                   (so that it rarely reaches its cap of 1), seeds 1-3: S8D (patch tokens 8x8, move loss detached,
#                   60k), S8DW (S8D + drive_changed_weight 10), SD (DMI, the CNN model, 60k), SL (the planner agent L,
#                   40k). 12 lifetimes on the GPU.
#   make analysis — the frozen tests (basic agents), hazard contact (all) and the planner's imagined hunger (SL)
# Controls at the old rates (0.005 per step): vm-00014 P8D (hazards 0.1), vm-00011 TH and vm-00012 LTH (hazards 0.02).

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00015
T = --set env.respawn=true --set env.end_on_success=false --set env.max_steps=1000000000 --set env.wrap=true \
    --set env.hazards=2 --set env.hazard_cost=0.02 --set env.h_rate=0.001
U = --set online.store_relief=0 --set online.store_surprise=0 --set online.store_base=1
P8 = cd $(DEV) && $(PY) online.py --config configs/AHR_gated.json --set model.pi_hidden=128 $(U) --set online.device=cuda \
     --set online.steps=60000 --set model.patch=8 --set ppo.pi_detach=1 $(T)
DMI = cd $(DEV) && $(PY) online.py --config configs/AHR_gated.json --set ppo.pi_detach=1 --set model.pi_hidden=128 \
      --set model.inv_hidden=128 --set ppo.inv_coef=0.1 $(U) --set online.device=cuda --set online.steps=60000 $(T)
L = cd $(DEV) && $(PY) online.py --config configs/L_gated.json --set online.device=cuda --set online.steps=40000 $(T)

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	for s in 1 2 3; do \
	  $(P8) --seed $$s --set name=S8D -o $(OUT)/00$$s\_S8D_s$$s > $(OUT)/00$$s\_S8D.log 2>&1 & \
	  $(P8) --seed $$s --set name=S8DW --set ppo.drive_changed_weight=10 -o $(OUT)/00$$((s+3))_S8DW_s$$s > $(OUT)/00$$((s+3))_S8DW.log 2>&1 & \
	  $(DMI) --seed $$s --set name=SD -o $(OUT)/00$$((s+6))_SD_s$$s > $(OUT)/00$$((s+6))_SD.log 2>&1 & \
	  $(L) --seed $$s --set name=SL -o $(OUT)/$$(printf %03d $$((s+9)))_SL_s$$s > $(OUT)/$$(printf %03d $$((s+9)))_SL.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/00[1-9]_*_s*; do $(PY) online.py --eval $$r --ckpts 0 20000 40000 60000 --eval-device cuda; done
	cd $(DEV) && for st in 20000 40000; do $(PY) analyze_hazards.py contact $(OUT)/0*_s* --step $$st; done
	cd $(DEV) && $(PY) analyze_hazards.py contact $(OUT)/00[1-9]_*_s* --step 60000
	cd $(DEV) && for st in 20000 40000; do $(PY) analyze_hazards.py hazard $(OUT)/01*_SL_s* --step $$st --device cuda; done
