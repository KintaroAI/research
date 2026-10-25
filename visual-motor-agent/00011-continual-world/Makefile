# Makefile for experiment vm-00011-continual-world
#
# Reproduces the runs from the source tagged exp/vm-00011 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so).
# Outputs go to ~/data/research/visual-motor-agent/exp_00011/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs      — vm-00008's DMI agent (online.device=cuda), 60k steps, seeds 1-3, in four continual worlds:
#                    C (respawn, never reset), CM (+ moving circles), CS (+ small circles), CH (+ 2 hazards)
#                    (stopped at about 12k steps: the agent starves at the walls; see README)
#   make torus     — the same agent in the wrap-around world (env.wrap), 60k steps, seeds 1-3, runs 013-024:
#                    T (torus), TE (+ egocentric view), TM (+ moving circles), TH (+ 2 hazards)
#   make analysis  — the episodic test and the continual forage test every 20k steps
#
# Runs share the GPU; with the CUDA multi-process service (nvidia-cuda-mps-control -d, and the same
# CUDA_MPS_PIPE_DIRECTORY in the environment of the runs) their small kernels run concurrently.

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00011
DMI = cd $(DEV) && $(PY) online.py --config configs/AHR_gated.json --set ppo.pi_detach=1 --set model.pi_hidden=128 \
      --set model.inv_hidden=128 --set ppo.inv_coef=0.1 --set online.store_relief=0 --set online.store_surprise=0 \
      --set online.store_base=1 --set online.device=cuda --set online.steps=60000
C = --set env.respawn=true --set env.end_on_success=false --set env.max_steps=1000000000
M = --set env.target_speed=0.5
S = --set env.radius_min=6 --set env.radius_max=8
H = --set env.hazards=2

W = --set env.wrap=true
E = --set env.egocentric=true

.PHONY: all runs torus analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	for s in 1 2 3; do \
	  $(DMI) --seed $$s --set name=C $(C) -o $(OUT)/00$$s\_C_s$$s > $(OUT)/00$$s\_C.log 2>&1 & \
	  $(DMI) --seed $$s --set name=CM $(C) $(M) -o $(OUT)/00$$((s+3))_CM_s$$s > $(OUT)/00$$((s+3))_CM.log 2>&1 & \
	  $(DMI) --seed $$s --set name=CS $(C) $(S) -o $(OUT)/00$$((s+6))_CS_s$$s > $(OUT)/00$$((s+6))_CS.log 2>&1 & \
	  $(DMI) --seed $$s --set name=CH $(C) $(H) -o $(OUT)/$$(printf %03d $$((s+9)))_CH_s$$s > $(OUT)/$$(printf %03d $$((s+9)))_CH.log 2>&1 & \
	done; wait

torus:
	mkdir -p $(OUT)
	for s in 1 2 3; do \
	  $(DMI) --seed $$s --set name=T $(C) $(W) -o $(OUT)/$$(printf %03d $$((s+12)))_T_s$$s > $(OUT)/$$(printf %03d $$((s+12)))_T.log 2>&1 & \
	  $(DMI) --seed $$s --set name=TE $(C) $(W) $(E) -o $(OUT)/$$(printf %03d $$((s+15)))_TE_s$$s > $(OUT)/$$(printf %03d $$((s+15)))_TE.log 2>&1 & \
	  $(DMI) --seed $$s --set name=TM $(C) $(W) $(M) -o $(OUT)/$$(printf %03d $$((s+18)))_TM_s$$s > $(OUT)/$$(printf %03d $$((s+18)))_TM.log 2>&1 & \
	  $(DMI) --seed $$s --set name=TH $(C) $(W) $(H) -o $(OUT)/$$(printf %03d $$((s+21)))_TH_s$$s > $(OUT)/$$(printf %03d $$((s+21)))_TH.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_s*; do $(PY) online.py --eval $$r --ckpts 0 20000 40000 60000; done
