# Makefile for experiment vm-00014-unified-patch-transformer
#
# Reproduces the runs from the source tagged exp/vm-00014 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the CUDA multi-process service (nvidia-cuda-mps-control -d, the same
# CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00014/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs     — the basic agent with patch tokens in the continual torus with 2 hazards (0.1 per step), 60k steps,
#                   seeds 1-3: P8D (8x8 patches, move loss detached), P8A (8x8, move loss trains the Transformer),
#                   P16A (16x16 patches), P64A (the whole frame as one token). 12 lifetimes on the GPU.
#   make analysis — the episodic test and the forage test, hazard contact (20k, 40k, 60k); token probes and aim (60k)
# Controls: vm-00012 DTH10 (DMI, the CNN model, the same world), vm-00013 W10D.

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00014
T = --set env.respawn=true --set env.end_on_success=false --set env.max_steps=1000000000 --set env.wrap=true \
    --set env.hazards=2 --set env.hazard_cost=0.1
B = cd $(DEV) && $(PY) online.py --config configs/AHR_gated.json --set model.pi_hidden=128 --set online.store_relief=0 \
    --set online.store_surprise=0 --set online.store_base=1 --set online.device=cuda --set online.steps=60000 $(T)

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	for s in 1 2 3; do \
	  $(B) --seed $$s --set name=P8D --set model.patch=8 --set ppo.pi_detach=1 -o $(OUT)/00$$s\_P8D_s$$s > $(OUT)/00$$s\_P8D.log 2>&1 & \
	  $(B) --seed $$s --set name=P8A --set model.patch=8 --set ppo.pi_detach=0 -o $(OUT)/00$$((s+3))_P8A_s$$s > $(OUT)/00$$((s+3))_P8A.log 2>&1 & \
	  $(B) --seed $$s --set name=P16A --set model.patch=16 --set ppo.pi_detach=0 -o $(OUT)/00$$((s+6))_P16A_s$$s > $(OUT)/00$$((s+6))_P16A.log 2>&1 & \
	  $(B) --seed $$s --set name=P64A --set model.patch=64 --set ppo.pi_detach=0 -o $(OUT)/$$(printf %03d $$((s+9)))_P64A_s$$s > $(OUT)/$$(printf %03d $$((s+9)))_P64A.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/0*_s*; do $(PY) online.py --eval $$r --ckpts 0 20000 40000 60000 --eval-device cuda; done
	cd $(DEV) && for st in 0 20000 40000 60000; do $(PY) analyze_hazards.py contact $(OUT)/0*_s* --step $$st; done
	cd $(DEV) && $(PY) analyze_patch.py tokens $(OUT)/0*_s* --step 60000 && \
	  $(PY) analyze_self_actor.py aim $(OUT)/0*_s* --step 60000
