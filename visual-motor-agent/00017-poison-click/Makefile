# Makefile for experiment vm-00017-poison-click
#
# Reproduces the runs from the source tagged exp/vm-00017 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID: 0 = RTX 3090, 1 = RTX 4090) and the CUDA
# multi-process service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00017/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs     — poison circles: a click inside a hazard sets hunger to 1 (env.hazard_click=1), a click in the red
#                   circle relieves it, contact alone costs nothing; continual torus, 2 hazards, hunger 0.001 per step.
#                   Seeds 1-3: PD (DMI, the CNN basic agent, 60k, GPU 0), PP (8x8 patch tokens, move loss detached, 60k,
#                   GPU 1), PL (the planner agent L, 40k, GPUs 0, 0, 1)
#   make analysis — the click prediction test (analyze_hazards.py click) every 10k, the frozen tests
# The question is the prediction, not behavior: does each model predict what a food click, a poison click and a
# click on nothing do to hunger and to the frame?

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00017
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
T = --set env.respawn=true --set env.end_on_success=false --set env.max_steps=1000000000 --set env.wrap=true \
    --set env.hazards=2 --set env.hazard_cost=0 --set env.hazard_click=1 --set env.h_rate=0.001 --set online.device=cuda
U = --set online.store_relief=0 --set online.store_surprise=0 --set online.store_base=1
BASIC = $(PY) online.py --config configs/AHR_gated.json --set ppo.pi_detach=1 --set model.pi_hidden=128 $(U) \
        --set online.steps=60000 $(T)
DMI = --set model.inv_hidden=128 --set ppo.inv_coef=0.1
L = $(PY) online.py --config configs/L_gated.json --set online.steps=40000 $(T)

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=0 $(BASIC) $(DMI) --seed $$s --set name=PD -o $(OUT)/00$$s\_PD_s$$s > $(OUT)/00$$s\_PD.log 2>&1 & \
	  $(G)=1 $(BASIC) --set model.patch=8 --seed $$s --set name=PP -o $(OUT)/00$$((s+3))_PP_s$$s > $(OUT)/00$$((s+3))_PP.log 2>&1 & \
	  $(G)=$$((s / 3)) $(L) --seed $$s --set name=PL -o $(OUT)/00$$((s+6))_PL_s$$s > $(OUT)/00$$((s+6))_PL.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for st in 0 10000 20000 40000; do $(G)=1 $(PY) analyze_hazards.py click $(OUT)/0*_s* --step $$st --device cuda; done
	cd $(DEV) && $(G)=1 $(PY) analyze_hazards.py click $(OUT)/00[1-6]_*_s* --step 60000 --device cuda
	cd $(DEV) && for r in $(OUT)/00[1-6]_*_s*; do $(G)=1 $(PY) online.py --eval $$r --ckpts 0 20000 40000 60000 --eval-device cuda; done
