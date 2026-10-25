# Makefile for experiment vm-00018-additive-poison-bodies
#
# Reproduces the runs from the source tagged exp/vm-00018 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID: 0 = RTX 3090, 1 = RTX 4090) and the CUDA
# multi-process service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00018/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs     — additive poison (a click in a hazard adds 0.5 to hunger; a click in the red circle relieves it),
#                   continual torus, 2 hazards, hunger 0.001 per step; 8x8 patch tokens, move loss detached
#                   (vm-00017 PP), lifetime.py, 100k ticks, seeds 1-3:
#                   Q1 (one body, batch 8), Q32 (32 bodies for 10k ticks, halving every 5k; batch 39 -> 8),
#                   Q1B (one body, batch 39)
#   make analysis — the click prediction test and the frozen tests at 0, 10k, 20k, 40k, 60k, 100k

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00018
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
T = --set env.respawn=true --set env.end_on_success=false --set env.max_steps=1000000000 --set env.wrap=true \
    --set env.hazards=2 --set env.hazard_cost=0 --set env.hazard_click_add=0.5 --set env.h_rate=0.001
Q = $(PY) lifetime.py --config configs/AHR_gated.json --set ppo.pi_detach=1 --set model.pi_hidden=128 --set model.patch=8 \
    --set online.store_relief=0 --set online.store_surprise=0 --set online.store_base=1 --set online.device=cuda \
    --set online.steps=100000 $(T)
BOD = --set online.bodies=32 --set online.bodies_hold=10000 --set online.bodies_halve=5000
CK = 0 10000 20000 40000 60000 100000

.PHONY: all runs analysis

all: runs analysis

# GPUs: Q1B (the heaviest) on the 4090 for seeds 1-2; Q32 seed 1 and Q1 seed 1 there too; the rest on the 3090
runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=$$([ $$s = 1 ] && echo 1 || echo 0) $(Q) --seed $$s --set name=Q1 -o $(OUT)/00$$s\_Q1_s$$s > $(OUT)/00$$s\_Q1.log 2>&1 & \
	  $(G)=$$([ $$s = 1 ] && echo 1 || echo 0) $(Q) $(BOD) --seed $$s --set name=Q32 -o $(OUT)/00$$((s+3))_Q32_s$$s > $(OUT)/00$$((s+3))_Q32.log 2>&1 & \
	  $(G)=$$([ $$s = 3 ] && echo 0 || echo 1) $(Q) --set online.batch=39 --seed $$s --set name=Q1B -o $(OUT)/00$$((s+6))_Q1B_s$$s > $(OUT)/00$$((s+6))_Q1B.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for st in $(CK); do $(G)=1 $(PY) analyze_hazards.py click $(OUT)/0*_s* --step $$st --device cuda; done
	cd $(DEV) && for r in $(OUT)/0*_s*; do $(G)=1 $(PY) online.py --eval $$r --ckpts $(CK) --eval-device cuda; done
