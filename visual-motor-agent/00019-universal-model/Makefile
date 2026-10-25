# Makefile for experiment vm-00019-universal-model
#
# Reproduces the runs from the source tagged exp/vm-00019 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID: 0 = RTX 3090, 1 = RTX 4090) and the CUDA
# multi-process service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00019/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs     — the universal model (ARCHITECTURE.md, task 00023: 8x8 patches, 32 bins) in vm-00018's world
#                   (additive poison 0.5, food, slow hunger, continual torus), lifetime.py, 60k ticks, seeds 1-3:
#                   U0 (its own policy, no instinct), UI (+ the hunger instinct, 16 candidates), UIB (+ 32 bodies for 10k
#                   ticks, halving every 5k). No babbling, no command. Control: vm-00018 Q1 (the patch model).
#   make analysis — the click test and the frozen tests (with the run's instinct, and without: --eval-instinct 0) at 0,
#                   10k, 20k, 40k, 60k; token probes at 60k

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00019
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
T = --set env.respawn=true --set env.end_on_success=false --set env.max_steps=1000000000 --set env.wrap=true \
    --set env.hazards=2 --set env.hazard_cost=0 --set env.hazard_click_add=0.5 --set env.h_rate=0.001
U = $(PY) lifetime.py --config configs/AHR_gated.json --set model.patch=8 --set model.universal=1 --set model.bins=32 \
    --set online.goal=0 --set online.explore_eps=0 --set online.store_relief=0 --set online.store_surprise=0 \
    --set online.store_base=1 --set online.device=cuda --set online.steps=60000 $(T)
I = --set online.instinct_m=16
BOD = --set online.bodies=32 --set online.bodies_hold=10000 --set online.bodies_halve=5000
CK = 0 10000 20000 40000 60000

.PHONY: all runs analysis

all: runs analysis

# GPUs: UIB (the heaviest) on the 4090 for seeds 1-2, the rest spread
runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=0 $(U) --seed $$s --set name=U0 -o $(OUT)/00$$s\_U0_s$$s > $(OUT)/00$$s\_U0.log 2>&1 & \
	  $(G)=$$([ $$s = 3 ] && echo 1 || echo 0) $(U) $(I) --seed $$s --set name=UI -o $(OUT)/00$$((s+3))_UI_s$$s > $(OUT)/00$$((s+3))_UI.log 2>&1 & \
	  $(G)=$$([ $$s = 3 ] && echo 0 || echo 1) $(U) $(I) $(BOD) --seed $$s --set name=UIB -o $(OUT)/00$$((s+6))_UIB_s$$s > $(OUT)/00$$((s+6))_UIB.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for st in $(CK); do $(G)=1 $(PY) analyze_hazards.py click $(OUT)/0*_s* --step $$st --device cuda; done
	cd $(DEV) && for r in $(OUT)/0*_s*; do $(G)=1 $(PY) online.py --eval $$r --ckpts $(CK) --eval-device cuda; done
	cd $(DEV) && for r in $(OUT)/00[4-9]_*_s*; do $(G)=1 $(PY) online.py --eval $$r --ckpts 20000 60000 --eval-device cuda --eval-instinct 0; done
	cd $(DEV) && $(PY) analyze_patch.py tokens $(OUT)/0*_s* --step 60000 --threads 32
