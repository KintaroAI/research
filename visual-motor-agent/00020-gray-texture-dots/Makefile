# Makefile for experiment vm-00020-gray-texture-dots
#
# Reproduces the runs from the source tagged exp/vm-00020 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), the RTX 4090 (CUDA_DEVICE_ORDER=PCI_BUS_ID, device 1) and the CUDA multi-process
# service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00020/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs     — the universal model's defaults (configs/U_universal.json, task 00024): grayscale patches (one
#                   channel), food a solid disc, poison a checkerboard disc (+0.5 on a click), 8 moving grey dots, slow
#                   hunger, continual torus; lifetime.py, 60k ticks, seeds 1-3: G0 (its own policy, no instinct),
#                   GI (+ the hunger instinct, 16 candidates). Comparison: vm-00019's U0 / UI (RGB, solid blue poison,
#                   no dots).
#   make analysis — the click test and the frozen tests (with the run's instinct and without) at 0, 10k, 20k, 40k,
#                   60k; token probes at 60k

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00020
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
U = $(G) $(PY) lifetime.py --config configs/U_universal.json --set online.steps=60000
CK = 0 10000 20000 40000 60000

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(U) --set online.instinct_m=0 --seed $$s --set name=G0 -o $(OUT)/00$$s\_G0_s$$s > $(OUT)/00$$s\_G0.log 2>&1 & \
	  $(U) --seed $$s --set name=GI -o $(OUT)/00$$((s+3))_GI_s$$s > $(OUT)/00$$((s+3))_GI.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for st in $(CK); do $(G) $(PY) analyze_hazards.py click $(OUT)/0*_s* --step $$st --device cuda; done
	cd $(DEV) && for r in $(OUT)/0*_s*; do $(G) $(PY) online.py --eval $$r --ckpts $(CK) --eval-device cuda; done
	cd $(DEV) && for r in $(OUT)/00[4-6]_*_s*; do $(G) $(PY) online.py --eval $$r --ckpts 20000 60000 --eval-device cuda --eval-instinct 0; done
	cd $(DEV) && $(PY) analyze_patch.py tokens $(OUT)/0*_s* --step 60000 --threads 32
