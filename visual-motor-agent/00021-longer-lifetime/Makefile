# Makefile for experiment vm-00021-longer-lifetime
#
# Reproduces the runs from the source tagged exp/vm-00021 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID: 0 = RTX 3090, 1 = RTX 4090) and the CUDA
# multi-process service (nvidia-cuda-mps-control -d, the same CUDA_MPS_PIPE_DIRECTORY in the environment).
# Outputs go to ~/data/research/visual-motor-agent/exp_00021/RRR_<name>_s<seed>/.
#
# Usage:
#   make runs     — the universal model's default (configs/U_universal.json: gray, checkerboard poison, 8 dots, the
#                   instinct with 16 candidates), 200k ticks, seeds 1-6 (1-3 on the RTX 4090, 4-6 on the RTX 3090)
#   make analysis — the click test and the frozen tests at 25k, 50k, 100k, 150k, 200k (the policy alone at 100k and
#                   200k); token probes at 200k

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00021
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
U = $(PY) lifetime.py --config configs/U_universal.json --set online.steps=200000 --set online.save_every=25000
CK = 0 25000 50000 100000 150000 200000

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3 4 5 6; do \
	  $(G)=$$([ $$s -le 3 ] && echo 1 || echo 0) $(U) --seed $$s --set name=GI -o $(OUT)/00$$s\_GI_s$$s > $(OUT)/00$$s\_GI.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for st in $(CK); do $(G)=1 $(PY) analyze_hazards.py click $(OUT)/0*_s* --step $$st --device cuda; done
	cd $(DEV) && for r in $(OUT)/0*_s*; do $(G)=1 $(PY) online.py --eval $$r --ckpts $(CK) --eval-device cuda; done
	cd $(DEV) && for r in $(OUT)/0*_s*; do $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 200000 --eval-device cuda --eval-instinct 0; done
	cd $(DEV) && $(PY) analyze_patch.py tokens $(OUT)/0*_s* --step 200000 --threads 32
