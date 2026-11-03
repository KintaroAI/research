# Makefile for experiment vm-00074-clogged-channels
#
# Reproduces the runs of task 00085: K4 with log summaries, online.clog at .1 and .3. The control is vm-00073's K4, not rerun. Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00074/.
#
# Usage:
#   make runs — C10, C30, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, six lives on each at once
#   make runs SEEDS="1 2 3" — only those seeds, each still on its own GPU
#   make frozen — eval_clog.py and eval_hunger.py chain with damage while acting, on the arms and the control

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00074
CTRL = $(HOME)/data/research/visual-motor-agent/exp_00073
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4
SEEDS = 1 2 3 4 5 6

.PHONY: all runs frozen

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for a in 10:0.1 30:0.3; do for s in $(SEEDS); do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.clog=$${a#*:} --seed $$s --set name=C$${a%:*} -o $(OUT)/C$${a%:*}_s$$s > $(OUT)/C$${a%:*}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  all="$$(for s in 1 2 3 4 5 6; do echo $(OUT)/C10_s$$s $(OUT)/C30_s$$s $(CTRL)/K4_s$$s; done)"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_clog.py score $$all > $(OUT)/clog.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$all --clog .3 > $(OUT)/chain_clog.jsonl & \
	  wait; }
	cd $(DEV) && $(PY) eval_clog.py show $(OUT)/clog.jsonl --base K4
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/chain_clog.jsonl --base K4
