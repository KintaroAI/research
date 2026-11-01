# Makefile for experiment vm-00057-delayed-cue
#
# Reproduces the runs from the source tagged exp/vm-00057 (dev/ at that tag): the delayed-cue world beside a thought field (task
# 00065, configs/U_cue.json): two pieces, gain, age key, ring. Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon,
# as in vm-00056, on both GPUs: seeds 1 and 2 on the 3090, 3 and 4 on the 4090 (CUDA_DEVICE_ORDER=PCI_BUS_ID).
# Outputs go to ~/data/research/visual-motor-agent/exp_00057/.
#
# Usage:
#   make runs — 60k ticks, seeds 1-4: CH (the sample held), CS (flash, delay 1-2), CF (flash, delay 20-30; thought frozen in all three),
#               CT (flash, delay 20-30, thought the dream, pair decoding), CG (the same, mean decoding)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00057
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_cue.json --set online.steps=60000
FROZEN = --set env.thought.mode=frozen

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4; do \
	    g=$$(( (s - 1) / 2 )); \
	    $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) $(FROZEN) --set env.cue.hold=true --seed $$s --set name=CH -o $(OUT)/CH_s$$s > $(OUT)/CH_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) $(FROZEN) --set env.cue.delay_min=1 --set env.cue.delay_max=2 --seed $$s --set name=CS -o $(OUT)/CS_s$$s > $(OUT)/CS_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) $(FROZEN) --seed $$s --set name=CF -o $(OUT)/CF_s$$s > $(OUT)/CF_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --seed $$s --set name=CT -o $(OUT)/CT_s$$s > $(OUT)/CT_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.thought_decode=mean --seed $$s --set name=CG -o $(OUT)/CG_s$$s > $(OUT)/CG_s$$s.log 2>&1 & \
	  done; wait; }
