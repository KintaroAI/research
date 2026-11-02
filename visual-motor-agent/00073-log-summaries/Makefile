# Makefile for experiment vm-00073-log-summaries
#
# Reproduces the runs of task 00084: vm-00071's arms with model.band_win = 1. Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon.
# Outputs go to ~/data/research/visual-motor-agent/exp_00073/. The control is exp_00071.
#
# Usage:
#   make runs — K1, K4, K16, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, nine lives on each at once
#   make frozen — eval_hunger.py chain, reach and poison on all 18 lives, then against vm-00071
#   make scaling — does the cost grow in line with the world, every summary against log summaries? Alone on the RTX 4090, with nothing else on it

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00073
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set model.band_win=1

.PHONY: all runs frozen scaling

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for k in 1 4 16; do for s in 1 2 3 4 5 6; do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.future_k=$$k --seed $$s --set name=K$$k -o $(OUT)/K$${k}_s$$s > $(OUT)/K$${k}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { for w in chain reach poison; do \
	  { CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py $$w $(OUT)/K1_s1 $(OUT)/K4_s1 $(OUT)/K16_s1 $(OUT)/K1_s2 $(OUT)/K4_s2 $(OUT)/K16_s2 --ticks 1500 & \
	    CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py $$w $(OUT)/K1_s3 $(OUT)/K4_s3 $(OUT)/K16_s3 --ticks 1500 & \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py $$w $(OUT)/K1_s4 $(OUT)/K4_s4 $(OUT)/K16_s4 $(OUT)/K1_s5 $(OUT)/K4_s5 $(OUT)/K16_s5 --ticks 1500 & \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py $$w $(OUT)/K1_s6 $(OUT)/K4_s6 $(OUT)/K16_s6 --ticks 1500 & \
	    wait; } > $(OUT)/hunger_$$w.jsonl; \
	done; }
	cd $(DEV) && $(PY) eval_hunger.py compare $(HOME)/data/research/visual-motor-agent/exp_00071 $(OUT)

scaling:
	cd $(DEV) && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $(PY) bench_scale.py -o $(OUT)/scaling > $(OUT)/scaling.jsonl
	cd $(DEV) && $(PY) bench_scale.py --plot $(OUT)/scaling.jsonl ../00073-log-summaries/images/scaling.png
