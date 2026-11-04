# Makefile for experiment vm-00080-delayed-copies
#
# Reproduces the runs of task 00091: the frame a 2 x 2 composite of the view now and in the past (env.delays), 16 transformers of the default
# quadrant's size (model.img = 128, uni_pyr = 4, band_n = 16), K4, log summaries; every copy's target the present. D0 delays 0 0 0 0,
# D1 0 1 2 3, DX 0 1 4 16. Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon.
# Outputs go to ~/data/research/visual-motor-agent/exp_00080/. The reference is vm-00073's K4, not rerun.
#
# Usage:
#   make runs — D0, D1, DX, six seeds, seeds 1-4 on the RTX 4090 and 5-6 on the RTX 3090 (the 4090 does about twice the 3090's work), 18 lives at once
#   make frozen — eval_delay.py error and chain (seen and with the delayed copies blanked), without MPS
#   make show — tables

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00080
SEEDS = 1 2 3 4 5 6
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set model.img=128 --set model.uni_pyr=4 --set model.band_n=16 --set online.future_k=4

.PHONY: all runs frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for a in D0:0,0,0,0 D1:0,1,2,3 DX:0,1,4,16; do for s in $(SEEDS); do \
	  if [ $$s -le 4 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set "env.delays=[$${a#*:}]" --seed $$s --set name=$${a%:*} -o $(OUT)/$${a%:*}_s$$s > $(OUT)/$${a%:*}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  r="$$(for a in D0 D1 DX; do for s in $(SEEDS); do echo $(OUT)/$${a}_s$$s; done; done)"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_delay.py chain $$r > $(OUT)/chain.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_delay.py error $$r > $(OUT)/error.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_delay.py show $(OUT)/error.jsonl $(OUT)/chain.jsonl
