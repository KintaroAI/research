# Makefile for experiment vm-00071-hunger-ahead
#
# Reproduces the runs of task 00082: the hunger a move is scored on is the mean over the next future_k ticks. Needs the GPU library (cd ../dev && make libvmcuda.so)
# and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00071/.
#
# Usage:
#   make runs — K1, K4, K16, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, nine lives on each at once
#   make frozen — eval_hunger.py chain and reach on all 18 lives, then the tables, the readings against K1 and images/aim.png
#   make videos — K1, K4, K16 side by side from the same start, seeds 3 and 2

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00071
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json

.PHONY: all runs frozen videos

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for k in 1 4 16; do for s in 1 2 3 4 5 6; do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.future_k=$$k --seed $$s --set name=K$$k -o $(OUT)/K$${k}_s$$s > $(OUT)/K$${k}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { for w in chain reach; do \
	  { CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py $$w $(OUT)/K1_s1 $(OUT)/K4_s1 $(OUT)/K16_s1 $(OUT)/K1_s2 $(OUT)/K4_s2 $(OUT)/K16_s2 --ticks 1500 & \
	    CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py $$w $(OUT)/K1_s3 $(OUT)/K4_s3 $(OUT)/K16_s3 --ticks 1500 & \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py $$w $(OUT)/K1_s4 $(OUT)/K4_s4 $(OUT)/K16_s4 $(OUT)/K1_s5 $(OUT)/K4_s5 $(OUT)/K16_s5 --ticks 1500 & \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py $$w $(OUT)/K1_s6 $(OUT)/K4_s6 $(OUT)/K16_s6 --ticks 1500 & \
	    wait; } > $(OUT)/hunger_$$w.jsonl; \
	done; }
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/hunger_chain.jsonl $(OUT)/hunger_reach.jsonl --base K1 --plot ../00071-hunger-ahead/images/aim.png

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py --cols 3 --worlds 16 --step 60000 -o $(OUT)/videos/seed3 \
	  "$(OUT)/K1_s3:K1, next tick:gain" "$(OUT)/K4_s3:K4, 4 ticks ahead:gain" "$(OUT)/K16_s3:K16, 16 ticks ahead:gain"
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py --cols 3 --worlds 16 --step 60000 -o $(OUT)/videos/seed2 \
	  "$(OUT)/K1_s2:K1, next tick:gain" "$(OUT)/K4_s2:K4, 4 ticks ahead:gain" "$(OUT)/K16_s2:K16, 16 ticks ahead:gain"
