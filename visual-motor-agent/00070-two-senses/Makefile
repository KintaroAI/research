# Makefile for experiment vm-00070-two-senses
#
# Reproduces the runs from the source tagged exp/vm-00070 (dev/ at that tag): a view and a text sense side by side, each going off by itself (task 00081). Needs the GPU
# library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00070/.
#
# Usage:
#   make runs-st — ST, the true teacher, seeds 1-8, on the RTX 4090
#   make runs-ss — SS, shuffled words, seeds 1-8, on the RTX 4090 too
#   make eval    — eval_sense.py on both arms, as lived and cut, on GPU $(GPU)
#   make steer   — eval_steer.py on both arms, on GPU $(GPU): who steers with the eyes closed, and where to

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00070
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_sense.json
GPU = 1

.PHONY: all runs-st runs-ss eval steer

all: runs-st runs-ss eval steer

runs-st:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in 1 2 3 4 5 6 7 8; do \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(RUN) --seed $$s --set name=ST -o $(OUT)/ST_s$$s > $(OUT)/ST_s$$s.log 2>&1 & \
	done; wait; }

runs-ss:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in 1 2 3 4 5 6 7 8; do \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(RUN) --set env.sense.teacher=shuffle --seed $$s --set name=SS -o $(OUT)/SS_s$$s > $(OUT)/SS_s$$s.log 2>&1 & \
	done; wait; }

eval:
	cd $(DEV) && for a in ST SS; do \
	  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(GPU) $(PY) eval_sense.py $$(ls -d $(OUT)/$${a}_s?) > $(OUT)/eval_$$a.jsonl; \
	done

steer:
	cd $(DEV) && for a in ST SS; do \
	  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(GPU) $(PY) eval_steer.py $$(ls -d $(OUT)/$${a}_s?) > $(OUT)/steer_$$a.jsonl; \
	done
