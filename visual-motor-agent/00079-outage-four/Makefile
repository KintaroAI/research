# Makefile for experiment vm-00079-outage-four
#
# Reproduces the runs of task 00090: the default model (4 quadrant transformers, K4, log summaries), each transformer off at random,
# online.off .25 (Q25) and .5 (Q50), spells of 32 ticks on average, six seeds. The control is vm-00073's K4, seeds 1-6, not rerun.
# Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00079/.
#
# Usage:
#   make runs — Q25 and Q50, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, 12 lives at once
#   make frozen — eval_outage.py chain at test off shares 0, .25 and .5, restore and alone, for Q25, Q50 and K4, without MPS
#   make show — tables against K4

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00079
CTRL = $(HOME)/data/research/visual-motor-agent/exp_00073
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4

.PHONY: all runs frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for a in Q25:0.25 Q50:0.5; do for s in 1 2 3 4 5 6; do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.off=$${a#*:} --seed $$s --set name=$${a%:*} -o $(OUT)/$${a%:*}_s$$s > $(OUT)/$${a%:*}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  k="$$(for s in 1 2 3 4 5 6; do echo $(CTRL)/K4_s$$s; done)"; q="$$(for s in 1 2 3 4 5 6; do echo $(OUT)/Q25_s$$s $(OUT)/Q50_s$$s; done)"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_outage.py chain $$k --offs 0 0.25 0.5 > $(OUT)/chain_k4.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_outage.py chain $$q --offs 0 0.25 0.5 > $(OUT)/chain_q.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_outage.py restore $$k $$q > $(OUT)/restore.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_outage.py alone $$k $$q > $(OUT)/alone.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_outage.py show $(OUT)/chain_k4.jsonl $(OUT)/chain_q.jsonl $(OUT)/restore.jsonl $(OUT)/alone.jsonl --base K4
