# Makefile for experiment vm-00078-outage
#
# Reproduces the runs of task 00089: 16 transformers, K4 with log summaries, the default world; R0 all on, R1 each transformer off a quarter of the time
# (online.off = 0.25, spells of 32 ticks on average), six seeds. Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon.
# Outputs go to ~/data/research/visual-motor-agent/exp_00078/.
#
# Usage:
#   make runs — R0 and R1, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, 12 lives at once
#   make frozen — eval_outage.py chain at test off shares 0, .25 and .5, restore and alone, without MPS
#   make show — tables, R1 against R0

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00078
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set model.band_n=16 --set online.future_k=4

.PHONY: all runs frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for a in R0:0.0 R1:0.25; do for s in 1 2 3 4 5 6; do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.off=$${a#*:} --seed $$s --set name=$${a%:*} -o $(OUT)/$${a%:*}_s$$s > $(OUT)/$${a%:*}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  r0="$$(for s in 1 2 3 4 5 6; do echo $(OUT)/R0_s$$s; done)"; r1="$$(for s in 1 2 3 4 5 6; do echo $(OUT)/R1_s$$s; done)"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_outage.py chain $$r0 --offs 0 0.25 0.5 > $(OUT)/chain_r0.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_outage.py chain $$r1 --offs 0 0.25 0.5 > $(OUT)/chain_r1.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_outage.py restore $$r0 $$r1 > $(OUT)/restore.jsonl & \
	  wait; }
	# added after the results: each transformer foraging by itself
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_outage.py alone $$(for s in 1 2 3 4 5 6; do echo $(OUT)/R0_s$$s; done) > $(OUT)/alone_r0.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_outage.py alone $$(for s in 1 2 3 4 5 6; do echo $(OUT)/R1_s$$s; done) > $(OUT)/alone_r1.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_outage.py show $(OUT)/chain_r0.jsonl $(OUT)/chain_r1.jsonl $(OUT)/restore.jsonl $(OUT)/alone_r0.jsonl $(OUT)/alone_r1.jsonl --base R0
