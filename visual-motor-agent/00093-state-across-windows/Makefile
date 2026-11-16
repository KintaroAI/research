# Makefile for experiment vm-00093-state-across-windows
#
# Reproduces the runs of task 00104: vm-00091's value agent (VW, here C8) in the same world (configs/U_view.json: 128 x 128, 2 foods,
# a 64 x 64 window), with a state carried across windows (model.state = 1): a write and a read token after the body of every step,
# the read being the write from 8 ticks back. Updates on pairs of consecutive windows (C8's 8 windows as 4 pairs). Two arms: SB, the
# gradient through the carry (online.state_grad = 1); SA, the carry detached (0). C8 is vm-00091's VW, not rerun: its lives are
# linked as C8_sN, and its frozen rows are vm-00092's (which repeat vm-00091's exactly). Twelve seeds per arm, seeds 1-8 of both arms
# on the RTX 4090 (16 lives), 9-12 on the RTX 3090 (8): about 5.3 h each, by the timing below. 60k ticks. Needs the GPU library (cd ../dev && make libvmcuda.so) and the
# MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00093/.
#
# Usage:
#   make runs — the 24 lives at once (SPLIT = the last seed on GPU 1, the 4090; CUDA_DEVICE_ORDER=PCI_BUS_ID)
#   make link — vm-00091's VW_sN as C8_sN, and C8's frozen rows from vm-00092
#   make frozen — eval_hunger.py chain (1,500 ticks), value and reach on SB and SA, then chain with the state blank, without MPS, two
#                 per card (frozen_a = seeds 1-6 on GPU 1, frozen_b = seeds 7-12 on GPU 0; either can run alone)
#   make show — SB and SA against C8, SB against SA, and each with its state blank against itself, paired by seed
#   make videos — compare_video.py, C8, SA and SB of seeds 5 and 12, each view with the whole world beside it

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00093
C8 = $(HOME)/data/research/visual-motor-agent/exp_00091
C92 = $(HOME)/data/research/visual-motor-agent/exp_00092
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set online.value_gamma=0.9 --set online.value_refresh=1 --set model.state=1
SPLIT ?= 8   # timing (600-tick lives, 2026-11-16): 3.58 ticks/s per life with 14 on the 4090, 2.60 with 10 on the 3090

.PHONY: all runs link frozen frozen_a frozen_b show videos

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le $(SPLIT) ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --set online.state_grad=1 --seed $$s --set name=SB -o $(OUT)/SB_s$$s > $(OUT)/SB_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --set online.state_grad=0 --seed $$s --set name=SA -o $(OUT)/SA_s$$s > $(OUT)/SA_s$$s.log 2>&1 & \
	  done; wait; }

link:
	mkdir -p $(OUT)
	for s in 1 2 3 4 5 6 7 8 9 10 11 12; do ln -sfn $(C8)/VW_s$$s $(OUT)/C8_s$$s; done
	for f in chain value reach; do cat $(C92)/$${f}_a.jsonl $(C92)/$${f}_b.jsonl | grep '"run": "C8_s' > $(OUT)/c8_$$f.jsonl; done

# frozen_a (seeds 1-6, GPU 1) and frozen_b (seeds 7-12, GPU 0): chain and value at once, then reach and the blank chain at once.
A = $$(for s in 1 2 3 4 5 6; do echo $(OUT)/SB_s$$s $(OUT)/SA_s$$s; done | tr "\n" " ")
B = $$(for s in 7 8 9 10 11 12; do echo $(OUT)/SB_s$$s $(OUT)/SA_s$$s; done | tr "\n" " ")
EVAL = cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID &&

frozen: link
	$(MAKE) frozen_a & $(MAKE) frozen_b & wait

frozen_a:
	$(EVAL) a="$(A)" && { \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1500 > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$a --ticks 1500 > $(OUT)/reach_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1500 --state blank > $(OUT)/blank_a.jsonl & \
	  wait; }

frozen_b:
	$(EVAL) b="$(B)" && { \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b --ticks 1500 > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$b --ticks 1500 > $(OUT)/reach_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b --ticks 1500 --state blank > $(OUT)/blank_b.jsonl & \
	  wait; }

CHAIN = $(OUT)/chain_a.jsonl $(OUT)/chain_b.jsonl
show:
	cd $(DEV) && $(PY) eval_hunger.py show $(CHAIN) $(OUT)/value_a.jsonl $(OUT)/value_b.jsonl $(OUT)/c8_chain.jsonl $(OUT)/c8_value.jsonl --base C8
	cd $(DEV) && $(PY) eval_hunger.py show $(CHAIN) --base SA
	cd $(DEV) && $(PY) eval_hunger.py show $(CHAIN) $(OUT)/blank_a.jsonl $(OUT)/blank_b.jsonl --base SB
	cd $(DEV) && $(PY) eval_hunger.py show $(CHAIN) $(OUT)/blank_a.jsonl $(OUT)/blank_b.jsonl --base SA
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/reach_a.jsonl $(OUT)/reach_b.jsonl $(OUT)/c8_reach.jsonl

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && for s in 5 12; do \
	  CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py $(OUT)/C8_s$$s:"C8 (no state)" $(OUT)/SA_s$$s:"SA (state, detached)" \
	    $(OUT)/SB_s$$s:"SB (state, gradient)" --steps 1000 --worlds 8 --cols 2 \
	    -o $(HOME)/data/research/visual-motor-agent/videos/vm93_s$$s; done
