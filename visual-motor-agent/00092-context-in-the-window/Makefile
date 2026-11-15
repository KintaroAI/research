# Makefile for experiment vm-00092-context-in-the-window
#
# Reproduces the runs of task 00103: vm-00091's value agent (VW) in the same world (configs/U_view.json: 128 x 128, 2 foods, a 64 x 64 window),
# with a context of 32 ticks instead of 8 (C32). The control C8 is vm-00091's VW, not rerun: its lives are linked here as C8_sN. Twelve seeds,
# seeds 1-6 on the RTX 4090, 7-12 on the RTX 3090 (about 2.9 GB per life at ctx 32: twelve per card do not fit). 60k ticks. Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00092/.
#
# Usage:
#   make runs — C32 seeds 1-6 on GPU 1 (CUDA_DEVICE_ORDER=PCI_BUS_ID: the 4090), seeds 7-12 on GPU 0 (the 3090); 12 lives at once
#   make link — vm-00091's VW_sN as C8_sN
#   make frozen — eval_hunger.py chain (1,500 ticks), value and reach on C32 and C8, without MPS, two per card
#                 (frozen_a = seeds 1-6 on GPU 1; frozen_b = seeds 7-12, chain and value on GPU 1, reach on GPU 0; either can run alone)
#   make show — C32 against C8 paired by seed, with aim at unseen food by ticks since the last sight
#   make videos — compare_video.py, C8 and C32 of seeds 5 and 12, each view with the whole world beside it

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00092
C8 = $(HOME)/data/research/visual-motor-agent/exp_00091
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set online.value_gamma=0.9 --set online.value_refresh=1 --set model.ctx=32

.PHONY: all runs link frozen frozen_a frozen_b show videos

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le 6 ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) --seed $$s --set name=C32 -o $(OUT)/C32_s$$s > $(OUT)/C32_s$$s.log 2>&1 & \
	  done; wait; }

link:
	mkdir -p $(OUT)
	for s in 1 2 3 4 5 6 7 8 9 10 11 12; do ln -sfn $(C8)/VW_s$$s $(OUT)/C8_s$$s; done

# frozen = frozen_a (seeds 1-6) and frozen_b (seeds 7-12), without MPS. frozen_a: chain + value at once on GPU 1, then reach.
# frozen_b: chain + value on GPU 1 and reach on GPU 0, all at once (both cards are free once the lives end).
# frozen_a is skipped when its three outputs exist (on 2026-11-14 it ran early, on the idle 4090, while seeds 7-12 were still living).
A = $$(for s in 1 2 3 4 5 6; do echo $(OUT)/C32_s$$s $(OUT)/C8_s$$s; done | tr "\n" " ")
B = $$(for s in 7 8 9 10 11 12; do echo $(OUT)/C32_s$$s $(OUT)/C8_s$$s; done | tr "\n" " ")
EVAL = cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID &&

frozen: frozen_a frozen_b

frozen_a: link
	if [ -s $(OUT)/chain_a.jsonl ] && [ -s $(OUT)/value_a.jsonl ] && [ -s $(OUT)/reach_a.jsonl ]; then echo "frozen_a done already"; else \
	  $(EVAL) a="$(A)" && { \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1500 > $(OUT)/chain_a.jsonl & \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & \
	    wait; \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$a --ticks 1500 > $(OUT)/reach_a.jsonl; }; fi

frozen_b: link
	$(EVAL) b="$(B)" && { \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$b --ticks 1500 > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$b --ticks 1500 > $(OUT)/reach_b.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/chain_a.jsonl $(OUT)/value_a.jsonl $(OUT)/chain_b.jsonl $(OUT)/value_b.jsonl --base C8
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/reach_a.jsonl $(OUT)/reach_b.jsonl

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && for s in 5 12; do \
	  CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py $(OUT)/C8_s$$s:"C8 (context 8)" $(OUT)/C32_s$$s:"C32 (context 32)" --steps 1000 --worlds 8 --cols 2 \
	    -o $(HOME)/data/research/visual-motor-agent/videos/vm92_s$$s; done
