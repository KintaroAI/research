# Makefile for experiment vm-00091-world-larger-than-view
#
# Reproduces the runs of task 00102: a world larger than the view. configs/U_view.json is vm-00072's 128 x 128 world with 2 foods (8 poisons, 32 dots),
# seen through a 64 x 64 window centered on the cursor (env.view = 64); the model is the small world's (img 64). vm-00090's two agents, unchanged:
# K4's label (KW) and the fresh value, gamma 0.9 (VW), four quarter transformers of 8 layers, online.instinct_tol = 0.002. Twelve seeds, each seed's two
# lives on one card: seeds 1-7 on the RTX 4090, 8-12 on the RTX 3090 (the 4090 is about 1.5 times as fast). 60k ticks. Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS
# daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00091/.
#
# Usage:
#   make walkers — eval_food_gap.py in U_view.json with the window and without it (CPU, no model)
#   make runs — KW and VW seeds 1-7 on GPU 1 (CUDA_DEVICE_ORDER=PCI_BUS_ID: the 4090), seeds 8-12 on GPU 0 (the 3090); 24 lives at once
#   make frozen — eval_hunger.py chain (1,500 ticks, with the window's measures), value and reach on every run, without MPS, two per card
#   make show — KW and VW, aim at unseen food against the blind baseline (also without the ticks after a meal), VW against KW paired by seed
#   make videos — compare_video.py, KW and VW of seeds 5 and 12 side by side, each view with the whole world beside it

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00091
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002
KW =
VW = --set online.value_gamma=0.9 --set online.value_refresh=1

.PHONY: all walkers runs frozen show videos

all: runs

walkers:
	mkdir -p $(OUT)
	cd $(DEV) && { $(PY) eval_food_gap.py configs/U_view.json > $(OUT)/walkers_view64.txt & \
	  $(PY) eval_food_gap.py configs/U_view.json --view 0 > $(OUT)/walkers_view0.txt & wait; }

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    if [ $$s -le 7 ]; then d=1; else d=0; fi; \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(VW) --seed $$s --set name=VW -o $(OUT)/VW_s$$s > $(OUT)/VW_s$$s.log 2>&1 & \
	    $(G) CUDA_VISIBLE_DEVICES=$$d $(RUN) $(KW) --seed $$s --set name=KW -o $(OUT)/KW_s$$s > $(OUT)/KW_s$$s.log 2>&1 & \
	  done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  a="$$(for s in 1 2 3 4 5 6 7; do echo $(OUT)/VW_s$$s $(OUT)/KW_s$$s; done | tr "\n" " ")"; \
	  b="$$(for s in 8 9 10 11 12; do echo $(OUT)/VW_s$$s $(OUT)/KW_s$$s; done | tr "\n" " ")"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py chain $$a --ticks 1500 > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py chain $$b --ticks 1500 > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & \
	  wait; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py reach $$a --ticks 1500 > $(OUT)/reach_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py reach $$b --ticks 1500 > $(OUT)/reach_b.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/chain_a.jsonl $(OUT)/value_a.jsonl $(OUT)/chain_b.jsonl $(OUT)/value_b.jsonl --base KW
	cd $(DEV) && $(PY) eval_hunger.py show $(OUT)/reach_a.jsonl $(OUT)/reach_b.jsonl

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && for s in 5 12; do \
	  CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py $(OUT)/KW_s$$s:"KW (K4 label)" $(OUT)/VW_s$$s:"VW (value)" --steps 1000 --worlds 8 --cols 2 \
	    -o $(HOME)/data/research/visual-motor-agent/videos/vm91_s$$s; done
