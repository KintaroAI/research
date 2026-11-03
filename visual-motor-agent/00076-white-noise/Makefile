# Makefile for experiment vm-00076-white-noise
#
# Reproduces the runs of task 00087: K4 with log summaries, env.noise at eight levels, three seeds. Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon.
# Outputs go to ~/data/research/visual-motor-agent/exp_00076/. The control is vm-00073's K4, seeds 1-3, not rerun.
#
# Usage:
#   make runs — N5 to N100, seeds 1 and 2 on the RTX 4090 and 3 on the RTX 3090, 24 lives at once
#   make frozen — eval_noise.py chain and error, every run at every test level in LEVELS, without MPS
#   make show — tables of train level by test level
#   make figures — the README's images (noise_figures.py)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00076
CTRL = $(HOME)/data/research/visual-motor-agent/exp_00073
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4
LEVELS = 0 0.1 0.5 0.9 0.95 1

.PHONY: all runs frozen show figures

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for a in 5:0.05 10:0.1 25:0.25 50:0.5 75:0.75 90:0.9 95:0.95 100:1.0; do for s in 1 2 3; do \
	  if [ $$s -le 2 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set env.noise=$${a#*:} --seed $$s --set name=N$${a%:*} -o $(OUT)/N$${a%:*}_s$$s > $(OUT)/N$${a%:*}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  a="$$(for s in 1 2 3; do echo $(CTRL)/K4_s$$s $(OUT)/N5_s$$s $(OUT)/N10_s$$s $(OUT)/N25_s$$s $(OUT)/N50_s$$s; done)"; \
	  b="$$(for s in 1 2 3; do echo $(OUT)/N75_s$$s $(OUT)/N90_s$$s $(OUT)/N95_s$$s $(OUT)/N100_s$$s; done)"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_noise.py chain $$a --levels $(LEVELS) > $(OUT)/chain_a.jsonl & \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_noise.py chain $$b --levels $(LEVELS) > $(OUT)/chain_b.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_noise.py error $$a $$b --levels $(LEVELS) > $(OUT)/error.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_noise.py show $(OUT)/chain_a.jsonl $(OUT)/chain_b.jsonl $(OUT)/error.jsonl

figures:
	cd $(DEV) && $(PY) noise_figures.py
