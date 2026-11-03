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
#   make stage2 — N10, seeds 4-6, on the RTX 4090 (the control: vm-00073's K4, seeds 4-6)
#   make frozen2 — chain at 0 and .1 for K4 and N10, seeds 4-6, on the RTX 3090 without MPS
#   Stage 3, noise on the input only, is vm-00077 (00077-input-noise)
#   make videos — the two videos: each level's model in its own noise, and K4 against N10 at .1 (compare_video.py)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00076
CTRL = $(HOME)/data/research/visual-motor-agent/exp_00073
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4
LEVELS = 0 0.1 0.5 0.9 0.95 1

.PHONY: all runs frozen show figures stage2 frozen2 videos

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

stage2:
	cd $(DEV) && { for s in 4 5 6; do \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(RUN) --set env.noise=0.1 --seed $$s --set name=N10 -o $(OUT)/N10_s$$s > $(OUT)/N10_s$$s.log 2>&1 & \
	done; wait; }

frozen2:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_noise.py chain $(CTRL)/K4_s4 $(CTRL)/K4_s5 $(CTRL)/K4_s6 --levels 0 0.1 > $(OUT)/chain2_k4.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_noise.py chain $(OUT)/N10_s4 $(OUT)/N10_s5 $(OUT)/N10_s6 --levels 0 0.1 > $(OUT)/chain2_n10.jsonl & \
	  wait; }

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 && mkdir -p $(OUT)/videos && \
	  $(PY) compare_video.py $(CTRL)/K4_s1:"q 0, clean" $(OUT)/N10_s1:"q .1" $(OUT)/N25_s1:"q .25" $(OUT)/N50_s1:"q .5" $(OUT)/N75_s1:"q .75" $(OUT)/N100_s1:"q 1" --steps 1000 -o $(OUT)/videos/levels && \
	  $(PY) compare_video.py $(CTRL)/K4_s1:"raised clean, tested at .1" $(OUT)/N10_s1:"raised at .1, tested at .1" --noise 0.1 --cols 2 --steps 1000 -o $(OUT)/videos/at_q10
