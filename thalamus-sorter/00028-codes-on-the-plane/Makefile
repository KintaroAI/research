# Makefile for experiment ts-00028-codes-on-the-plane
#
# Reproduces the runs of task 00002: per-window FSQ bottleneck columns predicting their inputs two ticks ahead, the
# codes fed back one tick late and sorted with the pixels (64 x 64 crop of garden.png, smooth drift), then rendered
# layer by layer. Outputs go to ~/data/research/thalamus-sorter/exp_00028/.
#
# Usage:
#   make runs    — F, F0, FP, F512, 100k ticks each; F and F0 on GPU 1 (the 4090), FP and F512 on GPU 0 (the 3090)
#   make render  — render_layers.py on each run (replay 4,000 ticks, video of 300)
#   make test    — test_fsq_column.py

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/thalamus-sorter/exp_00028
G = CUDA_DEVICE_ORDER=PCI_BUS_ID
RUN = $(PY) main.py word2vec --mode correlation -W 64 -H 64 --signal-source garden.png \
  --saccade-step 2 --saccade-momentum 0.95 --no-use-deriv-corr --threshold 0.5 --max-hit-ratio 0.1 \
  --dims 8 --lr 0.001 --k-sample 200 --signal-T 1000 --cluster-max-k 2 \
  --column-type fsq --column-outputs 4 --column-max-inputs 48 --column-window 4 --column-feedback \
  --cluster-report-every 10000 --log-every 5000 --save-every 9999999 --save-model --eval -f 100000

RUNS = 002_F 003_F0 004_FP 005_F512

.PHONY: all runs runs_a runs_b f8 f16 checks render test

all: runs render

runs:
	$(MAKE) runs_a & $(MAKE) runs_b & wait

# F and F0 on GPU 1 (the 4090)
runs_a:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(RUN) --cluster-neurons-per 20 -o $(OUT)/002_F > $(OUT)/002_F.log 2>&1 & \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(RUN) --cluster-neurons-per 20 --column-lr 0 -o $(OUT)/003_F0 > $(OUT)/003_F0.log 2>&1 & \
	  wait; }

# FP and F512 on GPU_B (default 0, the 3090; the 2026-11-14 runs used GPU_B=1 while the 3090 was busy)
GPU_B ?= 0
runs_b:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  $(G) CUDA_VISIBLE_DEVICES=$(GPU_B) $(RUN) --cluster-neurons-per 20 --corr-sign pos -o $(OUT)/004_FP > $(OUT)/004_FP.log 2>&1 & \
	  $(G) CUDA_VISIBLE_DEVICES=$(GPU_B) $(RUN) --cluster-neurons-per 12 -o $(OUT)/005_F512 > $(OUT)/005_F512.log 2>&1 & \
	  wait; }

# F8: F with an 8-number bottleneck (levels 8,5,5,5 twice); neurons-per 24 keeps 256 windows (4096 // (24 - 8))
# F8_200k: the same for 200k ticks (make f8 F8_TICKS=200000 F8_NAME=007_F8_200k)
F8_TICKS ?= 100000
F8_NAME ?= 006_F8
f8:
	mkdir -p $(OUT)
	cd $(DEV) && $(G) CUDA_VISIBLE_DEVICES=1 $(subst -f 100000,-f $(F8_TICKS),$(subst --column-outputs 4,--column-outputs 8,$(RUN))) \
	  --column-fsq-levels 8,5,5,5,8,5,5,5 --cluster-neurons-per 24 -o $(OUT)/$(F8_NAME) > $(OUT)/$(F8_NAME).log 2>&1
	cd $(DEV) && $(G) CUDA_VISIBLE_DEVICES=1 $(PY) render_layers.py $(OUT)/$(F8_NAME) > $(OUT)/$(F8_NAME).render.log 2>&1

# F16: F with a 16-number bottleneck (levels 8,5,5,5 four times) and the column scaled with it, keeping F's proportions:
# neurons-per 32 keeps 256 windows (4096 // (32 - 16)); input cap 48 -> 80 (1.6x the ring members per window);
# hidden width 64 -> 256 (16 per number); k-sample 200 -> 320 (3.9% of 8,192 neurons)
F16_TICKS ?= 100000
F16_NAME ?= 008_F16
F16_RUN = $(subst --k-sample 200,--k-sample 320,$(subst --column-max-inputs 48,--column-max-inputs 80,$(subst \
  -f 100000,-f $(F16_TICKS),$(subst --column-outputs 4,--column-outputs 16,$(RUN)))))
f16:
	mkdir -p $(OUT)
	cd $(DEV) && $(G) CUDA_VISIBLE_DEVICES=1 $(F16_RUN) --column-fsq-levels 8,5,5,5,8,5,5,5,8,5,5,5,8,5,5,5 \
	  --column-fsq-hidden 256 --cluster-neurons-per 32 -o $(OUT)/$(F16_NAME) > $(OUT)/$(F16_NAME).log 2>&1
	cd $(DEV) && $(G) CUDA_VISIBLE_DEVICES=1 $(PY) render_layers.py $(OUT)/$(F16_NAME) > $(OUT)/$(F16_NAME).render.log 2>&1

# checks (2026-11-15): F2 = F with a 2-number bottleneck (levels 8,5), only the numbers change as in F8, neurons-per 18 keeps
# 256 windows (4096 // (18 - 2)); F_rerun = F's exact command again (numpy's draws are unseeded, the columns start from the
# same weights). Both at once on GPU 1, then rendered.
checks:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(subst --column-outputs 4,--column-outputs 2,$(RUN)) --column-fsq-levels 8,5 \
	    --cluster-neurons-per 18 -o $(OUT)/009_F2 > $(OUT)/009_F2.log 2>&1 & \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(RUN) --cluster-neurons-per 20 -o $(OUT)/010_F_rerun > $(OUT)/010_F_rerun.log 2>&1 & \
	  wait; }
	cd $(DEV) && for r in 009_F2 010_F_rerun; do \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(PY) render_layers.py $(OUT)/$$r > $(OUT)/$$r.render.log 2>&1 || exit 1; done

render:
	cd $(DEV) && for r in $(RUNS); do \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(PY) render_layers.py $(OUT)/$$r > $(OUT)/$$r.render.log 2>&1 || exit 1; done

test:
	cd $(DEV) && $(PY) test_fsq_column.py
