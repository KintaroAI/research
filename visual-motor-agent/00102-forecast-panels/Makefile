# Makefile for experiment vm-00102-forecast-panels
#
# Reproduces the runs of task 00115: forecast panels. The frame is a 128 x 128 composite of four 64 x 64 panels, one per
# quarter transformer: panel 1 the real view, panels 2-4 transformers 0-2's forecasts of their own panel from the tick
# before (pair decoding, Gaussian noise sigma .03); panel i is trained against the real frame i ticks after panel 1's.
# PB: panels 2-4 random noise instead (the baseline). Both on RD's settings (vm-00100) with the candidates at temperature 2
# (vm-00101); neither is a default. Five seeds per arm, 10 lives at once on the RTX 3090 (GPU 0 under
# CUDA_DEVICE_ORDER=PCI_BUS_ID), under MPS, 60k ticks. Outputs go to ~/data/research/visual-motor-agent/exp_00102/.
#
# Usage:
#   make runs — PB and PN, seeds 1-5, 10 lives at once on GPU 0
#   make frozen — (after the runs; 5 lives at a time) per checkpoint (10k ... 60k) and life: 600 ticks with learning off (ppo.lr = 0) in new worlds (seed 100 + s);
#                 PN also with its forecasts blanked (rows PN_sN_B)
#   make video — panel_video.py of seed 1 at 60k: PN as seen beside the real frames, 400 ticks
#   Options: GPU=1 (the RTX 4090), MPS= (no MPS daemon), T="10000 30000" (frozen's checkpoints),
#   e.g. both cards: make frozen GPU=0 MPS= T="10000 30000 50000" & make frozen GPU=1 MPS= T="20000 40000 60000"

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00102
GPU ?= 0
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
T ?= 10000 20000 30000 40000 50000 60000
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS) CUDA_VISIBLE_DEVICES=$(GPU)
SET = --config configs/U_view.json --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set model.body_label=1 --set online.label=meal --set online.label_gamma=0.95 --set env.inertia=4 --set online.full_dose=1 \
  --set online.label_refresh=1 --set online.instinct_temp=2 --set model.img=128 --set online.panels=1 --set online.panel_decode=pair
PB = --set online.panel_blank=1
PN = --set online.panel_noise=0.03
FROZEN = --set ppo.lr=0 --set online.steps=600 --set online.log_every=100 --set online.save_every=100000 --set online.panel_snap=100000

.PHONY: all runs frozen video

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for a in PB PN; do for s in 1 2 3 4 5; do \
	    x=$$([ $$a = PB ] && echo "$(PB)" || echo "$(PN)"); \
	    env $(G) $(PY) lifetime.py $(SET) $$x --set online.save_every=10000 --set online.panel_snap=10000 --seed $$s --set name=$$a \
	      -o $(OUT)/$${a}_s$$s > $(OUT)/$${a}_s$$s.log 2>&1 & \
	  done; done; wait; }

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; \
	  for a in PB PN; do for s in 1 2 3 4 5; do \
	    x=$$([ $$a = PB ] && echo "$(PB)" || echo "$(PN)"); \
	    env $(G) $(PY) lifetime.py $(SET) $$x $(FROZEN) --init $(OUT)/$${a}_s$$s/ckpt_$$(printf %08d $$t) --seed 10$$s \
	      -o $$o/$${a}_s$$s > $$o/$${a}_s$$s.log 2>&1 & \
	  done; wait; done; \
	  for s in 1 2 3 4 5; do \
	    env $(G) $(PY) lifetime.py $(SET) $(PB) $(FROZEN) --init $(OUT)/PN_s$$s/ckpt_$$(printf %08d $$t) --seed 10$$s \
	      -o $$o/PN_s$${s}_B > $$o/PN_s$${s}_B.log 2>&1 & \
	  done; wait; \
	done

video:
	mkdir -p $(D)/videos/vm102
	cd $(DEV) && env $(G) $(PY) lifetime.py $(SET) $(PN) $(FROZEN) --set online.steps=400 --set online.panel_record=400 \
	  --init $(OUT)/PN_s1/ckpt_00060000 --seed 201 -o $(OUT)/video_PN_s1 > $(OUT)/video_PN_s1.log 2>&1 && \
	  $(PY) panel_video.py $(OUT)/video_PN_s1:"PN (forecasts + noise .03), seed 1 at 60k" -o $(D)/videos/vm102/panels_60k.mp4 --fps 6
