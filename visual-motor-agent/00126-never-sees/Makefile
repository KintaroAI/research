# Makefile for experiment vm-00126-never-sees
#
# Task 00139 (the owner, 2026-11-27: "lets have empty input (black for example) and train model on a real image ... model
# do not actually ever see the image in the first place - I want o see what will happen"): the one watching transformer
# with online.blind over the whole frame: every frame it is given is black, the real frame is only the target of its
# updates. One seed an arm, 60k.
#   EB, ED     the court, the discs: the usual update (the newest window and 7 replayed)
#   EB1, ED1   (mine) online.batch = 1: the newest window only
#   EB1x, ED1x (mine, added after the first lives were seen) the newest window only, the learning rate 10 times larger
# online.pred_score: at every tick the model's picture of the frame before it learns from it, its error and copy's and
# the mean frame's in the metrics; pred_record, pred_every: the frames kept (pred_frames.npz).
# GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00126/.
#
# Usage:
#   make runs CARD=1 ARMS="EB ED"       — the lives named, together on one card
#   make frozen CARD=1                  — the last checkpoint frozen: the exam as the life saw frames (black), on the
#                                         life's own world after its end and on the working set; and shown whole frames
#   make table                          — the README's tables (no GPU)
#   make images                         — images/pictures_<arm>.png, images/errors.png (no GPU)
#   make video                          — the time-lapse of the real frame beside the model's picture (no GPU)
#   Options: MPS= (no MPS daemon), SEEDS (1), TICKS (60000)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00126
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
SEEDS ?= 1
TICKS ?= 60000
CHUNK ?= 64
ARMS ?= EB EB1 EB1x ED ED1 ED1x
CFG_EB = configs/W1_breakout.json
CFG_EB1 = configs/W1_breakout.json
CFG_ED = configs/W1_discs.json
CFG_ED1 = configs/W1_discs.json
CFG_EB1x = configs/W1_breakout.json
CFG_ED1x = configs/W1_discs.json
MORE_EB =
MORE_ED =
MORE_EB1 = --set online.batch=1
MORE_ED1 = --set online.batch=1
MORE_EB1x = --set online.batch=1 --set ppo.lr=0.003
MORE_ED1x = --set online.batch=1 --set ppo.lr=0.003
BLACK = --set 'online.blind=[0,0,64,64]' --set online.pred_score=1 --set online.pred_record=64 --set online.pred_every=100
WM = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py score

.PHONY: runs frozen table images video

# the lives named in ARMS, all at once on one card
runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in $(SEEDS); do \
	  $(foreach a,$(ARMS),{ [ -e $(OUT)/$(a)_s$$s/ckpt_final.bin ] || env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config $(CFG_$(a)) --seed $$s --set name=$(a) --set online.steps=$(TICKS) $(BLACK) $(MORE_$(a)) -o $(OUT)/$(a)_s$$s > $(OUT)/$(a)_s$$s.log 2>&1; } & ) \
	done; wait

# the last checkpoint with learning off: shown black frames as in its life (--lived), and whole real frames for the first time
frozen:
	cd $(DEV) && for s in $(SEEDS); do for a in $(ARMS); do \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --set home --lived --name lived_home --parts regular >> $(OUT)/$${a}_s$$s.wm.log 2>&1 && \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --lived --name lived --parts regular >> $(OUT)/$${a}_s$$s.wm.log 2>&1 && \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --parts regular >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	done; done

table:
	cd $(DEV) && $(PY) pred_show.py table $(foreach a,$(ARMS),$(OUT)/$(a)_s1)

images:
	cd $(DEV) && $(PY) pred_show.py curves $(foreach a,$(ARMS),$(OUT)/$(a)_s1) -o $(CURDIR)/images/errors.png && for a in $(ARMS); do \
	  $(PY) pred_show.py strip $(OUT)/$${a}_s1 -o $(CURDIR)/images/pictures_$$a.png || exit 1; done

video:
	mkdir -p $(OUT)/videos
	cd $(DEV) && $(PY) pred_show.py video $(foreach a,$(ARMS),$(OUT)/$(a)_s1) -o $(OUT)/videos/never_sees.mp4
