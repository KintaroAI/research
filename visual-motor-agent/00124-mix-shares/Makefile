# Makefile for experiment vm-00124-mix-shares
#
# Task 00137 (the owner, 2026-11-27: "lets do 70% mix from next frame and 30% keep from real and yet another experiment
# - 30% real, 50% from prediction and 20% just grey noise"): vm-00123's mix with other shares. One seed an arm, 60k.
#   M7D, M7B   online.mix = .7                          .7 from its prediction, .3 real
#   MND, MNB   online.mix = .5, online.mix_noise = .2   .5 from its prediction, .2 a random grey, .3 real
# D: configs/W1_discs.json, B: configs/W1_breakout.json. Against vm-00123's MD and MB (.5) and vm-00122's lives at the
# same ticks. GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00124/.
#
# Usage:
#   make runs CARD=1 ARMS="M7B MNB"                — the lives named, together on one card (as run: M7B and MNB on the
#                                                    RTX 4090, M7D on the RTX 3090, then MND on the RTX 4090)
#   make anchors                                   — the floors of the exam, for the marks in the picture (no GPU)
#   make score CARD=1 ARMS="..." STEPS="20000 ..." — the checkpoints on the exam, shown whole real frames
#   make home CARD=1 DISCS="..." STEPS="..."       — the disc lives on their own world, the stretch vm-00122's lives were scored on
#   make lived CARD=1                              — the last checkpoint with the exam shown as the life saw frames
#   make free CARD=1                               — the free dream of 200 ticks
#   make curves                                    — the numbers by tick beside vm-00123's and vm-00122's lives (no GPU)
#   make images                                    — images/curves.png, images/seen_<arm>.png
#   make videos-long CARD=1                        — 20 watched ticks, then a dream of 1,000 (video-court, video-discs: one world)
#   Options: MPS= (no MPS daemon), SEEDS (1), TICKS (60000)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00124
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
SEEDS ?= 1
TICKS ?= 60000
CHUNK ?= 64
ARMS ?= M7D MND M7B MNB
DISCS ?= M7D MND
STEPS ?= 20000 40000 60000
CFG_M7D = configs/W1_discs.json
CFG_MND = configs/W1_discs.json
CFG_M7B = configs/W1_breakout.json
CFG_MNB = configs/W1_breakout.json
MIX_M7D = --set online.mix=0.7
MIX_M7B = --set online.mix=0.7
MIX_MND = --set online.mix=0.5 --set online.mix_noise=0.2
MIX_MNB = --set online.mix=0.5 --set online.mix_noise=0.2
WM = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py score

.PHONY: runs anchors score home lived free curves images videos-long video-court video-discs

# the lives named in ARMS, all at once on one card
runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in $(SEEDS); do \
	  $(foreach a,$(ARMS),{ [ -e $(OUT)/$(a)_s$$s/ckpt_final.bin ] || env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config $(CFG_$(a)) --seed $$s --set name=$(a) --set online.steps=$(TICKS) $(MIX_$(a)) --set online.mix_record=64 -o $(OUT)/$(a)_s$$s > $(OUT)/$(a)_s$$s.log 2>&1; } & ) \
	done; wait

# the exam's floors, for the marks in the picture (no GPU; the same exam as vm-00119's)
anchors:
	mkdir -p $(OUT)
	cd $(DEV) && $(PY) wm_measures.py anchors configs/W1_discs.json -o $(OUT) && $(PY) wm_measures.py anchors configs/W1_breakout.json -o $(OUT)

# the checkpoints on the exam, shown whole real frames, as every other model is
score:
	cd $(DEV) && for s in $(SEEDS); do for a in $(ARMS); do \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(STEPS) --parts regular course >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	done; done

# the disc lives' own world on the stretch vm-00122's lives were scored on (seed 1's world from tick 200,000 on, 64 starts)
home:
	cd $(DEV) && for s in $(SEEDS); do for a in $(DISCS); do \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(STEPS) --set home --home-starts 64 --home-from 200000 --name home64 --parts regular course >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	done; done

# the last checkpoint with the exam's real ticks shown as the life saw frames (MND, MNB: with their random grey)
lived:
	cd $(DEV) && for s in $(SEEDS); do \
	  for a in $(ARMS); do \
	    $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --lived --name lived --parts regular course >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	  done; \
	  for a in $(DISCS); do \
	    $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --set home --home-starts 64 --home-from 200000 --lived --name lived_home64 --parts regular course >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	  done; \
	done

# the free dream of 200 ticks, in a file of its own (vm-00123's Makefile scores its own lives and vm-00122's)
free:
	cd $(DEV) && for s in $(SEEDS); do for a in $(ARMS); do \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --parts free --name free >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	done; done

B = ../exp_00122
M = ../exp_00123
CURVES = $(PY) wm_curves.py $(OUT) --arms $(B)/OD $(B)/SD3 $(M)/MD M7D MND $(B)/OB $(B)/SB3 $(M)/MB M7B MNB --seeds $(SEEDS) --steps $(STEPS)
curves:
	cd $(DEV) && $(CURVES)

images:
	cd $(DEV) && $(CURVES) -o $(CURDIR)/images/curves.png && for a in $(ARMS); do \
	  $(PY) mix_strip.py $(OUT)/$${a}_s1 -o $(CURDIR)/images/seen_$$a.png || exit 1; done

# 20 watched ticks, then a dream of 1,000, the three mixes side by side; each watches as it lived
VID = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_video.py
LV = --step $(TICKS) --real 20 --warm 20 --dream 1000 --outline 32 --fps 12 --lived
MIXED = $(D)/exp_00123
videos-long: video-court video-discs

video-court:
	cd $(DEV) && for s in $(SEEDS); do \
	  $(VID) $(MIXED)/MB_s$$s:".5 own prediction" $(OUT)/M7B_s$$s:".7 own prediction" $(OUT)/MNB_s$$s:".5 own, .2 random grey" -o $(OUT)/videos/court_long_seed$$s.mp4 $(LV) \
	    --title "The court after $(TICKS) ticks of life: 20 watched ticks, then a dream of 1000 (vm-00124, seed $$s)" || exit 1; \
	done

video-discs:
	cd $(DEV) && for s in $(SEEDS); do \
	  $(VID) $(MIXED)/MD_s$$s:".5 own prediction" $(OUT)/M7D_s$$s:".7 own prediction" $(OUT)/MND_s$$s:".5 own, .2 random grey" -o $(OUT)/videos/discs_new_long_seed$$s.mp4 $(LV) \
	    --title "New discs after $(TICKS) ticks of life: 20 watched ticks, then a dream of 1000 (vm-00124, seed $$s)" || exit 1; \
	done
