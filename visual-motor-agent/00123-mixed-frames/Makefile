# Makefile for experiment vm-00123-mixed-frames
#
# Task 00136 (the owner, 2026-11-27: "each time lets say 50% of pixels in current image comes from dream (prediction of
# the next) ... model sees only these 50% mixed images and newer full frame"; "replay - mix should be re-made each
# replay"): the one watching transformer of vm-00120 with online.mix = .5. Each pixel of every frame it sees comes from
# its own prediction of that frame with probability .5; the targets and memory are the real frames; the mix is made
# again at every update. One seed an arm (few seeds first), 60k ticks.
#   MD  configs/W1_discs.json     online.mix = .5
#   MB  configs/W1_breakout.json  online.mix = .5
# Against seed 1 of vm-00122's lives at the same ticks (OD, SD3, OB, SB3: ../00122-long-lives, scored every 20k).
# GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS. One life alone on the RTX 4090 runs
# 12.7 ticks per second (the mix made again for 8 windows of 8 frames costs 66 ms a tick).
# Outputs go to ~/data/research/visual-motor-agent/exp_00123/.
#
# Usage:
#   make runs CARD=1 ARMS=MB                       — a life (as run: MB on the RTX 4090, MD on the RTX 3090)
#   make anchors                                   — the floors of the exam, for the marks in the picture (no GPU)
#   make score CARD=1 ARMS="..." STEPS="20000 ..." — the checkpoints on the exam, shown whole real frames
#   make home CARD=1 STEPS="..."                   — the disc life on its own world, the stretch vm-00122's lives were scored on
#   make lived CARD=1                              — the last checkpoint with the exam shown as the life saw frames
#   make free CARD=1                               — the free dream of 200 ticks
#   make free-base CARD=1                          — the same for vm-00122's four lives at 60k
#   make curves                                    — the numbers by tick beside vm-00122's lives (no GPU)
#   make images                                    — images/curves.png, images/seen_MB.png, images/seen_MD.png
#   make videos-long CARD=1                        — 20 watched ticks, then a dream of 1,000, beside vm-00122's lives
#   Options: MPS= (no MPS daemon), SEEDS (1), TICKS (60000)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00123
BASE = $(D)/exp_00122
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
SEEDS ?= 1
TICKS ?= 60000
CHUNK ?= 64
MIX ?= 0.5
ARMS ?= MD MB
DISCS ?= MD
STEPS ?= 20000 40000 60000
CFG_MD = configs/W1_discs.json
CFG_MB = configs/W1_breakout.json
WM = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py score

.PHONY: runs anchors score home lived free free-base curves images videos-long video-court video-discs

# the lives named in ARMS, all at once on one card
runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in $(SEEDS); do \
	  $(foreach a,$(ARMS),{ [ -e $(OUT)/$(a)_s$$s/ckpt_final.bin ] || env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config $(CFG_$(a)) --seed $$s --set name=$(a) --set online.steps=$(TICKS) --set online.mix=$(MIX) --set online.mix_record=64 -o $(OUT)/$(a)_s$$s > $(OUT)/$(a)_s$$s.log 2>&1; } & ) \
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

# the disc life's own world, on the stretch vm-00122's lives were scored on: the same world (seed 1) from tick 200,000
# on, 64 starts (wm_measures.home_skip: a later tick than the life's end may be asked for, never an earlier one)
home:
	cd $(DEV) && for s in $(SEEDS); do for a in $(DISCS); do \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(STEPS) --set home --home-starts 64 --home-from 200000 --name home64 --parts regular course >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	done; done

# the last checkpoint with the exam's real ticks shown as the life saw frames: its own prediction mixed into each
lived:
	cd $(DEV) && for s in $(SEEDS); do \
	  for a in $(ARMS); do \
	    $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --lived --name lived --parts regular course >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	  done; \
	  for a in $(DISCS); do \
	    $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --set home --home-starts 64 --home-from 200000 --lived --name lived_home64 --parts regular course >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	  done; \
	done

# the free dream of 200 ticks, in a file of its own (step_<N>_pair_free.json)
free:
	cd $(DEV) && for s in $(SEEDS); do for a in $(ARMS); do \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --parts free --name free >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	done; done

# the same for vm-00122's four lives at the same tick (a new file in each of their folders; their scored files are not touched)
BASES ?= OD SD3 OB SB3
free-base:
	cd $(DEV) && for s in $(SEEDS); do for a in $(BASES); do \
	  $(WM) $(BASE)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(TICKS) --parts free --name free >> $(OUT)/base_$${a}_s$$s.wm.log 2>&1 || exit 1; \
	done; done

B = ../exp_00122
CURVES = $(PY) wm_curves.py $(OUT) --arms $(B)/OD $(B)/SD3 MD $(B)/OB $(B)/SB3 MB --seeds $(SEEDS) --steps $(STEPS)
curves:
	cd $(DEV) && $(CURVES)

images:
	cd $(DEV) && $(CURVES) -o $(CURDIR)/images/curves.png && for a in $(ARMS); do \
	  $(PY) mix_strip.py $(OUT)/$${a}_s1 -o $(CURDIR)/images/seen_$$a.png || exit 1; done

# 20 watched ticks, then a dream of 1,000, beside vm-00122's lives at the same tick; the life with the mix watches as it lived
VID = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_video.py
LV = --step $(TICKS) --real 20 --warm 20 --dream 1000 --outline 32 --fps 12 --lived
videos-long: video-court video-discs

video-court:
	cd $(DEV) && for s in $(SEEDS); do \
	  $(VID) $(BASE)/OB_s$$s:"no self-forcing" $(BASE)/SB3_s$$s:"up to 3 dreamed steps" $(OUT)/MB_s$$s:"own prediction mixed in" -o $(OUT)/videos/court_long_seed$$s.mp4 $(LV) \
	    --title "The court after $(TICKS) ticks of life: 20 watched ticks, then a dream of 1000 (vm-00123, seed $$s)" || exit 1; \
	done

video-discs:
	cd $(DEV) && for s in $(SEEDS); do \
	  $(VID) $(BASE)/OD_s$$s:"no self-forcing" $(BASE)/SD3_s$$s:"up to 3 dreamed steps" $(OUT)/MD_s$$s:"own prediction mixed in" -o $(OUT)/videos/discs_new_long_seed$$s.mp4 $(LV) \
	    --title "New discs after $(TICKS) ticks of life: 20 watched ticks, then a dream of 1000 (vm-00123, seed $$s)" || exit 1; \
	done
