# Makefile for experiment vm-00122-long-lives
#
# Task 00135 (the owner: "how about we run 200k runs for both worlds to see if it improves over time?"): the one
# watching transformer of vm-00120 for 200k ticks in both watched worlds, with and without training on its own dreamed
# frames (up to 3 steps, vm-00121). One seed an arm (few seeds first).
#   OD   configs/W1_discs.json     no self-forcing
#   SD3  configs/W1_discs.json     online.sf_windows = 2, online.sf_steps = 3
#   OB   configs/W1_breakout.json  no self-forcing
#   SB3  configs/W1_breakout.json  as SD3
# A checkpoint every 10k ticks. GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00122/.
#
# Usage:
#   make runs CARD=1 ARMS="SD3 SB3"                — the lives named, together on one card (as run: the two self-forced
#   make runs CARD=0 ARMS="OD OB"                    ones on the RTX 4090, the two others on the RTX 3090)
#   make anchors                                   — the floors of the exam, for the marks in the picture (no GPU)
#   make score CARD=1 ARMS="..." STEPS="20000 ..." — the checkpoints on the exam: the regular starts and the course part
#   make home CARD=1 STEPS="..."                   — the disc lives on their own world after the life's end (64 starts)
#   make curves                                    — the numbers by tick: a row a measure and arm, a column a tick (no GPU)
#   make images                                    — images/curves.png: the main curves as a picture
#   make videos CARD=1                             — the world and the dreams at 200k, in the output folder's videos/
#   make videos-long CARD=1                        — the same with 20 watched ticks and a dream of 1,000 (LONG)
#   Options: MPS= (no MPS daemon), SEEDS (1), TICKS (200000)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00122
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
SEEDS ?= 1
TICKS ?= 200000
CHUNK ?= 64
ARMS ?= OD SD3 OB SB3
DISCS ?= OD SD3
STEPS ?= 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000
CFG_OD = configs/W1_discs.json
CFG_SD3 = configs/W1_discs.json
CFG_OB = configs/W1_breakout.json
CFG_SB3 = configs/W1_breakout.json
SF_OD =
SF_OB =
SF_SD3 = --set online.sf_windows=2 --set online.sf_steps=3
SF_SB3 = --set online.sf_windows=2 --set online.sf_steps=3
WM = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py score

.PHONY: runs anchors score home curves images videos videos-long

# the lives named in ARMS, all at once on one card
runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in $(SEEDS); do \
	  $(foreach a,$(ARMS),{ [ -e $(OUT)/$(a)_s$$s/ckpt_final.bin ] || env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config $(CFG_$(a)) --seed $$s --set name=$(a) --set online.steps=$(TICKS) $(SF_$(a)) -o $(OUT)/$(a)_s$$s > $(OUT)/$(a)_s$$s.log 2>&1; } & ) \
	done; wait

# the same exam as vm-00119's: its floors need no model
anchors:
	mkdir -p $(OUT)
	cd $(DEV) && $(PY) wm_measures.py anchors configs/W1_discs.json -o $(OUT) && $(PY) wm_measures.py anchors configs/W1_breakout.json -o $(OUT)

score:
	cd $(DEV) && for s in $(SEEDS); do for a in $(ARMS); do \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(STEPS) --parts regular course >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	done; done

# the life's own world from its last tick on (wm_measures.home_skip), 64 starts, in a file of its own (home64)
home:
	cd $(DEV) && for s in $(SEEDS); do for a in $(DISCS); do \
	  $(WM) $(OUT)/$${a}_s$$s --chunk $(CHUNK) --decode pair --step $(STEPS) --set home --home-starts 64 --name home64 --parts regular course >> $(OUT)/$${a}_s$$s.wm.log 2>&1 || exit 1; \
	done; done

curves:
	cd $(DEV) && $(PY) wm_curves.py $(OUT) --seeds $(SEEDS) --steps $(STEPS)

images:
	cd $(DEV) && $(PY) wm_curves.py $(OUT) --seeds $(SEEDS) --steps $(STEPS) -o $(CURDIR)/images/curves.png

VID = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_video.py
videos:
	cd $(DEV) && for s in $(SEEDS); do \
	  $(VID) $(OUT)/OB_s$$s:"no self-forcing" $(OUT)/SB3_s$$s:"up to 3 dreamed steps" -o $(OUT)/videos/court_seed$$s.mp4 --step $(TICKS) \
	    --dream 32 --title "The court after $(TICKS) ticks, seed $$s (vm-00122)" && \
	  $(VID) $(OUT)/OD_s$$s:"no self-forcing" $(OUT)/SD3_s$$s:"up to 3 dreamed steps" -o $(OUT)/videos/discs_new_seed$$s.mp4 --step $(TICKS) \
	    --dream 24 --title "Discs the life has never seen, after $(TICKS) ticks, seed $$s (vm-00122)" && \
	  $(VID) $(OUT)/OD_s$$s:"no self-forcing" $(OUT)/SD3_s$$s:"up to 3 dreamed steps" -o $(OUT)/videos/discs_own_seed$$s.mp4 --step $(TICKS) \
	    --set home --dream 24 --title "The life's own discs after $(TICKS) ticks, seed $$s (vm-00122)" || exit 1; \
	done

# the owner, 2026-11-27: "it starts for 20 tickes and then dream 1000 ticks". The model is given the 20 ticks shown and
# nothing before them (--warm 20); the true places are outlined on the first 32 dreamed ticks, the exam's own dream
LONG ?= 1000
LV = --step $(TICKS) --real 20 --warm 20 --dream $(LONG) --outline 32 --fps 12
videos-long:
	cd $(DEV) && for s in $(SEEDS); do \
	  $(VID) $(OUT)/OB_s$$s:"no self-forcing" $(OUT)/SB3_s$$s:"up to 3 dreamed steps" -o $(OUT)/videos/court_long_seed$$s.mp4 $(LV) \
	    --title "The court after $(TICKS) ticks of life: 20 watched ticks, then a dream of $(LONG) (vm-00122, seed $$s)" && \
	  $(VID) $(OUT)/OD_s$$s:"no self-forcing" $(OUT)/SD3_s$$s:"up to 3 dreamed steps" -o $(OUT)/videos/discs_new_long_seed$$s.mp4 $(LV) \
	    --title "New discs after $(TICKS) ticks of life: 20 watched ticks, then a dream of $(LONG) (vm-00122, seed $$s)" && \
	  $(VID) $(OUT)/OD_s$$s:"no self-forcing" $(OUT)/SD3_s$$s:"up to 3 dreamed steps" -o $(OUT)/videos/discs_own_long_seed$$s.mp4 $(LV) \
	    --set home --title "Own discs after $(TICKS) ticks of life: 20 watched ticks, then a dream of $(LONG) (vm-00122, seed $$s)" || exit 1; \
	done
