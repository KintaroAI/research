# Makefile for experiment vm-00125-dream-dice
#
# Task 00138 (the owner, 2026-11-27: "yes - lets do that. can temperature help?"): a dream that rolls dice. No new life:
# the 60k checkpoints of five court lives (seed 1) dreamed with today's decoding and with dice.
#   OB, SB3    vm-00122: neither; up to 3 dreamed steps (self-forcing)
#   MB         vm-00123: .5 its own prediction mixed in
#   M7B, MNB   vm-00124: .7 its own prediction; .5 with .2 random grey
# The decodings (wm_measures.Pieces):
#   pair                    today's: each pixel's most probable bin and its more probable neighbour
#   draw0                   each pixel's likeliest value (its heaviest hump), no dice
#   draw, draw0.7, draw0.5  dice by the humps' chances, at temperatures 1, .7, .5
#   drawp, drawp0.7, drawp0.5   dice by the humps' tallest bins (pair's ranking; without dice it is pair), at 1, .7, .5
# Everything is written to ~/data/research/visual-motor-agent/exp_00125/; nothing into the earlier experiments' folders.
# GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS.
#
# Usage:
#   make free CARD=1 [ARMS="M7B"] [DECODES="pair drawp0.5"]   — the free dream of 200 ticks (32 stretches), the court's balls
#   make regular CARD=1 [ARMS=...] [DECODES=...]              — the regular part (64 real ticks, a dream of 32) and the course part
#   make table                                                — the README's tables (no GPU)
#   make images                                               — images/balls.png (no GPU)
#   make video CARD=1 [LIFE=M7B]                              — 20 watched ticks, then a dream of 1,000: today's decoding beside the dice
#   Options: MPS= (no MPS daemon), DICE (0: the dice's number), STEP (60000)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00125
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
STEP ?= 60000
DICE ?= 0
CHUNK ?= 64
ARMS ?= OB SB3 MB M7B MNB
DECODES ?= pair draw0 draw draw0.7 draw0.5 drawp drawp0.7 drawp0.5
RUN_OB = $(D)/exp_00122/OB_s1
RUN_SB3 = $(D)/exp_00122/SB3_s1
RUN_MB = $(D)/exp_00123/MB_s1
RUN_M7B = $(D)/exp_00124/M7B_s1
RUN_MNB = $(D)/exp_00124/MNB_s1
WM = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py score
TO = --step $(STEP) --chunk $(CHUNK) --decode $(DECODES) --dice $(DICE)

.PHONY: free regular table images video

# the free dream of 200 ticks from 32 stretches of the exam: the zones' counts and the court's balls (served, back, specks)
free:
	mkdir -p $(OUT)
	cd $(DEV) && $(foreach a,$(ARMS),$(WM) $(RUN_$(a)) $(TO) --parts free --name free --out $(OUT)/$(a)_s1/wm >> $(OUT)/$(a)_s1.wm.log 2>&1 && ) true

# the regular part and the course part: what the dice cost where today's decoding works
regular:
	mkdir -p $(OUT)
	cd $(DEV) && $(foreach a,$(ARMS),$(WM) $(RUN_$(a)) $(TO) --parts regular course --out $(OUT)/$(a)_s1/wm >> $(OUT)/$(a)_s1.wm.log 2>&1 && ) true

table:
	cd $(DEV) && $(PY) wm_dice.py $(OUT) --arms $(ARMS) --decodes $(DECODES) --step $(STEP)

# the picture: the dreams with a ball by dreamed tick, today's decoding and the dice by the tallest bin at .7 and 1
SHOWN ?= pair drawp0.7 drawp
images:
	cd $(DEV) && $(PY) wm_dice.py $(OUT) --arms $(ARMS) --decodes $(DECODES) --step $(STEP) --show $(SHOWN) -o $(CURDIR)/images/balls.png

# one life's long dream, today's decoding beside the dice (the dreams start from the same watched ticks)
LIFE ?= M7B
R = $(RUN_$(LIFE))
video:
	mkdir -p $(OUT)/videos
	cd $(DEV) && env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_video.py $(R)@pair:"today (pair)" $(R)@drawp0.5:"dice, T .5" \
	  $(R)@drawp0.7:"dice, T .7" $(R)@draw:"dice by chance, T 1" \
	  -o $(OUT)/videos/court_dice_$(LIFE).mp4 --step $(STEP) --real 20 --warm 20 --dream 1000 --outline 0 --fps 12 --dice $(DICE) \
	  --title "The court, life $(LIFE) at $(STEP) ticks: 20 watched ticks, then a dream of 1000 (vm-00125)"
