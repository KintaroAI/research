# Makefile for experiment vm-00121-self-forcing-watching
#
# Task 00134: the one watching transformer of vm-00120 trained on its own dreamed frames (self-forcing, task 00039):
# of the 7 replayed windows of every update, 2 get the inputs (frame and body) of their last 1..K steps from the
# model's own predictions (pair decoding, 2 passes a frame); the targets stay real.
#   SD3  configs/W1_discs.json     online.sf_windows = 2, online.sf_steps = 3   (vm-00032's rule)
#   SD7  configs/W1_discs.json     online.sf_windows = 2, online.sf_steps = 7   (as deep as the window of 8 allows)
#   SB3  configs/W1_breakout.json  as SD3
#   SB7  configs/W1_breakout.json  as SD7
# Against OD and OB of vm-00120 (no self-forcing) by seed: a seed's lives watched the same world and started from the
# same weights. Seeds 1-6; SB7 seeds 1-2 only (seeds 7-12 and four SB7 lives were dropped by the owner's word after
# the first table: few seeds first). `compare` and `table` take the seeds that are there. 60k ticks,
# checkpoints every 10k. GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS. Outputs go to
# ~/data/research/visual-motor-agent/exp_00121/.
#
# The measures are dev/wm_measures.py's, with pair, on vm-00119's exam (seed 1001), 64 windows a pass (CHUNK).
#
# Usage:
#   make runs CARD=1 SEEDS="1 2 3 4 5 6" SLOTS=6   — the lives, SLOTS at a time
#   make anchors                                   — the floors and the ceilings on the working set (no GPU)
#   make score CARD=1 SEEDS="..."                  — the 60k weights: regular starts, events, warm-up
#   make warm-dream CARD=1 SEEDS="..."             — 8 dreamed ticks from a reset after 2, 4 and 8 real ticks
#   make home CARD=1 SEEDS="..."                   — the life's own world from tick 60,000 on
#   make learn CARD=1 SEEDS="..."                  — the checkpoints of 10k to 50k: 8 dreamed ticks, the next-tick events
#   make free CARD=1 SEEDS="1 2 3"                 — a dream of 200 ticks
#   make course CARD=1 SEEDS="..."                 — added after the first results: is a dreamed thing behind, ahead or
#                                                    aside of its course; the working set and 64 starts at home
#   make course-base CARD=1 SEEDS="..."            — the same for vm-00120's lives without self-forcing
#   make table                                     — each arm's tables (no GPU)
#   make compare                                   — each arm against vm-00120's lives without self-forcing, and the
#                                                    deeper arm against the shallower, paired by seed (no GPU)
#   make images CARD=1 ARMS="SD3 SB3"              — images/curves_<arm>.png (the arm against none, all seeds) and
#                                                    images/dream_<arm>.png (seed 1: real ticks and dreamed ones);
#                                                    the README shows those of the two 3-step arms
#   Options: MPS= (no MPS daemon), ARMS="SD3 SD7", JOBS (scoring processes at a time; one holds about 8 GB)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00121
BASE = $(D)/exp_00120
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
SEEDS ?= 1 2 3 4 5 6
SLOTS ?= 6
JOBS ?= 2
CHUNK ?= 64
ARMS ?= SD3 SD7 SB3 SB7
SF_WINDOWS ?= 2
CFG_SD3 = configs/W1_discs.json
CFG_SD7 = configs/W1_discs.json
CFG_SB3 = configs/W1_breakout.json
CFG_SB7 = configs/W1_breakout.json
K_SD3 = 3
K_SD7 = 7
K_SB3 = 3
K_SB7 = 7
BASE_SD3 = OD
BASE_SD7 = OD
BASE_SB3 = OB
BASE_SB7 = OB
BASES ?= OD OB
NAME_SD3 = "self-forcing, up to 3 steps"
NAME_SD7 = "self-forcing, up to 7 steps"
NAME_SB3 = "self-forcing, up to 3 steps"
NAME_SB7 = "self-forcing, up to 7 steps"
STEPS ?= 10000 20000 30000 40000 50000
WM = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py score
# one line per life (arm, seed); xargs keeps JOBS scoring at a time. $$0: the arm, $$1: the seed
LIVES = for s in $(SEEDS); do $(foreach a,$(ARMS),echo $(a) $$s;) done | xargs -P $(JOBS) -n 2 sh -c
ONE = $(OUT)/$$0_s$$1 --chunk $(CHUNK) --decode pair
LOG = >> $(OUT)/$$0_s$$1.wm.log 2>&1

.PHONY: runs anchors score warm-dream home learn free course course-base table compare images

# $$0: the arm, $$1: its config, $$2: the deepest dreamed step, $$3: the seed
runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in $(SEEDS); do \
	  $(foreach a,$(ARMS),echo $(a) $(CFG_$(a)) $(K_$(a)) $$s;) \
	done | xargs -P $(SLOTS) -n 4 sh -c \
	  '[ -e $(OUT)/$$0_s$$3/ckpt_final.bin ] || env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config $$1 --seed $$3 --set name=$$0 --set online.sf_windows=$(SF_WINDOWS) --set online.sf_steps=$$2 -o $(OUT)/$$0_s$$3 > $(OUT)/$$0_s$$3.log 2>&1'

# the floors and the ceilings of the exam (the same as vm-00119's: they need no model), beside the arms for `table`
anchors:
	cd $(DEV) && $(PY) wm_measures.py anchors configs/W1_discs.json -o $(OUT) && $(PY) wm_measures.py anchors configs/W1_breakout.json -o $(OUT)

score:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step 60000 --parts regular events warm $(LOG)'

warm-dream:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step 60000 --parts warmdream $(LOG)'

home:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step 60000 --set home --parts regular $(LOG)'

learn:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step $(STEPS) --horizon 8 --parts regular next --name learn $(LOG)'

free:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step 60000 --parts free $(LOG)'

# added after the first results: the things on their course. The working set's part goes into the life's summary (never
# while another scoring job of that life runs); the 64 starts of the life's own world are a file of their own (home64)
COURSE = --step 60000 --parts course
HOME64 = --step 60000 --set home --home-starts 64 --name home64 --parts course
course:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) $(COURSE) $(LOG) && $(WM) $(ONE) $(HOME64) $(LOG)'

# $$0: OD or OB, $$1: the seed
course-base:
	cd $(DEV) && for s in $(SEEDS); do $(foreach a,$(BASES),echo $(a) $$s;) done | xargs -P $(JOBS) -n 2 sh -c \
	  '$(WM) $(BASE)/$$0_s$$1 --chunk $(CHUNK) --decode pair $(COURSE) >> $(BASE)/$$0_s$$1.wm.log 2>&1 && $(WM) $(BASE)/$$0_s$$1 --chunk $(CHUNK) --decode pair $(HOME64) >> $(BASE)/$$0_s$$1.wm.log 2>&1'

table:
	cd $(DEV) && for a in $(ARMS); do $(PY) wm_measures.py table $(OUT) $$a --seeds $(SEEDS); done

# every arm against the lives without self-forcing; then up to 7 dreamed steps against up to 3
compare:
	cd $(DEV) && $(foreach a,$(ARMS),$(PY) wm_measures.py compare $(OUT) $(a) $(BASE) $(BASE_$(a)) --seeds $(SEEDS);)
	cd $(DEV) && $(PY) wm_measures.py compare $(OUT) SD7 $(OUT) SD3 --seeds $(SEEDS) && $(PY) wm_measures.py compare $(OUT) SB7 $(OUT) SB3 --seeds $(SEEDS)

images:
	cd $(DEV) && $(foreach a,$(ARMS),$(PY) wm_measures.py figure $(OUT) $(a) -o $(CURDIR)/images/curves_$(a).png --seeds $(SEEDS) \
	    --against $(BASE) $(BASE_$(a)) $(NAME_$(a)) "none" && \
	  env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py strip $(OUT)/$(a)_s1 --step 60000 --chunk $(CHUNK) \
	    -o $(CURDIR)/images/dream_$(a).png &&) true
