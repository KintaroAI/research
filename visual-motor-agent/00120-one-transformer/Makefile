# Makefile for experiment vm-00120-one-transformer
#
# Task 00133 (the owner, 2026-11-26): the world-model measures of vm-00119 on ONE transformer that reads the whole
# 64 x 64 frame, in place of the four quarter transformers. The model only watches (online.watch = 1), as in vm-00119:
#   OD  configs/W1_discs.json     configs/W_discs.json without model.band_n, band_cols, band_win and online.pieces
#   OB  configs/W1_breakout.json  configs/W_breakout.json likewise
# Seeds 1-12, 60k ticks, checkpoints every 10k, against WD and WB of vm-00119 by seed (a seed's lives watched the same
# world). GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS. As run: seeds 1-8 on the
# RTX 4090 (6 lives at a time), seeds 9-12 on the RTX 3090 (4 at a time), as vm-00119's.
# Outputs go to ~/data/research/visual-motor-agent/exp_00120/.
#
# The measures are dev/wm_measures.py's, with pair, on vm-00119's exam (seed 1001). The one transformer is scored 64
# windows at a time (CHUNK; 256 do not fit a card: its attention spans 696 tokens a window).
#
# Usage:
#   make runs CARD=1 SEEDS="1 2 3 4 5 6 7 8" SLOTS=6   — the lives of both worlds, SLOTS at a time
#   make anchors                                       — the floors and the ceilings on the working set (no GPU)
#   make score CARD=1 SEEDS="..."                      — the 60k weights: regular starts, events, warm-up
#   make warm-dream CARD=1 SEEDS="..."                 — 8 dreamed ticks from a reset after 2, 4 and 8 real ticks
#   make seam CARD=1 SEEDS="..."                       — the first 8 dreamed ticks, the error near the quarters' borders
#                                                        and away from them
#   make seam-four CARD=1 SEEDS="..."                  — the same part for vm-00119's four transformers (WD, WB)
#   make home CARD=1 SEEDS="..."                       — the life's own world from tick 60,000 on
#   make learn CARD=1 SEEDS="..."                      — the checkpoints of 10k to 50k: 8 dreamed ticks, the next-tick events
#   make free CARD=1 SEEDS="1 2 3"                     — a dream of 200 ticks
#   make borders                                       — the things held by where they have been (inside one quarter so
#                                                        far, or across a border between the quarters), for the one and
#                                                        the four, from the kept distances (no GPU; after score)
#   make table                                         — each arm's tables (no GPU)
#   make compare                                       — one against four, paired by seed (no GPU)
#   make images CARD=1                                 — images/curves_<arm>.png (one against four, all seeds) and
#                                                        images/dream_<arm>.png (seed 1: real ticks and dreamed ones)
#   Options: MPS= (no MPS daemon), ARMS="OD" or "OB", JOBS (scoring processes at a time; one holds about 8 GB)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00120
FOUR = $(D)/exp_00119
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
SEEDS ?= 1 2 3 4 5 6 7 8 9 10 11 12
SLOTS ?= 6
JOBS ?= 2
CHUNK ?= 64
ARMS ?= OD OB
CFG_OD = configs/W1_discs.json
CFG_OB = configs/W1_breakout.json
FOUR_OD = WD
FOUR_OB = WB
STEPS ?= 10000 20000 30000 40000 50000
WM = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py score
# one line per life (arm, seed); xargs keeps JOBS scoring at a time. $$0: the arm, $$1: the seed
LIVES = for s in $(SEEDS); do $(foreach a,$(ARMS),echo $(a) $$s;) done | xargs -P $(JOBS) -n 2 sh -c
ONE = $(OUT)/$$0_s$$1 --chunk $(CHUNK) --decode pair
LOG = >> $(OUT)/$$0_s$$1.wm.log 2>&1

.PHONY: runs anchors score warm-dream seam seam-four home learn free borders table compare images

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in $(SEEDS); do \
	  $(foreach a,$(ARMS),echo $(a) $(CFG_$(a)) $$s;) \
	done | xargs -P $(SLOTS) -n 3 sh -c \
	  '[ -e $(OUT)/$$0_s$$2/ckpt_final.bin ] || env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config $$1 --seed $$2 --set name=$$0 -o $(OUT)/$$0_s$$2 > $(OUT)/$$0_s$$2.log 2>&1'

# the floors and the ceilings of the exam (the same as vm-00119's: they need no model), beside the new arms for `table`
anchors:
	cd $(DEV) && $(PY) wm_measures.py anchors $(CFG_OD) -o $(OUT) && $(PY) wm_measures.py anchors $(CFG_OB) -o $(OUT)

score:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step 60000 --parts regular events warm $(LOG)'

warm-dream:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step 60000 --parts warmdream $(LOG)'

seam:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step 60000 --parts seam $(LOG)'

# vm-00119's four transformers, the seam part only (256 windows at a time, as they were scored)
seam-four:
	cd $(DEV) && for s in $(SEEDS); do $(foreach a,$(ARMS),echo $(FOUR_$(a)) $$s;) done | xargs -P $(JOBS) -n 2 sh -c \
	  '$(WM) $(FOUR)/$$0_s$$1 --step 60000 --decode pair --parts seam >> $(FOUR)/$$0_s$$1.wm.log 2>&1'

home:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step 60000 --set home --parts regular $(LOG)'

learn:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step $(STEPS) --horizon 8 --parts regular next --name learn $(LOG)'

free:
	cd $(DEV) && $(LIVES) '$(WM) $(ONE) --step 60000 --parts free $(LOG)'

# added after the first results: a thing that stayed inside one quarter was drawn by one of the four transformers alone
borders:
	cd $(DEV) && $(PY) wm_measures.py borders $(foreach a,$(ARMS),$(foreach s,$(SEEDS),$(OUT)/$(a)_s$(s) $(FOUR)/$(FOUR_$(a))_s$(s)))

table:
	cd $(DEV) && for a in $(ARMS); do $(PY) wm_measures.py table $(OUT) $$a --seeds $(SEEDS); done

compare:
	cd $(DEV) && $(foreach a,$(ARMS),$(PY) wm_measures.py compare $(OUT) $(a) $(FOUR) $(FOUR_$(a)) --seeds $(SEEDS);)

images:
	cd $(DEV) && $(foreach a,$(ARMS),$(PY) wm_measures.py figure $(OUT) $(a) -o $(CURDIR)/images/curves_$(a).png --seeds $(SEEDS) \
	    --against $(FOUR) $(FOUR_$(a)) "one transformer" "four transformers" && \
	  env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py strip $(OUT)/$(a)_s1 --step 60000 --chunk $(CHUNK) \
	    -o $(CURDIR)/images/dream_$(a).png &&) true
