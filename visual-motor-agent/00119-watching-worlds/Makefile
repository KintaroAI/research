# Makefile for experiment vm-00119-watching-worlds
#
# Task 00132: the default agent's model without the label only watches (online.watch = 1: the null move every tick)
# two worlds that move by themselves:
#   WD  configs/W_discs.json     2 food discs at 1.5 px a tick and 2 poison discs at .75 on a torus, bouncing off each other
#   WB  configs/W_breakout.json  Breakout stage 1 with the paddle given to a script (env.auto_lag = 6); the score is the hunger
# Seeds 1-12, 60k ticks, checkpoints every 10k. A life's world is make_world(1, config, seed) stepped with null moves, so
# every frame a life saw can be made again from its seed (checked: 400 of 400 frames in both worlds).
# GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS. As run: seeds 1-8 on the RTX 4090
# (6 lives at a time), seeds 9-12 on the RTX 3090 (4 at a time).
# Outputs go to ~/data/research/visual-motor-agent/exp_00119/.
#
# The measures (dev/wm_measures.py) read a model through three calls (reset, observe, dream) and never its loss, bins or
# tokens. The exam is one world of 16 copies, seed 1001, 2,300 ticks; 256 regular starts, each a dream of 32 ticks after
# 64 real ones. Each life's numbers go to its folder, wm/step_<step>_<decode>_<set>.json.
#
# Usage:
#   make runs CARD=1 SEEDS="1 2 3 4 5 6 7 8" SLOTS=6   — the lives of both worlds, SLOTS at a time; a life whose final
#                                                        checkpoint exists is skipped
#   make anchors                                       — the floors and the ceilings on the working set (no GPU)
#   make score CARD=1 SEEDS="..." JOBS=2               — the 60k weights with `pair`: regular starts, events, warm-up
#   make mean CARD=1 SEEDS="..."                       — the same weights with `mean`: regular starts
#   make instinct CARD=1 SEEDS="..."                   — the same weights dreamed as the agent's instinct dreams (the
#                                                        expected value, 2 passes, each quarter alone): regular starts;
#                                                        added after the first results
#   make instinct-pair CARD=1 SEEDS="..."              — the instinct's way with `pair` in place of the expected value
#                                                        (online.instinct_decode = pair); added after the first results
#   make warm-dream CARD=1 SEEDS="..."                 — `pair`, 8 dreamed ticks from a reset after 2, 4 and 8 real ticks;
#                                                        added after the first results
#   make home CARD=1 SEEDS="..."                       — `pair` on the life's own world from tick 60,000 on
#   make learn CARD=1 SEEDS="..."                      — `pair` at the checkpoints of 10k to 50k: 8 dreamed ticks, the
#                                                        next-tick events (STEPS=0 SEEDS=1: the untrained weights)
#   make free CARD=1 SEEDS="1 2 3"                     — `pair`, a dream of 200 ticks
#   make table                                         — the README's tables (no GPU)
#   make images CARD=1                                 — images/curves_<arm>.png (the dream's curves, all seeds) and
#                                                        images/dream_<arm>.png (seed 1: real ticks and dreamed ones)
#   make test CARD=1                                   — dev/tests/test_wm_measures.py and test_watch.py
#   Options: MPS= (no MPS daemon), ARMS="WD" or "WB" (one world only), JOBS (scoring processes at a time; one holds
#   about 10 GB of the card). As run: one target for one life at a time on each card, the lives scored as they ended
#   (score first, then mean, home, learn, free), beside the lives still running; then, on the cards whose lives had
#   ended, the readings added after the first results (instinct, instinct-pair, warm-dream, learn STEPS=0 SEEDS=1).
#   A life could be scored on either card.

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00119
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
SEEDS ?= 1 2 3 4 5 6 7 8 9 10 11 12
SLOTS ?= 6
ARMS ?= WD WB
CFG_WD = configs/W_discs.json
CFG_WB = configs/W_breakout.json

JOBS ?= 2
# the learning curve's checkpoints; 60000 is read from the full run's first 8 ticks. The untrained weights (STEPS=0)
# were scored for seed 1 only: they are far below 0 in every life and cost the most (45 decoding passes a tick)
STEPS ?= 10000 20000 30000 40000 50000
WM = env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py score
# one line per life (arm, seed); xargs keeps JOBS scoring at a time. $$0: the arm, $$1: the seed
LIVES = for s in $(SEEDS); do $(foreach a,$(ARMS),echo $(a) $$s;) done | xargs -P $(JOBS) -n 2 sh -c

.PHONY: runs anchors score mean instinct instinct-pair warm-dream home learn free table images test

# one line per life (arm, config, seed), seed by seed so that both worlds come in together; xargs keeps SLOTS running
runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in $(SEEDS); do \
	  $(foreach a,$(ARMS),echo $(a) $(CFG_$(a)) $$s;) \
	done | xargs -P $(SLOTS) -n 3 sh -c \
	  '[ -e $(OUT)/$$0_s$$2/ckpt_final_p0.bin ] || env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config $$1 --seed $$2 --set name=$$0 -o $(OUT)/$$0_s$$2 > $(OUT)/$$0_s$$2.log 2>&1'

anchors:
	cd $(DEV) && $(PY) wm_measures.py anchors $(CFG_WD) -o $(OUT) && $(PY) wm_measures.py anchors $(CFG_WB) -o $(OUT)

score:
	cd $(DEV) && $(LIVES) '$(WM) $(OUT)/$$0_s$$1 --step 60000 --decode pair --parts regular events warm >> $(OUT)/$$0_s$$1.wm.log 2>&1'

mean:
	cd $(DEV) && $(LIVES) '$(WM) $(OUT)/$$0_s$$1 --step 60000 --decode mean --parts regular >> $(OUT)/$$0_s$$1.wm.log 2>&1'

instinct:
	cd $(DEV) && $(LIVES) '$(WM) $(OUT)/$$0_s$$1 --step 60000 --decode instinct --parts regular >> $(OUT)/$$0_s$$1.wm.log 2>&1'

instinct-pair:
	cd $(DEV) && $(LIVES) '$(WM) $(OUT)/$$0_s$$1 --step 60000 --decode instinct_pair --parts regular >> $(OUT)/$$0_s$$1.wm.log 2>&1'

warm-dream:
	cd $(DEV) && $(LIVES) '$(WM) $(OUT)/$$0_s$$1 --step 60000 --decode pair --parts warmdream >> $(OUT)/$$0_s$$1.wm.log 2>&1'

home:
	cd $(DEV) && $(LIVES) '$(WM) $(OUT)/$$0_s$$1 --step 60000 --decode pair --set home --parts regular >> $(OUT)/$$0_s$$1.wm.log 2>&1'

learn:
	cd $(DEV) && $(LIVES) '$(WM) $(OUT)/$$0_s$$1 --step $(STEPS) --decode pair --horizon 8 --parts regular next --name learn >> $(OUT)/$$0_s$$1.wm.log 2>&1'

free:
	cd $(DEV) && $(LIVES) '$(WM) $(OUT)/$$0_s$$1 --step 60000 --decode pair --parts free >> $(OUT)/$$0_s$$1.wm.log 2>&1'

table:
	cd $(DEV) && for a in $(ARMS); do $(PY) wm_measures.py table $(OUT) $$a --seeds $(SEEDS); done

images:
	cd $(DEV) && for a in $(ARMS); do \
	  $(PY) wm_measures.py figure $(OUT) $$a -o $(CURDIR)/images/curves_$$a.png --seeds $(SEEDS) && \
	  env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) wm_measures.py strip $(OUT)/$${a}_s1 --step 60000 -o $(CURDIR)/images/dream_$$a.png; \
	done

test:
	cd $(DEV) && env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) -m pytest tests/test_wm_measures.py tests/test_watch.py -q
