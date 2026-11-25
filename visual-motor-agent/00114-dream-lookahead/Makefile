# Makefile for experiment vm-00114-dream-lookahead
#
# Reproduces the runs of task 00127: the instinct holds each of its 16 candidates for H = 4 ticks in the model's dream
# (online.instinct_h = 4, full windows: online.ring = 0) and scores it by the dream's 4-step return (dream_return:
# each imagined step's meal or poison read from the predicted hunger by the stored label's rule, then the 4th move's
# label). Two worlds, each against its existing arm: the food world (configs/U_agent.json, TT of vm-00101) and Breakout
# stage 1 (configs/U_breakout.json, BD of vm-00103). GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID,
# under MPS. Outputs go to ~/data/research/visual-motor-agent/exp_00114/.
#
# Usage:
#   make check — step 1, no training: TT's and BD's 60k checkpoints, 2000 ticks with learning off (ppo.lr = 0) in new
#                worlds (seed 100 + s), acting three ways: as they lived (ring, H = 1; TT only, BD's is vm-00103's
#                frozen60k), full windows H = 1, full windows H = 4. CK= picks world:mode pairs, CARD= the GPU.
#   make runs W=TT|BD — step 2, arm LA (H = 4, full windows), seeds SEEDS (60k ticks) on CARD
#   make frozen W=TT|BD — (after the runs) LA's checkpoints T (30k, 60k), 2000 ticks with learning off, seed 100 + s
#   Options: MPS= (no MPS daemon)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00114
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
T ?= 30000 60000
SEEDS ?= 1 2 3 4 5 6
W ?= TT
TT_SET = --config configs/U_agent.json
BD_SET = --config configs/U_breakout.json
WSET = $(if $(filter BD,$(W)),$(BD_SET),$(TT_SET))
LA = --set online.ring=0 --set online.instinct_h=4
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000
CK ?= BD:h4 TT:h4 BD:full TT:full TT:ring

.PHONY: check runs frozen

check:
	cd $(DEV) && for wm in $(CK); do \
	  w=$${wm%%:*}; m=$${wm##*:}; \
	  case $$w in TT) ws="$(TT_SET)"; dir=$(D)/exp_00101/TT;; BD) ws="$(BD_SET)"; dir=$(D)/exp_00103/BD;; esac; \
	  case $$m in ring) x="";; full) x="--set online.ring=0";; h4) x="$(LA)";; esac; \
	  o=$(OUT)/check/$${w}_$$m; mkdir -p $$o; \
	  for half in "1 2 3 4 5 6" "7 8 9 10 11 12"; do \
	    for s in $$half; do \
	      env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $$ws $$x $(FROZEN) --init $${dir}_s$$s/ckpt_00060000 \
	        --seed 10$$s -o $$o/s$$s > $$o/s$$s.log 2>&1 & \
	    done; wait; \
	  done; \
	done

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(WSET) $(LA) --seed $$s --set name=LA$(W) \
	      -o $(OUT)/LA$(W)_s$$s > $(OUT)/LA$(W)_s$$s.log 2>&1 & \
	  done; wait; }

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(WSET) $(LA) $(FROZEN) \
	      --init $(OUT)/LA$(W)_s$$s/ckpt_$$(printf %08d $$t) --seed 10$$s -o $$o/LA$(W)_s$$s > $$o/LA$(W)_s$$s.log 2>&1 & \
	  done; wait; \
	done
