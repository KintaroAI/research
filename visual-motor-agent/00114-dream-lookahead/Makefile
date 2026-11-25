# Makefile for experiment vm-00114-dream-lookahead
#
# Reproduces the frozen check of task 00127: the instinct holds each of its 16 candidates for H ticks in the model's
# dream (online.instinct_h = H, full windows: online.ring = 0) and scores it by the dream's H-step return (dream_return:
# each imagined step's meal or poison read from the predicted hunger by the stored label's rule, then the H-th move's
# label). Two worlds: the food world (configs/U_agent.json, TT of vm-00101) and Breakout stage 1 (configs/U_breakout.json,
# BD of vm-00103). The result was negative frozen, so the lives (step 2) were not run (the owner, 2026-11-25).
# GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00114/.
#
# Usage:
#   make check — TT's and BD's 60k checkpoints, seeds 1-12, 2000 ticks with learning off (ppo.lr = 0) in new worlds
#                (seed 100 + s), acting as they lived (ring, H = 1; TT only: BD's is vm-00103's frozen60k), on full
#                windows at H = 1, H = 2 and H = 4. CK= picks world:mode pairs, CARD= the GPU (6 at a time).
#   make probe — seeds 1-3 of each world at H = 4, 600 ticks frozen, saving every dream_return input (probe/*.npz)
#   make table — the check's table and the probe's reading (analyze_lookahead.py)
#   Options: MPS= (no MPS daemon)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00114
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
TT_SET = --config configs/U_agent.json
BD_SET = --config configs/U_breakout.json
LA = --set online.ring=0 --set online.instinct_h=4
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000
CK ?= BD:h4 TT:h4 BD:full TT:full TT:ring TT:h2 BD:h2

.PHONY: check probe table

check:
	cd $(DEV) && for wm in $(CK); do \
	  w=$${wm%%:*}; m=$${wm##*:}; \
	  case $$w in TT) ws="$(TT_SET)"; dir=$(D)/exp_00101/TT;; BD) ws="$(BD_SET)"; dir=$(D)/exp_00103/BD;; esac; \
	  case $$m in ring) x="";; full) x="--set online.ring=0";; h4) x="$(LA)";; h2) x="--set online.ring=0 --set online.instinct_h=2";; esac; \
	  o=$(OUT)/check/$${w}_$$m; mkdir -p $$o; \
	  for half in "1 2 3 4 5 6" "7 8 9 10 11 12"; do \
	    for s in $$half; do \
	      env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $$ws $$x $(FROZEN) --init $${dir}_s$$s/ckpt_00060000 \
	        --seed 10$$s -o $$o/s$$s > $$o/s$$s.log 2>&1 & \
	    done; wait; \
	  done; \
	done

probe:
	mkdir -p $(OUT)/probe
	cd $(DEV) && { for w in TT BD; do \
	  case $$w in TT) ws="$(TT_SET)"; dir=$(D)/exp_00101/TT;; BD) ws="$(BD_SET)"; dir=$(D)/exp_00103/BD;; esac; \
	  for s in 1 2 3; do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) analyze_lookahead.py record $(OUT)/probe/$${w}_s$$s.npz $$ws $(LA) \
	      --set ppo.lr=0 --set online.steps=600 --set online.log_every=100 --set online.save_every=100000 \
	      --init $${dir}_s$$s/ckpt_00060000 --seed 10$$s -o $(OUT)/probe/$${w}_s$$s > $(OUT)/probe/$${w}_s$$s.log 2>&1 & \
	  done; done; wait; }

table:
	cd $(DEV) && $(PY) analyze_lookahead.py table $(OUT)/check && $(PY) analyze_lookahead.py read $(OUT)/probe
