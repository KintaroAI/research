# Makefile for experiment vm-00116-why-it-misses
#
# Task 00129: why the trained default agent still misses balls in Breakout stage 1, measured on checkpoints that exist.
# Part A, the ring control: BD's (vm-00103) 60k checkpoints, seeds 1-12, learning off (ppo.lr = 0) for 6000 ticks in new
# courts (seed 100 + s), acting on the ring as they lived and on full windows (online.ring = 0), both on one build.
# vm-00114 read .765 on full windows against .644 on the ring, but its ring arm was vm-00103's frozen60k, older code.
# Part B, the miss autopsy (dev/analyze_misses.py record and read): the frozen pieces of an arm's checkpoint play 16 new
# courts for 1500 ticks (seed 100 + s), every tick recorded with the court's state (for scoring only).
# GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00116/.
#
# Usage:
#   make check [CARD=1] — ring and full, seeds 1-12, 6 at a time (about 10 minutes a batch on the RTX 4090)
#   make table          — hit share by mode, the paired difference, and the ring's first 2000 ticks against vm-00103's
#                         frozen60k (the same weights, courts and ticks: equal counts mean the build did not change the play)
#   make autopsy ARM=BD STEP=60000 [MODE=--full] [CARD=1] — the recordings, seeds 1-12, 4 at a time (ARM: BD of
#                         vm-00103, or RB of vm-00110; MODE=--full acts on full windows instead of as the run lived)
#   make read ARM=BD STEP=60000 [MODE=--full]             — the autopsy's tables over those recordings
#   Options: MPS= (no MPS daemon), TICKS=6000

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00116
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
TICKS ?= 6000
FROZEN = --set ppo.lr=0 --set online.steps=$(TICKS) --set online.log_every=100 --set online.save_every=100000
ARM ?= BD
STEP ?= 60000
MODE ?=
FROM = $(if $(filter RB,$(ARM)),$(D)/exp_00110/RB,$(D)/exp_00103/BD)
REC = $(OUT)/autopsy/$(ARM)_$(STEP)$(if $(MODE),_full,)

.PHONY: check table autopsy read

check:
	cd $(DEV) && for m in ring full; do \
	  case $$m in ring) x="";; full) x="--set online.ring=0";; esac; \
	  o=$(OUT)/$$m; mkdir -p $$o; \
	  for half in "1 2 3 4 5 6" "7 8 9 10 11 12"; do \
	    for s in $$half; do \
	      env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config configs/U_breakout.json $$x $(FROZEN) \
	        --init $(D)/exp_00103/BD_s$$s/ckpt_00060000 --seed 10$$s -o $$o/s$$s > $$o/s$$s.log 2>&1 & \
	    done; wait; \
	    rm -f $$o/s*/ckpt_*.bin; \
	  done; \
	done

table:
	cd $(DEV) && $(PY) analyze_misses.py ring $(OUT)

autopsy:
	mkdir -p $(REC)
	cd $(DEV) && for part in "1 2 3 4" "5 6 7 8" "9 10 11 12"; do \
	  for s in $$part; do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) analyze_misses.py record $(REC)/s$$s.npz $(FROM)_s$$s --step $(STEP) $(MODE) \
	      --ticks 1500 --bodies 16 --seed 10$$s > $(REC)/s$$s.log 2>&1 & \
	  done; wait; \
	done

read:
	cd $(DEV) && $(PY) analyze_misses.py read $(REC)/s*.npz
