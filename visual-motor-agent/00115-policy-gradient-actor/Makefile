# Makefile for experiment vm-00115-policy-gradient-actor
#
# Reproduces the runs of task 00128: the move token as a policy-gradient actor on real windows (online.actor_pg = 1):
# the move made is one sample of the move token at temperature 1 (no instinct, online.piece_rule = random), its
# advantage per transformer is the mean predicted label of 16 sampled moves minus the move's, and the move token's loss on
# the newest windows is weighted by it (clipped at 3 running RMS; online.pg_replay = 0), plus an entropy bonus
# (online.pg_ent = .03, chosen on the 30k pilot). Two worlds: the food world (arm PG, configs/U_agent.json, against TT
# of vm-00101) and Breakout stage 1 (arm PB, configs/U_breakout.json, against BD of vm-00103). Seeds 1-12, 60k ticks.
# GPUs: 0 = RTX 3090, 1 = RTX 4090 under CUDA_DEVICE_ORDER=PCI_BUS_ID, under MPS, 6 lives per card at a time.
# Outputs go to ~/data/research/visual-motor-agent/exp_00115/.
#
# Usage:
#   make pilot — pg_ent .03 and .1, food world, seeds 1-3, 30k ticks (the choice of pg_ent)
#   make runs W=PG CARD=1 SEEDS="1 2 3 4 5 6" — one arm's lives (W = PG or PB)
#   make frozen W=PG CARD=1 — per checkpoint (30k, 60k) and life: 2000 ticks with learning off (ppo.lr = 0) in new
#                             worlds (seed 100 + s)
#   make base CARD=1 — TT frozen at 30k (TT at 60k is vm-00114's check/TT_ring; BD's are vm-00103's frozen30k, frozen60k)
#   make table — the README's tables (seeds 1-6, paired with BD and TT by seed)
# As run: seeds 1-6 of each arm (PB on the RTX 4090, PG on the RTX 3090); seeds 7-12, `frozen` and `base` were
# cancelled by the owner (2026-11-25) once every seed showed the whole effect.
#   Options: MPS= (no MPS daemon), T="30000 60000" (frozen's checkpoints)

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00115
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID $(MPS)
CARD ?= 1
W ?= PG
T ?= 30000 60000
SEEDS ?= 1 2 3 4 5 6 7 8 9 10 11 12
PGSET = --set online.actor_pg=1 --set online.piece_rule=random --set online.pg_ent=0.03
WSET = $(if $(filter PB,$(W)),--config configs/U_breakout.json,--config configs/U_agent.json)
FROZEN = --set ppo.lr=0 --set online.steps=2000 --set online.log_every=100 --set online.save_every=100000

.PHONY: pilot runs frozen base table

pilot:
	mkdir -p $(OUT)/pilot3
	cd $(DEV) && { for e in 0.03 0.1; do for s in 1 2 3; do \
	  env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config configs/U_agent.json $(PGSET) --set online.pg_ent=$$e \
	    --set online.steps=30000 --seed $$s -o $(OUT)/pilot3/e$${e}_s$$s > $(OUT)/pilot3/e$${e}_s$$s.log 2>&1 & \
	done; done; wait; }

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in $(SEEDS); do \
	  env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(WSET) $(PGSET) --seed $$s --set name=$(W) \
	    -o $(OUT)/$(W)_s$$s > $(OUT)/$(W)_s$$s.log 2>&1 & \
	done; wait; }

frozen:
	cd $(DEV) && for t in $(T); do \
	  o=$(OUT)/frozen$$((t / 1000))k; mkdir -p $$o; \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py $(WSET) $(PGSET) $(FROZEN) --init $(OUT)/$(W)_s$$s/ckpt_$$(printf %08d $$t) \
	      --seed 10$$s -o $$o/$(W)_s$$s > $$o/$(W)_s$$s.log 2>&1 & \
	  done; wait; \
	done

base:
	mkdir -p $(OUT)/frozen30k
	cd $(DEV) && for half in "1 2 3 4 5 6" "7 8 9 10 11 12"; do \
	  for s in $$half; do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PY) lifetime.py --config configs/U_agent.json $(FROZEN) \
	      --init $(D)/exp_00101/TT_s$$s/ckpt_00030000 --seed 10$$s -o $(OUT)/frozen30k/TT_s$$s > $(OUT)/frozen30k/TT_s$$s.log 2>&1 & \
	  done; wait; \
	done

table:
	cd $(DEV) && $(PY) analyze_lives.py pg
