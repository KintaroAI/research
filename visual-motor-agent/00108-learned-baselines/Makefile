# Makefile for experiment vm-00108-learned-baselines
#
# Reproduces the runs of task 00120: learned baselines on the same pixels, signal and ticks as the default agent
# (dev/baselines/: vm_env.py, the world's grey frame, drives and previous move, the meal label's immediate part as the
# reward, one life). Two worlds: breakout (configs/U_breakout.json, stage 1; ours: vm-00103's BD) and food
# (configs/U_agent.json's world; ours: vm-00101's TT). Seeds 1-6 (the same world seeds as ours).
#   DR: DreamerV3 (PyTorch port NM512/dreamerv3-torch, commit 6ef8646, in ~/src/dreamerv3-torch; ~/venvs/dreamer),
#       one body, 60k ticks, train ratio 512
#   PP: PPO, one body, 60k ticks          PL: PPO, 16 bodies, 2M samples
# Seeds 1-3 on the RTX 3090 (GPU 0 under CUDA_DEVICE_ORDER=PCI_BUS_ID), 4-6 on the RTX 4090 (GPU 1), under MPS.
# Outputs go to ~/data/research/visual-motor-agent/exp_00108/.
#
# Usage:
#   make all — pilot, ppo, dreamer
#   make pilot — Dreamer, 5000 ticks of breakout seed 1 on each card alone (timing; pilot_gpu0.log, pilot_gpu1.log)
#   make ppo — PP and PL, both worlds, seeds 1-6 (24 lives at once)
#   make dreamer — DR breakout seeds 1-6, then DR food seeds 1-6 (6 lives at once, 3 per card)
#   Options: MPS= (no MPS daemon), TR=1024 (Dreamer's train ratio)

DEV = ../dev
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00108
MPS ?= CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
G = CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 $(MPS)
DPY = $(HOME)/venvs/dreamer/bin/python -W ignore
PPY = $(HOME)/venvs/vm-gpu/bin/python
TR ?= 512
SEEDS = 1 2 3 4 5 6
CARD = $$([ $$s -le 3 ] && echo 0 || echo 1)
CFG = $$([ $$w = breakout ] && echo configs/U_breakout.json || echo configs/U_agent.json)

.PHONY: all pilot ppo dreamer

all: pilot ppo dreamer

pilot:
	mkdir -p $(OUT)
	cd $(DEV) && for c in 0 1; do \
	  ( t0=$$(date +%s); env $(G) CUDA_VISIBLE_DEVICES=$$c $(DPY) baselines/run_dreamer.py --config configs/U_breakout.json --seed 1 \
	      --steps 5000 --train-ratio $(TR) --out $(OUT)/pilot_gpu$$c > $(OUT)/pilot_gpu$$c.log 2>&1; \
	    echo "5000 ticks in $$(( $$(date +%s) - t0 )) s" >> $(OUT)/pilot_gpu$$c.log ); \
	done

ppo:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for w in breakout food; do for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PPY) baselines/run_ppo.py --config $(CFG) --seed $$s --bodies 1 --steps 60000 \
	      --out $(OUT)/PP_$${w}_s$$s > $(OUT)/PP_$${w}_s$$s.log 2>&1 & \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(PPY) baselines/run_ppo.py --config $(CFG) --seed $$s --bodies 16 --steps 2000000 \
	      --out $(OUT)/PL_$${w}_s$$s > $(OUT)/PL_$${w}_s$$s.log 2>&1 & \
	  done; done; wait; }

dreamer:
	mkdir -p $(OUT)
	cd $(DEV) && for w in breakout food; do \
	  for s in $(SEEDS); do \
	    env $(G) CUDA_VISIBLE_DEVICES=$(CARD) $(DPY) baselines/run_dreamer.py --config $(CFG) --seed $$s --steps 60000 --train-ratio $(TR) \
	      --out $(OUT)/DR_$${w}_s$$s > $(OUT)/DR_$${w}_s$$s.log 2>&1 & \
	  done; wait; \
	done
