# Makefile for experiment vm-00083-food-hides-gradually
#
# Reproduces the runs of task 00094: the default model (4 quadrant transformers, K4, log summaries), the food always seen for 20k ticks, then
# hidden after fewer and fewer frames (online.env_sched on env.food_show). G4: 64, 32, 16, 8 for 4k ticks each, then 4 from 36k; G1: as G4, then
# 2 at 40k and 1 from 44k; J4: 4 at once from 20k. Six seeds, 60k ticks. References, not rerun: vm-00073's K4 (always seen), vm-00081's H4
# and H1 (hidden from birth) and K4's frozen aim at food_show 4 and 1 (vm-00081's aim_k4.jsonl).
# Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00083/.
#
# Usage:
#   make runs — G4, G1, J4, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, 18 lives at once
#   make frozen — eval_hidden.py aim at food_show 4 (RTX 4090) and 1 (RTX 3090), without MPS
#   make show — tables

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00083
REF = $(HOME)/data/research/visual-motor-agent/exp_00081
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4 --set online.steps=60000
STEPS4 = [20001,64],[24001,32],[28001,16],[32001,8],[36001,4]
S_G4 = {"food_show":[$(STEPS4)]}
S_G1 = {"food_show":[$(STEPS4),[40001,2],[44001,1]]}
S_J4 = {"food_show":[[20001,4]]}

.PHONY: all runs frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for a in G4 G1 J4; do for s in 1 2 3 4 5 6; do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  case $$a in G4) sc='$(S_G4)';; G1) sc='$(S_G1)';; J4) sc='$(S_J4)';; esac; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set "online.env_sched=$$sc" --seed $$s --set name=$$a -o $(OUT)/$${a}_s$$s > $(OUT)/$${a}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  r="$$(for a in G4 G1 J4; do for s in 1 2 3 4 5 6; do echo $(OUT)/$${a}_s$$s; done; done | tr "\n" " ")"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hidden.py aim $$r --show 4 > $(OUT)/aim4.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_hidden.py aim $$r --show 1 > $(OUT)/aim1.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hidden.py show $(OUT)/aim4.jsonl $(OUT)/aim1.jsonl $(REF)/aim_k4.jsonl
