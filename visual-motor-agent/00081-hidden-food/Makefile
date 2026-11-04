# Makefile for experiment vm-00081-hidden-food
#
# Reproduces the runs of task 00092: the default model (4 quadrant transformers, K4, log summaries), the food drawn only in its first
# env.food_show frames after it appears: H16, H4, H1, six seeds. The control is vm-00073's K4 (always seen), not rerun.
# Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00081/.
#
# Usage:
#   make runs — H16, H4, H1, seeds 1-3 on the RTX 4090 and 4-6 on the RTX 3090, 18 lives at once
#   make copies — the conditional arms: vm-00080's D0 and DX with food_show 4 (C0, CX), seeds 1-4, 1-2 on the 4090 and 3-4 on the 3090
#   make frozen — eval_hidden.py aim (each arm at its own food_show; K4 at 16, 4 and 1) and walk, without MPS
#   make show — tables

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00081
CTRL = $(HOME)/data/research/visual-motor-agent/exp_00073
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4
BIG = --set model.img=128 --set model.uni_pyr=4 --set model.band_n=16

.PHONY: all runs copies frozen show

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for v in 16 4 1; do for s in 1 2 3 4 5 6; do \
	  if [ $$s -le 3 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set env.food_show=$$v --seed $$s --set name=H$$v -o $(OUT)/H$${v}_s$$s > $(OUT)/H$${v}_s$$s.log 2>&1 & \
	done; done; wait; }

copies:
	mkdir -p $(OUT)
	cd $(DEV) && { for a in C0:0,0,0,0 CX:0,1,4,16; do for s in 1 2 3 4; do \
	  if [ $$s -le 2 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) $(BIG) --set "env.delays=[$${a#*:}]" --set env.food_show=4 --seed $$s --set name=$${a%:*} -o $(OUT)/$${a%:*}_s$$s > $(OUT)/$${a%:*}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  h="$$(for v in 16 4 1; do for s in 1 2 3 4 5 6; do echo $(OUT)/H$${v}_s$$s; done; done)"; k="$$(for s in 1 2 3 4 5 6; do echo $(CTRL)/K4_s$$s; done | tr "\n" " ")"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_hidden.py aim $$h > $(OUT)/aim.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 sh -c "for v in 16 4 1; do $(PY) eval_hidden.py aim $$k --show \$$v; done" > $(OUT)/aim_k4.jsonl & \
	  $(PY) eval_hidden.py walk configs/U_default.json --shows 16 4 1 > $(OUT)/walk.jsonl & \
	  wait; }

show:
	cd $(DEV) && $(PY) eval_hidden.py show $(OUT)/walk.jsonl $(OUT)/aim.jsonl $(OUT)/aim_k4.jsonl
