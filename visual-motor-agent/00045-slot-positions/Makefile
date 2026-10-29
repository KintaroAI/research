# Makefile for experiment vm-00045-slot-positions
#
# Reproduces the runs from the source tagged exp/vm-00045 (dev/ at that tag): the position codes at birth placed by the
# slot layout (task 00052). Needs the GPU library (cd ../dev && make libvmcuda.so), both GPUs
# (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service for the lifetimes (not for the analysis: a 4-step
# frozen test takes ~10 GB, run at most two per GPU), the CUDA PyTorch venv (~/venvs/vm-gpu) for generation, the corpus
# (../../llm-fundamentals/dev: prepare_tinyshakespeare.py), the controls (vm-00042's EF / EFT, vm-00041's TP,
# vm-00036's EB) and vm-00033's GPT.
# Outputs go to ~/data/research/visual-motor-agent/exp_00045/.
#
# Usage:
#   make runs     — image on the 4090: XF (uni_pyr 3, full attention), XFT (the same with sum_thumb 1); text on the
#                   3090: XTP (attn_win 1, uni_pyr 3); 100k ticks, seeds 1-3, all nine at once
#   make analysis — image: frozen forage tests, dream curve, held-out prediction; text: val nats per letter every 10k
#   make dream    — text generation, seed 1 of XTP at its best checkpoint (BEST_XTP), T = 0.8
#   make tables   — the image tables (tables.py), with EB, EF and EFT

DEV = ../dev
PY = ./venv/bin/python
TPY = $(HOME)/venvs/vm-gpu/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00045
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
PYR = --set model.uni_pyr=3
LI = $(PY) lifetime.py --config configs/U_touch_ego.json --set online.steps=100000 --set online.save_every=25000 $(PYR)
LT = $(PY) lifetime.py --config configs/T_text_B.json --set online.steps=100000 --set online.save_every=10000 $(PYR)
F = --eval-device cuda --eval-tests forage --eval-worlds 16
PR = --ckpts 100000 --device cuda --steps 1000 --instinct 16
VAL = --env-set split=val --env-set tile=true --device cuda
CK = 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
BEST_XTP = 00100000
D = $(TPY) reference/text_dream.py --gpt $(DATA)/exp_00033/001_GPTlife30k_s1/params.pt --config configs/T_text_B.json \
    --windows 32 --pages 4 --samples 32 --temp 0.8
EV = for t in "--eval-instinct 0" "--eval-horizon 1" "--eval-horizon 4 --eval-decode mean" \
       "--eval-horizon 4 --eval-decode pair"; do $(PY) online.py --eval $$r --ckpts 100000 $(F) $$t; done; \
     $(PY) eval_dream.py $$r --ckpts 100000; $(PY) eval_predict.py $$r $(PR)

.PHONY: all runs analysis dream tables

all: runs analysis dream tables

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(LI) --seed $$s --set name=XF -o $(OUT)/00$${s}_XF_s$$s > $(OUT)/00$${s}_XF_s$$s.log 2>&1 & \
	  $(G)=1 $(LI) --set model.sum_thumb=1 --seed $$s --set name=XFT -o $(OUT)/00$$((s+3))_XFT_s$$s > $(OUT)/00$$((s+3))_XFT_s$$s.log 2>&1 & \
	  $(G)=0 $(LT) --set model.attn_win=1 --seed $$s --set name=XTP -o $(OUT)/00$$((s+6))_XTP_s$$s > $(OUT)/00$$((s+6))_XTP_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; \
	  (for r in $(OUT)/00?_XF_s?; do export $(G)=1; $(EV); done) & \
	  (for r in $(OUT)/00?_XFT_s?; do export $(G)=0; $(EV); done) & \
	  wait; \
	  for r in $(OUT)/00?_XTP_s?; do $(G)=0 $(PY) eval_predict.py $$r --ckpts $(CK) $(VAL) --tag lines; done

dream:
	cd $(DEV) && $(G)=1 $(D) --uni $(OUT)/007_XTP_s1/ckpt_$(BEST_XTP).bin -o $(OUT)/text_dream_XTP_t0.8.json

tables:
	cd $(DEV) && $(PY) tables.py $(DATA)/exp_00036/00?_EB_s? $(DATA)/exp_00042/00?_EF_s? $(OUT)/00?_XF_s? \
	  $(DATA)/exp_00042/00?_EFT_s? $(OUT)/00?_XFT_s? --step 100000
