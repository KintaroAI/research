# Makefile for experiment vm-00040-hierarchical-attention
#
# Reproduces the runs from the source tagged exp/vm-00040 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service, the CUDA
# PyTorch venv (~/venvs/vm-gpu) for generation, the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py),
# the controls (vm-00036's EB, vm-00030's text runs) and vm-00033's GPT.
# Outputs go to ~/data/research/visual-motor-agent/exp_00040/.
#
# Usage:
#   make runs     — image on the 4090: EW (attn_win 1), EH (attn_win 1, uni_sum 1), ER (uni_sum 1, full attention);
#                   text on the 3090: TW (attn_win 1), TH (attn_win 1, uni_sum 1); 100k ticks, seeds 1-3, all at once
#   make analysis — image: frozen forage tests, dream curve, held-out prediction by row / column; text: val nats per
#                   letter every 10k by line / column
#   make dream    — text generation, seed 1 of TW and TH at their best checkpoints (BEST_TW, BEST_TH), T = 0.8
#   make tables   — the image tables (tables.py), with EB
#   make video    — EH s3 against EB s3, side by side (forage_video.py, the 4-step pair lookahead)

DEV = ../dev
PY = ./venv/bin/python
TPY = $(HOME)/venvs/vm-gpu/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00040
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
WIN = --set model.attn_win=1
SUM = --set model.uni_sum=1
LI = $(PY) lifetime.py --config configs/U_touch_ego.json --set online.steps=100000 --set online.save_every=25000
LT = $(PY) lifetime.py --config configs/T_text_B.json --set online.steps=100000 --set online.save_every=10000
F = --eval-device cuda --eval-tests forage --eval-worlds 16
PR = --ckpts 100000 --device cuda --steps 1000 --instinct 16
VAL = --env-set split=val --env-set tile=true --device cuda
CK = 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
BEST_TW = 00060000
BEST_TH = 00090000
D = $(TPY) reference/text_dream.py --gpt $(DATA)/exp_00033/001_GPTlife30k_s1/params.pt --config configs/T_text_B.json \
    --windows 32 --pages 4 --samples 32 --temp 0.8

.PHONY: all runs analysis dream tables video

all: runs analysis dream tables video

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(LI) $(WIN) --seed $$s --set name=EW -o $(OUT)/00$${s}_EW_s$$s > $(OUT)/00$${s}_EW_s$$s.log 2>&1 & \
	  $(G)=1 $(LI) $(WIN) $(SUM) --seed $$s --set name=EH -o $(OUT)/00$$((s+3))_EH_s$$s \
	    > $(OUT)/00$$((s+3))_EH_s$$s.log 2>&1 & \
	  $(G)=1 $(LI) $(SUM) --seed $$s --set name=ER -o $(OUT)/00$$((s+6))_ER_s$$s > $(OUT)/00$$((s+6))_ER_s$$s.log 2>&1 & \
	  $(G)=0 $(LT) $(WIN) --seed $$s --set name=TW -o $(OUT)/0$$((s+9))_TW_s$$s > $(OUT)/0$$((s+9))_TW_s$$s.log 2>&1 & \
	  $(G)=0 $(LT) $(WIN) $(SUM) --seed $$s --set name=TH -o $(OUT)/0$$((s+12))_TH_s$$s \
	    > $(OUT)/0$$((s+12))_TH_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/00?_E?_s?; do \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-instinct 0; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 1; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode mean; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode pair; \
	  $(G)=1 $(PY) eval_dream.py $$r --ckpts 100000; \
	  $(G)=1 $(PY) eval_predict.py $$r $(PR); \
	done
	cd $(DEV) && for r in $(OUT)/0??_T?_s?; do $(G)=1 $(PY) eval_predict.py $$r --ckpts $(CK) $(VAL) --tag lines; done

dream:
	cd $(DEV) && $(G)=1 $(D) --uni $(OUT)/010_TW_s1/ckpt_$(BEST_TW).bin -o $(OUT)/text_dream_TW_t0.8.json
	cd $(DEV) && $(G)=1 $(D) --uni $(OUT)/013_TH_s1/ckpt_$(BEST_TH).bin -o $(OUT)/text_dream_TH_t0.8.json

tables:
	cd $(DEV) && $(PY) tables.py $(DATA)/exp_00036/00?_EB_s? $(OUT)/00?_EW_s? $(OUT)/00?_EH_s? $(OUT)/00?_ER_s? \
	  --step 100000

video:
	cd $(DEV) && $(G)=1 $(PY) forage_video.py $(OUT)/006_EH_s3 --step 100000 --steps 1000 --worlds 16 --horizon 4 \
	  --decode pair --title "EH seed 3 (3x3 window + row summaries)" --vs $(DATA)/exp_00036/003_EB_s3 \
	  --vs-title "EB seed 3 (full attention)" -o $(OUT)/video_EH_s3_vs_EB_s3
