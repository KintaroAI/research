# Makefile for experiment vm-00037-attention-regions
#
# Reproduces the runs from the source tagged exp/vm-00037 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service, the CUDA
# PyTorch venv (~/venvs/vm-gpu) for generation, the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py),
# vm-00036's EB runs and vm-00030's text runs (the full-attention controls) and vm-00033's GPT.
# Outputs go to ~/data/research/visual-motor-agent/exp_00037/.
#
# Usage:
#   make image    — ES (halves, one row of overlap) on the 4090 and ESg (ES + the move's predictor shared) on the 3090,
#                   egocentric touch world, 100k ticks, seeds 1-3
#   make text     — TS (by lines) and TV (by columns), the text world, 100k ticks, seeds 1-3
#   make analysis — image: frozen forage tests, dream curve, held-out prediction; text: val nats per letter every 10k
#                   (the controls rescored by line / column)
#   make dream    — text generation, seed 1 at each arm's best (full 30k, TS 40k, TV 60k), T = 0.8
#   make tables   — the image tables (tables.py)
#   make video    — ES s3 and ESg s3 against EB s3, side by side (forage_video.py, the 4-step pair lookahead)

DEV = ../dev
PY = ./venv/bin/python
TPY = $(HOME)/venvs/vm-gpu/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00037
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
SPLIT = --set model.attn_split=2 --set model.attn_overlap=1
LI = $(PY) lifetime.py --config configs/U_touch_ego.json --set online.steps=100000 --set online.save_every=25000 $(SPLIT)
LT = $(PY) lifetime.py --config configs/T_text_B.json --set online.steps=100000 --set online.save_every=10000 $(SPLIT)
F = --eval-device cuda --eval-tests forage --eval-worlds 16
PR = --ckpts 100000 --device cuda --steps 1000 --instinct 16
VAL = --env-set split=val --env-set tile=true --device cuda
CK = 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
D = $(TPY) reference/text_dream.py --gpt $(DATA)/exp_00033/001_GPTlife30k_s1/params.pt --config configs/T_text_B.json \
    --windows 32 --pages 4 --samples 32 --temp 0.8

.PHONY: all image text analysis dream tables video

all: image text analysis dream tables video

image:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(LI) --seed $$s --set name=ES -o $(OUT)/00$${s}_ES_s$$s > $(OUT)/00$${s}_ES_s$$s.log 2>&1 & \
	  $(G)=0 $(LI) --set model.attn_pred_global=1 --seed $$s --set name=ESg -o $(OUT)/00$$((s+3))_ESg_s$$s \
	    > $(OUT)/00$$((s+3))_ESg_s$$s.log 2>&1 & \
	done; wait

text:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(LT) --seed $$s --set name=TS -o $(OUT)/00$$((s+6))_TS_s$$s > $(OUT)/00$$((s+6))_TS_s$$s.log 2>&1 & \
	done; wait
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(LT) --set model.attn_cols=1 --seed $$s --set name=TV -o $(OUT)/0$$((s+9))_TV_s$$s \
	    > $(OUT)/0$$((s+9))_TV_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/00?_ES*_s?; do \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-instinct 0; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 1; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode mean; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode pair; \
	  $(G)=1 $(PY) eval_dream.py $$r --ckpts 100000; \
	  $(G)=1 $(PY) eval_predict.py $$r $(PR); \
	done
	cd $(DEV) && for s in 1 2 3; do $(G)=1 $(PY) eval_predict.py $(DATA)/exp_00036/00$${s}_EB_s$$s $(PR); done
	cd $(DEV) && for r in $(OUT)/0??_T?_s? $(DATA)/exp_00030/00?_Bib_s?; do \
	  $(G)=1 $(PY) eval_predict.py $$r --ckpts $(CK) $(VAL) --tag lines; \
	done

dream:
	cd $(DEV) && $(G)=1 $(D) --uni $(DATA)/exp_00030/001_Bib_s1/ckpt_00030000.bin -o $(OUT)/text_dream_base_t0.8.json
	cd $(DEV) && $(G)=1 $(D) --uni $(OUT)/007_TS_s1/ckpt_00040000.bin -o $(OUT)/text_dream_TS_t0.8.json
	cd $(DEV) && $(G)=1 $(D) --uni $(OUT)/010_TV_s1/ckpt_00060000.bin -o $(OUT)/text_dream_TV_t0.8.json

tables:
	cd $(DEV) && $(PY) tables.py $(DATA)/exp_00036/00?_EB_s? $(OUT)/00?_ES_s? $(OUT)/00?_ESg_s? --step 100000

video:
	cd $(DEV) && $(G)=1 $(PY) forage_video.py $(OUT)/003_ES_s3 --step 100000 --steps 1000 --worlds 16 --horizon 4 \
	  --decode pair --title "ES seed 3 (split halves)" --vs $(DATA)/exp_00036/003_EB_s3 \
	  --vs-title "EB seed 3 (full attention)" -o $(OUT)/video_ES_s3_vs_EB_s3
	cd $(DEV) && $(G)=0 $(PY) forage_video.py $(OUT)/006_ESg_s3 --step 100000 --steps 1000 --worlds 16 --horizon 4 \
	  --decode pair --title "ESg seed 3 (split, move predictor shared)" --vs $(DATA)/exp_00036/003_EB_s3 \
	  --vs-title "EB seed 3 (full attention)" -o $(OUT)/video_ESg_s3_vs_EB_s3
