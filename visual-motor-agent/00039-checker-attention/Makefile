# Makefile for experiment vm-00039-checker-attention
#
# Reproduces the runs from the source tagged exp/vm-00039 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service, the CUDA
# PyTorch venv (~/venvs/vm-gpu) for generation, the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py),
# the controls (vm-00036's EB, vm-00037's ES / ESg / TS / TV, vm-00030's text runs) and vm-00033's GPT.
# Outputs go to ~/data/research/visual-motor-agent/exp_00039/.
#
# Usage:
#   make runs     — EC (checker, move shared) and TC (text, checker) on the 4090, ECg (EC + the move's predictor in
#                   both halves) on the 3090; 100k ticks, seeds 1-3, all nine at once
#   make analysis — image: frozen forage tests, dream curve, held-out prediction by row / column; text: val nats per
#                   letter every 10k by line / column
#   make dream    — text generation, seed 1 at its best checkpoint (BEST, 90000), T = 0.8
#   make tables   — the image tables (tables.py), with EB, ES and ESg
#   make video    — EC s3 and ECg s3 against EB s3, side by side (forage_video.py, the 4-step pair lookahead)

DEV = ../dev
PY = ./venv/bin/python
TPY = $(HOME)/venvs/vm-gpu/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00039
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
CHECK = --set model.attn_split=2 --set model.attn_checker=1
LI = $(PY) lifetime.py --config configs/U_touch_ego.json --set online.steps=100000 --set online.save_every=25000 $(CHECK)
LT = $(PY) lifetime.py --config configs/T_text_B.json --set online.steps=100000 --set online.save_every=10000 $(CHECK)
F = --eval-device cuda --eval-tests forage --eval-worlds 16
PR = --ckpts 100000 --device cuda --steps 1000 --instinct 16
VAL = --env-set split=val --env-set tile=true --device cuda
CK = 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
BEST = 90000
D = $(TPY) reference/text_dream.py --gpt $(DATA)/exp_00033/001_GPTlife30k_s1/params.pt --config configs/T_text_B.json \
    --windows 32 --pages 4 --samples 32 --temp 0.8

.PHONY: all runs analysis dream tables video

all: runs analysis dream tables video

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(LI) --seed $$s --set name=EC -o $(OUT)/00$${s}_EC_s$$s > $(OUT)/00$${s}_EC_s$$s.log 2>&1 & \
	  $(G)=0 $(LI) --set model.attn_pred_global=1 --seed $$s --set name=ECg -o $(OUT)/00$$((s+3))_ECg_s$$s \
	    > $(OUT)/00$$((s+3))_ECg_s$$s.log 2>&1 & \
	  $(G)=1 $(LT) --seed $$s --set name=TC -o $(OUT)/00$$((s+6))_TC_s$$s > $(OUT)/00$$((s+6))_TC_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && for r in $(OUT)/00?_EC*_s?; do \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-instinct 0; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 1; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode mean; \
	  $(G)=1 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode pair; \
	  $(G)=1 $(PY) eval_dream.py $$r --ckpts 100000; \
	  $(G)=1 $(PY) eval_predict.py $$r $(PR); \
	done
	cd $(DEV) && for r in $(OUT)/00?_TC_s?; do $(G)=1 $(PY) eval_predict.py $$r --ckpts $(CK) $(VAL) --tag lines; done

dream:
	cd $(DEV) && $(G)=1 $(D) --uni $(OUT)/007_TC_s1/ckpt_000$(BEST).bin -o $(OUT)/text_dream_TC_t0.8.json

tables:
	cd $(DEV) && $(PY) tables.py $(DATA)/exp_00036/00?_EB_s? $(DATA)/exp_00037/00?_ES_s? $(DATA)/exp_00037/00?_ESg_s? \
	  $(OUT)/00?_EC_s? $(OUT)/00?_ECg_s? --step 100000

video:
	cd $(DEV) && $(G)=1 $(PY) forage_video.py $(OUT)/003_EC_s3 --step 100000 --steps 1000 --worlds 16 --horizon 4 \
	  --decode pair --title "EC seed 3 (checker, move shared)" --vs $(DATA)/exp_00036/003_EB_s3 \
	  --vs-title "EB seed 3 (full attention)" -o $(OUT)/video_EC_s3_vs_EB_s3
	cd $(DEV) && $(G)=1 $(PY) forage_video.py $(OUT)/006_ECg_s3 --step 100000 --steps 1000 --worlds 16 --horizon 4 \
	  --decode pair --title "ECg seed 3 (checker, move predictor shared)" --vs $(DATA)/exp_00036/003_EB_s3 \
	  --vs-title "EB seed 3 (full attention)" -o $(OUT)/video_ECg_s3_vs_EB_s3
