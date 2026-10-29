# Makefile for experiment vm-00041-pyramid-attention
#
# Reproduces the runs from the source tagged exp/vm-00041 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service for the
# lifetimes (not for the analysis: a 4-step frozen test takes ~8 GB, run at most two per GPU), the CUDA PyTorch venv
# (~/venvs/vm-gpu) for generation, the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py), the controls
# (vm-00040's EH / ER / TH / TW, vm-00036's EB, vm-00030's text runs) and vm-00033's GPT.
# Outputs go to ~/data/research/visual-motor-agent/exp_00041/.
#
# Usage:
#   make runs     — EP (image, attn_win 1, uni_pyr 3) on the 4090, TP (text, the same) on the 3090; 100k ticks,
#                   seeds 1-3, all six at once
#   make analysis — image: frozen forage tests, dream curve, held-out prediction; text: val nats per letter every 10k
#   make dream    — text generation, seed 1 of TP at its best checkpoint (BEST_TP), T = 0.8
#   make tables   — the image tables (tables.py), with EB, EH and ER
#   make video    — EP s3 against EB s3, side by side (forage_video.py, the 4-step pair lookahead)

DEV = ../dev
PY = ./venv/bin/python
TPY = $(HOME)/venvs/vm-gpu/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00041
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
PYR = --set model.attn_win=1 --set model.uni_pyr=3
LI = $(PY) lifetime.py --config configs/U_touch_ego.json --set online.steps=100000 --set online.save_every=25000 $(PYR)
LT = $(PY) lifetime.py --config configs/T_text_B.json --set online.steps=100000 --set online.save_every=10000 $(PYR)
F = --eval-device cuda --eval-tests forage --eval-worlds 16
PR = --ckpts 100000 --device cuda --steps 1000 --instinct 16
VAL = --env-set split=val --env-set tile=true --device cuda
CK = 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
BEST_TP = 00090000
D = $(TPY) reference/text_dream.py --gpt $(DATA)/exp_00033/001_GPTlife30k_s1/params.pt --config configs/T_text_B.json \
    --windows 32 --pages 4 --samples 32 --temp 0.8
EV = for t in "--eval-instinct 0" "--eval-horizon 1" "--eval-horizon 4 --eval-decode mean" \
       "--eval-horizon 4 --eval-decode pair"; do $(PY) online.py --eval $$r --ckpts 100000 $(F) $$t; done; \
     $(PY) eval_dream.py $$r --ckpts 100000; $(PY) eval_predict.py $$r $(PR)

.PHONY: all runs analysis dream tables video

all: runs analysis dream tables video

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(LI) --seed $$s --set name=EP -o $(OUT)/00$${s}_EP_s$$s > $(OUT)/00$${s}_EP_s$$s.log 2>&1 & \
	  $(G)=0 $(LT) --seed $$s --set name=TP -o $(OUT)/00$$((s+3))_TP_s$$s > $(OUT)/00$$((s+3))_TP_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; \
	  (for r in $(OUT)/001_EP_s1 $(OUT)/003_EP_s3; do export $(G)=1; $(EV); done) & \
	  (for r in $(OUT)/002_EP_s2; do export $(G)=0; $(EV); done; \
	   for r in $(OUT)/00?_TP_s?; do $(G)=0 $(PY) eval_predict.py $$r --ckpts $(CK) $(VAL) --tag lines; done) & \
	  wait

dream:
	cd $(DEV) && $(G)=1 $(D) --uni $(OUT)/004_TP_s1/ckpt_$(BEST_TP).bin -o $(OUT)/text_dream_TP_t0.8.json

tables:
	cd $(DEV) && $(PY) tables.py $(DATA)/exp_00036/00?_EB_s? $(DATA)/exp_00040/00?_EH_s? $(DATA)/exp_00040/00?_ER_s? \
	  $(OUT)/00?_EP_s? --step 100000

video:
	cd $(DEV) && $(G)=1 $(PY) forage_video.py $(OUT)/003_EP_s3 --step 100000 --steps 1000 --worlds 16 --horizon 4 \
	  --decode pair --title "EP seed 3 (3x3 window + summary pyramid)" --vs $(DATA)/exp_00036/003_EB_s3 \
	  --vs-title "EB seed 3 (full attention)" -o $(OUT)/video_EP_s3_vs_EB_s3
