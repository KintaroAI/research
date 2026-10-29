# Makefile for experiment vm-00042-summary-thumbnails
#
# Reproduces the runs from the source tagged exp/vm-00042 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service for the
# lifetimes (not for the analysis: a 4-step frozen test takes ~8 GB, run at most two per GPU), the CUDA PyTorch venv
# (~/venvs/vm-gpu) for generation, the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py), the controls
# (vm-00041's EP / TP, vm-00040's ER, vm-00036's EB) and vm-00033's GPT.
# Outputs go to ~/data/research/visual-motor-agent/exp_00042/.
#
# Usage:
#   make runs     — image on the 4090: EPT (attn_win 1, uni_pyr 3, sum_thumb 1), EFT (uni_pyr 3, sum_thumb 1, full
#                   attention), EF (uni_pyr 3, full attention); text on the 3090: TPT (attn_win 1, uni_pyr 3,
#                   sum_thumb 1); 100k ticks, seeds 1-3, all twelve at once
#   make analysis — image: frozen forage tests, dream curve, held-out prediction; text: val nats per letter every 10k
#                   (two jobs per GPU, without the multi-process service)
#   make dream    — text generation, seed 1 of TPT at its best checkpoint (BEST_TPT), T = 0.8
#   make tables   — the image tables (tables.py), with EB, ER and EP
#   make video    — EFT s3 against EB s3, side by side (forage_video.py, the 4-step pair lookahead)

DEV = ../dev
PY = ./venv/bin/python
TPY = $(HOME)/venvs/vm-gpu/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00042
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
PYR = --set model.uni_pyr=3
WIN = --set model.attn_win=1
TH = --set model.sum_thumb=1
LI = $(PY) lifetime.py --config configs/U_touch_ego.json --set online.steps=100000 --set online.save_every=25000 $(PYR)
LT = $(PY) lifetime.py --config configs/T_text_B.json --set online.steps=100000 --set online.save_every=10000 $(PYR)
F = --eval-device cuda --eval-tests forage --eval-worlds 16
PR = --ckpts 100000 --device cuda --steps 1000 --instinct 16
VAL = --env-set split=val --env-set tile=true --device cuda
CK = 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
BEST_TPT = 00100000
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
	  $(G)=1 $(LI) $(WIN) $(TH) --seed $$s --set name=EPT -o $(OUT)/00$${s}_EPT_s$$s > $(OUT)/00$${s}_EPT_s$$s.log 2>&1 & \
	  $(G)=1 $(LI) $(TH) --seed $$s --set name=EFT -o $(OUT)/00$$((s+3))_EFT_s$$s > $(OUT)/00$$((s+3))_EFT_s$$s.log 2>&1 & \
	  $(G)=1 $(LI) --seed $$s --set name=EF -o $(OUT)/00$$((s+6))_EF_s$$s > $(OUT)/00$$((s+6))_EF_s$$s.log 2>&1 & \
	  $(G)=0 $(LT) $(WIN) $(TH) --seed $$s --set name=TPT -o $(OUT)/0$$((s+9))_TPT_s$$s > $(OUT)/0$$((s+9))_TPT_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; \
	  (for r in $(OUT)/001_EPT_s1 $(OUT)/004_EFT_s1 $(OUT)/007_EF_s1; do export $(G)=1; $(EV); done) & \
	  (for r in $(OUT)/002_EPT_s2 $(OUT)/005_EFT_s2 $(OUT)/008_EF_s2; do export $(G)=1; $(EV); done) & \
	  (for r in $(OUT)/003_EPT_s3 $(OUT)/006_EFT_s3 $(OUT)/009_EF_s3; do export $(G)=0; $(EV); done) & \
	  (for r in $(OUT)/0??_TPT_s?; do $(G)=0 $(PY) eval_predict.py $$r --ckpts $(CK) $(VAL) --tag lines; done) & \
	  wait

dream:
	cd $(DEV) && $(G)=1 $(D) --uni $(OUT)/010_TPT_s1/ckpt_$(BEST_TPT).bin -o $(OUT)/text_dream_TPT_t0.8.json

tables:
	cd $(DEV) && $(PY) tables.py $(DATA)/exp_00036/00?_EB_s? $(DATA)/exp_00040/00?_ER_s? $(DATA)/exp_00041/00?_EP_s? \
	  $(OUT)/00?_EPT_s? $(OUT)/00?_EFT_s? $(OUT)/00?_EF_s? --step 100000

video:
	cd $(DEV) && $(G)=1 $(PY) forage_video.py $(OUT)/006_EFT_s3 --step 100000 --steps 1000 --worlds 16 --horizon 4 \
	  --decode pair --title "EFT seed 3 (full attention, summary pyramid with thumbnails)" \
	  --vs $(DATA)/exp_00036/003_EB_s3 --vs-title "EB seed 3 (full attention)" -o $(OUT)/video_EFT_s3_vs_EB_s3
