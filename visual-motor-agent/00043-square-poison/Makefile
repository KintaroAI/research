# Makefile for experiment vm-00043-square-poison
#
# Reproduces the runs from the source tagged exp/vm-00043 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service for the
# lifetimes (not for the analysis: at most two frozen tests per GPU), and vm-00042's circle-world runs to compare with.
# Outputs go to ~/data/research/visual-motor-agent/exp_00043/.
#
# Usage:
#   make runs     — the square-poison world (configs/U_touch_ego_sq.json), 100k ticks, seeds 1-3: SB (full attention)
#                   and SF (uni_pyr 3, full attention) on the 3090, SFT (uni_pyr 3, sum_thumb 1, full attention) and
#                   SPT (attn_win 1, uni_pyr 3, sum_thumb 1) on the 4090; all twelve at once
#   make analysis — frozen forage tests, dream curve, held-out prediction (two jobs per GPU)
#   make tables   — the tables (tables.py), with the circle world's EB, EF, EFT, EPT
#   make video    — SFT s3 against SB s3, side by side (forage_video.py, the 4-step pair lookahead)

DEV = ../dev
PY = ./venv/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00043
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
PYR = --set model.uni_pyr=3
WIN = --set model.attn_win=1
TH = --set model.sum_thumb=1
LI = $(PY) lifetime.py --config configs/U_touch_ego_sq.json --set online.steps=100000 --set online.save_every=25000
F = --eval-device cuda --eval-tests forage --eval-worlds 16
PR = --ckpts 100000 --device cuda --steps 1000 --instinct 16
EV = for t in "--eval-instinct 0" "--eval-horizon 1" "--eval-horizon 4 --eval-decode mean" \
       "--eval-horizon 4 --eval-decode pair"; do $(PY) online.py --eval $$r --ckpts 100000 $(F) $$t; done; \
     $(PY) eval_dream.py $$r --ckpts 100000; $(PY) eval_predict.py $$r $(PR)

.PHONY: all runs analysis tables video

all: runs analysis tables video

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=0 $(LI) --seed $$s --set name=SB -o $(OUT)/00$${s}_SB_s$$s > $(OUT)/00$${s}_SB_s$$s.log 2>&1 & \
	  $(G)=0 $(LI) $(PYR) --seed $$s --set name=SF -o $(OUT)/00$$((s+3))_SF_s$$s > $(OUT)/00$$((s+3))_SF_s$$s.log 2>&1 & \
	  $(G)=1 $(LI) $(PYR) $(TH) --seed $$s --set name=SFT -o $(OUT)/00$$((s+6))_SFT_s$$s \
	    > $(OUT)/00$$((s+6))_SFT_s$$s.log 2>&1 & \
	  $(G)=1 $(LI) $(PYR) $(WIN) $(TH) --seed $$s --set name=SPT -o $(OUT)/0$$((s+9))_SPT_s$$s \
	    > $(OUT)/0$$((s+9))_SPT_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; \
	  (for r in $(OUT)/001_SB_s1 $(OUT)/004_SF_s1 $(OUT)/007_SFT_s1; do export $(G)=1; $(EV); done) & \
	  (for r in $(OUT)/002_SB_s2 $(OUT)/005_SF_s2 $(OUT)/008_SFT_s2; do export $(G)=1; $(EV); done) & \
	  (for r in $(OUT)/003_SB_s3 $(OUT)/006_SF_s3 $(OUT)/009_SFT_s3; do export $(G)=0; $(EV); done) & \
	  (for r in $(OUT)/0??_SPT_s?; do export $(G)=0; $(EV); done) & \
	  wait

tables:
	cd $(DEV) && $(PY) tables.py $(DATA)/exp_00036/00?_EB_s? $(DATA)/exp_00042/00?_EF_s? $(DATA)/exp_00042/00?_EFT_s? \
	  $(DATA)/exp_00042/00?_EPT_s? $(OUT)/00?_SB_s? $(OUT)/00?_SF_s? $(OUT)/00?_SFT_s? $(OUT)/0??_SPT_s? --step 100000

video:
	cd $(DEV) && $(G)=1 $(PY) forage_video.py $(OUT)/009_SFT_s3 --step 100000 --steps 1000 --worlds 16 --horizon 4 \
	  --decode pair --title "SFT seed 3 (full attention, pyramid with thumbnails)" --vs $(OUT)/003_SB_s3 \
	  --vs-title "SB seed 3 (full attention)" -o $(OUT)/video_SFT_s3_vs_SB_s3
