# Makefile for experiment vm-00044-checker-cells
#
# Reproduces the runs from the source tagged exp/vm-00044 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service for the
# lifetimes (not for the analysis: at most two frozen tests per GPU), and the circle and square worlds' runs
# (vm-00042, vm-00043) to compare with.
# Outputs go to ~/data/research/visual-motor-agent/exp_00044/.
#
# Usage:
#   make runs     — the 2-pixel-checker world (configs/U_touch_ego_c2.json), 100k ticks, seeds 1-3: CB (full attention)
#                   and CF (uni_pyr 3, full attention) on the 3090, CFT (uni_pyr 3, sum_thumb 1, full attention) and
#                   CPT (attn_win 1, uni_pyr 3, sum_thumb 1) on the 4090; all twelve at once
#   make analysis — frozen forage tests, dream curve, held-out prediction (two jobs per GPU)
#   make tables   — the tables (tables.py), with the circle world (EB, EF, EFT, EPT) and the square world (vm-00043)
#   make video    — CFT s3 against CB s3, side by side (forage_video.py, the 4-step pair lookahead)

DEV = ../dev
PY = ./venv/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00044
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
PYR = --set model.uni_pyr=3
WIN = --set model.attn_win=1
TH = --set model.sum_thumb=1
LI = $(PY) lifetime.py --config configs/U_touch_ego_c2.json --set online.steps=100000 --set online.save_every=25000
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
	  $(G)=0 $(LI) --seed $$s --set name=CB -o $(OUT)/00$${s}_CB_s$$s > $(OUT)/00$${s}_CB_s$$s.log 2>&1 & \
	  $(G)=0 $(LI) $(PYR) --seed $$s --set name=CF -o $(OUT)/00$$((s+3))_CF_s$$s > $(OUT)/00$$((s+3))_CF_s$$s.log 2>&1 & \
	  $(G)=1 $(LI) $(PYR) $(TH) --seed $$s --set name=CFT -o $(OUT)/00$$((s+6))_CFT_s$$s \
	    > $(OUT)/00$$((s+6))_CFT_s$$s.log 2>&1 & \
	  $(G)=1 $(LI) $(PYR) $(WIN) $(TH) --seed $$s --set name=CPT -o $(OUT)/0$$((s+9))_CPT_s$$s \
	    > $(OUT)/0$$((s+9))_CPT_s$$s.log 2>&1 & \
	done; wait

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; \
	  (for r in $(OUT)/001_CB_s1 $(OUT)/004_CF_s1 $(OUT)/007_CFT_s1; do export $(G)=1; $(EV); done) & \
	  (for r in $(OUT)/002_CB_s2 $(OUT)/005_CF_s2 $(OUT)/008_CFT_s2; do export $(G)=1; $(EV); done) & \
	  (for r in $(OUT)/003_CB_s3 $(OUT)/006_CF_s3 $(OUT)/009_CFT_s3; do export $(G)=0; $(EV); done) & \
	  (for r in $(OUT)/0??_CPT_s?; do export $(G)=0; $(EV); done) & \
	  wait

tables:
	cd $(DEV) && $(PY) tables.py $(DATA)/exp_00036/00?_EB_s? $(DATA)/exp_00042/00?_EF_s? $(DATA)/exp_00042/00?_EFT_s? \
	  $(DATA)/exp_00042/00?_EPT_s? $(OUT)/00?_CB_s? $(OUT)/00?_CF_s? $(OUT)/00?_CFT_s? $(OUT)/0??_CPT_s? --step 100000

video:
	cd $(DEV) && $(G)=1 $(PY) forage_video.py $(OUT)/009_CFT_s3 --step 100000 --steps 1000 --worlds 16 --horizon 4 \
	  --decode pair --title "CFT seed 3 (full attention, pyramid with thumbnails)" --vs $(OUT)/003_CB_s3 \
	  --vs-title "CB seed 3 (full attention)" -o $(OUT)/video_CFT_s3_vs_CB_s3
