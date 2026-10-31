# Makefile for experiment vm-00054-age-key-ring
#
# Reproduces the runs from the source tagged exp/vm-00054 (dev/ at that tag): the best models so far with the age key
# (model.age_key: position read at attention by age) and acting on a key and value ring (online.ring; task 00062). Needs
# the GPU library (cd ../dev && make libvmcuda.so), the corpus for the text arm, ~/venvs/vm-gpu for its scoring. GPU 0
# (CUDA_DEVICE_ORDER=PCI_BUS_ID); all thirteen lifetimes on one card.
# Outputs go to ~/data/research/visual-motor-agent/exp_00054/.
#
# Usage:
#   make runs     — seeds 1-3, 60k ticks: QNB (vm-00052's QN, with the age key), QNA (with the ring as well), QCA (QCN:
#                   council, no overlap, with the age key and the ring); seed 1, 40k ticks: T0-T3 (vm-00050's text
#                   quarters Q, with the age key)
#   make analysis — next-letter cross-entropy of the text quarters

DEV = ../dev
PY = ./venv/bin/python
GPY = $(HOME)/venvs/vm-gpu/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00054
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
RUN = $(G) $(PY) lifetime.py --config configs/U_touch_ego_c2.json --set online.steps=60000 \
  --set online.save_every=10000 --set model.uni_pyr=3 --set model.sum_thumb=1 --set online.pieces=1 \
  --set model.band_n=4 --set model.band_cols=2 --set env.no_overlap=true --set model.age_key=1
TXT = $(G) $(PY) lifetime.py --config configs/T_text_B2.json --set online.steps=40000 --set online.save_every=10000 \
  --set model.sum_only=0 --set model.uni_pyr=4 --set model.band_n=4 --set model.band_cols=0 --set model.age_key=1 --seed 1

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3; do \
	    $(RUN) --set online.piece_rule=gain --seed $$s --set name=QNB -o $(OUT)/00$${s}_QNB_s$$s > $(OUT)/00$${s}_QNB_s$$s.log 2>&1 & \
	    $(RUN) --set online.piece_rule=gain --set online.ring=1 --seed $$s --set name=QNA \
	      -o $(OUT)/00$$((s + 3))_QNA_s$$s > $(OUT)/00$$((s + 3))_QNA_s$$s.log 2>&1 & \
	    $(RUN) --set online.piece_rule=council --set online.ring=1 --seed $$s --set name=QCA \
	      -o $(OUT)/00$$((s + 6))_QCA_s$$s > $(OUT)/00$$((s + 6))_QCA_s$$s.log 2>&1 & \
	  done; \
	  for k in 0 1 2 3; do \
	    $(TXT) --set model.band_k=$$k --set name=T$$k -o $(OUT)/01$${k}_T$${k}_s1 > $(OUT)/01$${k}_T$${k}_s1.log 2>&1 & \
	  done; wait; }

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; \
	  $(G) $(GPY) letter_ce.py --bands $(OUT)/010_T0_s1 $(OUT)/011_T1_s1 $(OUT)/012_T2_s1 $(OUT)/013_T3_s1 \
	    --ckpts 10000 20000 30000 40000 --device cuda | grep -v -i warn | tee $(OUT)/letter_ce.txt
