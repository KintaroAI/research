# Makefile for experiment vm-00049-separate-halves
#
# Reproduces the runs from the source tagged exp/vm-00049 (dev/ at that tag): vm-00046's NB3 (the text world at twice
# the scale, patches and the full pyramid) as two separate transformers, one per half, each with its own half's
# patches and every summary (task 00057). Needs the GPU library (cd ../dev && make libvmcuda.so), the corpus
# (../../llm-fundamentals/dev: prepare_tinyshakespeare.py), ~/venvs/vm-gpu for the scoring, and vm-00046's NB3 and
# vm-00047's VS as the controls. GPU 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID); all four runs on one card.
# Outputs go to ~/data/research/visual-motor-agent/exp_00049/.
#
# Usage:
#   make runs     — V0, V1 (vertical halves: band_n 2, band_cols 1) and H0, H1 (horizontal: band_cols 0); T_text_B2 with
#                   sum_only 0, uni_pyr 4; seed 1, 50k ticks, checkpoints every 10k
#   make analysis — next-letter cross-entropy on the val text of each pair (letter_ce.py --bands), by column and line

DEV = ../dev
PY = ./venv/bin/python
GPY = $(HOME)/venvs/vm-gpu/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00049
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
RUN = $(G) $(PY) lifetime.py --config configs/T_text_B2.json --set online.steps=50000 --set online.save_every=10000 \
  --set model.sum_only=0 --set model.uni_pyr=4 --set model.band_n=2 --seed 1

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  $(RUN) --set model.band_cols=1 --set model.band_k=0 --set name=V0 -o $(OUT)/001_V0_s1 > $(OUT)/001_V0_s1.log 2>&1 & \
	  $(RUN) --set model.band_cols=1 --set model.band_k=1 --set name=V1 -o $(OUT)/002_V1_s1 > $(OUT)/002_V1_s1.log 2>&1 & \
	  $(RUN) --set model.band_cols=0 --set model.band_k=0 --set name=H0 -o $(OUT)/003_H0_s1 > $(OUT)/003_H0_s1.log 2>&1 & \
	  $(RUN) --set model.band_cols=0 --set model.band_k=1 --set name=H1 -o $(OUT)/004_H1_s1 > $(OUT)/004_H1_s1.log 2>&1 & \
	  wait; }

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; { \
	  $(G) $(GPY) letter_ce.py --bands $(OUT)/001_V0_s1 $(OUT)/002_V1_s1 --ckpts 10000 20000 30000 40000 50000 \
	    --device cuda; \
	  $(G) $(GPY) letter_ce.py --bands $(OUT)/003_H0_s1 $(OUT)/004_H1_s1 --ckpts 10000 20000 30000 40000 50000 \
	    --device cuda; } | grep -v -i warn | tee $(OUT)/letter_ce.txt
