# Makefile for experiment vm-00046-summary-only-text
#
# Reproduces the runs from the source tagged exp/vm-00046 (dev/ at that tag): the text world at twice the scale with
# tokens from the B1 summaries alone (task 00054, model.sum_only), and with the usual patch tokens. Needs the GPU
# library (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service
# for the lifetimes, and the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py).
# Outputs go to ~/data/research/visual-motor-agent/exp_00046/.
#
# Usage:
#   make runs     — NB (T_text_B), NB2 (T_text_B2: B1 thumbnails only), NB3 (T_text_B2 with patches, uni_pyr 4),
#                   NB3a (with patches, uni_pyr 1); seed 1, 100k ticks, checkpoints every 10k
#   make analysis — next-letter cross-entropy on the val text at 20k, 30k, 40k, 100k (letter_ce.py)

DEV = ../dev
PY = ./venv/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00046
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --set online.steps=100000 --set online.save_every=10000 --seed 1

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  $(G)=1 $(L) --config configs/T_text_B.json --set name=NB -o $(OUT)/001_NB_s1 > $(OUT)/001_NB_s1.log 2>&1 & \
	  $(G)=0 $(L) --config configs/T_text_B2.json --set name=NB2 -o $(OUT)/002_NB2_s1 > $(OUT)/002_NB2_s1.log 2>&1 & \
	  wait; }
	cd $(DEV) && { \
	  $(G)=1 $(L) --config configs/T_text_B2.json --set model.sum_only=0 --set model.uni_pyr=4 --set name=NB3 \
	    -o $(OUT)/003_NB3_s1 > $(OUT)/003_NB3_s1.log 2>&1 & \
	  $(G)=0 $(L) --config configs/T_text_B2.json --set model.sum_only=0 --set model.uni_pyr=1 --set name=NB3a \
	    -o $(OUT)/004_NB3a_s1 > $(OUT)/004_NB3a_s1.log 2>&1 & \
	  wait; }

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; $(G)=0 $(PY) letter_ce.py $(OUT)/001_NB_s1 $(OUT)/002_NB2_s1 \
	  $(OUT)/004_NB3a_s1 $(OUT)/003_NB3_s1 --ckpts 20000 30000 40000 100000 | tee $(OUT)/letter_ce.txt
