# Makefile for experiment vm-00050-summary-windows
#
# Reproduces the runs from the source tagged exp/vm-00050 (dev/ at that tag): vm-00046's NB3 (the text world at twice
# the scale, patches and the full pyramid) as separate transformers, one per piece, each with its own piece's patches
# and either every summary or only those within band_win cells of the piece at each level (task 00058). Needs the GPU
# library (cd ../dev && make libvmcuda.so), the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py),
# ~/venvs/vm-gpu for the scoring, and vm-00046's NB3 and vm-00049's H as the controls. GPU 0
# (CUDA_DEVICE_ORDER=PCI_BUS_ID); all runs of a phase on one card.
# Outputs go to ~/data/research/visual-motor-agent/exp_00050/.
#
# Usage:
#   make phase1   — Q0-Q3 (horizontal quarters: band_n 4, every summary) and W0-W3 (quarters, band_win 1); T_text_B2
#                   with sum_only 0, uni_pyr 4; seed 1, 40k ticks, checkpoints every 10k
#   make analysis — next-letter cross-entropy on the val text of each set (letter_ce.py --bands), by column and line

DEV = ../dev
PY = ./venv/bin/python
GPY = $(HOME)/venvs/vm-gpu/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00050
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
RUN = $(G) $(PY) lifetime.py --config configs/T_text_B2.json --set online.steps=40000 --set online.save_every=10000 \
  --set model.sum_only=0 --set model.uni_pyr=4 --set model.band_n=4 --set model.band_cols=0 --seed 1
CK = --ckpts 10000 20000 30000 40000 --device cuda

.PHONY: all phase1 analysis

all: phase1 analysis

phase1:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for k in 0 1 2 3; do \
	    $(RUN) --set model.band_k=$$k --set name=Q$$k -o $(OUT)/00$$((k + 1))_Q$${k}_s1 > $(OUT)/00$$((k + 1))_Q$${k}_s1.log 2>&1 & \
	    $(RUN) --set model.band_k=$$k --set model.band_win=1 --set name=W$$k \
	      -o $(OUT)/00$$((k + 5))_W$${k}_s1 > $(OUT)/00$$((k + 5))_W$${k}_s1.log 2>&1 & \
	  done; wait; }

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; { \
	  $(G) $(GPY) letter_ce.py --bands $(OUT)/001_Q0_s1 $(OUT)/002_Q1_s1 $(OUT)/003_Q2_s1 $(OUT)/004_Q3_s1 $(CK); \
	  $(G) $(GPY) letter_ce.py --bands $(OUT)/005_W0_s1 $(OUT)/006_W1_s1 $(OUT)/007_W2_s1 $(OUT)/008_W3_s1 $(CK); \
	} | grep -v -i warn | tee $(OUT)/letter_ce.txt
