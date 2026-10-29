# Makefile for experiment vm-00047-pyramid-halves
#
# Reproduces the run from the source tagged exp/vm-00047 (dev/ at that tag): vm-00046's NB3 (the text world at twice
# the scale, patches and the full pyramid) split into two vertical halves that share the summaries (task 00055). Needs
# the GPU library (cd ../dev && make libvmcuda.so), the 4090 (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process
# service for the lifetime, the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py) and vm-00046's NB3 as
# the control.
# Outputs go to ~/data/research/visual-motor-agent/exp_00047/.
#
# Usage:
#   make runs     — VS: T_text_B2 with sum_only 0, uni_pyr 4, attn_split 2, attn_cols 1; seed 1, 100k ticks,
#                   checkpoints every 10k
#   make analysis — next-letter cross-entropy on the val text every 10k, by letter column (letter_ce.py)

DEV = ../dev
PY = ./venv/bin/python
DATA = $(HOME)/data/research/visual-motor-agent
OUT = $(DATA)/exp_00047
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES

.PHONY: all runs analysis

all: runs analysis

runs:
	mkdir -p $(OUT)
	cd $(DEV) && $(G)=1 $(PY) lifetime.py --config configs/T_text_B2.json --set online.steps=100000 \
	  --set online.save_every=10000 --set model.sum_only=0 --set model.uni_pyr=4 --set model.attn_split=2 \
	  --set model.attn_cols=1 --seed 1 --set name=VS -o $(OUT)/001_VS_s1 > $(OUT)/001_VS_s1.log 2>&1

analysis:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY; $(G)=0 $(PY) letter_ce.py $(OUT)/001_VS_s1 \
	  --ckpts 20000 30000 40000 50000 60000 70000 80000 90000 100000 | tee $(OUT)/letter_ce.txt
