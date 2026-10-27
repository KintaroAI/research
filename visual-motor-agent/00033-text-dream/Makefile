# Makefile for experiment vm-00033-text-dream
#
# Reproduces the results from the source tagged exp/vm-00033 (dev/ at that tag). Needs the CUDA PyTorch venv
# (~/venvs/vm-gpu), vm-00030's text lifetime (exp_00030/001_Bib_s1, its 30k checkpoint) and the corpus
# (../../llm-fundamentals/dev: prepare_tinyshakespeare.py). Outputs go to ~/data/research/visual-motor-agent/exp_00033/.
#
# Usage:
#   make gpt      — the classical GPT on the lifetime's batches, 30k updates (its best), seed 1
#   make dream    — generation and scoring: 64 val contexts, 4 pages each (reference/text_dream.py)

DEV = ../dev
TPY = $(HOME)/venvs/vm-gpu/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00033
UNI = $(HOME)/data/research/visual-motor-agent/exp_00030/001_Bib_s1/ckpt_00030000.bin
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1

.PHONY: all gpt dream

all: gpt dream

gpt:
	mkdir -p $(OUT)
	cd $(DEV) && $(G) $(TPY) reference/char_gpt.py --config configs/T_text_B.json --data lifetime --updates 30000 \
	  --eval-every 10000 --seed 1 -o $(OUT)/001_GPTlife30k_s1 > $(OUT)/001_GPTlife30k_s1.log 2>&1

dream:
	cd $(DEV) && $(G) $(TPY) reference/text_dream.py --uni $(UNI) --gpt $(OUT)/001_GPTlife30k_s1/params.pt \
	  --config configs/T_text_B.json --windows 64 --pages 4 --samples 3 -o $(OUT)/text_dream_s1.json
