# Makefile for experiment vm-00034-text-self-forcing
#
# Reproduces the results from the source tagged exp/vm-00034 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID), the CUDA multi-process service, the CUDA
# PyTorch venv (~/venvs/vm-gpu), the corpus (../../llm-fundamentals/dev: prepare_tinyshakespeare.py), vm-00030's text
# lifetime (exp_00030/001_Bib_s1, the baseline) and vm-00033's GPT. Outputs go to
# ~/data/research/visual-motor-agent/exp_00034/.
#
# Usage:
#   make runs     — the text lifetime (T_text_B) with self-forcing (2 windows, 1-3 dreamed pages, J = 2), 100k ticks,
#                   seeds 1-3
#   make predict  — val nats per letter every 10k
#   make dream    — generation at 30k, seed 1 (reference/text_dream.py): self-forcing at temperatures 1, 0.8, 0.6;
#                   the baseline at 0.8, 0.6 (temperature 1: vm-00033)

DEV = ../dev
PY = ./venv/bin/python
TPY = $(HOME)/venvs/vm-gpu/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00034
BASE = $(HOME)/data/research/visual-motor-agent/exp_00030/001_Bib_s1/ckpt_00030000.bin
GPT = $(HOME)/data/research/visual-motor-agent/exp_00033/001_GPTlife30k_s1/params.pt
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --config configs/T_text_B.json --set online.steps=100000 --set online.save_every=10000 \
    --set online.sf_windows=2 --set online.sf_steps=3 --set online.sf_jacobi=2
D = $(TPY) reference/text_dream.py --gpt $(GPT) --config configs/T_text_B.json --windows 64 --pages 4 --samples 3

.PHONY: all runs predict dream

all: runs predict dream

runs:
	mkdir -p $(OUT)
	cd $(DEV) && for spec in "1 1" "1 2" "0 3"; do set -- $$spec; \
	  $(G)=$$1 $(L) --seed $$2 --set name=TSF -o $(OUT)/00$$2_TSF_s$$2 > $(OUT)/00$$2_TSF_s$$2.log 2>&1 & \
	done; wait

predict:
	cd $(DEV) && for s in 1 2 3; do \
	  $(G)=1 $(PY) eval_predict.py $(OUT)/00$${s}_TSF_s$$s --ckpts 10000 20000 30000 40000 50000 60000 70000 80000 \
	    90000 100000 --env-set split=val --env-set tile=true --device cuda > $(OUT)/analysis_s$$s.log 2>&1; \
	done

dream:
	cd $(DEV) && $(G)=1 $(D) --uni $(OUT)/001_TSF_s1/ckpt_00030000.bin -o $(OUT)/text_dream_s1.json
	cd $(DEV) && for t in 0.8 0.6; do \
	  $(G)=1 $(D) --uni $(OUT)/001_TSF_s1/ckpt_00030000.bin --temp $$t -o $(OUT)/text_dream_TSF_t$$t.json; \
	  $(G)=1 $(D) --uni $(BASE) --temp $$t -o $(OUT)/text_dream_base_t$$t.json; \
	done
