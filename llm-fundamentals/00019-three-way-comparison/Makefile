# Makefile for experiment 00019-three-way-comparison
#
# Reproduces the full experiment from a clean state.
# Assumes llm-fundamentals/dev/ has been built from the exp/00019 tag:
#   cd llm-fundamentals/dev && make train
#
# Usage:
#   make run      — run all 3 training jobs sequentially
#   make baseline — run baseline only
#   make blend    — run blend-G8 only
#   make hebbian  — run hebbian-H4 only

DEV = ../dev
DATA = $(HOME)/data

.PHONY: all run baseline blend hebbian

all: run

run: baseline blend hebbian

baseline:
	cd $(DEV) && ./venv/bin/python wandb_train.py --name "exp19-baseline" --tags exp19 -- \
		./train -e model_50m.bin -b 8 -t 512 -n 50000 -q 0 -c $(DATA)/exp19-baseline.bin

blend:
	cd $(DEV) && ./venv/bin/python wandb_train.py --name "exp19-blend-G8" --tags exp19 -- \
		./train -e model_50m.bin -b 8 -t 512 -n 50000 -q 0 -G 8 -c $(DATA)/exp19-blend-G8.bin

hebbian:
	cd $(DEV) && ./venv/bin/python wandb_train.py --name "exp19-hebbian-H4" --tags exp19 -- \
		./train -e model_50m.bin -b 8 -t 512 -n 50000 -q 0 -H 4 -u 1e-5 -c $(DATA)/exp19-hebbian-H4.bin
