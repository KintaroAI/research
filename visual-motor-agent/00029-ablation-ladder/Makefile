# Makefile for experiment vm-00029-ablation-ladder
#
# Reproduces the runs from the source tagged exp/vm-00029 (dev/ at that tag). Needs CUDA PyTorch in ~/venvs/vm-gpu,
# libvmagent.so (the C++ model's initial weights), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID) and the corpus
# (../../llm-fundamentals/dev: prepare_tinyshakespeare.py). Outputs go to ~/data/research/visual-motor-agent/exp_00029/.
#
# Usage:
#   make runs — the ladder (reference/text_ladder.py): the universal model (R0) with one switch at a time toward the
#               classical GPT (R1-R5), the GPT (R6), the bin-embedding input (R7, R8); R0 and R5 with a second seed

DEV = ../dev
GPY = $(HOME)/venvs/vm-gpu/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00029
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
R = $(GPY) reference/text_ladder.py --updates 30000 --eval-every 5000

.PHONY: all runs

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && \
	$(G)=0 $(R) --seed 1 --name R0 -o $(OUT)/R0 > $(OUT)/R0.log 2>&1 & \
	$(G)=0 $(R) --seed 1 --name R1 --init gpt -o $(OUT)/R1 > $(OUT)/R1.log 2>&1 & \
	$(G)=0 $(R) --seed 1 --name R2 --output classes -o $(OUT)/R2 > $(OUT)/R2.log 2>&1 & \
	$(G)=0 $(R) --seed 1 --name R3 --tokens letters -o $(OUT)/R3 > $(OUT)/R3.log 2>&1 & \
	$(G)=1 $(R) --seed 1 --name R4 --pos absolute -o $(OUT)/R4 > $(OUT)/R4.log 2>&1 & \
	$(G)=1 $(R) --seed 1 --name R5 --input table -o $(OUT)/R5 > $(OUT)/R5.log 2>&1 & \
	$(G)=1 $(R) --seed 1 --name R6 --tokens letters --input table --pos absolute --output classes --init gpt \
	  -o $(OUT)/R6 > $(OUT)/R6.log 2>&1 & \
	wait
	cd $(DEV) && \
	$(G)=0 $(R) --seed 1 --name R7 --input bins -o $(OUT)/R7 > $(OUT)/R7.log 2>&1 & \
	$(G)=1 $(R) --seed 1 --name R8 --input bins --init gpt -o $(OUT)/R8 > $(OUT)/R8.log 2>&1 & \
	$(G)=0 $(R) --seed 2 --name R0_s2 -o $(OUT)/R0_s2 > $(OUT)/R0_s2.log 2>&1 & \
	$(G)=1 $(R) --seed 2 --name R5_s2 --input table -o $(OUT)/R5_s2 > $(OUT)/R5_s2.log 2>&1 & \
	wait
