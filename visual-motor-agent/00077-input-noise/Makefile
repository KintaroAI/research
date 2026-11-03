# Makefile for experiment vm-00077-input-noise
#
# Reproduces the runs of task 00088 (stage 3 of vm-00076): K4 with log summaries, online.noise_in .1 and .5, the world, memory and targets clean, three seeds.
# Needs the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00077/.
# The comparisons are vm-00076's N10 and N50 (the same noise in the world) and vm-00073's K4, seeds 1-3, not rerun.
#
# Usage:
#   make runs — I10 and I50, seeds 1 and 2 on the RTX 4090 and 3 on the RTX 3090, six lives at once
#   make frozen — eval_noise.py chain and error at 0, .1, .5 and 1, without MPS
#   make figures — the README's image (noise_figures.py input)

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00077
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.future_k=4

.PHONY: all runs frozen figures

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { for a in 10:0.1 50:0.5; do for s in 1 2 3; do \
	  if [ $$s -le 2 ]; then g=1; else g=0; fi; \
	  $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.noise_in=$${a#*:} --seed $$s --set name=I$${a%:*} -o $(OUT)/I$${a%:*}_s$$s > $(OUT)/I$${a%:*}_s$$s.log 2>&1 & \
	done; done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { \
	  r="$$(for s in 1 2 3; do echo $(OUT)/I10_s$$s $(OUT)/I50_s$$s; done)"; \
	  CUDA_VISIBLE_DEVICES=1 $(PY) eval_noise.py chain $$r --levels 0 0.1 0.5 1 > $(OUT)/chain3.jsonl & \
	  CUDA_VISIBLE_DEVICES=0 $(PY) eval_noise.py error $$r --levels 0 0.1 0.5 1 > $(OUT)/error3.jsonl & \
	  wait; }

figures:
	cd $(DEV) && $(PY) noise_figures.py input
