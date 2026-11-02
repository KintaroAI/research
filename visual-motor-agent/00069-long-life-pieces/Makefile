# Makefile for experiment vm-00069-long-life-pieces
#
# Reproduces the runs from the source tagged exp/vm-00069 (dev/ at that tag): 500k ticks with a memory of the whole life (task 00080). Needs the GPU library
# (cd ../dev && make libvmcuda.so) and the MPS daemon. Outputs go to ~/data/research/visual-motor-agent/exp_00069/.
#
# Usage:
#   make runs-lt — LT, seeds 1-4 on the RTX 4090: 16 pieces, shared tables; about 10 hours
#   make runs-ld — LD, seeds 1-4 on the RTX 3090: the default, 4 pieces; about 8 hours
#   make frozen — eval_pieces.py at the end of each life, gain and a random piece, without MPS, on the RTX 3090 (frozen_LD.txt, frozen_LT.txt)
#   make hunger — eval_hunger.py (newer than the tag: use master's dev/): food in reach by real hunger, a false hunger, chaos and a grab; LD on the RTX 3090, LT on the RTX 4090, without MPS, about 25 minutes; `$(PY) eval_hunger.py show` prints the tables
#   make videos — frozen agents at 500k from the same start: each arm's four seeds (videos/ld_seeds, videos/lt_seeds; the single ARM.mp4 is the last seed, since an arm's runs share a name)
#                 and LT seed 4 against LD seed 1, the seeds closest to their arms' mean food (videos/lt_vs_ld). Labels hold no colon: RUN:LABEL:RULE

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00069
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set online.steps=500000 --set online.replay=500000 --set online.save_every=50000

.PHONY: all runs-lt runs-ld frozen hunger videos

all: runs-lt runs-ld

runs-lt:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in 1 2 3 4; do \
	  $(G) CUDA_VISIBLE_DEVICES=1 $(RUN) --set model.band_n=16 --set online.tie_tables=inout --seed $$s --set name=LT -o $(OUT)/LT_s$$s > $(OUT)/LT_s$$s.log 2>&1 & \
	done; wait; }

runs-ld:
	mkdir -p $(OUT)
	cd $(DEV) && { for s in 1 2 3 4; do \
	  $(G) CUDA_VISIBLE_DEVICES=0 $(RUN) --seed $$s --set name=LD -o $(OUT)/LD_s$$s > $(OUT)/LD_s$$s.log 2>&1 & \
	done; wait; }

frozen:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 && for a in LD LT; do \
	  for s in 1 2 3 4; do \
	    $(PY) eval_pieces.py $(OUT)/$${a}_s$$s --rule gain --ring; \
	    $(PY) eval_pieces.py $(OUT)/$${a}_s$$s --rule random --ring; \
	  done > $(OUT)/frozen_$$a.txt 2>&1; \
	done

hunger:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID && { for w in reach:1500 told:750 chain:1500; do \
	  m=$${w%:*}; t=$${w#*:}; \
	  { CUDA_VISIBLE_DEVICES=0 $(PY) eval_hunger.py $$m $(OUT)/LD_s1 $(OUT)/LD_s2 $(OUT)/LD_s3 $(OUT)/LD_s4 --ticks $$t & \
	    CUDA_VISIBLE_DEVICES=1 $(PY) eval_hunger.py $$m $(OUT)/LT_s1 $(OUT)/LT_s2 $(OUT)/LT_s3 $(OUT)/LT_s4 --ticks $$t & \
	    wait; } > $(OUT)/hunger_$$m.jsonl; \
	done; }

videos:
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $(PY) compare_video.py --cols 2 --worlds 16 --step 500000 -o $(OUT)/videos/ld_seeds \
	  "$(OUT)/LD_s1:LD seed 1:gain" "$(OUT)/LD_s2:LD seed 2:gain" "$(OUT)/LD_s3:LD seed 3:gain" "$(OUT)/LD_s4:LD seed 4:gain"
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $(PY) compare_video.py --cols 2 --worlds 16 --step 500000 -o $(OUT)/videos/lt_seeds \
	  "$(OUT)/LT_s1:LT seed 1:gain" "$(OUT)/LT_s2:LT seed 2:gain" "$(OUT)/LT_s3:LT seed 3:gain" "$(OUT)/LT_s4:LT seed 4:gain"
	cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $(PY) compare_video.py --cols 2 --worlds 16 --step 500000 -o $(OUT)/videos/lt_vs_ld \
	  "$(OUT)/LT_s4:LT, 16 pieces, shared tables:gain" "$(OUT)/LD_s1:LD, the default, 4 pieces:gain"
