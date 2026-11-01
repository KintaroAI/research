# Makefile for experiment vm-00063-selection-rules
#
# Reproduces the runs from the source tagged exp/vm-00063 (dev/ at that tag): five ways to choose the body's move among the pieces (task 00072), on the default model in its circle world. Needs
# the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Seeds 1-3 on the RTX 3090, 4-8 on the RTX 4090 (CUDA_DEVICE_ORDER=PCI_BUS_ID).
# Outputs go to ~/data/research/visual-motor-agent/exp_00063/.
#
# Usage:
#   make runs  — 60k ticks, seeds 1-8: GS (gain_sharp), SO (soft), PE (pessimism), RA (rank_first: first run as rank, before the tie fix), VE (veto)
#   make runs2 — the same, after the fix: RT (rank, ties at random) and the same-build control NA (gain), CO (council)
#   make videos — frozen agents at 60k: all eight rules at once (videos/rules/compare.mp4, each RULE.mp4) and council against gain (videos/council_vs_gain/compare.mp4),
#                from the same start; each rule plays the seed closest to its own mean food, not its best

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00063
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json

.PHONY: all runs runs2 videos

all: runs runs2

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8; do \
	    g=$$(( s > 3 ? 1 : 0 )); \
	    for arm in GS:gain_sharp SO:soft PE:pessimism RA:rank_first VE:veto; do \
	      a=$${arm%%:*}; r=$${arm##*:}; \
	      $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.piece_rule=$$r --seed $$s --set name=$$a -o $(OUT)/$${a}_s$$s > $(OUT)/$${a}_s$$s.log 2>&1 & \
	    done; \
	  done; wait; }

runs2:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8; do \
	    g=$$(( s > 3 ? 1 : 0 )); \
	    for arm in RT:rank NA:gain CO:council; do \
	      a=$${arm%%:*}; r=$${arm##*:}; \
	      $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.piece_rule=$$r --seed $$s --set name=$$a -o $(OUT)/$${a}_s$$s > $(OUT)/$${a}_s$$s.log 2>&1 & \
	    done; \
	  done; wait; }

videos:
	cd $(DEV) && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py --cols 4 --worlds 16 --step 60000 -o $(OUT)/videos/rules \
	  "$(OUT)/NA_s7:gain (default):gain" "$(OUT)/CO_s8:council:council" "$(OUT)/RA_s1:rank_first:rank_first" "$(OUT)/RT_s1:rank (fair ties):rank" \
	  "$(OUT)/GS_s2:gain_sharp:gain_sharp" "$(OUT)/VE_s4:veto:veto" "$(OUT)/PE_s8:pessimism:pessimism" "$(OUT)/SO_s3:soft:soft"
	cd $(DEV) && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $(PY) forage_video.py $(OUT)/CO_s8 --vs $(OUT)/NA_s7 --title "council" --vs-title "gain (default)" \
	  --step 60000 --worlds 16 -o $(OUT)/videos/council_vs_gain
