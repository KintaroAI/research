# Makefile for experiment vm-00064-sixteen-pieces
#
# Reproduces the runs from the source tagged exp/vm-00064 (dev/ at that tag): the default model split into 16 transformers of 4 patches each (task 00073). Needs
# the GPU library (cd ../dev && make libvmcuda.so) and the MPS daemon. Seeds 1-3 on the RTX 3090, 4-8 on the RTX 4090 (CUDA_DEVICE_ORDER=PCI_BUS_ID).
# Outputs go to ~/data/research/visual-motor-agent/exp_00064/. The control is vm-00063's NA and CO.
#
# Usage:
#   make runs — 60k ticks, seeds 1-8: G16 (gain), C16 (council), both model.band_n=16
#   make videos — frozen agents at 60k from the same start: 16 and 4 pieces under both rules (videos/pieces/compare.mp4, each ARM.mp4) and council against gain at 16 pieces
#                (videos/council_vs_gain/compare.mp4); each arm plays the seed closest to its own mean food. The blue square is the block of the piece that made the move

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00064
CTL = $(HOME)/data/research/visual-motor-agent/exp_00063
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe
RUN = $(PY) lifetime.py --config configs/U_default.json --set model.band_n=16

.PHONY: all runs videos

all: runs

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8; do \
	    g=$$(( s > 3 ? 1 : 0 )); \
	    for arm in G16:gain C16:council; do \
	      a=$${arm%%:*}; r=$${arm##*:}; \
	      $(G) CUDA_VISIBLE_DEVICES=$$g $(RUN) --set online.piece_rule=$$r --seed $$s --set name=$$a -o $(OUT)/$${a}_s$$s > $(OUT)/$${a}_s$$s.log 2>&1 & \
	    done; \
	  done; wait; }

videos:
	cd $(DEV) && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $(PY) compare_video.py --cols 2 --worlds 16 --step 60000 -o $(OUT)/videos/pieces \
	  "$(OUT)/G16_s3:G16 (gain, 16 pieces):gain" "$(OUT)/C16_s4:C16 (council, 16 pieces):council" "$(CTL)/NA_s7:NA (gain, 4 pieces):gain" "$(CTL)/CO_s8:CO (council, 4 pieces):council"
	cd $(DEV) && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $(PY) forage_video.py $(OUT)/C16_s4 --vs $(OUT)/G16_s3 --title "council, 16 pieces" --vs-title "gain, 16 pieces" \
	  --step 60000 --worlds 16 -o $(OUT)/videos/council_vs_gain
