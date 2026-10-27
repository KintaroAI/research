# Makefile for experiment vm-00036-touch-world-egocentric
#
# Reproduces the runs from the source tagged exp/vm-00036 (dev/ at that tag). Needs the GPU library
# (cd ../dev && make libvmcuda.so), both GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID) and the CUDA multi-process service.
# Outputs go to ~/data/research/visual-motor-agent/exp_00036/ (the video compares with vm-00035's TSF4 s3).
#
# Usage:
#   make runs     — EB (no self-forcing, 1-step) s1-3 on the 3090; ESF4 (self-forcing + 4-step pair lookahead in life)
#                   s1-2 on the 4090, s3 on the 3090 after EB (run ESF4 alone on its GPU: next to other jobs it slows
#                   to ~1 tick/s)
#   make analysis — frozen forage tests at 100k (instinct off, 1-step, 4-step mean, 4-step pair) and the dream curve
#   make video    — ESF4 s3 (egocentric) vs vm-00035's TSF4 s3 (allocentric) side by side; ESF4 s3 and EB s3 dreaming

DEV = ../dev
PY = ./venv/bin/python
OUT = $(HOME)/data/research/visual-motor-agent/exp_00036
OUT35 = $(HOME)/data/research/visual-motor-agent/exp_00035
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES
L = $(PY) lifetime.py --config configs/U_touch_ego.json --set online.steps=100000 --set online.save_every=25000
X = --set online.sf_windows=8 --set online.sf_steps=3 --set online.sf_jacobi=2 \
    --set online.instinct_h=4 --set online.instinct_decode=pair
F = --eval-device cuda --eval-tests forage --eval-worlds 16

.PHONY: all runs analysis video

all: runs analysis video

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2; do $(G)=1 $(L) $(X) --seed $$s --set name=ESF4 -o $(OUT)/00$$((s+3))_ESF4_s$$s \
	    > $(OUT)/00$$((s+3))_ESF4_s$$s.log 2>&1 & done; \
	  for s in 1 2 3; do $(G)=0 $(L) --seed $$s --set name=EB -o $(OUT)/00$${s}_EB_s$$s \
	    > $(OUT)/00$${s}_EB_s$$s.log 2>&1 & done; \
	  while [ $$(ls $(OUT)/00?_EB_s?/ckpt_final.bin 2>/dev/null | wc -l) -lt 3 ]; do sleep 30; done; \
	  $(G)=0 $(L) $(X) --seed 3 --set name=ESF4 -o $(OUT)/006_ESF4_s3 > $(OUT)/006_ESF4_s3.log 2>&1; wait; }

analysis:
	cd $(DEV) && for r in $(OUT)/00?_E*_s?; do \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-instinct 0; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 1; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode mean; \
	  $(G)=0 $(PY) online.py --eval $$r --ckpts 100000 $(F) --eval-horizon 4 --eval-decode pair; \
	  $(G)=0 $(PY) eval_dream.py $$r --ckpts 100000; \
	done

video:
	cd $(DEV) && $(G)=1 $(PY) forage_video.py $(OUT)/006_ESF4_s3 --step 100000 --steps 1000 --worlds 16 \
	  --horizon 4 --decode pair --title "ESF4 seed 3 (egocentric)" --vs $(OUT35)/009_TSF4_s3 \
	  --vs-title "TSF4 seed 3 (allocentric, vm-00035)" -o $(OUT)/video_ESF4_s3_vs_TSF4_s3
	cd $(DEV) && for r in 006_ESF4_s3 003_EB_s3; do \
	  $(G)=0 $(PY) dream_video.py $(OUT)/$$r --step 100000 --real 200 --dream 200 -o $(OUT)/video_$${r#*_}; \
	done
