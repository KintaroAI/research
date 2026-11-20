# Makefile for experiment vm-00101-candidate-temperature
#
# Reproduces the runs of task 00113: the instinct's 16 candidates with the first drawn at temperature 1 (the model's own
# move) and the other 15 at temperature 2 (online.instinct_temp = 2), on the base chosen by the rule set before vm-00100's
# results: RD (vm-00100, the body label re-labeled at every update) if RD - SD food at 40-60k, paired by seed, has a 95%
# interval above 0, else SD (vm-00099, stored once). Both: the default agent C8 with the meal label as a body value and the
# full dose of poison, in the world with inertia 4 from birth. Twelve seeds on the RTX 3090 (GPU 0 under
# CUDA_DEVICE_ORDER=PCI_BUS_ID), under MPS, 60k ticks. Outputs go to ~/data/research/visual-motor-agent/exp_00101/.
#
# Usage:
#   make base — print the base the rule chooses (from vm-00100's and vm-00099's metrics)
#   make runs BASE=RD|SD — TT, 12 lives at once on GPU 0
#   make frozen — eval_hunger.py chain (1,500 ticks), value (2,000), reach (1,500) and move (1,500), without MPS, two at a time
#   make candidates — eval_candidates.py on TT (300 ticks x 16 bodies)
#   make videos BASE=RD|SD — compare_video.py of seeds 5 and 3: IB, MB, the base and TT

DEV = ../dev
PY = ./venv/bin/python
D = $(HOME)/data/research/visual-motor-agent
OUT = $(D)/exp_00101
BASE = RD
BASE_DIR = $(if $(filter RD,$(BASE)),$(D)/exp_00100,$(D)/exp_00099)
BASE_SET = $(if $(filter RD,$(BASE)),--set online.label_refresh=1,)
BASE_NAME = $(if $(filter RD,$(BASE)),RD (re-labeled),SD (stored once))
G = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MPS_PIPE_DIRECTORY=/tmp/claude-1000/mps_pipe CUDA_VISIBLE_DEVICES=0
RUN = $(PY) lifetime.py --config configs/U_view.json --set model.n_layer=8 --set online.instinct_tol=0.002 \
  --set model.body_label=1 --set online.label=meal --set online.label_gamma=0.95 --set env.inertia=4 --set online.full_dose=1 \
  $(BASE_SET) --set online.instinct_temp=2

.PHONY: all base runs frozen candidates videos

all: runs

base:
	@cd $(DEV) && $(PY) -c 'import json, numpy as np; \
	f = lambda e, a, s: np.mean([r["step_hits"] for r in map(json.loads, open("$(D)/exp_%s/%s_s%d/metrics.jsonl" % (e, a, s))) if 40000 < r["step"] <= 60000]); \
	d = np.array([f("00100", "RD", s) - f("00099", "SD", s) for s in range(1, 13)]); \
	h = 2.201 * d.std(ddof=1) / 12 ** .5; \
	print("RD - SD food at 40-60k: %+.2f +- %.2f (%d of 12 above 0)" % (d.mean(), h, (d > 0).sum())); \
	print("BASE=" + ("RD" if d.mean() - h > 0 else "SD"))'

runs:
	mkdir -p $(OUT)
	cd $(DEV) && { \
	  for s in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	    $(G) $(RUN) --seed $$s --set name=TT -o $(OUT)/TT_s$$s > $(OUT)/TT_s$$s.log 2>&1 & \
	  done; wait; }

A = $$(for s in 1 2 3 4 5 6; do echo $(OUT)/TT_s$$s; done | tr "\n" " ")
B = $$(for s in 7 8 9 10 11 12; do echo $(OUT)/TT_s$$s; done | tr "\n" " ")
EVAL = cd $(DEV) && unset CUDA_MPS_PIPE_DIRECTORY && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 &&

frozen:
	$(EVAL) a="$(A)" && b="$(B)" && { \
	  $(PY) eval_hunger.py chain $$a > $(OUT)/chain_a.jsonl & $(PY) eval_hunger.py chain $$b > $(OUT)/chain_b.jsonl & wait; \
	  $(PY) eval_hunger.py value $$a --ticks 2000 > $(OUT)/value_a.jsonl & $(PY) eval_hunger.py value $$b --ticks 2000 > $(OUT)/value_b.jsonl & wait; \
	  $(PY) eval_hunger.py reach $$a > $(OUT)/reach_a.jsonl & $(PY) eval_hunger.py reach $$b > $(OUT)/reach_b.jsonl & wait; \
	  $(PY) eval_hunger.py move $$a > $(OUT)/move_a.jsonl & $(PY) eval_hunger.py move $$b > $(OUT)/move_b.jsonl & wait; }

candidates:
	$(EVAL) $(PY) eval_candidates.py $$(for s in 1 2 3 4 5 6 7 8 9 10 11 12; do echo $(OUT)/TT_s$$s; done) > $(OUT)/candidates.jsonl

videos:
	$(EVAL) for s in 5 3; do \
	  $(PY) compare_video.py $(D)/exp_00095/IB_s$$s:"IB (hunger value)" $(D)/exp_00097/MB_s$$s:"MB (meal label, re-labeled)" \
	    $(BASE_DIR)/$(BASE)_s$$s:"$(BASE_NAME)" $(OUT)/TT_s$$s:"TT ($(BASE) + candidates at temperature 2)" \
	    --steps 1000 --worlds 8 --cols 2 -o $(D)/videos/vm101_s$$s; \
	done
