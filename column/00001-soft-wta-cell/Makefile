DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

# Baseline: 8 clusters in 16-dim, 8 output units, 10k frames
run:
	cd $(DEV) && $(PYTHON) main.py \
		--n-inputs 16 --n-outputs 8 --frames 10000 \
		-o $$($(PYTHON) output_name.py 1 baseline_16x8)

test:
	cd $(DEV) && make test
