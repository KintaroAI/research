DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run run-baseline test

# Temporal correlation mode on co-variation data
run:
	cd $(DEV) && $(PYTHON) main.py \
		--n-inputs 16 --n-outputs 4 --frames 5000 \
		--temporal-mode correlation --temporal-window 10 \
		-o $$($(PYTHON) output_name.py 2 correlation_16x4)

# Baseline: instantaneous mode on same temporal data (should fail to separate)
run-baseline:
	cd $(DEV) && $(PYTHON) main.py \
		--n-inputs 16 --n-outputs 4 --frames 5000 \
		--temporal-mode none --temporal-window 10 \
		-o $$($(PYTHON) output_name.py 2 instantaneous_16x4)

test:
	cd $(DEV) && make test
