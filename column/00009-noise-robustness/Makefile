DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_noise.py \
		-o $$($(PYTHON) output_name.py 9 noise_robustness)

test:
	cd $(DEV) && make test
