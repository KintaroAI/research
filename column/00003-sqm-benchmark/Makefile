DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark.py \
		-o $$($(PYTHON) output_name.py 3 sqm_benchmark)

test:
	cd $(DEV) && make test
