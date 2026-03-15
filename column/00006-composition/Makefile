DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_composition.py --samples 15000 \
		-o $$($(PYTHON) output_name.py 6 composition)

test:
	cd $(DEV) && make test
