DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_composition.py --noise 0.1 \
		-o $$($(PYTHON) output_name.py 7 composition)

test:
	cd $(DEV) && make test
