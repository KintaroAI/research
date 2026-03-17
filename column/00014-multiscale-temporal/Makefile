DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_multiscale.py --frames 20000 \
		-o $$($(PYTHON) output_name.py 14 multiscale)

test:
	cd $(DEV) && make test
