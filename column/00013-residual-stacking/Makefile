DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_residual.py --frames 20000 \
		-o $$($(PYTHON) output_name.py 13 residual)

test:
	cd $(DEV) && make test
