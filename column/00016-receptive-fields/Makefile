DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_receptive.py --frames 20000 \
		-o $$($(PYTHON) output_name.py 16 receptive)

test:
	cd $(DEV) && make test
