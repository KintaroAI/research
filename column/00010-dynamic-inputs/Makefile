DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_dynamic_inputs.py \
		-o $$($(PYTHON) output_name.py 10 dynamic_inputs)

test:
	cd $(DEV) && make test
