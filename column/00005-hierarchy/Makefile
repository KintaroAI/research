DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_hierarchy.py --samples 8000 \
		-o $$($(PYTHON) output_name.py 5 hierarchy)

test:
	cd $(DEV) && make test
