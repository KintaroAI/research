DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_3d.py \
		-o $$($(PYTHON) output_name.py 4 movement_3d)

test:
	cd $(DEV) && make test
