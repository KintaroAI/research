DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: run test

run:
	cd $(DEV) && $(PYTHON) benchmark_recurrent.py --frames 20000 \
		-o $$($(PYTHON) output_name.py 15 recurrent)

test:
	cd $(DEV) && make test
