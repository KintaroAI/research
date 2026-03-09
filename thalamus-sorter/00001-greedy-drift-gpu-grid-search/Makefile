# Reproduce experiment 00001: greedy drift GPU optimization & grid search
# Run from thalamus-sorter/dev/

DEV := ../dev
PYTHON := $(DEV)/venv/bin/python

.PHONY: setup grid grid-images single-80 single-160

setup:
	cd $(DEV) && make setup

# Full grid search: 5 MF x 6 K = 30 runs, 80x80, 100k frames each
grid:
	cd $(DEV) && $(PYTHON) run_grid.py

# Generate composite grid visualization (requires grid results)
grid-images:
	cd $(DEV) && $(PYTHON) make_grid_image.py

# Single convergence runs
single-80:
	cd $(DEV) && $(PYTHON) main.py greedy --image K_80_g.png -W 80 -H 80 \
		--k 24 --frames 500000 --save-every 1000 -o output/ --gpu

single-160:
	cd $(DEV) && $(PYTHON) main.py greedy --image K_160_g.png -W 160 -H 160 \
		--k 24 --frames 500000 --save-every 1000 -o output/ --gpu
