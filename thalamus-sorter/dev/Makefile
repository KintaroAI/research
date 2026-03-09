VENV := venv
PYTHON := $(VENV)/bin/python

.PHONY: setup test greedy mst sa camera-sa camera-spatial clean

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

test:
	$(PYTHON) -c "\
from utils.weights import inverse_distance_1d, decay_distance_2d, topk_decay2d, topk_inv1d, OnlineWeightMatrix; \
from utils.correlation import pearson, temporal_correlation; \
from utils.geometry import coords, to_index, move_closer; \
from utils.graph import UnionFind, build_mst, tree_to_adjacency, dfs_order; \
from solvers.greedy_drift import GreedyDrift; \
from solvers.simulated_annealing import SimulatedAnnealing; \
from solvers.spatial_coherence import SpatialCoherence; \
from solvers.mst_sort import mst_sort; \
import numpy as np; \
W = decay_distance_2d(10, 10, 0.1); \
tk = topk_decay2d(10, 10, 8); \
g = GreedyDrift(10, 10, top_k=tk, k=8); g.tick(); \
g2 = GreedyDrift(10, 10, W, k=8); g2.tick(); \
tk1d = topk_inv1d(100, 8); assert tk1d.shape == (100, 8); \
sa = SimulatedAnnealing(W, 10, 10); sa.tick(50); \
m = mst_sort(W, 10, 10); \
sc = SpatialCoherence(10, 10); sc.optimize(np.random.rand(100), epochs=10); \
assert coords(7, 5) == (2, 1); \
assert to_index(2, 1, 5) == 7; \
assert move_closer(1, 1, 5, 5) == (2, 2, 4, 4); \
ow = OnlineWeightMatrix(10); ow.update(0, 1, 0.8); ow.update(0, 1, 0.6); \
assert abs(ow.weights[0, 1] - 0.7) < 1e-10; \
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]); \
assert abs(pearson(x, 2*x) - 1.0) < 1e-10; \
print('All tests passed.')"

# Grid modes (no camera required)
greedy:
	$(PYTHON) main.py greedy $(ARGS)

mst:
	$(PYTHON) main.py mst $(ARGS)

sa:
	$(PYTHON) main.py sa $(ARGS)

# Camera modes
camera-sa:
	$(PYTHON) main.py camera-sa $(ARGS)

camera-spatial:
	$(PYTHON) main.py camera-spatial $(ARGS)

clean:
	rm -rf $(VENV) __pycache__ utils/__pycache__ solvers/__pycache__
