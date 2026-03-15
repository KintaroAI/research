VENV := venv
PYTHON := $(VENV)/bin/python

.PHONY: setup test clean

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

test:
	$(PYTHON) -c "\
	import torch; \
	from column import SoftWTACell; \
	\
	# Test 1: forward produces valid probabilities \
	cell = SoftWTACell(16, 8); \
	x = torch.randn(16); \
	p = cell.forward(x); \
	assert p.shape == (8,), f'bad shape: {p.shape}'; \
	assert abs(p.sum().item() - 1.0) < 1e-5, f'probs sum to {p.sum()}'; \
	assert (p >= 0).all(), 'negative probabilities'; \
	\
	# Test 2: batch forward \
	xb = torch.randn(4, 16); \
	pb = cell.forward(xb); \
	assert pb.shape == (4, 8), f'bad batch shape: {pb.shape}'; \
	\
	# Test 3: update returns valid winner and match quality \
	w, mq = cell.update(x); \
	assert 0 <= w < 8, f'bad winner: {w}'; \
	assert -1.0 <= mq <= 1.0, f'bad match quality: {mq}'; \
	\
	# Test 4: anti-collapse — white noise should spread winners \
	cell2 = SoftWTACell(16, 8, temperature=0.5, lr=0.05); \
	winners = set(); \
	for _ in range(500): \
	    xn = torch.randn(16); \
	    w2, _ = cell2.update(xn); \
	    winners.add(w2); \
	assert len(winners) >= 4, f'collapse: only {len(winners)} unique winners on noise'; \
	\
	# Test 5: cluster learning — purity should be decent \
	import numpy as np; \
	rng = np.random.default_rng(0); \
	centers = [torch.tensor(c, dtype=torch.float32) for c in \
	    (rng.standard_normal((4, 16)) / np.sqrt(16)).tolist()]; \
	centers = [c / c.norm() for c in centers]; \
	cell3 = SoftWTACell(16, 4, temperature=0.3, lr=0.05); \
	cluster_winners = {i: [] for i in range(4)}; \
	for _ in range(2000): \
	    ci = rng.integers(4); \
	    xc = centers[ci] + torch.randn(16) * 0.1; \
	    w3, _ = cell3.update(xc); \
	    cluster_winners[ci].append(w3); \
	modes = set(); \
	for ci in range(4): \
	    last = cluster_winners[ci][-200:]; \
	    mode = max(set(last), key=last.count); \
	    modes.add(mode); \
	assert len(modes) >= 3, f'poor separation: only {len(modes)} distinct modes'; \
	\
	# Test 6: state save/load roundtrip \
	sd = cell3.state_dict(); \
	cell4 = SoftWTACell.from_state_dict(sd); \
	assert torch.allclose(cell3.prototypes, cell4.prototypes); \
	assert torch.allclose(cell3.usage, cell4.usage); \
	\
	# Test 7: output_name \
	from output_name import next_output; \
	import tempfile, os; \
	d = tempfile.mkdtemp(); \
	p1 = next_output(1, 'test', base_dir=d); \
	assert '/exp_00001/001_test' in p1; \
	p2 = next_output(1, 'test2', base_dir=d); \
	assert '/exp_00001/002_test2' in p2; \
	\
	# Test 8: SQM — perfect separation \
	from metrics import compute_sqm, winner_entropy, usage_gini, confidence_gap, \
	    prototype_spread, normalized_mutual_info, purity; \
	w_perfect = np.array([0,0,0,1,1,1,2,2,2,3,3,3]); \
	l_perfect = np.array([0,0,0,1,1,1,2,2,2,3,3,3]); \
	assert abs(purity(w_perfect, l_perfect, 4) - 1.0) < 1e-6, 'perfect purity should be 1'; \
	assert abs(normalized_mutual_info(w_perfect, l_perfect) - 1.0) < 1e-6, 'perfect NMI should be 1'; \
	assert abs(winner_entropy(w_perfect, 4) - 1.0) < 1e-6, 'uniform entropy should be 1'; \
	assert abs(usage_gini(w_perfect, 4)) < 1e-6, 'uniform gini should be 0'; \
	\
	# Test 9: SQM — collapsed (single winner) \
	w_collapse = np.zeros(100, dtype=int); \
	assert abs(winner_entropy(w_collapse, 4)) < 1e-6, 'collapsed entropy should be 0'; \
	assert usage_gini(w_collapse, 4) > 0.5, 'collapsed gini should be high'; \
	\
	# Test 10: SQM — random assignment degrades NMI \
	w_random = np.array([0,1,2,3,0,1,2,3,0,1,2,3]); \
	l_ordered = np.array([0,0,0,1,1,1,2,2,2,3,3,3]); \
	nmi_rand = normalized_mutual_info(w_random, l_ordered); \
	assert nmi_rand < 0.5, f'random assignment NMI should be low, got {nmi_rand}'; \
	\
	# Test 11: SQM — confidence gap \
	p_sharp = np.array([[0.9, 0.05, 0.03, 0.02], [0.85, 0.1, 0.03, 0.02]]); \
	p_flat = np.array([[0.3, 0.25, 0.25, 0.2], [0.28, 0.27, 0.25, 0.2]]); \
	assert confidence_gap(p_sharp) > confidence_gap(p_flat), 'sharp should have larger gap'; \
	\
	# Test 12: SQM — prototype spread \
	eye_protos = np.eye(4, dtype=np.float32); \
	assert abs(prototype_spread(eye_protos) - 1.0) < 1e-6, 'orthogonal spread should be 1'; \
	dup_protos = np.ones((4, 4), dtype=np.float32); \
	assert abs(prototype_spread(dup_protos)) < 1e-6, 'identical spread should be 0'; \
	\
	# Test 13: SQM — compute_sqm full pipeline \
	sqm = compute_sqm(w_perfect, p_sharp[:1].repeat(12, axis=0)[:12], \
	    eye_protos, 4, labels=l_perfect); \
	assert 'winner_entropy' in sqm; \
	assert 'nmi' in sqm; \
	assert 'confidence_gap' in sqm; \
	assert 'prototype_spread' in sqm; \
	\
	# Test 14: zero input — uniform output, no prototype drift \
	cell_z = SoftWTACell(16, 4, temperature=0.3, lr=0.05); \
	rng_z = np.random.default_rng(7); \
	cz = [torch.tensor(c, dtype=torch.float32) for c in \
	    (rng_z.standard_normal((4, 16))).tolist()]; \
	cz = [c / c.norm() for c in cz]; \
	for _ in range(3000): \
	    ci = rng_z.integers(4); \
	    xc = cz[ci] + torch.randn(16) * 0.1; \
	    cell_z.update(xc); \
	protos_before = cell_z.prototypes.clone(); \
	zero_x = torch.zeros(16); \
	pz = cell_z.forward(zero_x); \
	assert abs(pz.max().item() - pz.min().item()) < 1e-3, \
	    f'zero input should give uniform output, got range {pz.max()-pz.min()}'; \
	for _ in range(5000): \
	    cell_z.update(zero_x); \
	drift_z = (cell_z.prototypes - protos_before).norm(dim=1); \
	assert drift_z.max().item() < 0.01, \
	    f'prototypes drifted {drift_z.max():.4f} after 5k zero frames'; \
	\
	print('All tests passed.')"

clean:
	rm -rf $(VENV) __pycache__ *.pyc
