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
	assert 'consistency' in sqm; \
	\
	# Test 13b: winner consistency \
	from metrics import winner_consistency; \
	assert abs(winner_consistency(w_perfect, l_perfect) - 1.0) < 1e-6, \
	    'perfect assignment should have consistency 1.0'; \
	w_scattered = np.array([0,1,2,3,1,2,3,0,2,3,0,1]); \
	assert winner_consistency(w_scattered, l_perfect) < 0.5, \
	    'scattered assignment should have low consistency'; \
	\
	# Test 13c: lock-in score \
	from metrics import lock_in_score; \
	fake_snaps = [ \
	    {'nmi': 0.2, 'frame': 1000}, \
	    {'nmi': 0.7, 'frame': 2000}, \
	    {'nmi': 0.85, 'frame': 3000}, \
	    {'nmi': 0.9, 'frame': 4000}, \
	    {'nmi': 0.91, 'frame': 5000}]; \
	lv, lf = lock_in_score(fake_snaps, 'nmi'); \
	assert abs(lv - 0.905) < 0.01, f'lock-in value wrong: {lv}'; \
	assert lf == 3000, f'lock-in frame wrong: {lf}'; \
	\
	# Test 14: temporal — forward produces valid probs for (n, T) input \
	cell_t = SoftWTACell(16, 4, temporal_mode='correlation'); \
	xt = torch.randn(16, 10); \
	pt = cell_t.forward(xt); \
	assert pt.shape == (4,), f'temporal bad shape: {pt.shape}'; \
	assert abs(pt.sum().item() - 1.0) < 1e-5, f'temporal probs sum to {pt.sum()}'; \
	assert (pt >= 0).all(), 'temporal negative probs'; \
	\
	# Test 15: temporal — update returns valid winner \
	wt, mqt = cell_t.update(xt); \
	assert 0 <= wt < 4, f'temporal bad winner: {wt}'; \
	assert 0.0 <= mqt <= 1.0, f'temporal bad match quality: {mqt}'; \
	\
	# Test 16: temporal — separates co-variation clusters \
	rng_t = np.random.default_rng(42); \
	dirs = rng_t.standard_normal((4, 16)); \
	dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True); \
	cell_tc = SoftWTACell(16, 4, temperature=0.5, lr=0.05, \
	    match_threshold=0.1, temporal_mode='correlation'); \
	cluster_winners_t = {i: [] for i in range(4)}; \
	for _ in range(3000): \
	    ci = rng_t.integers(4); \
	    sig = rng_t.standard_normal(10); \
	    trace = np.outer(dirs[ci], sig) + rng_t.standard_normal((16, 10)) * 0.3; \
	    xt2 = torch.tensor(trace, dtype=torch.float32); \
	    wt2, _ = cell_tc.update(xt2); \
	    cluster_winners_t[ci].append(wt2); \
	modes_t = set(); \
	for ci in range(4): \
	    last = cluster_winners_t[ci][-200:]; \
	    mode = max(set(last), key=last.count); \
	    modes_t.add(mode); \
	assert len(modes_t) >= 3, \
	    f'temporal poor separation: only {len(modes_t)} distinct modes'; \
	\
	# Test 17: temporal — zero trace produces uniform output \
	cell_t2 = SoftWTACell(16, 4, temporal_mode='correlation'); \
	pz_t = cell_t2.forward(torch.zeros(16, 10)); \
	assert abs(pz_t.max().item() - pz_t.min().item()) < 1e-3, \
	    f'temporal zero should be uniform, range={pz_t.max()-pz_t.min()}'; \
	\
	# Test 18: temporal — state roundtrip preserves temporal_mode \
	sd_t = cell_tc.state_dict(); \
	cell_tc2 = SoftWTACell.from_state_dict(sd_t); \
	assert cell_tc2.temporal_mode == 'correlation'; \
	assert torch.allclose(cell_tc.prototypes, cell_tc2.prototypes); \
	\
	# Test 19: zero input (instantaneous) — uniform output, no prototype drift \
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
	# Test 20: extend_inputs — zero disruption \
	cell_ext = SoftWTACell(8, 4, temperature=0.3, lr=0.05); \
	rng_ext = np.random.default_rng(20); \
	cx = [torch.tensor(c, dtype=torch.float32) for c in \
	    rng_ext.standard_normal((4, 8)).tolist()]; \
	cx = [c / c.norm() for c in cx]; \
	for _ in range(2000): \
	    ci = rng_ext.integers(4); \
	    cell_ext.update(cx[ci] + torch.randn(8) * 0.1); \
	x_test = cx[0] + torch.randn(8) * 0.1; \
	w_before_ext = cell_ext.forward(x_test).argmax().item(); \
	protos_old = cell_ext.prototypes.clone(); \
	cell_ext.extend_inputs(2); \
	assert cell_ext.n_inputs == 10, 'extend_inputs wrong n_inputs'; \
	assert cell_ext.prototypes.shape == (4, 10), 'extend_inputs wrong shape'; \
	x_test_ext = torch.cat([x_test, torch.zeros(2)]); \
	w_after_ext = cell_ext.forward(x_test_ext).argmax().item(); \
	assert w_before_ext == w_after_ext, 'extend_inputs changed winner'; \
	\
	# Test 21: remove_inputs — preserves n_inputs and re-normalizes \
	cell_rm = SoftWTACell(8, 4, temperature=0.3, lr=0.05); \
	cell_rm.remove_inputs([6, 7]); \
	assert cell_rm.n_inputs == 6, 'remove_inputs wrong n_inputs'; \
	assert cell_rm.prototypes.shape == (4, 6), 'remove_inputs wrong shape'; \
	norms_rm = cell_rm.prototypes.norm(dim=1); \
	assert (norms_rm - 1.0).abs().max() < 1e-5, 'remove_inputs not normalized'; \
	\
	# Test 22: streaming — forward produces valid probs \
	cell_s = SoftWTACell(16, 4, temporal_mode='streaming'); \
	xs = torch.randn(16); \
	ps = cell_s.forward(xs); \
	assert ps.shape == (4,), f'streaming bad shape: {ps.shape}'; \
	assert abs(ps.sum().item() - 1.0) < 1e-5, 'streaming probs bad sum'; \
	assert (ps >= 0).all(), 'streaming negative probs'; \
	\
	# Test 23: streaming — update returns valid winner \
	ws, mqs = cell_s.update(xs); \
	assert 0 <= ws < 4, f'streaming bad winner: {ws}'; \
	assert 0.0 <= mqs <= 1.0, f'streaming bad match quality: {mqs}'; \
	\
	# Test 24: streaming — separates co-variation clusters \
	rng_s = np.random.default_rng(42); \
	dirs_s = rng_s.standard_normal((4, 16)); \
	dirs_s = dirs_s / np.linalg.norm(dirs_s, axis=1, keepdims=True); \
	cell_sc = SoftWTACell(16, 4, temperature=0.5, lr=0.05, \
	    match_threshold=0.1, temporal_mode='streaming', streaming_decay=0.9); \
	cluster_ws = {i: [] for i in range(4)}; \
	for _ in range(5000): \
	    ci = rng_s.integers(4); \
	    sig = rng_s.standard_normal(); \
	    x_s = torch.tensor(dirs_s[ci] * sig + rng_s.standard_normal(16) * 0.3, \
	        dtype=torch.float32); \
	    w_s, _ = cell_sc.update(x_s); \
	    cluster_ws[ci].append(w_s); \
	modes_s = set(); \
	for ci in range(4): \
	    last = cluster_ws[ci][-300:]; \
	    mode = max(set(last), key=last.count); \
	    modes_s.add(mode); \
	assert len(modes_s) >= 3, \
	    f'streaming poor separation: only {len(modes_s)} distinct modes'; \
	\
	# Test 25: streaming — accepts (n, T) trace \
	cell_st = SoftWTACell(16, 4, temporal_mode='streaming'); \
	xt_s = torch.randn(16, 10); \
	pt_s = cell_st.forward(xt_s); \
	assert pt_s.shape == (4,), f'streaming (n,T) bad shape: {pt_s.shape}'; \
	assert abs(pt_s.sum().item() - 1.0) < 1e-5, 'streaming (n,T) bad sum'; \
	wt_s, mqt_s = cell_st.update(xt_s, pt_s); \
	assert 0 <= wt_s < 4; \
	\
	# Test 26: streaming (n,T) separates co-variation clusters \
	rng_st = np.random.default_rng(99); \
	dirs_st = rng_st.standard_normal((4, 16)); \
	dirs_st = dirs_st / np.linalg.norm(dirs_st, axis=1, keepdims=True); \
	cell_st2 = SoftWTACell(16, 4, temperature=0.5, lr=0.05, \
	    match_threshold=0.1, temporal_mode='streaming'); \
	cw_st = {i: [] for i in range(4)}; \
	for _ in range(5000): \
	    ci = rng_st.integers(4); \
	    sig = rng_st.standard_normal(10); \
	    trace = np.outer(dirs_st[ci], sig) + rng_st.standard_normal((16, 10)) * 0.3; \
	    xt_s2 = torch.tensor(trace, dtype=torch.float32); \
	    w_st2, _ = cell_st2.update(xt_s2); \
	    cw_st[ci].append(w_st2); \
	modes_st = set(); \
	for ci in range(4): \
	    last = cw_st[ci][-300:]; \
	    mode = max(set(last), key=last.count); \
	    modes_st.add(mode); \
	assert len(modes_st) >= 3, \
	    f'streaming (n,T) poor separation: only {len(modes_st)} modes'; \
	\
	# Test 27: streaming — state roundtrip \
	sd_s = cell_st2.state_dict(); \
	cell_st3 = SoftWTACell.from_state_dict(sd_s); \
	assert cell_st3.temporal_mode == 'streaming'; \
	assert torch.allclose(cell_st2.prototypes, cell_st3.prototypes); \
	assert torch.allclose(cell_st2.proj_mean, cell_st3.proj_mean); \
	assert torch.allclose(cell_st2.proj_var, cell_st3.proj_var); \
	\
	print('All tests passed.')"

clean:
	rm -rf $(VENV) __pycache__ *.pyc
