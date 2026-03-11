# ts-00006: 3D Reconstruction

**Date:** 2026-03-10
**Status:** In progress

## Hypothesis

The Euclidean ContinuousDrift solver generalizes from 2D to 3D: if neurons are arranged in a 3D volume with 3D proximity neighbors, the solver with dims=3 will specialize all 3 dimensions to capture x, y, z — reconstructing the 3D structure.

## Method

1. Generate an 80x80x80 volume containing a tree shape (3 layered cones for canopy, cylinder trunk, star on top). ~8% filled, values 0-1.
2. Compute 3D top-K neighbors using KDTree on grid coordinates (k=24).
3. Run Euclidean ContinuousDrift: positions lerp toward neighbor centroids, LayerNorm per dimension.
4. Render via 3D Voronoi: each grid cell gets nearest neuron's voxel value.
5. Visualize as matplotlib 3D scatter from multiple angles.

## Key: PyTorch GPU solver

Rewrote the drift solver in PyTorch (`solvers/drift_torch.py`) for GPU acceleration.

**Benchmark (RTX 4090):**
| Neurons | Mode | Per tick (CPU) | Per tick (GPU) | Speedup |
|---------|------|---------------|----------------|---------|
| 6,400 | Euclidean D2 | ~0.001s | 0.00026s | ~4x |
| 512,000 | Euclidean D3 | 0.510s | 0.0026s | **~200x** |

The GPU solver (`DriftSolver`) supports both Euclidean and dot product (word2vec) modes with a unified API.

## Results

### Test 1: 20x20x20 tree, CPU, 2k ticks
- Command: `python run_3d.py tree_20.npy -f 2000 --save-every 200 -o output_6_test1 -s 1`
- Tree shape visible by tick 1000. Clear cone + trunk structure.
- 8000 neurons, fast on CPU.

### Test 2: 80x80x80 tree, CPU, 5k ticks
- Command: `python run_3d.py tree_80.npy -f 5000 --save-every 500 -o output_6_test2 -s 3`
- 512k neurons, ~0.5s/tick → would take ~42 min for 5k ticks.
- Voxel count dropped (418→94) with subsample=3 — too aggressive subsampling.
- Killed in favor of GPU run.

### Test 3: 80x80x80 tree, GPU, 5k ticks
- Command: `python run_3d_gpu.py tree_80.npy -f 5000 --save-every 500 -o output_6_test3 -s 2`
- **36 seconds total** (vs ~42 min CPU estimate).
- 3D structure forming — canopy mass visible, separation starting.

### Test 4: 80x80x80 tree, GPU, 20k ticks
- Command: `python run_3d_gpu.py tree_80.npy -f 20000 --save-every 2000 -o output_6_test4 -s 2`
- 94 seconds total.
- Tree shape emerging: cone canopy on top, trunk going down.
- Needs more ticks for full convergence.

### Test 5: 80x80x80 tree, GPU, 100k ticks
- Command: `python run_3d_gpu.py tree_80.npy -f 100000 --save-every 1000 -o output_6_test5 -s 2`
- 18 min total. Voxel count peaked at ~3600 (tick 11k) then declined to ~1400.
- Tree shape visible but compact. Subsample=2 loses detail at 80³.

### Test RGB-1: RGB cube 80³, GPU, D3, 20k ticks
- Command: `python run_3d_gpu.py rgb_cube_80.npy -f 20000 --save-every 2000 -o output_6_rgb_test1 -s 4`
- RGB cube: x→red, y→green, z→blue. Every cell filled (no sparsity issue).
- 121 seconds. Color gradients forming by tick 2000, good structure by tick 20k.
- Smooth transitions but still some noise in center.

### Test RGB-2: RGB cube 80³, GPU, D4, 500k ticks (1000 frames)
- Command: `python run_3d_gpu.py rgb_cube_80.npy -f 500000 --save-every 500 -o output_6_rgb_test2 -s 4 --dims 4`
- 8 hours total (rendering 1000 frames was the bottleneck, solver itself ~25 min).
- **Excellent reconstruction.** Smooth color gradients in all 3 directions at 500k ticks.
- Corners show correct colors: black(0,0,0), red(x), green(y), blue(z), white(max).
- D4 solver correctly discovered all 3 spatial dimensions; 4th dimension ignored by PCA during rendering.
- At tick 50k already good; tick 500k has tighter, smoother gradients.

## RGB cube as benchmark

The RGB cube (x→R, y→G, z→B) is a better 3D benchmark than the tree:
- **Every cell filled** — no sparsity collapse issues.
- **Color encodes position** — reconstruction quality visible at a glance without needing to recognize shapes.
- **Tests all 3 axes simultaneously** — any axis that fails shows immediately as wrong colors.
- **Works with any volume size** — just `generate_rgb_cube.py` with different size.

## Files

- `generate_tree_3d.py` — Generate 3D tree volumes (.npy)
- `generate_rgb_cube.py` — Generate RGB cube volumes (.npy)
- `view_3d.py` — 3D matplotlib scatter renderer (multi-angle, scalar + RGB)
- `run_3d.py` — CPU runner (numpy, ContinuousDrift-style)
- `run_3d_gpu.py` — GPU runner (PyTorch DriftSolver, scalar + RGB)
- `solvers/drift_torch.py` — PyTorch GPU drift solver (Euclidean + dot product modes)
- `tree_80.npy` — 80x80x80 tree volume
- `rgb_cube_80.npy` — 80x80x80 RGB gradient cube

## Findings

1. **Euclidean ContinuousDrift generalizes to 3D.** Same algorithm, same LayerNorm normalization, same Voronoi rendering — just with 3D coordinates and dims=3.
2. **GPU acceleration is essential for 3D.** 512k neurons at 0.5s/tick on CPU vs 0.003s/tick on GPU (RTX 4090). ~200x speedup makes interactive experimentation possible.
3. **PyTorch solver matches numpy solver behavior.** Same LayerNorm, same margin/tanh dead zone, identical stats (std=1.0000).
4. **Extra dimensions are harmless.** D4 solver on a 3D signal correctly discovers the 3 spatial axes; the 4th dimension becomes noise that PCA ignores.
5. **RGB cube is the ideal 3D benchmark.** Color-encodes position, fully filled (no sparsity), all axes tested simultaneously.
6. **Rendering is the bottleneck at scale.** 1000 frames of 3D matplotlib scatter took ~8 hours; the 500k solver ticks themselves took ~25 min.
