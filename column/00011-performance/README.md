# 00011 — Performance Characterization

**Status:** Complete

## Goal

Document time and space complexity of SoftWTACell. Pure CPU, no GPU.

## Complexity

### Space: `O(m × n)`

| Config (n×m) | Prototypes | Usage | Total |
|---|---|---|---|
| 16×4 | 256 B | 16 B | 272 B |
| 64×16 | 4,096 B | 64 B | 4.1 KB |
| 256×32 | 32,768 B | 128 B | 32.1 KB |
| 1024×64 | 262,144 B | 256 B | 256.3 KB |

### Time per step

| Mode | Complexity | Bottleneck |
|---|---|---|
| Instantaneous | `O(m × n)` | matmul `x @ prototypes.T` |
| Correlation | `O(n²T + mn²)` | covariance `x_c @ x_c.T` |

## Measured throughput (CPU, single thread, PyTorch)

| Config | Forward | Update | Total | Steps/s |
|---|---|---|---|---|
| instant 16×4 | 38 us | 156 us | 195 us | 5,132 |
| instant 16×8 | 40 us | 154 us | 194 us | 5,161 |
| instant 64×16 | 41 us | 155 us | 196 us | 5,099 |
| instant 256×32 | 42 us | 159 us | 202 us | 4,961 |
| corr 16×4 T=10 | 49 us | 207 us | 256 us | 3,906 |
| corr 16×8 T=10 | 49 us | 208 us | 257 us | 3,890 |
| corr 64×16 T=10 | 18,676 us | 8,931 us | 27,607 us | 36 |

## Analysis

**Instantaneous mode is flat ~200us** regardless of size (16×4 to 256×32). PyTorch
CPU overhead dominates at these small tensor sizes — the actual matmul is negligible.
~5,000 steps/s is sufficient for online learning.

**Forward is cheap (~40us), update is the bottleneck (~155us).** The Hebbian pull +
re-normalization + usage counter update costs 4x more than the similarity computation.

**Correlation mode scales as O(n²).** At 16 inputs, the overhead is small (+30%).
At 64 inputs with T=10, the covariance matrix (64×64 from 64×10 matmul) takes 18ms
alone — drops throughput to 36 steps/s.

**The cell is tiny.** Even 1024×64 fits in 256 KB. No GPU needed — transfer overhead
would exceed the computation. A CUDA C kernel (as in llm-fundamentals) would eliminate
the PyTorch overhead for instantaneous mode, potentially reaching ~100,000 steps/s.

**Practical limits:**
- Instantaneous mode: limited by PyTorch overhead, not by n or m
- Correlation mode: limited by O(n²T) covariance — keep n ≤ ~30 for real-time use
- Memory is never a concern at these scales
