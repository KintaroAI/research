# Task 00006: Banded Matmul Kernels (True Sparse Compute)

**Date:** 2026-02-26
**Status:** Done

## Context

Banded sparsity currently uses full dense matmuls with post-hoc masking — zero
weights outside the band are still multiplied. With bandwidth=512 on C=1024,
FC density is only 12%, but we do 100% of the compute. Custom banded kernels
that skip out-of-band elements give actual speedup.

**Key insight:** Conv1d does NOT map to this — convolution has weight sharing
(same kernel across positions), but banded FC has unique weights per output
neuron. This is a "locally connected layer." Need custom CUDA kernels.

## Deliverables

- Banded forward kernel: only iterates within the band per output row
- Banded backward-to-input kernel: same idea for gradient propagation
- Backward-to-weight stays cuBLAS + mask (hard to beat cuBLAS for reductions)
- Correctness tests: banded kernels must match dense+mask exactly
- Runtime switch: banded kernels used when bandwidth > 0, dense path unchanged

## Design Decisions

### Keep full (OC, IC) weight storage — no compact format
- Zero changes to ParameterTensors, checkpoint format, optimizer state
- Existing mask-based path stays as correctness reference
- Memory cost of storing zeros is tolerable (768 MB on 24 GB GPU)
- Compact storage (OC, BW) can be a follow-up for memory savings

### Backward-to-weight stays dense (cuBLAS + mask)
- This is a reduction over BT — cuBLAS is extremely optimized for this shape
- A custom kernel would need atomicAdd or multi-pass reduction
- Mask application is cheap (element-wise multiply)
- Forward + backward-to-input are where the real savings are

## Compute Savings (BW=512, C=1024, BT=4096)

| Operation | Dense FLOPs | Banded FLOPs | Speedup |
|-----------|------------|-------------|---------|
| FC1 fwd (C→4C) | BT×4C×C | BT×4C×BW | 2x |
| FC1 bwd-input | BT×C×4C | BT×C×2BW | 2x |
| FC1 bwd-weight | BT×4C×C | same (cuBLAS+mask) | 1x |
| FC2 fwd (4C→C) | BT×C×4C | BT×C×BW | 8x |
| FC2 bwd-input | BT×4C×C | BT×4C×(BW/4) | 8x |
| FC2 bwd-weight | BT×C×4C | same (cuBLAS+mask) | 1x |

FC2 benefits much more because input dimension (4C) is 4x larger than band.
Expected end-to-end training step speedup: **10-20%**.

## File to modify

`llm-fundamentals/dev/src/train_gpt2_fp32.cu`

## Implementation

### 1. Banded forward kernel (~after line 1240)

```c
__global__ void banded_matmul_forward_kernel(
    float* out, const float* inp, const float* weight, const float* bias,
    int BT, int IC, int OC, int bandwidth, float ic_over_oc)
```

One thread per output element `out[bt][j]`. For output row `j`:
- Compute band center: `i_center = j * IC / OC`
- Iterate `col_start..col_end` (bandwidth elements clamped to [0,IC))
- Sum `weight[j][i] * inp[bt][i]` + bias

Band boundary formula must exactly match `create_band_mask_fc1/fc2_kernel`
to ensure correctness: `fabsf((float)i - i_center) <= bandwidth / 2.0f`.

Wrapper function:
```c
void banded_matmul_forward(float* out, const float* inp, const float* weight,
                           const float* bias, int B, int T, int IC, int OC,
                           int bandwidth)
```

### 2. Banded backward-to-input kernel (~after banded forward)

```c
__global__ void banded_matmul_backward_input_kernel(
    float* dinp, const float* dout, const float* weight,
    int BT, int IC, int OC, int bandwidth, float ic_over_oc)
```

One thread per input element `dinp[bt][i]`. For input column `i`:
- Compute reverse band: which output rows `j` include `i` in their band
- `j_start..j_end` = inverse of the forward band mapping
- Sum `dout[bt][j] * weight[j][i]` over the reverse band

### 3. Combined backward launcher (~after line 1352)

```c
void banded_matmul_backward(float* dinp, float* dweight, float* dbias,
                            float* dout, float* inp, float* weight, float* mask,
                            int B, int T, int IC, int OC, int bandwidth)
```

- dinp: banded backward-to-input kernel (actual savings)
- dweight: cuBLAS SGEMM + apply_mask_kernel (same as current)
- dbias: existing matmul_backward_bias_kernel4 (same as current)

### 4. Integration into gpt2_forward (~line 2432)

```c
if (BANDWIDTH_FC1 > 0) {
    banded_matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, BANDWIDTH_FC1);
} else {
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
}
```

Same pattern for FC2 and in gpt2_backward.

### 5. generate.cu (follow-up, not this task)

Port banded forward kernel for inference speedup.

## Tests

Add to `test_banded_sparsity.cu`:

- **Banded forward matches dense+mask** — apply mask to weights, run both
  kernels, compare outputs (should be bit-exact or very close)
- **Banded backward-to-input matches dense** — same pattern for dinp
- **End-to-end training equivalence** — 5 steps old vs new, losses match (1e-2)
- **Edge cases** — bandwidth=IC (matches dense), bandwidth=1, non-power-of-2

## Verification

```bash
make test_banded_sparsity && ./test_banded_sparsity  # all tests pass
make train && ./train -e model.bin -1 256 -2 256 -n 100  # trains without crash
```

Compare step times with/without banded kernels to measure actual speedup.

## Risks

- **FC1 banded may be slower than dense** — only 2x FLOPs reduction, naive
  kernel has worse memory patterns than optimized `matmul_forward_kernel4`.
  Mitigation: keep dense for FC1 if banded is slower, focus on FC2 (8x win).
- Floating-point summation order differs → losses not bit-exact. Use 1e-2 tol.
