# Experiment 00004: CUDA Probe

**Date:** 2026-02-03  
**Status:** ✅ Complete

## Objective

Verify CUDA GPU is accessible and working from C code.

## Results

```
=== GPU Probe - CUDA Test ===

CUDA Devices found: 1

Device 0: NVIDIA GeForce RTX 4090
  Compute capability: 8.9
  Total memory: 23.52 GB
  SM count: 128
  Max threads per block: 1024
  Max threads per SM: 1536
  Warp size: 32
  Clock rate: 2.52 GHz
  Memory clock: 10.50 GHz
  Memory bus width: 384 bits

✓ Verification PASSED: all 1048576 elements correct!
Throughput: 254.59 billion elements/sec

✓ CUDA is working!
```

## GPU Specs (RTX 4090)

| Spec | Value |
|------|-------|
| Architecture | Ada Lovelace |
| Compute Capability | 8.9 |
| VRAM | 23.52 GB |
| Streaming Multiprocessors | 128 |
| CUDA Cores | ~16,384 |
| Memory Bus | 384-bit |
| Memory Bandwidth | ~1 TB/s |

## What the Probe Tests

1. **Device detection** - cudaGetDeviceCount
2. **Property query** - cudaGetDeviceProperties
3. **Memory allocation** - cudaMalloc (12MB total)
4. **Host-to-device copy** - cudaMemcpy H2D
5. **Kernel execution** - vectorAdd kernel
6. **Device-to-host copy** - cudaMemcpy D2H
7. **Result verification** - CPU check
8. **Benchmark** - 1000 kernel launches

## How to Run

```bash
cd ~/research/llm-fundamentals/00004-cuda-probe
nvcc -o gpu_probe src/gpu_probe.cu -O3
./gpu_probe
```

## Files

```
00004-cuda-probe/
├── README.md
└── src/
    └── gpu_probe.cu   # CUDA test program
```

## Next Steps

- Build llm.c with CUDA support
- Run GPU-accelerated training (~100x faster than CPU)
