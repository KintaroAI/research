# Task 00005: Training Resume Support

**Date:** 2026-02-25
**Status:** Done

## Context

Checkpoints currently save only model weights. When resuming training from a
checkpoint, the optimizer state (Adam m/v), step counter, dataloader position,
and LR schedule all reset — so "resuming at 20k for 30k more steps" is not
equivalent to a continuous 50k run. We need full training state save/restore.

Discovered this when continuing 355M warmup checkpoints from 20k to 50k steps
in experiment 00019 — each segment got a fresh warmup and optimizer reset.

## Deliverables

- Save/restore step count, dataloader position, and Adam optimizer state
- New `-R` CLI flag to enable resume mode
- Backward compatible: old checkpoints and runs without `-R` behave identically

## Existing Infrastructure

- **Header slots [8-255]** are unused and zeroed — free for step/dataloader metadata
- **`dataloader_resume()`** in `llmc/dataloader.h` (line 233) already exists but is never called
- **Sidecar pattern** established by `.sort` and `.blend` files — use same for `.optim`
- Checkpoint save called at periodic intervals (line 3088) and at end (line 3195)

## Implementation

### What to save

| Data | Where | Size |
|------|-------|------|
| Step count | Header slot [8] | 4 bytes |
| Dataloader shard_idx | Header slot [9] | 4 bytes |
| Dataloader sample_idx | Header slot [10] | 4 bytes |
| Adam m (main params) | `.optim` sidecar | num_parameters × 4 bytes |
| Adam v (main params) | `.optim` sidecar | num_parameters × 4 bytes |

Sort/blend optimizer state (sort_m/v, blend_m/v) is tiny (<100 floats) —
append to `.optim` sidecar after main m/v.

### File to modify

`llm-fundamentals/dev/src/train_gpt2_fp32.cu`

### Changes

#### 1. Add CLI flag and variable (~line 2896)

```c
int resume = 0;  // 0 = fresh start, 1 = resume from checkpoint state
```

Flag parsing: `else if (argv[i][1] == 'R') { resume = atoi(argv[i+1]); }`
Help text: `fprintf(stderr, "  -R <int>    resume training from checkpoint state (default = 0, off)\n");`
Config table: `printf("| resume                | %-50d |\n", resume);`

#### 2. Extend `gpt2_save_checkpoint` signature (~line 1745)

Add step and dataloader state parameters:

```c
void gpt2_save_checkpoint(GPT2 *model, const char* checkpoint_path,
                          int step, int current_shard_idx, int current_sample_idx)
```

Write step/dataloader to header slots [8-10] before writing.

#### 3. Save optimizer state in `gpt2_save_checkpoint` (~after line 1770)

After writing model weights, save `.optim` sidecar with m and v (same
GPU→CPU→fwrite pattern as main params). Also write sort/blend m/v if present.

#### 4. Add optimizer load function (~after gpt2_build_from_checkpoint)

`gpt2_load_optimizer_state(GPT2 *model, const char* checkpoint_path)` —
reads `.optim` sidecar, allocates m/v on GPU, copies data.

#### 5. Resume logic in main() (~after model build, line 2997)

Re-read header slots [8-10] from checkpoint. If resume_step > 0:
- Call `dataloader_resume()` with saved shard/sample indices
- Call `gpt2_load_optimizer_state()` to restore Adam state
- Print resume info

#### 6. Training loop starts from resume_step (~line 3064)

Change: `for (int step = 0; ...)` → `for (int step = resume_step; ...)`

#### 7. Update all checkpoint save call sites

Pass step and dataloader state to extended `gpt2_save_checkpoint` at both
periodic (line 3089) and final (line 3196) save points.

## Backward Compatibility

- Old checkpoints have header[8-10] = 0 → resume_step = 0 → no resume (safe)
- Without `-R`, behavior is identical to current code
- `.optim` sidecar is optional — if missing and `-R` is set, print warning and
  start with fresh optimizer
- `generate.cu` reads only header[0-7] and weights — unaffected

## Verification

1. `make train` — builds
2. `make test` — passes (all changes inside `#ifndef TESTING` main)
3. Train 100 steps with `-k 50 -c test_ckpt.bin`, verify `.optim` sidecar created
4. Resume with `-R 1 -e test_ckpt.bin -n 100`, verify:
   - Log says "Resuming from step 50"
   - Step counter starts at 50
   - LR schedule is continuous
5. Compare: 100-step continuous run vs 50+50 resumed run — val loss should be
   identical (or very close, modulo atomicAdd nondeterminism)
