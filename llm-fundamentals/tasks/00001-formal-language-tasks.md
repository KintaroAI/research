# Task 00001: Formal Language Tasks (Eval Roadmap 2b)

**Date:** 2026-02-21
**Status:** Planned

## Context

Experiment 00014 established a sequential eval protocol on modular arithmetic and showed complete catastrophic forgetting in the dense model. The all-position ablation (no `-p 4`) further showed that position masking amplifies forgetting — shared sequence structure provides a partial anchor.

The next eval roadmap item is **2b: formal language tasks** — parity, copy, reverse, majority. These test different algorithmic primitives (counting, identity mapping, position reversal, thresholding) using the same infrastructure. They complement modular arithmetic by testing structurally different computations.

## Deliverables

1. **`gen_formal_data.py`** — data generator for all 4 tasks
2. **`-P` flag in `train_gpt2_fp32.cu`** — multi-position loss masking for copy/reverse
3. **`formal` preset in `create_model.py`** — block_size=24 to handle longer sequences
4. **EVAL.md update** — Phase 2b protocol

## Task Designs

All tasks use the existing vocab_size=101. Special tokens: BOS=99, EOS=100, EQ=98, SEP=97. Value tokens: 0,1 (binary alphabet).

| Task | Sequence (buffer) | T | `-p` | Examples |
|------|-------------------|---|------|----------|
| Parity-8 | `BOS b1..b8 EQ ans EOS` + 8 EOS pad | 20 | `-p 9` | 256 (2^8) |
| Copy-8 | `BOS x1..x8 SEP x1..x8 EOS EOS` | 20 | `-p 9 -P 16` | 256 |
| Reverse-8 | `BOS x1..x8 SEP x8..x1 EOS EOS` | 20 | `-p 9 -P 16` | 256 |
| Majority-9 | `BOS t1..t9 EQ ans EOS` + 6 EOS pad | 20 | `-p 10` | 512 (2^9) |

### Design rationale

- **T=20 for all tasks**: The train binary locks (B,T) on first forward pass, so sequential eval requires uniform T. T=20 is the natural length for copy/reverse-8 (2*8+4=20) and divisible by 4 (attention kernel requires `T%4==0`). Shorter tasks are padded with trailing EOS.
- **Binary alphabet**: Gives exhaustive enumeration (256-512 examples), matching the grokking paradigm of training on the complete dataset. Can scale up later with `--alphabet-size`.
- **50/50 train/val split**: Matches modular arithmetic protocol.
- **Full-batch training**: B=127 (128 train examples, need B*T+1 tokens) or B=255 (majority). Same grokking dynamics as modular arithmetic.
- **Majority uses odd input length (9)**: Guarantees no ties with binary alphabet.

### Token layout detail

```
Parity-8 (T=20):
  buffer: [BOS b1 b2 b3 b4 b5 b6 b7 b8 EQ ans EOS EOS EOS EOS EOS EOS EOS EOS EOS]
  index:    0   1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19
  targets[9] = buffer[10] = answer → task_position = 9

Copy-8 (T=20):
  buffer: [BOS x1 x2 x3 x4 x5 x6 x7 x8 SEP x1 x2 x3 x4 x5 x6 x7 x8 EOS EOS]
  index:    0   1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17  18  19
  targets[9..16] = buffer[10..17] = x1..x8 → task_position = 9, task_position_end = 16

Reverse-8 (T=20):
  buffer: [BOS x1 x2 x3 x4 x5 x6 x7 x8 SEP x8 x7 x6 x5 x4 x3 x2 x1 EOS EOS]
  (same positions as copy, reversed output)

Majority-9 (T=20):
  buffer: [BOS t1 t2 t3 t4 t5 t6 t7 t8 t9 EQ ans EOS EOS EOS EOS EOS EOS EOS EOS]
  index:    0   1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19
  targets[10] = buffer[11] = answer → task_position = 10
```

## Implementation

### 1. `gen_formal_data.py` (new file)

Follow `gen_modular_data.py` pattern. CLI:
```bash
python gen_formal_data.py --task parity --input-len 8 --seed 42 --output-dir data/parity_8
python gen_formal_data.py --task copy --input-len 8 --seed 42 --output-dir data/copy_8
python gen_formal_data.py --task reverse --input-len 8 --seed 42 --output-dir data/reverse_8
python gen_formal_data.py --task majority --input-len 9 --seed 42 --output-dir data/majority_9
```

Key args: `--task`, `--input-len`, `--alphabet-size` (default 2), `--train-frac` (default 0.5), `--pad-to` (default 20), `--seed`, `--output-dir`.

Exhaustive enumeration for small datasets (alphabet^input_len), shuffled with seed, split into train/val. Pad all sequences to `--pad-to` with EOS. Print 3 sample sequences and the suggested training command.

Binary format: same header (magic=20240520, version=1, num_tokens as int64) + uint16 tokens.

### 2. `src/train_gpt2_fp32.cu` — add `-P` flag (~20 lines changed)

Add `task_position_end` field to GPT2 struct alongside `task_position`. New flag `-P <int>` (default -1, meaning same as `-p`). Changes:

- **Struct** (line ~1489): add `int task_position_end;`
- **Init** (line ~1556): `model->task_position_end = -1;`
- **Arg parsing** (line ~2575): parse `-P`, normalize: if `-p` set but `-P` not, set `task_position_end = task_position`
- **Mask construction** (lines 2109-2118): loop from `task_position` to `task_position_end`, weight = `1/(B * n_positions)`
- **Mean loss** (lines 2127-2131): sum over position range, divide by `B * n_positions`
- **Set on model** (line ~2637): `model.task_position_end = task_position_end;`
- **Print** (line ~2604): print both values

Fully backward-compatible — without `-P`, behavior identical to current.

### 3. `create_model.py` — add preset (1 line)

```python
'formal': dict(block_size=24, vocab_size=101, n_layer=2, n_head=4, n_embd=128),
```

Same architecture as grokking but block_size=24 (supports T up to 24).

### 4. `EVAL.md` — add Phase 2b section

Document the formal language sequential eval protocol with exact commands:
- Data generation
- Model creation (`--preset formal`)
- 4-phase sequential training (parity → copy → reverse → majority)
- Cross-task eval after each phase (with per-task `-p`/`-P` and `-t` values)

## Verification

1. **Data gen**: Generate all 4 datasets, inspect samples, verify token values < 101 and sequence lengths = 20
2. **`-P` flag regression**: Run modular arithmetic with only `-p 4` (no `-P`), verify identical loss to before
3. **`-P` flag new**: Train copy-8 with `-p 9 -P 16`, verify loss is computed correctly
4. **Individual task training**: Train each task standalone to verify it's learnable
5. **Sequential eval**: Run full 4-phase protocol, record forgetting matrix

## Key constraints discovered during planning

- `T % 4 == 0` — hard assert in attention kernel (`train_gpt2_fp32.cu:588`)
- B,T locked on first `gpt2_forward()` call — sequential eval phases must share the same T (`train_gpt2_fp32.cu:2005-2008`)
- `task_position` is currently single-int — copy/reverse need the new `-P` range flag
- Dataloader requires `B*T + 1 <= num_tokens` per shard
