# Evaluation Protocol

Repeatable evaluation protocol for comparing model architectures. Run the full
protocol for each variant (dense, banded sparsity, sort layer, etc.) and compare
the resulting metrics tables.

---

## Phase 2a: Sequential Modular Arithmetic

The primary eval for architectural comparisons. Trains a single checkpoint on 4
algorithmic tasks **sequentially**, evaluating retention of all previous tasks after
each phase. This measures both learning ability and catastrophic forgetting.

### Why sequential?

A model that learns task 2 but forgets task 1 has brittle representations. A model
that retains all tasks has learned more general structure (modular arithmetic shares
algebraic structure across operations). Architecture changes that improve retention
are strong evidence of better internal representations.

### Tasks (in training order)

| # | Operation | Formula | OP token |
|---|-----------|---------|----------|
| 1 | Addition | `(a + b) mod 97` | `+` |
| 2 | Subtraction | `(a - b) mod 97` | `-` |
| 3 | Multiplication | `(a * b) mod 97` | `*` |
| 4 | Squared sum | `(a^2 + b^2) mod 97` | `S` |

All tasks: p=97, vocab_size=101, sequence format `BOS a OP b EQ c EOS EOS`,
50/50 train/val split, seed 42.

### Step 1: Generate data

```bash
python gen_modular_data.py --op add    --prime 97 --seed 42 --output-dir data/modular_add
python gen_modular_data.py --op sub    --prime 97 --seed 42 --output-dir data/modular_sub
python gen_modular_data.py --op mul    --prime 97 --seed 42 --output-dir data/modular_mul
python gen_modular_data.py --op sq_sum --prime 97 --seed 42 --output-dir data/modular_sq_sum
```

Each produces `train.bin` (4704 equations) and `val.bin` (4705 equations).

### Step 2: Create model

```bash
python create_model.py --preset grokking --prime 97 -o model_grok.bin
```

### Step 3: Sequential training + cross-task eval

Train on each task in order, carrying the checkpoint forward. After each training
phase, evaluate on **all** tasks' val sets. Training runs use `wandb_train.py` for
real-time metric logging (gracefully degrades if wandb is not installed).

Note: B=4703 (not 4704) because the dataloader requires B*T+1 tokens.

```bash
# --- Phase 1: Train on addition ---
python wandb_train.py --project gpt2-cuda --group seq-eval --name phase1-add -- \
    ./train -e model_grok.bin -c ckpt_phase1.bin \
    -i data/modular_add/train.bin -j data/modular_add/val.bin \
    -t 8 -b 4703 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
    -s 0 -p 4 -v 1000 -q 1337

# Eval phase 1 checkpoint on all available tasks
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval1-add -- \
    ./train -e ckpt_phase1.bin -i data/modular_add/train.bin -j data/modular_add/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_add

# --- Phase 2: Train on subtraction (from phase 1 checkpoint) ---
python wandb_train.py --project gpt2-cuda --group seq-eval --name phase2-sub -- \
    ./train -e ckpt_phase1.bin -c ckpt_phase2.bin \
    -i data/modular_sub/train.bin -j data/modular_sub/val.bin \
    -t 8 -b 4703 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
    -s 0 -p 4 -v 1000 -q 1337

# Eval phase 2 checkpoint on tasks 1-2
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval2-add -- \
    ./train -e ckpt_phase2.bin -i data/modular_sub/train.bin -j data/modular_add/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_add (retention)
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval2-sub -- \
    ./train -e ckpt_phase2.bin -i data/modular_sub/train.bin -j data/modular_sub/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_sub

# --- Phase 3: Train on multiplication (from phase 2 checkpoint) ---
python wandb_train.py --project gpt2-cuda --group seq-eval --name phase3-mul -- \
    ./train -e ckpt_phase2.bin -c ckpt_phase3.bin \
    -i data/modular_mul/train.bin -j data/modular_mul/val.bin \
    -t 8 -b 4703 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
    -s 0 -p 4 -v 1000 -q 1337

# Eval phase 3 checkpoint on tasks 1-3
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval3-add -- \
    ./train -e ckpt_phase3.bin -i data/modular_mul/train.bin -j data/modular_add/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_add (retention)
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval3-sub -- \
    ./train -e ckpt_phase3.bin -i data/modular_mul/train.bin -j data/modular_sub/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_sub (retention)
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval3-mul -- \
    ./train -e ckpt_phase3.bin -i data/modular_mul/train.bin -j data/modular_mul/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_mul

# --- Phase 4: Train on squared sum (from phase 3 checkpoint) ---
python wandb_train.py --project gpt2-cuda --group seq-eval --name phase4-sq_sum -- \
    ./train -e ckpt_phase3.bin -c ckpt_phase4.bin \
    -i data/modular_sq_sum/train.bin -j data/modular_sq_sum/val.bin \
    -t 8 -b 4703 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
    -s 0 -p 4 -v 1000 -q 1337

# Eval phase 4 checkpoint on all tasks
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval4-add -- \
    ./train -e ckpt_phase4.bin -i data/modular_sq_sum/train.bin -j data/modular_add/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_add (retention)
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval4-sub -- \
    ./train -e ckpt_phase4.bin -i data/modular_sq_sum/train.bin -j data/modular_sub/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_sub (retention)
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval4-mul -- \
    ./train -e ckpt_phase4.bin -i data/modular_sq_sum/train.bin -j data/modular_mul/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_mul (retention)
python wandb_train.py --project gpt2-cuda --group seq-eval --name eval4-sq_sum -- \
    ./train -e ckpt_phase4.bin -i data/modular_sq_sum/train.bin -j data/modular_sq_sum/val.bin \
    -n 0 -v 1 -t 8 -b 4703 -s 0 -p 4    # => val_sq_sum
```

**How eval-only works:** With `-n 0`, `max_steps` doesn't override
`train_num_batches` (the check is `max_steps > 0`), so the loop runs a full
epoch. However, val eval runs at step 0 before any training — this first
`val loss` line is the eval result. For small datasets (grokking/formal tasks)
the loop finishes quickly. We point `-j` at each task's val data to get the
loss. The `-p 4` flag ensures loss is computed only at the answer position
(matching training). No C changes needed.

### Step 4: Record results

Fill in the evaluation matrix (val loss for each task after each phase):

| Phase | Trained on | val_add | val_sub | val_mul | val_sq_sum |
|-------|------------|---------|---------|---------|------------|
| 1 | add | ___ | - | - | - |
| 2 | sub | ___ | ___ | - | - |
| 3 | mul | ___ | ___ | ___ | - |
| 4 | sq_sum | ___ | ___ | ___ | ___ |

Key metrics:
- **Diagonal**: learning ability (should reach low loss)
- **Below diagonal**: retention (should stay low if representations are stable)
- **Forgetting**: increase in a task's val loss between phases

### Comparing architectures

Run the full 4-phase sequence for each variant. Compare the matrices:

```
Variant A (dense):       Variant B (banded-256):
val_add  val_sub ...     val_add  val_sub ...
  0.05    -              0.04     -
  0.08   0.05            0.05    0.05
  0.15   0.12  0.06      0.06    0.06   0.07
  ...                    ...
```

Better architecture = lower diagonal (learns faster) + smaller off-diagonal
increase (less forgetting).

---

## Phase 1: TinyStories

Natural language generalization on TinyStories (train/val/test splits).
See [GENERALIZATION_PROTOCOL.md](GENERALIZATION_PROTOCOL.md) for the full
procedure including multi-seed averaging and matched-train-loss comparisons.

---

## Phase 1a: Held-Out Domain Perplexity

Evaluate a TinyStories-trained checkpoint on text from different distributions.
Tests whether architectural variants generalize beyond the training domain.
No C changes needed — uses existing eval-only mode (`-n 0 -v 1`).

### Step 1: Prepare held-out data

```bash
python prepare_heldout.py
```

Produces:
- `data/heldout/shakespeare.bin` — TinyShakespeare (~300K tokens)
- `data/heldout/wikitext2.bin` — WikiText-2 test set (~240K tokens)

The script prints token counts. Use them to compute B:
`B = floor(num_tokens / T) - 1`

### Step 2: Eval a checkpoint on each domain

The `-i` flag is required by the binary but unused with `-n 0`. Point `-j` at
the held-out data to get cross-domain val loss. Perplexity = exp(val_loss).

```bash
# In-domain baseline (TinyStories val)
./train -e checkpoint.bin -n 0 -v 1 -t 256 -b 16 \
    -i data/tinystories/TinyStories_train.bin \
    -j data/tinystories/TinyStories_val.bin

# Cross-domain: Shakespeare
./train -e checkpoint.bin -n 0 -v 1 -t 256 -b 16 \
    -i data/tinystories/TinyStories_train.bin \
    -j data/heldout/shakespeare.bin

# Cross-domain: WikiText-2
./train -e checkpoint.bin -n 0 -v 1 -t 256 -b 16 \
    -i data/tinystories/TinyStories_train.bin \
    -j data/heldout/wikitext2.bin
```

Adjust `-b` based on token counts if needed (B*T+1 must fit in the val file).

### Step 3: Record results

| Domain | Val Loss | Perplexity |
|--------|----------|------------|
| TinyStories (in-domain) | ___ | ___ |
| TinyShakespeare | ___ | ___ |
| WikiText-2 | ___ | ___ |

### Comparing architectures

Run the eval for each variant's checkpoint. Lower held-out perplexity = better
cross-domain generalization.

| Domain | Dense | Banded-256 | ... |
|--------|-------|------------|-----|
| TinyStories (in-domain) | ___ | ___ | |
| TinyShakespeare | ___ | ___ | |
| WikiText-2 | ___ | ___ | |

---

## Phase 2b: Sequential Formal Language Tasks

Tests catastrophic forgetting across structurally different algorithmic tasks:
parity (counting), copy (identity mapping), reverse (position reversal), and
majority (thresholding). Uses the same sequential protocol as Phase 2a but with
the `-P` flag for multi-position loss on copy/reverse tasks.

### Tasks (in training order)

| # | Task | Sequence layout | T | `-p`/`-P` | Examples |
|---|------|-----------------|---|-----------|----------|
| 1 | Parity-8 | `BOS b1..b8 EQ ans EOS` + 8 EOS pad | 20 | `-p 9` | 256 |
| 2 | Copy-8 | `BOS x1..x8 SEP x1..x8 EOS EOS` | 20 | `-p 9 -P 16` | 256 |
| 3 | Reverse-8 | `BOS x1..x8 SEP x8..x1 EOS EOS` | 20 | `-p 9 -P 16` | 256 |
| 4 | Majority-9 | `BOS t1..t9 EQ ans EOS` + 6 EOS pad | 20 | `-p 10` | 512 |

All tasks: binary alphabet (0/1), vocab_size=101, 50/50 train/val split, seed 42.
Special tokens: BOS=99, EOS=100, EQ=98, SEP=97.

### Step 1: Generate data

```bash
python gen_formal_data.py --task parity --input-len 8 --seed 42 --output-dir data/parity_8
python gen_formal_data.py --task copy --input-len 8 --seed 42 --output-dir data/copy_8
python gen_formal_data.py --task reverse --input-len 8 --seed 42 --output-dir data/reverse_8
python gen_formal_data.py --task majority --input-len 9 --seed 42 --output-dir data/majority_9
```

Parity/copy/reverse: 256 examples (128 train, 128 val). Majority: 512 examples (256 train, 256 val).

### Step 2: Create model

```bash
python create_model.py --preset formal -o model_formal.bin
```

Uses block_size=24, vocab_size=101, 2-layer/4-head/128-dim (same architecture as
grokking preset but with block_size=24 to support T=20 sequences).

### Step 3: Sequential training + cross-task eval

Train on each task in order, carrying the checkpoint forward. After each training
phase, evaluate on **all** tasks' val sets. B=127 for parity/copy/reverse (128
train sequences, dataloader needs B*T+1 tokens). B=255 for majority (256 train).

Note: for cross-task eval, each eval command must use the correct `-p`/`-P` for
the task being evaluated (not the task that was trained), and `-t 20` is shared.

```bash
# --- Phase 1: Train on parity ---
python wandb_train.py --project gpt2-cuda --group formal-eval --name phase1-parity -- \
    ./train -e model_formal.bin -c ckpt_formal_p1.bin \
    -i data/parity_8/train.bin -j data/parity_8/val.bin \
    -t 20 -b 127 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
    -s 0 -p 9 -v 1000 -q 1337

# Eval phase 1 checkpoint
./train -e ckpt_formal_p1.bin -i data/parity_8/train.bin -j data/parity_8/val.bin \
    -n 0 -v 1 -t 20 -b 127 -s 0 -p 9    # => val_parity

# --- Phase 2: Train on copy (from phase 1 checkpoint) ---
python wandb_train.py --project gpt2-cuda --group formal-eval --name phase2-copy -- \
    ./train -e ckpt_formal_p1.bin -c ckpt_formal_p2.bin \
    -i data/copy_8/train.bin -j data/copy_8/val.bin \
    -t 20 -b 127 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
    -s 0 -p 9 -P 16 -v 1000 -q 1337

# Eval phase 2 checkpoint on tasks 1-2
./train -e ckpt_formal_p2.bin -i data/copy_8/train.bin -j data/parity_8/val.bin \
    -n 0 -v 1 -t 20 -b 127 -s 0 -p 9    # => val_parity (retention)
./train -e ckpt_formal_p2.bin -i data/copy_8/train.bin -j data/copy_8/val.bin \
    -n 0 -v 1 -t 20 -b 127 -s 0 -p 9 -P 16    # => val_copy

# --- Phase 3: Train on reverse (from phase 2 checkpoint) ---
python wandb_train.py --project gpt2-cuda --group formal-eval --name phase3-reverse -- \
    ./train -e ckpt_formal_p2.bin -c ckpt_formal_p3.bin \
    -i data/reverse_8/train.bin -j data/reverse_8/val.bin \
    -t 20 -b 127 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
    -s 0 -p 9 -P 16 -v 1000 -q 1337

# Eval phase 3 checkpoint on tasks 1-3
./train -e ckpt_formal_p3.bin -i data/reverse_8/train.bin -j data/parity_8/val.bin \
    -n 0 -v 1 -t 20 -b 127 -s 0 -p 9    # => val_parity (retention)
./train -e ckpt_formal_p3.bin -i data/reverse_8/train.bin -j data/copy_8/val.bin \
    -n 0 -v 1 -t 20 -b 127 -s 0 -p 9 -P 16    # => val_copy (retention)
./train -e ckpt_formal_p3.bin -i data/reverse_8/train.bin -j data/reverse_8/val.bin \
    -n 0 -v 1 -t 20 -b 127 -s 0 -p 9 -P 16    # => val_reverse

# --- Phase 4: Train on majority (from phase 3 checkpoint) ---
python wandb_train.py --project gpt2-cuda --group formal-eval --name phase4-majority -- \
    ./train -e ckpt_formal_p3.bin -c ckpt_formal_p4.bin \
    -i data/majority_9/train.bin -j data/majority_9/val.bin \
    -t 20 -b 255 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
    -s 0 -p 10 -v 1000 -q 1337

# Eval phase 4 checkpoint on all tasks
./train -e ckpt_formal_p4.bin -i data/majority_9/train.bin -j data/parity_8/val.bin \
    -n 0 -v 1 -t 20 -b 127 -s 0 -p 9    # => val_parity (retention)
./train -e ckpt_formal_p4.bin -i data/majority_9/train.bin -j data/copy_8/val.bin \
    -n 0 -v 1 -t 20 -b 127 -s 0 -p 9 -P 16    # => val_copy (retention)
./train -e ckpt_formal_p4.bin -i data/majority_9/train.bin -j data/reverse_8/val.bin \
    -n 0 -v 1 -t 20 -b 127 -s 0 -p 9 -P 16    # => val_reverse (retention)
./train -e ckpt_formal_p4.bin -i data/majority_9/train.bin -j data/majority_9/val.bin \
    -n 0 -v 1 -t 20 -b 255 -s 0 -p 10    # => val_majority
```

**Note on B mismatch:** Majority has 256 train sequences (B=255) while the other
tasks have 128 (B=127). For eval-only runs (`-n 0`), B just needs to satisfy
`B*T+1 <= num_tokens` for whichever val set is loaded, so use B=127 for
parity/copy/reverse val sets and B=255 for majority val sets.

**Note on `-p`/`-P` per task:** The dlosses mask is built lazily on the first
forward pass, so each eval run must use the correct flags for the task being
evaluated. Single-position tasks (parity, majority) use only `-p`. Multi-position
tasks (copy, reverse) use `-p 9 -P 16`.

### Step 4: Record results

| Phase | Trained on | val_parity | val_copy | val_reverse | val_majority |
|-------|------------|------------|----------|-------------|--------------|
| 1 | parity | ___ | - | - | - |
| 2 | copy | ___ | ___ | - | - |
| 3 | reverse | ___ | ___ | ___ | - |
| 4 | majority | ___ | ___ | ___ | ___ |

Key metrics (same interpretation as Phase 2a):
- **Diagonal**: learning ability (should reach low loss)
- **Below diagonal**: retention (should stay low if representations are stable)
- **Forgetting**: increase in a task's val loss between phases

---

## Future Phases

- **2c. In-context learning probes** — see EVAL_ROADMAP.md
- **3. Scaling laws** — see EVAL_ROADMAP.md
- **4. Standard benchmarks** — see EVAL_ROADMAP.md
