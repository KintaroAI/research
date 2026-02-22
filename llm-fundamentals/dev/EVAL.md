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
phase, evaluate on **all** tasks' val sets.

```bash
# --- Phase 1: Train on addition ---
./train -e model_grok.bin -c ckpt_phase1.bin \
        -i data/modular_add/train.bin -j data/modular_add/val.bin \
        -t 8 -b 4704 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
        -s 0 -p 4 -q 1337 -o log_phase1_train.txt

# Eval phase 1 checkpoint on all available tasks
./train -e ckpt_phase1.bin -i data/modular_add/train.bin -j data/modular_add/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_add

# --- Phase 2: Train on subtraction (from phase 1 checkpoint) ---
./train -e ckpt_phase1.bin -c ckpt_phase2.bin \
        -i data/modular_sub/train.bin -j data/modular_sub/val.bin \
        -t 8 -b 4704 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
        -s 0 -p 4 -q 1337 -o log_phase2_train.txt

# Eval phase 2 checkpoint on tasks 1-2
./train -e ckpt_phase2.bin -i data/modular_sub/train.bin -j data/modular_add/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_add (retention)
./train -e ckpt_phase2.bin -i data/modular_sub/train.bin -j data/modular_sub/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_sub

# --- Phase 3: Train on multiplication (from phase 2 checkpoint) ---
./train -e ckpt_phase2.bin -c ckpt_phase3.bin \
        -i data/modular_mul/train.bin -j data/modular_mul/val.bin \
        -t 8 -b 4704 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
        -s 0 -p 4 -q 1337 -o log_phase3_train.txt

# Eval phase 3 checkpoint on tasks 1-3
./train -e ckpt_phase3.bin -i data/modular_mul/train.bin -j data/modular_add/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_add (retention)
./train -e ckpt_phase3.bin -i data/modular_mul/train.bin -j data/modular_sub/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_sub (retention)
./train -e ckpt_phase3.bin -i data/modular_mul/train.bin -j data/modular_mul/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_mul

# --- Phase 4: Train on squared sum (from phase 3 checkpoint) ---
./train -e ckpt_phase3.bin -c ckpt_phase4.bin \
        -i data/modular_sq_sum/train.bin -j data/modular_sq_sum/val.bin \
        -t 8 -b 4704 -n 50000 -l 0.001 -w 1.0 -a 0.98 \
        -s 0 -p 4 -q 1337 -o log_phase4_train.txt

# Eval phase 4 checkpoint on all tasks
./train -e ckpt_phase4.bin -i data/modular_sq_sum/train.bin -j data/modular_add/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_add (retention)
./train -e ckpt_phase4.bin -i data/modular_sq_sum/train.bin -j data/modular_sub/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_sub (retention)
./train -e ckpt_phase4.bin -i data/modular_sq_sum/train.bin -j data/modular_mul/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_mul (retention)
./train -e ckpt_phase4.bin -i data/modular_sq_sum/train.bin -j data/modular_sq_sum/val.bin \
        -n 0 -v 1 -t 8 -b 256 -s 0    # => val_sq_sum
```

**How eval-only works:** With `-n 0`, the training loop runs step=0 which is
`last_step=true`, so it does val eval and exits. We point `-j` at each task's val
data to get the loss. No C changes needed.

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

## Future Phases

- **2b. Formal language tasks** (parity, copy/reverse) — see EVAL_ROADMAP.md
- **2c. In-context learning probes** — see EVAL_ROADMAP.md
- **3. Scaling laws** — see EVAL_ROADMAP.md
- **4. Standard benchmarks** — see EVAL_ROADMAP.md
