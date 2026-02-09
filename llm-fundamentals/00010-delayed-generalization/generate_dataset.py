#!/usr/bin/env python3
"""
Generate modular arithmetic dataset in llm.c shard format.

Produces train.bin and val.bin containing equations: a + b ≡ c (mod p)
encoded as 8-token sequences [BOS, a, OP, b, EQ, c, EOS, EOS] in uint16.

Token encoding:
  0..96  = residues (for p=97)
  97     = OP  (the + operator)
  98     = EQ  (the = sign)
  99     = BOS (beginning of sequence)
  100    = EOS (end of sequence)

Usage:
    python3 generate_dataset.py [--prime 97] [--train-frac 0.5] [--seed 42]
"""

import argparse
import os
import numpy as np


# llm.c shard format constants
HEADER_SIZE = 256
DATA_MAGIC = 20240520
DATA_VERSION = 1


def generate_modular_addition(p, train_frac, seed):
    """Generate all p*p equations and split into train/val.
    
    Returns:
        train_tokens: flat numpy array of uint16 tokens (train split)
        val_tokens:   flat numpy array of uint16 tokens (val split)
    """
    OP  = p       # token id for +
    EQ  = p + 1   # token id for =
    BOS = p + 2   # token id for beginning of sequence
    EOS = p + 3   # token id for end of sequence
    
    # Generate all equations as 8-token sequences: BOS a OP b EQ c EOS EOS
    equations = []
    for a in range(p):
        for b in range(p):
            c = (a + b) % p
            equations.append([BOS, a, OP, b, EQ, c, EOS, EOS])
    
    equations = np.array(equations, dtype=np.uint16)
    
    # Shuffle and split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(equations))
    
    n_train = int(len(equations) * train_frac)
    train_eqs = equations[perm[:n_train]]
    val_eqs = equations[perm[n_train:]]
    
    # Shuffle within each split (so equations appear in random order in the stream)
    rng.shuffle(train_eqs)
    rng.shuffle(val_eqs)
    
    # Flatten to token streams
    train_tokens = train_eqs.flatten()
    val_tokens = val_eqs.flatten()
    
    return train_tokens, val_tokens


def write_shard(filename, tokens):
    """Write tokens in llm.c shard format (256-int header + uint16 tokens)."""
    assert tokens.dtype == np.uint16
    assert len(tokens) < 2**31
    
    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = DATA_MAGIC
    header[1] = DATA_VERSION
    header[2] = len(tokens)
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    
    file_size = os.path.getsize(filename)
    expected_size = HEADER_SIZE * 4 + len(tokens) * 2
    assert file_size == expected_size, f"Size mismatch: {file_size} != {expected_size}"


def main():
    parser = argparse.ArgumentParser(description="Generate modular arithmetic dataset")
    parser.add_argument("--prime", type=int, default=97,
                        help="Prime modulus p (default: 97)")
    parser.add_argument("--train-frac", type=float, default=0.5,
                        help="Fraction of data for training (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory (default: data)")
    args = parser.parse_args()
    
    p = args.prime
    vocab_size = p + 4  # residues + OP + EQ + BOS + EOS
    
    print(f"Generating modular addition dataset (mod {p})")
    print(f"  Total equations: {p * p}")
    print(f"  Train fraction:  {args.train_frac}")
    print(f"  Vocab size:      {vocab_size} ({p} residues + OP + EQ + BOS + EOS)")
    
    train_tokens, val_tokens = generate_modular_addition(p, args.train_frac, args.seed)
    
    SEQ_LEN = 8  # BOS a OP b EQ c EOS EOS
    n_train_eqs = len(train_tokens) // SEQ_LEN
    n_val_eqs = len(val_tokens) // SEQ_LEN
    print(f"  Train equations: {n_train_eqs} ({len(train_tokens)} tokens)")
    print(f"  Val equations:   {n_val_eqs} ({len(val_tokens)} tokens)")
    
    # Sanity check: verify first few equations
    OP, EQ_tok, BOS_tok, EOS_tok = p, p + 1, p + 2, p + 3
    for i in range(min(3, n_train_eqs)):
        bos = train_tokens[i*SEQ_LEN + 0]
        a   = train_tokens[i*SEQ_LEN + 1]
        op  = train_tokens[i*SEQ_LEN + 2]
        b   = train_tokens[i*SEQ_LEN + 3]
        eq  = train_tokens[i*SEQ_LEN + 4]
        c   = train_tokens[i*SEQ_LEN + 5]
        eos1 = train_tokens[i*SEQ_LEN + 6]
        eos2 = train_tokens[i*SEQ_LEN + 7]
        assert bos == BOS_tok, f"BOS token error at equation {i}"
        assert op == OP and eq == EQ_tok, f"Token format error at equation {i}"
        assert eos1 == EOS_tok and eos2 == EOS_tok, f"EOS token error at equation {i}"
        assert (a + b) % p == c, f"Equation error: {a} + {b} != {c} (mod {p})"
        print(f"  Sample: BOS {a} + {b} = {c} EOS EOS (mod {p}) ✓")
    
    # Write shard files
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")
    
    write_shard(train_path, train_tokens)
    print(f"\n  Wrote {train_path} ({os.path.getsize(train_path)} bytes)")
    
    write_shard(val_path, val_tokens)
    print(f"  Wrote {val_path} ({os.path.getsize(val_path)} bytes)")
    
    # Print info for training command
    print(f"\nReady! Use with T=8, B up to {n_train_eqs}")
    print(f"  ./train -e model.bin -i {train_path} -j {val_path} -t 8 -b 512 -n 100000")


if __name__ == "__main__":
    main()
