#!/usr/bin/env python3
"""
Generate modular arithmetic dataset in llm.c shard format.

Produces train.bin and val.bin containing equations:
  a OP b = c (mod p)
encoded as 8-token sequences [BOS, a, OP, b, EQ, c, EOS, EOS] in uint16.

Token encoding:
  0..p-1  = residues
  p       = OP  (+ or *)
  p+1     = EQ  (=)
  p+2     = BOS
  p+3     = EOS
  vocab_size = p + 4

Usage:
    python gen_modular_data.py [options]
    python gen_modular_data.py --prime 97 --op add --seed 42
    python gen_modular_data.py --prime 97 --op sub --seed 42 --output-dir data/modular_sub
    python gen_modular_data.py --prime 97 --op mul --seed 42 --output-dir data/modular_mul
    python gen_modular_data.py --prime 97 --op sq_sum --seed 42 --output-dir data/modular_sq_sum
"""

import argparse
import os
import numpy as np


# llm.c shard format constants
HEADER_SIZE = 256
DATA_MAGIC = 20240520
DATA_VERSION = 1

OPERATIONS = {
    'add': lambda a, b, p: (int(a) + int(b)) % p,
    'sub': lambda a, b, p: (int(a) - int(b)) % p,
    'mul': lambda a, b, p: (int(a) * int(b)) % p,
    'sq_sum': lambda a, b, p: (int(a) * int(a) + int(b) * int(b)) % p,
}

OP_SYMBOLS = {'add': '+', 'sub': '-', 'mul': '*', 'sq_sum': 'S'}


def generate_equations(p, op, train_frac, seed):
    """Generate all p*p equations and split into train/val.

    Returns:
        (train_tokens, val_tokens): flat uint16 arrays
    """
    OP_tok  = p
    EQ_tok  = p + 1
    BOS_tok = p + 2
    EOS_tok = p + 3

    compute = OPERATIONS[op]

    # Generate all p*p equations as 8-token sequences
    equations = []
    for a in range(p):
        for b in range(p):
            c = compute(a, b, p)
            equations.append([BOS_tok, a, OP_tok, b, EQ_tok, c, EOS_tok, EOS_tok])

    equations = np.array(equations, dtype=np.uint16)

    # Shuffle and 2-way split (train + val, no test)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(equations))

    n_train = int(len(equations) * train_frac)

    train_eqs = equations[perm[:n_train]]
    val_eqs   = equations[perm[n_train:]]

    # Shuffle within each split
    rng.shuffle(train_eqs)
    rng.shuffle(val_eqs)

    return train_eqs.flatten(), val_eqs.flatten()


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
    parser.add_argument("--op", choices=list(OPERATIONS.keys()), default="add",
                        help="Operation (default: add)")
    parser.add_argument("--train-frac", type=float, default=0.5,
                        help="Fraction of data for training (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data/modular",
                        help="Output directory (default: data/modular)")
    args = parser.parse_args()

    p = args.prime
    op = args.op
    vocab_size = p + 4
    sym = OP_SYMBOLS[op]
    SEQ_LEN = 8

    print(f"Generating modular {op} dataset (mod {p})")
    print(f"  Total equations: {p * p}")
    print(f"  Train fraction:  {args.train_frac}")
    print(f"  Vocab size:      {vocab_size} ({p} residues + OP + EQ + BOS + EOS)")
    print(f"  Seed:            {args.seed}")

    train_tokens, val_tokens = generate_equations(
        p, op, args.train_frac, args.seed)

    n_train = len(train_tokens) // SEQ_LEN
    n_val   = len(val_tokens)   // SEQ_LEN
    print(f"  Train equations: {n_train} ({len(train_tokens)} tokens)")
    print(f"  Val equations:   {n_val} ({len(val_tokens)} tokens)")

    # Sanity check first few train equations
    OP_tok, EQ_tok, BOS_tok, EOS_tok = p, p + 1, p + 2, p + 3
    compute = OPERATIONS[op]
    for i in range(min(3, n_train)):
        seq = train_tokens[i*SEQ_LEN:(i+1)*SEQ_LEN]
        bos, a, op_t, b, eq, c, eos1, eos2 = seq
        assert bos == BOS_tok and op_t == OP_tok and eq == EQ_tok
        assert eos1 == EOS_tok and eos2 == EOS_tok
        assert compute(a, b, p) == c, f"Equation error: {a} {sym} {b} != {c} (mod {p})"
        print(f"  Sample: BOS {a} {sym} {b} = {c} EOS EOS (mod {p})")

    # Write shard files
    os.makedirs(args.output_dir, exist_ok=True)

    for name, tokens in [("train.bin", train_tokens),
                         ("val.bin", val_tokens)]:
        path = os.path.join(args.output_dir, name)
        write_shard(path, tokens)
        n_eqs = len(tokens) // SEQ_LEN
        print(f"  Wrote {path} ({os.path.getsize(path)} bytes, {n_eqs} equations)")

    # Print suggested training command
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path   = os.path.join(args.output_dir, "val.bin")
    print(f"\nSuggested command:")
    print(f"  python create_model.py --preset grokking --prime {p} -o model_grok.bin")
    print(f"  ./train -e model_grok.bin -i {train_path} -j {val_path} \\")
    print(f"          -t {SEQ_LEN} -b {n_train} -n 50000 -l 0.001 -w 1.0 \\")
    print(f"          -a 0.98 -s 0 -p 4 -q 1337 -o log_grok_{op}_s1337.txt")


if __name__ == "__main__":
    main()
