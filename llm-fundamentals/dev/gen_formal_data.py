#!/usr/bin/env python3
"""
Generate formal language datasets in llm.c shard format.

Produces train.bin and val.bin for 4 tasks: parity, copy, reverse, majority.
All use a binary alphabet (0/1) with exhaustive enumeration, padded to uniform
sequence length for sequential eval compatibility.

Token encoding (shared with modular arithmetic, vocab_size=101):
  0, 1     = binary alphabet (value tokens)
  97       = SEP (separator for copy/reverse)
  98       = EQ  (equals sign for parity/majority)
  99       = BOS
  100      = EOS
  vocab_size = 101

Sequence layouts (all padded to T=20):
  Parity-8:    BOS b1..b8 EQ ans EOS [8 EOS pad]       -> task_position = 9
  Copy-8:      BOS x1..x8 SEP x1..x8 EOS EOS           -> task_position = 9..16
  Reverse-8:   BOS x1..x8 SEP x8..x1 EOS EOS           -> task_position = 9..16
  Majority-9:  BOS t1..t9 EQ ans EOS [6 EOS pad]        -> task_position = 10

Usage:
    python gen_formal_data.py --task parity --input-len 8 --seed 42 --output-dir data/parity_8
    python gen_formal_data.py --task copy --input-len 8 --seed 42 --output-dir data/copy_8
    python gen_formal_data.py --task reverse --input-len 8 --seed 42 --output-dir data/reverse_8
    python gen_formal_data.py --task majority --input-len 9 --seed 42 --output-dir data/majority_9
"""

import argparse
import os
import itertools
import numpy as np

# llm.c shard format constants
HEADER_SIZE = 256
DATA_MAGIC = 20240520
DATA_VERSION = 1

# Special tokens (matching modular arithmetic convention: vocab_size=101)
BOS = 99
EOS = 100
EQ = 98
SEP = 97


def generate_parity(input_len, alphabet_size):
    """BOS b1..bN EQ ans EOS [pad]  ->  ans = XOR of all bits."""
    sequences = []
    for bits in itertools.product(range(alphabet_size), repeat=input_len):
        ans = sum(bits) % alphabet_size
        seq = [BOS] + list(bits) + [EQ, ans, EOS]
        sequences.append(seq)
    tp_start, tp_end = input_len + 1, input_len + 1  # single position
    return sequences, tp_start, tp_end


def generate_copy(input_len, alphabet_size):
    """BOS x1..xN SEP x1..xN EOS EOS  ->  copy input after separator."""
    sequences = []
    for bits in itertools.product(range(alphabet_size), repeat=input_len):
        seq = [BOS] + list(bits) + [SEP] + list(bits) + [EOS, EOS]
        sequences.append(seq)
    tp_start = input_len + 1  # first output position (after SEP)
    tp_end = 2 * input_len    # last output position
    return sequences, tp_start, tp_end


def generate_reverse(input_len, alphabet_size):
    """BOS x1..xN SEP xN..x1 EOS EOS  ->  reverse input after separator."""
    sequences = []
    for bits in itertools.product(range(alphabet_size), repeat=input_len):
        seq = [BOS] + list(bits) + [SEP] + list(reversed(bits)) + [EOS, EOS]
        sequences.append(seq)
    tp_start = input_len + 1
    tp_end = 2 * input_len
    return sequences, tp_start, tp_end


def generate_majority(input_len, alphabet_size):
    """BOS t1..tN EQ ans EOS [pad]  ->  ans = most frequent token (odd N, no ties)."""
    assert input_len % 2 == 1, "majority requires odd input length (no ties)"
    assert alphabet_size == 2, "majority only supports binary alphabet"
    sequences = []
    threshold = input_len // 2  # majority if count > threshold
    for bits in itertools.product(range(alphabet_size), repeat=input_len):
        ans = 1 if sum(bits) > threshold else 0
        seq = [BOS] + list(bits) + [EQ, ans, EOS]
        sequences.append(seq)
    tp_start, tp_end = input_len + 1, input_len + 1
    return sequences, tp_start, tp_end


TASKS = {
    'parity': generate_parity,
    'copy': generate_copy,
    'reverse': generate_reverse,
    'majority': generate_majority,
}


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
    parser = argparse.ArgumentParser(description="Generate formal language dataset")
    parser.add_argument("--task", choices=list(TASKS.keys()), required=True,
                        help="Task type")
    parser.add_argument("--input-len", type=int, default=8,
                        help="Input sequence length (default: 8)")
    parser.add_argument("--alphabet-size", type=int, default=2,
                        help="Alphabet size (default: 2, binary)")
    parser.add_argument("--train-frac", type=float, default=0.5,
                        help="Fraction of data for training (default: 0.5)")
    parser.add_argument("--pad-to", type=int, default=20,
                        help="Pad all sequences to this length with EOS (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/<task>_<input_len>)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"data/{args.task}_{args.input_len}"

    task = args.task
    input_len = args.input_len
    alphabet_size = args.alphabet_size
    pad_to = args.pad_to

    print(f"Generating {task} dataset (input_len={input_len}, alphabet={alphabet_size})")

    # Generate all sequences
    sequences, tp_start, tp_end = TASKS[task](input_len, alphabet_size)
    n_total = len(sequences)
    raw_len = len(sequences[0])

    print(f"  Total sequences: {n_total} ({alphabet_size}^{input_len})")
    print(f"  Raw sequence length: {raw_len}")

    # Validate and pad
    assert raw_len <= pad_to, f"Sequence length {raw_len} exceeds pad_to {pad_to}"
    pad_len = pad_to - raw_len

    padded = []
    for seq in sequences:
        assert len(seq) == raw_len
        padded.append(seq + [EOS] * pad_len)

    padded = np.array(padded, dtype=np.uint16)
    assert padded.shape == (n_total, pad_to)

    # Validate all tokens are in vocab range
    assert padded.max() <= EOS, f"Token {padded.max()} exceeds vocab (max={EOS})"

    # Shuffle and split
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_total)
    n_train = int(n_total * args.train_frac)

    train_seqs = padded[perm[:n_train]]
    val_seqs = padded[perm[n_train:]]

    rng.shuffle(train_seqs)
    rng.shuffle(val_seqs)

    train_tokens = train_seqs.flatten()
    val_tokens = val_seqs.flatten()

    n_val = n_total - n_train
    print(f"  Train: {n_train} sequences ({len(train_tokens)} tokens)")
    print(f"  Val:   {n_val} sequences ({len(val_tokens)} tokens)")
    print(f"  Padded length: {pad_to} (task_position: {tp_start}..{tp_end})")

    # Print sample sequences
    TOKEN_NAMES = {BOS: 'BOS', EOS: 'EOS', EQ: 'EQ', SEP: 'SEP'}
    for i in range(min(3, n_train)):
        seq = train_seqs[i]
        parts = []
        for t in seq:
            parts.append(TOKEN_NAMES.get(int(t), str(int(t))))
        print(f"  Sample: {' '.join(parts)}")

    # Write shard files
    os.makedirs(args.output_dir, exist_ok=True)
    for name, tokens in [("train.bin", train_tokens), ("val.bin", val_tokens)]:
        path = os.path.join(args.output_dir, name)
        write_shard(path, tokens)
        n_seqs = len(tokens) // pad_to
        print(f"  Wrote {path} ({os.path.getsize(path)} bytes, {n_seqs} sequences)")

    # Print suggested training command
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")
    B = n_train - 1  # dataloader needs B*T+1 <= num_tokens
    p_flag = f"-p {tp_start}"
    P_flag = f" -P {tp_end}" if tp_end != tp_start else ""
    print(f"\nSuggested command:")
    print(f"  python create_model.py --preset formal -o model_formal.bin")
    print(f"  ./train -e model_formal.bin -i {train_path} -j {val_path} \\")
    print(f"          -t {pad_to} -b {B} -n 50000 -l 0.001 -w 1.0 \\")
    print(f"          -a 0.98 -s 0 {p_flag}{P_flag} -q 1337")


if __name__ == "__main__":
    main()
