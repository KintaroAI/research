"""
Download, preprocess and serve the Shakespeare dataset for llama2.c training.
Follows the same pattern as tinystories.py
"""

import argparse
import glob
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data_shakespeare"

def download():
    """Downloads the TinyShakespeare dataset"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "shakespeare.txt")
    
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        resp = requests.get(data_url)
        with open(data_filename, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"Downloaded {len(resp.text)} characters")
    else:
        print(f"{data_filename} already exists, skipping download...")
    
    # Show a sample
    with open(data_filename, "r") as f:
        text = f.read()
    print(f"Total characters: {len(text)}")
    print(f"Sample:\n{text[:500]}")
    print("Download done.")


def train_vocab(vocab_size):
    """
    Trains a custom sentencepiece tokenizer on Shakespeare.
    """
    assert vocab_size > 0, "Vocab size must be positive"
    
    input_file = os.path.join(DATA_CACHE_DIR, "shakespeare.txt")
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    
    if not os.path.exists(input_file):
        print("Shakespeare data not found. Run 'python shakespeare.py download' first.")
        return
    
    print(f"Training vocab of size {vocab_size}...")
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity"
    )
    
    print(f"Trained tokenizer saved to {prefix}.model")
    print("Done.")


def pretokenize(vocab_size):
    """
    Pretokenizes Shakespeare into train/val .bin files
    """
    input_file = os.path.join(DATA_CACHE_DIR, "shakespeare.txt")
    
    if vocab_size > 0:
        tokenizer_model = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)
    else:
        print("Error: For Shakespeare, you must train a custom vocab first.")
        print("Run: python shakespeare.py train_vocab --vocab_size=512")
        return
    
    enc = Tokenizer(tokenizer_model)
    
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Tokenizing {len(text)} characters...")
    
    # Tokenize with BOS tokens between chunks
    # Split into ~1000 char chunks to add BOS tokens periodically
    chunk_size = 1000
    all_tokens = []
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        tokens = enc.encode(chunk, bos=True, eos=False)
        all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens)}")
    
    # Split 90/10 for train/val
    n = len(all_tokens)
    train_tokens = all_tokens[:int(n*0.9)]
    val_tokens = all_tokens[int(n*0.9):]
    
    # Save as .bin files
    train_file = os.path.join(bin_dir, "train.bin")
    val_file = os.path.join(bin_dir, "val.bin")
    
    with open(train_file, "wb") as f:
        f.write(train_tokens.tobytes())
    print(f"Saved {train_file} ({len(train_tokens)} tokens)")
    
    with open(val_file, "wb") as f:
        f.write(val_tokens.tobytes())
    print(f"Saved {val_file} ({len(val_tokens)} tokens)")
    
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized Shakespeare from disk"""
    
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created Shakespeare PretokDataset with rng seed {seed}")
        
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
        filename = "train.bin" if self.split == "train" else "val.bin"
        filepath = os.path.join(bin_dir, filename)
        
        while True:
            m = np.memmap(filepath, dtype=np.uint16, mode="r")
            num_batches = len(m) // self.max_seq_len
            num_batches -= 1
            assert num_batches > 0, f"File {filepath} too small"
            ixs = list(range(num_batches))
            rng.shuffle(ixs)
            for ix in ixs:
                start = ix * self.max_seq_len
                end = start + self.max_seq_len + 1
                chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y


def get_tokenizer_model_path(vocab_size):
    if vocab_size == 0:
        return None
    return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_vocab", "pretokenize"])
    parser.add_argument("--vocab_size", type=int, default=512, help="vocab size for custom tokenizer")
    args = parser.parse_args()
    
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
