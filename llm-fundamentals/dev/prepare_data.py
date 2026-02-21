"""
Download and tokenize the TinyStories dataset for training.
Adapted from llm.c by Andrej Karpathy (MIT License).

Usage:
    python prepare_data.py

Downloads ~1.5GB, produces:
    data/tinystories/TinyStories_train.bin (~1.7GB, ~887M tokens)
    data/tinystories/TinyStories_val.bin   (~38MB, ~19M tokens)
    data/tinystories/TinyStories_test.bin  (~38MB, ~19M tokens)
"""

import os
import glob
import json
import random
import struct
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
import numpy as np
from tqdm import tqdm

# Use tiktoken for GPT-2 tokenization
import tiktoken

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "tinystories")


def download_file(url: str, fname: str, chunk_size=1024):
    """Download a file with progress bar"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=os.path.basename(fname),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Download TinyStories dataset from HuggingFace"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # Unpack
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print(f"Download done. Number of shards: {len(shard_filenames)}")
    return shard_filenames


def process_shard(shard_index, shard_filename):
    """Tokenize one shard of the dataset"""
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    
    with open(shard_filename, "r") as f:
        data = json.load(f)
    
    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    
    all_tokens = []
    for example in data:
        text = example["story"].strip()
        tokens = enc.encode_ordinary(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    
    return all_tokens


def write_datafile(filename, toks):
    """
    Write tokens to binary file in llm.c format:
    - 256 int32 header (magic, version, token_count, ...)
    - uint16 tokens
    """
    assert len(toks) < 2**31, "token count too large"
    
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic for data files
    header[1] = 1         # version
    header[2] = len(toks)
    
    toks_np = np.array(toks, dtype=np.uint16)
    
    print(f"Writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def tokenize(shard_filenames):
    """Tokenize all shards into train/val/test splits"""
    # shard 0 = validation, shard 1 = test, rest = training
    val_shards = [shard_filenames[0]]
    test_shards = [shard_filenames[1]]
    train_shards = shard_filenames[2:]

    for split_name, split_shards in [("val", val_shards), ("test", test_shards), ("train", train_shards)]:
        print(f"\nTokenizing {split_name} split ({len(split_shards)} shards)...")
        
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_shard, i, fn)
                for i, fn in enumerate(split_shards)
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                all_tokens.extend(future.result())
        
        output_file = os.path.join(DATA_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(output_file, all_tokens)


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories dataset")
    parser.add_argument("--download-only", action="store_true", help="Only download, don't tokenize")
    args = parser.parse_args()
    
    shard_filenames = download()
    
    if not args.download_only:
        tokenize(shard_filenames)
        print("\nDone! Data files ready in data/tinystories/")


if __name__ == "__main__":
    main()
