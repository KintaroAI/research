#!/usr/bin/env python3
"""Tokenize TinyStories JSON files into binary format for training."""

import os
import json
import glob
import numpy as np
import tiktoken
from tqdm import tqdm

def main():
    data_dir = "data/tinystories"
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    
    # Process all files
    all_tokens = []
    for json_file in tqdm(json_files, desc="Processing files"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            story = item.get('story', '')
            if story:
                tokens = enc.encode_ordinary(story)
                all_tokens.extend(tokens)
                all_tokens.append(eot)
    
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens):,}")
    
    # Split into train/val (95/5)
    n = len(all_tokens)
    split_idx = int(n * 0.95)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Write in llm.c format
    def write_data(filename, tokens):
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520  # magic
        header[1] = 1  # version
        header[2] = len(tokens)
        
        with open(filename, 'wb') as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())
        
        print(f"Wrote {filename}: {len(tokens):,} tokens ({os.path.getsize(filename) / 1e6:.1f} MB)")
    
    write_data(os.path.join(data_dir, "TinyStories_train.bin"), train_tokens)
    write_data(os.path.join(data_dir, "TinyStories_val.bin"), val_tokens)
    
    print("\nDone! Ready for training.")

if __name__ == "__main__":
    main()
