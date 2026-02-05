#!/usr/bin/env python3
"""Create small synthetic tokenized data for testing banded sparsity experiment."""

import numpy as np
import os
import struct

def main():
    os.makedirs("data/tinystories", exist_ok=True)
    
    # Create synthetic token sequences
    # Use tokens in range [0, 50257) - GPT-2 vocab size
    np.random.seed(42)
    
    # ~100K tokens for training (small but enough to test)
    num_train_tokens = 100_000
    train_tokens = np.random.randint(0, 50257, size=num_train_tokens, dtype=np.uint16)
    
    # ~10K tokens for validation
    num_val_tokens = 10_000
    val_tokens = np.random.randint(0, 50257, size=num_val_tokens, dtype=np.uint16)
    
    # Write in llm.c format: header (256 ints) + tokens (uint16)
    def write_data(filename, tokens):
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520  # magic
        header[1] = 1  # version
        header[2] = len(tokens)  # num tokens
        
        with open(filename, 'wb') as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())
        
        print(f"Wrote {filename}: {len(tokens):,} tokens")
    
    write_data("data/tinystories/TinyStories_train.bin", train_tokens)
    write_data("data/tinystories/TinyStories_val.bin", val_tokens)
    
    print("\nSynthetic data created! Ready for testing.")
    print("Note: This is random tokens - loss won't improve meaningfully,")
    print("but it's sufficient to verify the banded sparsity code works.")

if __name__ == "__main__":
    main()
