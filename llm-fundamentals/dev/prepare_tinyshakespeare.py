"""
Download and tokenize TinyShakespeare for test reference data generation.
Matches llm.c convention exactly (section-based EOT, first 32768 tokens as val).

Usage:
    python prepare_tinyshakespeare.py

Produces:
    data/tinyshakespeare/tiny_shakespeare_val.bin   (32,768 tokens)
    data/tinyshakespeare/tiny_shakespeare_train.bin  (~305K tokens)
"""

import os
import numpy as np
import requests
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def write_datafile(filename, toks):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks)
    toks_np = np.array(toks, dtype=np.uint16)
    print(f"  {filename}: {len(toks):,} tokens")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # download
    cache = os.path.join(DATA_DIR, "input.txt")
    if os.path.exists(cache):
        print(f"cached: {cache}")
        with open(cache, "r") as f:
            text = f.read()
    else:
        print(f"downloading {URL}")
        text = requests.get(URL).text
        with open(cache, "w") as f:
            f.write(text)

    # tokenize — matches llm.c convention exactly
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    sections = text.split("\n\n")
    tokens = []
    for i, s in enumerate(sections):
        tokens.append(eot)
        # llm.c re-adds \n\n after each section except the last
        spad = s + "\n\n" if i != len(sections) - 1 else s
        tokens.extend(enc.encode_ordinary(spad))

    # first 32768 tokens as val, rest as train (llm.c convention)
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]

    write_datafile(os.path.join(DATA_DIR, "tiny_shakespeare_val.bin"), val_tokens)
    write_datafile(os.path.join(DATA_DIR, "tiny_shakespeare_train.bin"), train_tokens)
    print("done")


if __name__ == "__main__":
    main()
