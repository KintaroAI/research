"""
Download and tokenize held-out text corpora for cross-domain perplexity evaluation.

Usage:
    python prepare_heldout.py

Downloads and produces:
    data/heldout/shakespeare.bin  (~300K tokens, TinyShakespeare)
    data/heldout/wikitext2.bin    (~240K tokens, WikiText-2 test set)

Each file is a single .bin shard in the same format as prepare_data.py
(magic=20240520, version=1, uint16 tokens). Documents are separated by EOT.
"""

import os
import requests
import numpy as np
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "heldout")


def download_text(url, cache_path):
    """Download a text file, caching locally."""
    if os.path.exists(cache_path):
        print(f"  cached: {cache_path}")
        with open(cache_path, "r") as f:
            return f.read()
    print(f"  downloading: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    text = resp.text
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        f.write(text)
    return text


def write_datafile(filename, toks):
    """Write tokens to binary file in llm.c data format."""
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks)
    toks_np = np.array(toks, dtype=np.uint16)
    print(f"  writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def prepare_shakespeare(enc, eot):
    """TinyShakespeare: ~1MB of Shakespeare text from Karpathy's char-rnn repo."""
    print("\n=== TinyShakespeare ===")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache = os.path.join(DATA_DIR, "raw", "shakespeare.txt")
    text = download_text(url, cache)

    # Treat the whole file as one document
    tokens = [eot] + enc.encode_ordinary(text)
    write_datafile(os.path.join(DATA_DIR, "shakespeare.bin"), tokens)
    return len(tokens)


def download_wikitext2_test(cache_path):
    """Download WikiText-2 test set via HuggingFace datasets API."""
    if os.path.exists(cache_path):
        print(f"  cached: {cache_path}")
        with open(cache_path, "r") as f:
            return f.read()

    print("  downloading via HuggingFace datasets API...")
    base_url = ("https://datasets-server.huggingface.co/rows"
                "?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1"
                "&split=test")
    lines = []
    offset = 0
    page_size = 100
    while True:
        resp = requests.get(f"{base_url}&offset={offset}&length={page_size}")
        resp.raise_for_status()
        data = resp.json()
        rows = data["rows"]
        if not rows:
            break
        for row in rows:
            lines.append(row["row"]["text"])
        offset += len(rows)
        if offset >= data["num_rows_total"]:
            break
    print(f"  fetched {offset} rows")

    text = "".join(lines)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        f.write(text)
    return text


def prepare_wikitext2(enc, eot):
    """WikiText-2 test set: standard perplexity benchmark (~240K tokens)."""
    print("\n=== WikiText-2 (test) ===")
    cache = os.path.join(DATA_DIR, "raw", "wikitext2_test.txt")
    text = download_wikitext2_test(cache)

    # WikiText format: paragraphs separated by blank lines, section headers
    # prefixed with " = ". Split on double newlines to get documents.
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]

    tokens = []
    for doc in docs:
        tokens.append(eot)
        tokens.extend(enc.encode_ordinary(doc))
    write_datafile(os.path.join(DATA_DIR, "wikitext2.bin"), tokens)
    return len(tokens)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    n_shk = prepare_shakespeare(enc, eot)
    n_wkt = prepare_wikitext2(enc, eot)

    print("\n=== Summary ===")
    print(f"  shakespeare.bin : {n_shk:>10,} tokens")
    print(f"  wikitext2.bin   : {n_wkt:>10,} tokens")
    print(f"\nUse these token counts to compute B for eval:")
    print(f"  B = floor(num_tokens / T) - 1   (e.g. T=256 => B={n_shk // 256 - 1} for shakespeare)")
    print(f"\nDone! Held-out data ready in {DATA_DIR}/")


if __name__ == "__main__":
    main()
