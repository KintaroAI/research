# Experiment 00006: TinyStories End-to-End Training

Self-contained GPT training experiment using CUDA. Train a small language model from scratch on the TinyStories dataset with all C/CUDA code included.

## Features

- **Self-contained**: All source code in this folder (no external dependencies on llm.c)
- **TinyStories dataset**: ~925M tokens of children's stories (much larger than Shakespeare)
- **Checkpoint saving**: Resume training from saved checkpoints
- **GPU training**: CUDA-accelerated with cuBLAS

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA toolkit (tested with 12.0)
- Python 3 with: `tiktoken`, `numpy`, `requests`, `tqdm`, `torch`

```bash
pip install tiktoken numpy requests tqdm torch
```

## Quick Start

```bash
# Full setup (downloads ~1.5GB, takes several minutes)
make setup

# Train
./train -e model.bin -c checkpoint.bin -n 10000 -b 32 -t 256 -l 1e-3 -v 500 -s 500
```

## Step-by-Step

### 1. Prepare training data

```bash
python prepare_data.py
```

Downloads TinyStories from HuggingFace and tokenizes with GPT-2 tokenizer:
- `data/tinystories/TinyStories_train.bin` (~1.8GB, 925M tokens)
- `data/tinystories/TinyStories_val.bin` (~38MB, 19M tokens)

### 2. Create the model

```bash
python create_model.py
```

Generates `model.bin` with random weights (7.3M parameters by default).

Edit the script to change model size:
```python
@dataclass
class GPTConfig:
    block_size: int = 256      # sequence length
    vocab_size: int = 50257    # GPT-2 tokenizer vocab  
    n_layer: int = 4           # transformer layers
    n_head: int = 4            # attention heads
    n_embd: int = 128          # embedding dimension
```

### 3. Download tokenizer

```bash
wget https://huggingface.co/karpathy/llmc-data/resolve/main/gpt2_tokenizer.bin
```

### 4. Build the trainer

```bash
make train
```

### 5. Train

```bash
./train -e model.bin -c checkpoint.bin -n 10000 \
    -b 32 -t 256 -l 1e-3 -v 500 -s 500
```

## Training Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-e` | Model checkpoint to load | `model.bin` |
| `-c` | Checkpoint output path | `NULL` (no saving) |
| `-k` | Save checkpoint every N steps | `0` (disabled) |
| `-i` | Training data path | `data/tinystories/TinyStories_train.bin` |
| `-j` | Validation data path | `data/tinystories/TinyStories_val.bin` |
| `-n` | Max training steps | `-1` (1 epoch) |
| `-b` | Batch size | `4` |
| `-t` | Sequence length | `1024` |
| `-l` | Learning rate | `3e-4` |
| `-v` | Validate every N steps | `20` |
| `-s` | Sample every N steps | `20` |
| `-g` | Generation length | `64` |

## Resume Training

Simply load from checkpoint:

```bash
# First run - save checkpoint
./train -e model.bin -c checkpoint.bin -n 5000

# Resume from checkpoint
./train -e checkpoint.bin -c checkpoint.bin -n 5000
```

## Expected Results

With the 7.3M parameter model on TinyStories:
- **Initial loss**: ~10.5 (random weights)
- **After 10k steps**: ~4.5
- **After 50k steps**: ~3.5 (starts generating coherent stories)

Training speed on RTX 4090: ~95ms/step with batch=32, seq=256

## Inference

After training, generate text with the standalone inference binary:

```bash
./generate -e checkpoint.bin -n 256
```

Flags:
- `-e` — checkpoint path (default: checkpoint.bin)
- `-n` — number of tokens to generate (default: 256)
- `-s` — random seed (default: current time)

## File Structure

```
00006-tinystories-e2e/
├── README.md
├── Makefile
├── create_model.py      # Generate random model weights
├── prepare_data.py      # Download & tokenize TinyStories
└── src/
    ├── train_gpt2_fp32.cu  # Training code (CUDA)
    ├── generate.cu         # Inference code (CUDA)
    └── llmc/
        ├── dataloader.h    # Data loading utilities
        ├── tokenizer.h     # Tokenizer utilities
        ├── utils.h         # General utilities
        └── rand.h          # Random number utilities
```

## License

Training code adapted from [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy (MIT License).

TinyStories dataset from [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories).
