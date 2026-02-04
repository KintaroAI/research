# Experiment 00005: Small Model GPU Training from Scratch

Train a small GPT model (7.3M params) entirely from scratch using llm.c CUDA.

## Prerequisites

- Complete [Experiment 00003](../00003-llmc-full-c/README.md) to set up llm.c and verify CUDA training works
- NVIDIA GPU with CUDA toolkit (tested on RTX 4090, CUDA 12.0)
- Python 3 with PyTorch

## Files

| File | Description |
|------|-------------|
| `create_small_model.py` | Generates random 7.3M param GPT model in llm.c format |
| `train_gpt2_fp32_custom.patch` | Patch to add `-e` flag for custom model path |

## Steps

### 1. Apply the CUDA patch

```bash
cd ~/llm.c
patch -p0 < ~/research/llm-fundamentals/00005-small-model-gpu/train_gpt2_fp32_custom.patch
```

This adds `-e <model_path>` flag to `train_gpt2_fp32.cu`.

### 2. Compile the modified trainer

```bash
cd ~/llm.c
make train_gpt2_fp32_custom
```

### 3. Generate the small model

```bash
cd ~/llm.c
python ../research/llm-fundamentals/00005-small-model-gpu/create_small_model.py
```

Creates `small_model.bin` (7.3M params: 4 layers, 4 heads, 128 embed dim, 256 seq len).

### 4. Prepare training data

Use TinyShakespeare (already prepared in exp 00003):

```bash
cd ~/llm.c
python dev/data/tinyshakespeare/prepare.py
```

### 5. Train from scratch

```bash
cd ~/llm.c
./train_gpt2_fp32_custom \
    -e small_model.bin \
    -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
    -j dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
    -b 32 -t 256 -l 1e-3 -v 100 -s 100 -n 1000
```

Flags:
- `-e small_model.bin` — custom model path
- `-b 32` — batch size
- `-t 256` — sequence length (matches model)
- `-l 1e-3` — learning rate
- `-v 100` — validate every 100 steps
- `-s 100` — sample every 100 steps  
- `-n 1000` — train for 1000 steps

## Results

### Initial run (1 epoch, ~300 steps)
- Loss: 10.93 → 8.84
- Speed: ~95ms/step on RTX 4090
- Output: Still random (not enough training)

### Known limitations
- No checkpoint saving in fp32 code — each run starts from random weights
- Need multi-epoch training for coherent output
- Data format requires 256-int header (magic=20240520, version, token_count)

## Model Architecture

```
Parameters: 7,340,417 (7.3M)
- Layers: 4
- Heads: 4  
- Embedding dim: 128
- Sequence length: 256
- Vocab size: 50257 (GPT-2 tokenizer)
```

## Next Steps

1. Implement checkpoint saving to persist weights between runs
2. Create properly formatted multi-epoch dataset
3. Train long enough for coherent text generation

## License

`create_small_model.py` and patch are MIT licensed.
Original llm.c by Andrej Karpathy, MIT License.
