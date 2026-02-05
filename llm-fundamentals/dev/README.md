# dev/ - Working Directory

Active development folder. Snapshots get copied to numbered experiment folders.

## Quick Start

```bash
# Create venv and install deps
python3 -m venv venv
source venv/bin/activate
pip install torch numpy tqdm requests tiktoken

# Download and prepare TinyStories
python prepare_data.py

# Create initial model
python create_model.py

# Build
make all

# Train (dense)
./train -e model.bin -c checkpoint.bin -i data/tinystories/train.bin -j data/tinystories/val.bin -n 3000

# Train (banded)
./train_banded -e model.bin -c checkpoint_banded.bin -i data/tinystories/train.bin -j data/tinystories/val.bin -n 3000

# Generate
./generate -e checkpoint.bin -n 256
```

## Structure

```
dev/
├── src/
│   ├── train_gpt2_fp32.cu      # Dense training
│   ├── train_gpt2_fp32_banded.cu  # Banded FC1 training
│   ├── generate.cu             # Dense inference
│   ├── generate_banded.cu      # Banded inference
│   └── llmc/                   # Headers
├── create_model.py             # Generate model.bin
├── prepare_data.py             # Download & tokenize TinyStories
├── tokenize_tinystories.py     # Tokenizer helper
├── gpt2_tokenizer.bin          # GPT-2 tokenizer
└── Makefile
```

## Snapshotting to Experiments

When an experiment is ready to archive:

```bash
# From llm-fundamentals/
cp -r dev 00009-experiment-name
cd 00009-experiment-name
# Add experiment-specific README, commit
```
