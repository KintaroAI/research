# Makefile for TinyStories E2E Training & Inference
# Requires: CUDA toolkit, nvcc compiler

NVCC = nvcc
NVCC_FLAGS = -O3 -Isrc
CUDA_LIBS = -lcublas -lcuda

# Default target
all: train generate

# Build the training binary
train: src/train_gpt2_fp32.cu src/llmc/*.h
	$(NVCC) $(NVCC_FLAGS) -o train src/train_gpt2_fp32.cu $(CUDA_LIBS)

# Build the inference binary
generate: src/generate.cu src/llmc/*.h
	$(NVCC) $(NVCC_FLAGS) -o generate src/generate.cu $(CUDA_LIBS)

# Download and prepare data
data: prepare_data.py
	python prepare_data.py

# Create model (small by default)
model: create_model.py
	python create_model.py

# Download GPT-2 tokenizer (try llm.c first, then HuggingFace)
tokenizer:
	@if [ ! -f gpt2_tokenizer.bin ]; then \
		if [ -f ~/llm.c/gpt2_tokenizer.bin ]; then \
			echo "Copying tokenizer from llm.c..."; \
			cp ~/llm.c/gpt2_tokenizer.bin .; \
		else \
			echo "Downloading GPT-2 tokenizer..."; \
			curl -sLO https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe || \
			echo "Download failed - get gpt2_tokenizer.bin manually from llm.c repo"; \
		fi \
	else \
		echo "gpt2_tokenizer.bin already exists"; \
	fi

# Full setup: data + model + tokenizer + compile
setup: data model tokenizer all
	@echo "Setup complete!"
	@echo "  Train: ./train -e model.bin -c checkpoint.bin -n 10000"
	@echo "  Generate: ./generate -e checkpoint.bin -n 256"

# Quick train (small test)
quick: all tokenizer
	./train -e model.bin -c checkpoint.bin -n 100 -b 8 -t 256 -v 50 -s 50

# Quick test inference
test-generate: generate tokenizer
	@if [ -f checkpoint.bin ]; then \
		./generate -e checkpoint.bin -n 64; \
	elif [ -f model.bin ]; then \
		./generate -e model.bin -n 64; \
	else \
		echo "No model found. Run 'make model' or train first."; \
	fi

# Clean build artifacts
clean:
	rm -f train generate

# Clean all generated files
cleanall: clean
	rm -rf data/
	rm -f model.bin checkpoint.bin gpt2_tokenizer.bin

.PHONY: all data model tokenizer setup quick clean cleanall
