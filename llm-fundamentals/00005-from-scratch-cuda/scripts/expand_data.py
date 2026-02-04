"""Expand training data by repeating tokens (with correct header format)"""
import struct
import numpy as np

# Read original data
with open('dev/data/tinyshakespeare/tiny_shakespeare_train.bin', 'rb') as f:
    # Read header (256 ints = 1024 bytes)
    header = list(struct.unpack('256i', f.read(1024)))
    magic = header[0]
    version = header[1]
    num_tokens = header[2]
    
    print(f"Original: magic={magic}, version={version}, tokens={num_tokens}")
    
    # Read tokens
    tokens = np.frombuffer(f.read(), dtype=np.uint16)
    print(f"Read {len(tokens)} tokens")

# Expand by repeating 50x
tokens_expanded = np.tile(tokens, 50)
new_num_tokens = len(tokens_expanded)

print(f"Expanded to {new_num_tokens} tokens ({new_num_tokens/1e6:.1f}M)")

# Write expanded file with updated header
with open('dev/data/tinyshakespeare/tiny_shakespeare_train_50x.bin', 'wb') as f:
    header[2] = new_num_tokens  # update token count
    f.write(struct.pack('256i', *header))
    f.write(tokens_expanded.tobytes())

print("Saved expanded data")
