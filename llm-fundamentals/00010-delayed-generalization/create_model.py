"""
Create a tiny GPT model for modular arithmetic (delayed generalization experiment).
Outputs model.bin in llm.c checkpoint format.

Architecture matches the paper:
  - 2 layers, 128 dim, 4 heads
  - Vocab: 101 tokens (97 residues + OP + EQ + BOS + EOS), padded to 128
  - Sequence length: 8 (BOS a OP b EQ c EOS EOS)

Usage:
    python3 create_model.py
"""
import struct
import os
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class GPTConfig:
    block_size: int = 8        # sequence length: BOS a OP b EQ c EOS EOS
    vocab_size: int = 101      # 97 residues + OP(97) + EQ(98) + BOS(99) + EOS(100)
    n_layer: int = 2           # 2 transformer layers
    n_head: int = 4            # 4 attention heads
    n_embd: int = 128          # 128 dim embeddings


class NewGELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)


def write_fp32(tensor, file):
    t = tensor.detach().cpu().contiguous().to(torch.float32).numpy()
    file.write(t.tobytes())


def write_model(model, filename):
    """Write model in llm.c format (version 3)"""
    config = model.config

    # Pad vocab to multiple of 64 (llm.c requirement)
    padded_vocab_size = ((config.vocab_size + 63) // 64) * 64

    C = config.n_embd
    L = config.n_layer
    maxT = config.block_size
    Vp = padded_vocab_size

    with open(filename, 'wb') as f:
        # Header: 256 ints
        header = [0] * 256
        header[0] = 20240326  # magic
        header[1] = 3         # version
        header[2] = config.block_size   # max_seq_len
        header[3] = config.vocab_size   # vocab_size
        header[4] = config.n_layer      # num_layers
        header[5] = config.n_head       # num_heads
        header[6] = config.n_embd       # channels
        header[7] = padded_vocab_size   # padded_vocab_size

        for val in header:
            f.write(struct.pack('i', val))

        sd = model.state_dict()

        # 1. wte (Vp * C) - pad vocab
        wte = sd['transformer.wte.weight']
        wte_padded = torch.zeros(padded_vocab_size, C)
        wte_padded[:config.vocab_size] = wte
        write_fp32(wte_padded, f)

        # 2. wpe (maxT * C)
        write_fp32(sd['transformer.wpe.weight'], f)

        # 3. ln1w (L * C)
        ln1w = torch.cat([sd[f'transformer.h.{i}.ln_1.weight'] for i in range(L)])
        write_fp32(ln1w, f)

        # 4. ln1b (L * C)
        ln1b = torch.cat([sd[f'transformer.h.{i}.ln_1.bias'] for i in range(L)])
        write_fp32(ln1b, f)

        # 5. qkvw (L * 3C * C)
        qkvw = torch.cat([sd[f'transformer.h.{i}.attn.c_attn.weight'] for i in range(L)])
        write_fp32(qkvw, f)

        # 6. qkvb (L * 3C)
        qkvb = torch.cat([sd[f'transformer.h.{i}.attn.c_attn.bias'] for i in range(L)])
        write_fp32(qkvb, f)

        # 7. attprojw (L * C * C)
        attprojw = torch.cat([sd[f'transformer.h.{i}.attn.c_proj.weight'] for i in range(L)])
        write_fp32(attprojw, f)

        # 8. attprojb (L * C)
        attprojb = torch.cat([sd[f'transformer.h.{i}.attn.c_proj.bias'] for i in range(L)])
        write_fp32(attprojb, f)

        # 9. ln2w (L * C)
        ln2w = torch.cat([sd[f'transformer.h.{i}.ln_2.weight'] for i in range(L)])
        write_fp32(ln2w, f)

        # 10. ln2b (L * C)
        ln2b = torch.cat([sd[f'transformer.h.{i}.ln_2.bias'] for i in range(L)])
        write_fp32(ln2b, f)

        # 11. fcw (L * 4C * C)
        fcw = torch.cat([sd[f'transformer.h.{i}.mlp.c_fc.weight'] for i in range(L)])
        write_fp32(fcw, f)

        # 12. fcb (L * 4C)
        fcb = torch.cat([sd[f'transformer.h.{i}.mlp.c_fc.bias'] for i in range(L)])
        write_fp32(fcb, f)

        # 13. fcprojw (L * C * 4C)
        fcprojw = torch.cat([sd[f'transformer.h.{i}.mlp.c_proj.weight'] for i in range(L)])
        write_fp32(fcprojw, f)

        # 14. fcprojb (L * C)
        fcprojb = torch.cat([sd[f'transformer.h.{i}.mlp.c_proj.bias'] for i in range(L)])
        write_fp32(fcprojb, f)

        # 15. lnfw (C)
        write_fp32(sd['transformer.ln_f.weight'], f)

        # 16. lnfb (C)
        write_fp32(sd['transformer.ln_f.bias'], f)

    # Verify file size
    expected_params = (
        Vp * C +       # wte
        maxT * C +     # wpe
        L * C +        # ln1w
        L * C +        # ln1b
        L * 3*C * C +  # qkvw
        L * 3*C +      # qkvb
        L * C * C +    # attprojw
        L * C +        # attprojb
        L * C +        # ln2w
        L * C +        # ln2b
        L * 4*C * C +  # fcw
        L * 4*C +      # fcb
        L * C * 4*C +  # fcprojw
        L * C +        # fcprojb
        C +            # lnfw
        C              # lnfb
    )
    expected_size = 256 * 4 + expected_params * 4

    actual_size = os.path.getsize(filename)
    print(f"Expected size: {expected_size} bytes ({expected_params:,} params)")
    print(f"Actual size:   {actual_size} bytes")

    if actual_size == expected_size:
        print(f"✓ Saved model to {filename}")
    else:
        print(f"✗ Size mismatch!")


if __name__ == "__main__":
    config = GPTConfig()

    model = GPT(config)
    n_params = sum(p.numel() for p in model.parameters())

    padded_vocab = ((config.vocab_size + 63) // 64) * 64

    print(f"Model config (delayed generalization experiment):")
    print(f"  block_size:  {config.block_size}")
    print(f"  vocab_size:  {config.vocab_size} (padded to {padded_vocab})")
    print(f"  n_layer:     {config.n_layer}")
    print(f"  n_head:      {config.n_head}")
    print(f"  n_embd:      {config.n_embd}")
    print(f"  Parameters:  {n_params:,}")
    print()

    write_model(model, "model.bin")
