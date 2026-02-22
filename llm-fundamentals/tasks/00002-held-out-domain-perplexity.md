# Task 00002: Held-Out Domain Perplexity (Eval Roadmap 1a)

**Date:** 2026-02-21
**Status:** Done

## Context

The eval roadmap item 1a is "held-out domain perplexity" — train on TinyStories,
evaluate perplexity on different text distributions (TinyShakespeare, etc.). This
tests whether architectural variants generalize beyond the training domain.

The codebase already has eval-only mode (`-n 0 -v 1 -j <file>`) used in Phase 2a/2b
cross-task eval. No C changes are needed — we just need held-out data in .bin format
and a documented protocol.

## Deliverables

1. **`prepare_heldout.py`** — download and tokenize held-out text corpora
2. **EVAL.md update** — Phase 1a protocol
3. **EVAL_ROADMAP.md update** — mark 1a as done

## Data Sources

| Domain | Source | Tokens | File |
|--------|--------|--------|------|
| TinyShakespeare | Karpathy's char-rnn repo | ~338K | `data/heldout/shakespeare.bin` |
| WikiText-2 test | HuggingFace datasets API (Salesforce/wikitext) | ~283K | `data/heldout/wikitext2.bin` |

Both files use the standard llm.c data format (magic=20240520, version=1, uint16
tokens). Documents are EOT-separated.

## Eval Mechanism

Uses existing eval-only mode — no C changes:

```bash
# Eval a TinyStories checkpoint on Shakespeare
./train -e checkpoint.bin -n 0 -v 1 -t 256 -b 16 \
    -i data/tinystories/TinyStories_train.bin \
    -j data/heldout/shakespeare.bin
```

The `-i` flag is required by the binary but not used with `-n 0`. The val loss
reported is the held-out domain perplexity (perplexity = exp(val_loss)).

## Implementation Notes

- Shakespeare: single document, downloaded as raw text from GitHub
- WikiText-2: original S3 URL is dead; downloaded via HuggingFace datasets server
  API (paginated, 100 rows/page, 4358 total rows), then split into documents on
  double newlines
- Both cached as raw text in `data/heldout/raw/` to avoid re-downloading
- `write_datafile` reused from `prepare_data.py` pattern (256-int header + uint16 tokens)

## Verification

1. `prepare_heldout.py` runs successfully, both .bin files created
2. Binary format verified: magic, version, token counts all correct
3. First token is EOT (50256) in both files
