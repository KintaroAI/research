# KintaroAI Research

Experimental research artifacts for AGI development.

## ⚠️ Security Warning

**DO NOT commit API keys, credentials, tokens, or sensitive data to this repository.**

- Use environment variables or `.env` files (gitignored)
- Use credential managers or secret vaults
- Review your commits before pushing

## Repository Structure

```
research/
├── README.md                          # This file
├── topic-name/                        # Research topic (e.g., efficient-attention)
│   ├── README.md                      # Topic overview
│   └── 00001-experiment-name/         # Individual experiment
│       ├── README.md                  # Experiment documentation
│       ├── config/                    # Configuration files
│       ├── src/                       # Source code (if any)
│       ├── notebooks/                 # Jupyter notebooks (if any)
│       ├── results/                   # Output data, metrics, plots
│       └── artifacts/                 # Models, checkpoints, etc.
└── _templates/                        # Templates for new topics/experiments
```

## Topic README Template

Each research topic folder should have a README.md with:

1. **Objective** — What are we trying to achieve?
2. **Hypothesis** — What do we believe and why?
3. **Success Criteria** — How do we know if we succeeded?
4. **Background** — Relevant papers, prior work, context
5. **Setup** — Environment, dependencies, hardware requirements
6. **Progress** — Summary of experiments and findings
7. **Open Questions** — What we still don't know

## Experiment README Template

Each experiment folder (e.g., `00001-baseline-transformer`) should have a README.md with:

1. **Date** — When the experiment was conducted
2. **Goal** — What specific question does this experiment answer?
3. **Method** — What was the approach/setup?
4. **Configuration** — Key hyperparameters, settings
5. **Results** — What happened? Include metrics, plots
6. **Conclusions** — What did we learn?
7. **Next Steps** — What should follow from this?

## Experiment Naming Convention

```
NNNNN-short-descriptive-name
```

- `NNNNN` — 5-digit zero-padded sequential number (00001, 00002, ...)
- `short-descriptive-name` — Lowercase, hyphen-separated, concise

Examples:
- `00001-baseline-gpt2-small`
- `00002-linear-attention-test`
- `00003-thalamic-sorting-sim`

## Creating a New Topic

```bash
mkdir -p topic-name
cp _templates/TOPIC_README.md topic-name/README.md
# Edit the README with your topic details
```

## Creating a New Experiment

```bash
cd topic-name
# Find the next experiment number
NEXT=$(printf "%05d" $(($(ls -d [0-9]*-* 2>/dev/null | wc -l) + 1)))
mkdir -p ${NEXT}-experiment-name/{config,src,notebooks,results,artifacts}
cp ../_templates/EXPERIMENT_README.md ${NEXT}-experiment-name/README.md
# Edit the README with your experiment details
```

## Git Workflow

1. Create a branch for significant experiments: `git checkout -b topic/experiment-name`
2. Commit frequently with clear messages
3. Merge to main when experiment is complete and documented
4. Tag significant milestones: `git tag -a v0.1.0 -m "Description"`

## Large Files

For large files (models, datasets, checkpoints):
- Use Git LFS for files that must be versioned
- Or store externally and document the location
- Keep artifacts/ for essential outputs only

---

*"It's not about pre-programming every detail – it's about designing systems that can learn to wire themselves."*
