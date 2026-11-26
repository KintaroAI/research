# KintaroAI Research

Experimental research artifacts for AGI development.

## ⚠️ Security Warning

**DO NOT commit API keys, credentials, tokens, or sensitive data to this repository.**

- Use environment variables or `.env` files (gitignored)
- Use credential managers or secret vaults
- Review your commits before pushing

## Topics

State as of 2026-11-26. Each topic's README has its own progress table, one row per experiment.

| Topic | Question | Experiments | State |
|---|---|---|---|
| [visual-motor-agent](visual-motor-agent/README.md) | What does a small agent learn about its own actions from prediction alone? Can one universal next-token model see, predict and act, learning online in one life? | 117 (vm-00001 to vm-00118) | **Active** since 2026-10-23. The default agent eats 71.6 food and 8.3 poison per 1k ticks in its food world and returns .68 of the balls in Breakout after 60k ticks. Against DreamerV3 it is ahead in the food world and behind in Breakout (vm-00108). vm-00116 found why it misses in Breakout: its label barely orders the candidate moves. A policy-gradient actor in place of its instinct learned neither world (vm-00115), and a move decided every 4 ticks and held is chosen better but plays a little worse (vm-00118). Weights taken from one world into the other transfer one way (the food world helps Breakout, Breakout hurts the food world) and lose the old world's play within 10k ticks (vm-00117). |
| [thalamus-sorter](thalamus-sorter/README.md) | Can neurons be arranged so that correlated ones become neighbours, from their temporal signals alone? | 27 (ts-00001 to ts-00028) | Parked. A pixel layout is recovered from correlation alone (97–98% of neighbours within 5 px at 80 x 80, ts-00010); clusters are wired to columns in a feedback loop with a forage benchmark (ts-00020 to ts-00027). Last: bottleneck codes on the plane (ts-00028, 2026-11-14). |
| [column](column/README.md) | Can a small cell learn categories online with local Hebbian rules and no backprop (a cortical minicolumn model)? | 16 (00001 to 00016) | Parked since 2026-03-18. The cell separates 4 clusters at NMI 0.97 and learns temporal co-variation; stacking by receptive fields or time scales helps, a recurrent ring does not. |
| [llm-fundamentals](llm-fundamentals/README.md) | How do transformer language models work, learned by implementing GPT-2 training in raw CUDA/C? | 22 (00001 to 00021) | Parked since 2026-11-12. A self-contained CUDA trainer, banded sparsity, grokking on mod 97, and fact memory: 2 to 21 showings to learn a fact, forgotten within a few thousand steps (00020, 00021). |

Plans: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) is the routed-prediction plan of March 2026; its phases have not been run as written. The plan being followed in `visual-motor-agent` is section 13 of its [`ARCHITECTURE.md`](visual-motor-agent/ARCHITECTURE.md).

## Repository Structure

```
research/
├── README.md                          # This file
├── CLAUDE.md                          # Guidance for Claude Code: state of the repository, conventions, doc upkeep
├── RESEARCH_PLAN.md                   # The routed-prediction plan (March 2026)
├── visual_motor_learning_*.md         # The starting plans of visual-motor-agent
├── topic-name/                        # Research topic (e.g., visual-motor-agent)
│   ├── README.md                      # Topic overview, current state, progress table
│   ├── CLAUDE.md                      # How to work in the topic (llm-fundamentals: the root CLAUDE.md)
│   ├── dev/                           # Active development: code, configs, tests
│   ├── tasks/                         # Numbered task plans, written before implementing
│   └── 00001-experiment-name/         # Individual experiment
│       ├── README.md                  # Experiment report
│       ├── Makefile                   # Reproduces the experiment
│       └── images/                    # Small PNGs for the README (optional)
└── _templates/                        # Templates for new topics/experiments
```

Source code is not copied into experiment directories: the git tag of each experiment preserves it. Outputs (checkpoints, logs, videos) live in `~/data/research/<topic>/`, outside the repo.

## Topic README Template

Each research topic folder should have a README.md with:

1. **Objective** — What are we trying to achieve?
2. **Hypothesis** — What do we believe and why?
3. **Success Criteria** — How do we know if we succeeded?
4. **Background** — Relevant papers, prior work, context
5. **Setup** — Environment, dependencies, hardware requirements
6. **Progress** — Summary of experiments and findings, one table row per experiment
7. **Open Questions** — What we still don't know

An active topic also keeps a dated **Current state** section near the top.

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

Add the topic to the Topics table above and to the table in `CLAUDE.md`.

## Creating a New Experiment

```bash
cd topic-name
ls -d [0-9]* | sort -V | tail -1                 # the last number used; take the next one
mkdir NNNNN-experiment-name
cp ../_templates/EXPERIMENT_README.md NNNNN-experiment-name/README.md
cp ../_templates/EXPERIMENT_MAKEFILE NNNNN-experiment-name/Makefile
# Edit the README with your experiment details
```

The full workflow (plan, run, write up, tag) is [`llm-fundamentals/dev/EXPERIMENT_PROTOCOL.md`](llm-fundamentals/dev/EXPERIMENT_PROTOCOL.md). Add the experiment's row to the topic README's progress table when the directory is created, and complete it at the write-up.

## Git Workflow

1. Work on `master` and commit frequently with clear messages
2. When an experiment is written up, tag its final state and push the tag: `exp/NNNNN` (llm-fundamentals), `exp/vm-NNNNN`, `exp/ts-NNNNN`, `c/NNNNN`
3. A last commit marks the experiment complete with its source tag

## Large Files

Checkpoints, logs, videos and datasets stay out of the repo, in `~/data/research/<topic>/exp_NNNNN/`.

- Experiment READMEs quote the key numbers, not full logs
- Small PNGs for a README go in the experiment's `images/` folder (at most 960 px wide and 200 KB each)

---

*"It's not about pre-programming every detail – it's about designing systems that can learn to wire themselves."*
