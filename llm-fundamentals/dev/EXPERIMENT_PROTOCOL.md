# Experiment Protocol

Step-by-step workflow for running a reproducible experiment.

## Phase 1 — Prepare (before any training)

1. **Commit all changes.** Every source modification needed for the experiment must be
   committed to `llm-fundamentals/dev/`. Nothing uncommitted, nothing stashed.

2. **Run tests.** From `llm-fundamentals/dev/`:
   ```bash
   make test_all
   ```
   All tests must pass. Do not proceed with failures.

3. **Tag the commit.** Use the next sequential experiment number:
   ```bash
   git tag exp/NNNNN
   ```
   Find the next number: `git tag -l 'exp/*' | sort -V | tail -1`

4. **Push the tag:**
   ```bash
   git push origin exp/NNNNN
   ```

## Phase 2 — Run (training)

5. **Build binaries:**
   ```bash
   make train       # or: make all
   ```

6. **Generate data** if the experiment requires it. Record the exact commands:
   ```bash
   # Example:
   python gen_modular_data.py --op add --mod 113 --split 0.5
   ```

7. **Create model** if needed:
   ```bash
   make model       # or: python create_model.py <args>
   ```

8. **Run training.** Record the exact command with all flags:
   ```bash
   # Example:
   ./train -1 256 -2 256 -p 4
   ```

9. **Save key results.** Capture loss milestones and final metrics, not full logs.
   Full training logs are too large for git — use head/tail excerpts or sample
   key lines (e.g., val loss at checkpoints).

## Phase 3 — Record (after training)

10. **Create the experiment README.** Place it at
    `llm-fundamentals/NNNNN-experiment-name/README.md` using the template in
    `_templates/EXPERIMENT_README.md`.

    The README must include:
    - Source tag and commit hash in the header
    - Exact reproduction commands using `build_from_tag.sh`
    - Do **not** copy source code into the experiment directory

11. **Commit the experiment README** — only the README. No binaries, logs, data files,
    or source copies:
    ```bash
    git add llm-fundamentals/NNNNN-experiment-name/README.md
    git commit -m "Snapshot experiment NNNNN: short description"
    ```

12. **Push:**
    ```bash
    git push
    ```

---

## Quick Reference Checklist

```
[ ] All dev/ changes committed
[ ] make test_all passes
[ ] git tag exp/NNNNN
[ ] git push origin exp/NNNNN
[ ] Training run complete, results recorded
[ ] llm-fundamentals/NNNNN-name/README.md created (with tag + reproduction steps)
[ ] Committed README only (no binaries/logs/data/source copies)
[ ] Pushed
```
