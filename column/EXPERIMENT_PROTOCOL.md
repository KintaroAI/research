# Experiment Protocol

Step-by-step workflow for running a reproducible experiment.

## Phase 1 — Plan (before any code changes)

1. **Pick the next experiment number.** Find it with:
   ```bash
   ls -d [0-9]* 2>/dev/null | sort -V | tail -1
   ```

2. **Create the experiment directory and README.** Use the template:
   ```bash
   mkdir NNNNN-experiment-name
   cp ../_templates/EXPERIMENT_README.md NNNNN-experiment-name/README.md
   ```
   Fill in: Goal, Hypothesis, and planned Method. Status = In Progress.

3. **Create a Makefile** in the experiment directory (copy from template):
   ```bash
   cp ../_templates/EXPERIMENT_MAKEFILE NNNNN-experiment-name/Makefile
   ```
   Edit it with the exact commands for data generation and training.

4. **Commit the initial README and Makefile:**
   ```bash
   git add NNNNN-experiment-name/
   git commit -m "Start experiment NNNNN: short description"
   ```

## Phase 2 — Run (development + training)

5. **Make code changes** in `dev/` as needed. Commit as you go.

6. **Run tests** before training:
   ```bash
   cd dev && make test
   ```

7. **Run training.** Document exact commands in the Makefile and README.

8. **Update the README** periodically as the experiment progresses:
   - Record intermediate results and observations
   - Note any methodology changes or surprises
   - Update the Makefile if commands change

9. **Save key results.** Capture loss milestones and final metrics, not full logs.
   Full training logs are too large for git — use head/tail excerpts or sample
   key lines (e.g., loss at checkpoints).

10. **Commit progress** as you go. README updates, code changes, results — all
    committed incrementally.

## Phase 3 — Finalize (after training)

11. **Finish the README.** Fill in Results, Analysis, and Conclusions. Set Status = Complete.

12. **Verify the Makefile** reproduces the experiment end-to-end.

13. **Run tests** one final time:
    ```bash
    cd dev && make test
    ```

14. **Tag the final state.** This captures all code changes made during the experiment:
    ```bash
    git tag c/NNNNN
    git push origin c/NNNNN
    ```

15. **Update the README header** with the tag and commit hash:
    ```
    **Source:** `c/NNNNN` (`<commit-hash>`)
    ```

16. **Commit and push:**
    ```bash
    git add NNNNN-experiment-name/
    git commit -m "Complete experiment NNNNN: short description"
    git push
    ```

---

## Quick Reference Checklist

```
[ ] Experiment directory + README created (Goal, Hypothesis, Method)
[ ] Makefile created with reproduction commands
[ ] Initial commit
[ ] Code changes committed incrementally
[ ] Tests pass
[ ] README updated with results, analysis, conclusions
[ ] Makefile verified
[ ] git tag c/NNNNN && git push origin c/NNNNN
[ ] README updated with source tag + commit hash, status = Complete
[ ] Final commit + push
```
