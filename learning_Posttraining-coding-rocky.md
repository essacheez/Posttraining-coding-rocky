# Learnings: Essa's preferences

- Never modify an earlier stage's pipeline or results (`SFT/`, `RLAIF/`, `EVALS/results_sft/`, `EVALS/results_rlaif/`) when building a later one; add new files and new results dirs instead.
- Persona adherence is reported with the five dimension rubric judge (`judge/`), not the RLAIF reward judge in `RLAIF/training/judge.txt`.
- Pull from `main` before starting work on a new piece.
- Verify against the real repo state before claiming something is missing; the eval harness is more complete than a quick grep suggests.
- A later stage reuses an earlier stage's reward or eval code by importing it, never by copying or editing it, so the instrument stays identical across stages.
- Always pass `--results-dir` to `EVALS/evaluation.py`; its default is `EVALS/results_rlaif`, so a bare run overwrites committed RLAIF results.
