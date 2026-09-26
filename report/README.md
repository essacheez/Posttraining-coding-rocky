# Final report

`REPORT.md` is the source and `REPORT.pdf` is the rendered report, with the figures embedded.
Edit the markdown, then rebuild the PDF. Never edit the PDF directly.

## Rebuild

From the repo root:

```bash
uv pip install matplotlib numpy scipy markdown pypdf   # if not already in the env
python report/scripts/collect.py        # gathers generations and surface metrics into report/data/
python report/scripts/stats.py          # every CI, p-value and penalty number cited in the report -> data/stats.json
python report/scripts/fig_tradeoff.py   # Figure 1
python report/scripts/fig_markers.py    # Figure 2
python report/scripts/fig_canon_rater.py  # Figure 4
python report/scripts/fig_collapse.py   # Figure 5
python report/scripts/build_pdf.py      # REPORT.md -> REPORT.pdf (uses headless Google Chrome)
```

## Open slots in REPORT.md

Each open slot is marked in the text, and the PDF highlights them in yellow.

- `[TODO RLVR]`: RLVR results. `RLVR/eval_stages.sh` on the `rlvr` branch writes
  `EVALS/results_rlvr/<stage>/`. Fill the Section 2 table, add the RLVR runs to Figure 1,
  and add Figure 3 (HumanEval+ against judge score; `EVALS/plot_stages.py` is a starting
  point). Use HumanEval+ as the main STEM axis, not MBPP, because the RLVR prompts show
  the MBPP asserts.
- `[TODO JUDGE]`: held-out judge scores for every stage, including the first RLAIF attempt,
  whose generations are in `data/first_rlaif_attempt/`. Report the rubric's five dimensions
  separately and lead with `human_plausibility`: the `markers` dimension rewards habits the
  report shows are off-canon.
- `[OPEN]`: two items that need per-problem completions from `EVALS/evaluation.py`: the
  split between formatting loss and logic loss in SFT, and a re-score of step 75 of the
  first attempt with the fixed extractor.

## Data

| file | what it is |
| --- | --- |
| `data/all_generations.json` | held-out persona generations for every stage, plus the reference replies |
| `data/first_rlaif_attempt/` | the abandoned first RLAIF run, recovered from commit `070e788`: trainer log, evals, generations |
| `data/rocky_canon.md` | the canon speech profile of book Rocky, written by a research agent with cited sources |
| `data/opus_rater_key.json`, `data/ratings_{1,2}.json` | the blind canon-grounded rating: the answer key, and the raw ratings from two rater agents (160 replies) |
| `data/live_key.json`, `data/live_ratings.json` | the blind human rating (16 replies) and its answer key |
| `data/surface_metrics.json`, `data/stats.json` | derived numbers; regenerate with the scripts above |

Figure 3 is not produced yet, so the figure numbering skips from 2 to 4 until the RLVR results land.
