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
python RLVR/figures/plot_rlvr.py          # Figures 3 and 4
python report/scripts/fig_canon_rater.py  # Figure 5
python report/scripts/fig_collapse.py     # Figure 6
python report/scripts/build_pdf.py      # REPORT.md -> REPORT.pdf (uses headless Google Chrome)
```

## Open slot in REPORT.md

All training and code-benchmark results are incorporated. One result remains: the
independent AI-judge evaluation of `RLVR/model/RLVR_model`. Add it to Section 2 and
Section 6 when it lands. Report the rubric's five dimensions separately and lead with
`human_plausibility`: the `markers` dimension rewards habits the report shows are
off-canon. Do not label a checkpoint the final winner until this comparison is present.

Two optional follow-ups remain marked `[OPEN]`: split SFT's coding loss into formatting
versus logic, and re-score step 75 of the failed first RLAIF attempt with the fixed
extractor. Neither blocks the final results narrative.
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

Figures 3 and 4 are generated from `RLVR/figures/plot_rlvr.py` and embedded directly in
the report, so the report and README share one reproducible source of truth.
