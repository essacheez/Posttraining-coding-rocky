# Skeptical expert review

## Verdict

The paper is complete and suitable for submission as a class-project report. Its best
feature is the evidence chain: it preserves the failed RLAIF run, separates training and
evaluation judges, reports both final coding snapshots, and does not hide disagreement
between persona metrics.

The central claim is appropriately narrow: one RLVR run recovered much of the measured
coding loss while retaining recognizable surface features of the target voice. The paper
does not claim that RLVR removed the trade-off in general or that the final checkpoint is
best on every metric.

## Points a skeptical reviewer should still notice

1. **One run per stage.** The results describe these training runs; they do not estimate
   variation across random seeds.
2. **Repeated benchmark inspection.** HumanEval, HumanEval+, and MBPP were measured at
   several RLVR checkpoints. The trajectory is useful diagnostics, not a pristine final
   test. LiveCodeBench is the cleaner final check.
3. **Two final snapshots.** The trajectory evaluation reports 0.695 HumanEval and 0.646
   HumanEval+, while the shared final rerun reports 0.683 and 0.628. The paper correctly
   reports both instead of selecting or averaging them.
4. **Mixed persona evidence.** Aggregate style cosine improves from 0.930 to 0.947, but
   mean per-response centroid distance worsens from 1.147 to 1.208. This supports
   retention of the surface style, not uniform improvement.
5. **AI judges are not ground truth.** The final GPT-5.4 rubric is separate from the
   GPT-5.4-nano training judge, which is good experimental hygiene, but both encode the
   team's style definition. The Claude audit and small teammate read are useful checks,
   not substitutes for a larger human evaluation.
6. **The strongest simpler alternative matters.** The self-generated-code SFT ablation
   is competitive on several coding and persona measurements. The paper treats it as a
   serious alternative rather than implying that every stage is necessary.

## Checks against misleading or AI-generated prose

- Training counts and splits are explicit: SFT uses 950 optimization examples, RLAIF
  uses 570, and RLVR uses all 700 selected problems twice with no internal split.
- `pass@1` is defined at first use.
- The 117-feature style cosine and the two judge roles are explained.
- Synthetic target rewrites are not called human references or canon ground truth.
- The paper avoids unsupported causal language, statistical ceremony, and claims about
  uncomputed results.
- All scheduled evaluations are reported as complete.

## Recommended final framing

SFT installed measurable surface style and coincided with a coding regression. The
revised RLAIF objective removed the observed catchphrase exploit. In one final run, RLVR
recovered a substantial fraction of the recorded coding performance, and the completed
persona evaluation found recognizable style without the same marker-gaming pattern.
