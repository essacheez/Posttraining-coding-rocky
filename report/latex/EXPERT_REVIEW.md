# Skeptical expert review

## Verdict

**Complete as a class-project paper; weak reject only if judged as a fully reproducible
research result.**

The project has a worthwhile empirical story, candid failure analysis, and useful saved
outputs. The main risk is not weak prose. It is that a polished narrative can obscure
contradictions between the manuscript, code, and committed data. A demanding reviewer
would still question whether the training run can be independently reproduced.

All scheduled evaluations are now present, including the final persona judge,
LiveCodeBench, and verifier check. The manuscript reports both final coding snapshots
instead of selecting the higher one.

## Remaining validity issues

### 1. The latest RLVR run is not reproducible

The paper describes 700 problems selected from a 1,700-problem pool. The repository does
not contain that pool, the selected dataset, or the builder and deduplication code. The
tracked `RLVR/data/rlvr_dataset.jsonl` is an older 677-row MBPP/persona mixture with a
`tests` field. The latest `RLVR/training/train_rlvr.py` expects `test_cases`. The checked-in
command therefore cannot reproduce the reported model.

The curriculum plot also hard-codes the category counts instead of loading them from an
auditable manifest.

Required fix: commit the real builder plus source identifiers and hashes for the pool and
selected set. If examples cannot be redistributed, commit immutable dataset revisions,
selection code, selected IDs, source counts, and content hashes.

### 2. The final-checkpoint claim was factually wrong

HumanEval and HumanEval+ reach their highest observed values at step 174. MBPP reaches
0.586 at steps 125 and 150 and ends at 0.584. The benchmark curves also contain several
intermediate declines; improvement is not monotonic.

Required wording:

> HumanEval and HumanEval+ reached their highest observed scores at step 174. MBPP
> peaked at 0.586 at steps 125 and 150 and ended at 0.584. Because these suites were
> inspected throughout training, the curves are diagnostic rather than an unbiased final
> test.

### 3. Evaluation snapshots differ

The training-trajectory evaluation gives the final checkpoint 0.695 HumanEval and 0.646
HumanEval+, while the shared final rerun gives 0.683 and 0.628. MBPP is 0.584 in both.
The paper now reports both snapshots without averaging them, but cross-stage comparisons
still do not come from one frozen evaluation launch.

Required fix: evaluate base, SFT, RLAIF, and the selected RLVR checkpoint together under
one environment and command, save per-problem outputs, and compute paired intervals.
Until then, label the values as recorded point estimates from separate launches.

### 4. “Benchmark-disjoint generalization” is too strong

All curriculum checkpoints were inspected on HumanEval, HumanEval+, and MBPP, so those
suites function as development data for checkpoint inspection. HumanEval+ is also the
same 164 prompts and completions as HumanEval with expanded tests, not a third independent
problem set. Public pretraining contamination cannot be ruled out.

Required fix: describe two problem sets evaluated with three test suites. Say that
benchmark tasks were intended to be excluded from this post-training dataset, conditional
on an auditable deduplication process. Evaluate the frozen checkpoint once on a genuinely
unused suite.

### 5. Persona evidence is mixed but complete

Corpus-level style cosine improves from 0.930 to 0.947, while mean per-response centroid
distance worsens from 1.147 to 1.208. The completed five-dimension judge scores all 200
final responses at 18.5/25, including 3.19/5 for human plausibility. This completes the
evaluation but does not make the handcrafted metrics agree.

Required wording:

> RLVR retained prominent surface markers. Aggregate style cosine increased by 0.017,
> while mean per-response centroid distance worsened by 0.061. The completed rubric
> judge scores human plausibility at 3.19/5, so persona retention is supported but not
> uniformly improved across metrics.

### 6. The strongest alternative needs a fair comparison

The earlier `SFT + self-generated code` ablation reaches 0.701 HumanEval, 0.689
HumanEval+, 0.576 MBPP, and a 19.725 aggregate judge score. It is a serious alternative
to a multi-stage RL pipeline, although its style cosine and catchphrase behavior are
worse. A final Pareto plot should include this checkpoint, verifier-only RLVR,
multi-objective RLVR, RLAIF, and curriculum RLVR under one evaluation protocol.

## Additional corrections

- SFT has 1,000 rows but trains on 950 after a 5% split.
- RLAIF has 600 rows but trains on 570 after a 5% split. Its builder deliberately copies
  60 narrative prompts from the SFT training file.
- The reproducible SFT count is 21/1,000 targets with a fenced code block. The prior
  report's count of 34 was unsupported.
- The so-called “human references” are GPT-4o-mini rewrites of human-authored answers.
  Call them **synthetic target-style references**, not human references or a ceiling.
- Distinguish the GPT-5.4-nano training reward judge, the project-authored GPT-5.4
  evaluation rubric, the Claude canon-grounded audit, and the teammate blind read.
- The GPT evaluation rubric is not fully independent because it uses the same project
  style definition that shaped the training targets.
- Each canon reply is rated by only one of two model-rater instances; prompt and rater
  effects are confounded.
- The Wilcoxon p-value of 0.02 is exploratory and does not survive a simple Bonferroni
  correction across the three displayed comparisons.
- One training seed cannot estimate run-to-run variance. Benchmark binomial intervals do
  not substitute for repeated training.
- The pinned Python environment is unresolved because sentence-transformers requires
  Transformers below 5 while the project pins Transformers 5.17.

## AI-authorship warning signs

The earlier draft read as potentially AI-generated because it had a smooth causal story
that contradicted its own tables, promotional phrases such as “cleanly asks” and “won
back,” anthropomorphic explanations of what the model “learned,” selective emphasis on
the improving style metric, and no bibliography despite naming many methods and datasets.

The right remedy is not to make the prose more ornate. It is to make every sentence
traceable to an artifact, state what is missing, distinguish observation from causal
explanation, and give the strongest alternative pipeline equal space.

## Recommended framing

The defensible conclusion is:

> SFT strongly changed measurable surface style and coincided with coding regression.
> The revised RLAIF objective suppressed obvious verbal tics. One curriculum RLVR run
> recovered a substantial fraction of the recorded coding performance, and the completed
> rubric judge found moderate persona plausibility without a marker-gaming gap. The
> evidence does not establish that the curriculum caused the gain or that the final
> checkpoint is globally preferable.
