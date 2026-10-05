# Post-training Qwen2.5-3B for the Rocky persona and coding

We teach `Qwen/Qwen2.5-3B-Instruct` to talk like Rocky, the alien engineer from
*Project Hail Mary*, and then try to win back the coding ability that the persona costs.

The full write-up is [`Report.pdf`](Report.pdf).

## The short version

Training runs in three stages. Each stage adds a LoRA adapter on top of the one before.

1. **SFT** teaches the voice. The model learns Rocky's grammar, but HumanEval drops by about 20 points.
2. **RLAIF** cleans up the voice. An LLM judge rewards style and correctness, and a rule penalizes catchphrase spam. Coding does not change.
3. **RLVR** wins back coding. The model trains on 700 coding problems and gets reward only when its code passes every test. The voice survives.

| model | HumanEval | HumanEval+ | MBPP | style cosine |
| --- | --- | --- | --- | --- |
| base Qwen | **76.2%** | **73.8%** | **58.8%** | 0.583 |
| after SFT | 55.5% | 50.6% | 55.6% | 0.879 |
| after RLAIF | 57.9% | 53.7% | 56.2% | 0.930 |
| after RLVR (final) | 69.5% | 64.6% | 58.4% | **0.947** |

Coding scores are pass@1 on the full benchmarks with greedy decoding. Style cosine
measures how close the writing is to the Rocky training answers (higher is closer). The
final model scores 18.5 out of 25 on a separate persona judge, against 19.9 for the
Rocky-style reference answers.

## Repo layout

| folder | what it holds |
| --- | --- |
| [`SFT/`](SFT/) | stage 1: data building, training code, and the trained adapter |
| [`RLAIF/`](RLAIF/) | stage 2: prompt dataset, judge prompt, training code, and the trained adapter |
| [`RLVR/`](RLVR/) | stage 3: problem pool, training data, verifier, training code, figures, and the trained adapter |
| [`EVALS/`](EVALS/) | coding and persona evaluation scripts, and the saved results of every run |
| [`judge/`](judge/) | the rubric judge used for the final persona score |
| [`ABLATION/`](ABLATION/) | the experiment that adds coding data to SFT and RLAIF |

Each stage folder has the same shape: `data/` for the dataset and the scripts that
build it, `training/` for the trainer, and `model/` for the trained adapter. Adapters
are stored with git-lfs, so run `git lfs pull` after you clone.

## Setup

```bash
uv sync
```

The RLAIF judge and the persona judge call the OpenAI API. Put `OPENAI_API_KEY` in a
`.env` file at the repo root. Nothing else needs a key.

## Stage 1: SFT

**Data.** We took 1,200 conversations from No Robots, Dolly-15k and OASST2 and
rewrote the answers in Rocky's voice. 1,000 are for training and 200 are held out for
the persona evaluation. The trainer splits the 1,000 into 950 for training and 50 for
validation.

The scripts in `SFT/data/script/` build the master dataset, rewrite it, and split it.
`prepare_sft_data.py` then converts the split into training format:

```bash
uv run python SFT/data/prepare_sft_data.py
```

**Train.** LoRA, three epochs, loss on the response only.

```bash
uv run python SFT/training/train_sft.py
```

**Result.** The adapter is `SFT/model/final_adapter`. You can chat with it using
`SFT/chat.py`.

The voice transfers well. The clearest single sign is the word "the", because Rocky
drops articles: it falls from 61.5 per 1,000 tokens in the base model to 4.9 after SFT.

The cost is coding. Only 21 of the 1,000 training answers contain code, and the model
also stops putting its code in markdown fences. Part of the drop may be formatting and
not wrong answers. We did not separate the two.

## Stage 2: RLAIF

**Data.** 600 prompts: 500 assistant prompts from No Robots, Dolly-15k and OASST2, 60
story prompts from the SFT set, and 40 coding prompts from the MBPP validation split.

```bash
uv run python RLAIF/data/build_grpo_dataset.py
```

**Reward.** GRPO with four reward terms:

| reward | what it checks | weight |
| --- | --- | --- |
| task | judge score for answering the question correctly, 0 to 5 | 1.0 |
| style | judge score for Rocky's voice, 1 to 5 | 0.6 |
| marker penalty | rule: too many "listen", "you", "yes", "good", or opening with "Listen" | 1.0 |
| format | rule: exactly one code block on coding prompts, and no overly long replies | 1.0 |

The judge is `gpt-5.4-nano` at temperature 0. Its prompt is `RLAIF/training/judge.txt`.

**Train.** Merge the SFT adapter into the base model, then run GRPO on top.
LoRA rank 32, 8 prompts x 8 samples per step, learning rate 1e-5, KL weight 0.04, 120 steps.

```bash
uv run python RLAIF/training/merge_adapter.py
uv run python RLAIF/training/train_rlaif.py
```

**Result.** The adapter is `RLAIF/model/RLAIF_model`, on top of `RLAIF/model/sft_merged`.
Style cosine rises from 0.879 to 0.930 and coding stays flat. The model also uses fewer
catchphrases: "you" drops from 39 to 17 per 1,000 words. In a blind audit, forced
catchphrases fall from 42% of replies after SFT to 0%.

**The first attempt failed.** The first reward was just the judge's task score plus
its style score, and the judge prompt named Rocky's catchphrases. The model learned to
open 91% of replies with "Listen. Listen." and to split code into many small blocks.
The judge reward kept rising while HumanEval fell from 0.58 to 0.07. The fix had three
parts: separate reward terms, a judge prompt that describes the voice without listing
catchphrases, and a stronger KL term (0.04, up from 0.004) at a lower learning rate.

## Stage 3: RLVR

**Problem pool.** We collected 2,550 coding problems from four public datasets and
kept 1,700. A problem stays only if it has at least five tests, a clear input format,
one correct answer, and a reference solution that passes its own tests. We removed
near duplicates, and we removed anything that overlaps HumanEval, MBPP or LiveCodeBench.

| source | test type | candidates | kept | used in training |
| --- | --- | --- | --- | --- |
| [KodCode-Light-RL-10K](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K) | function | 1200 | 800 | 562 |
| [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) (code) | stdin | 525 | 350 | 73 |
| [Klear-CodeTest](https://huggingface.co/datasets/Kwai-Klear/Klear-CodeTest) | stdin | 450 | 300 | 34 |
| [DeepCoder PrimeIntellect](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) | stdin | 375 | 250 | 31 |
| **total** | | **2550** | **1700** | **700** |

**Picking problems by difficulty.** GRPO learns nothing from a problem if all eight
samples pass or all eight fail. So before training, the RLAIF model answered every pool
problem eight times. We grouped problems by how many of the eight passed and picked 700.

| band | passes out of 8 | in pool | picked |
| --- | --- | --- | --- |
| too easy | 7 to 8 | 38 | 0 |
| easy | 5 to 6 | 113 | 113 |
| mid | 3 to 4 | 201 | 201 |
| hard | 1 to 2 | 318 | 316 |
| very hard | 0 | 1030 | 70 |

![Problems in the pool against problems picked, by difficulty](RLVR/figures/rlvr_curriculum.png)

Easy and mid problems were rare, so all of them went in. Training is not shuffled. It
goes from easy to very hard in each epoch.

**Reward.** 1.0 if the reply has exactly one code block and the code passes every test,
otherwise 0.0. Each test gets 4 seconds and 2 GB of memory. There is no persona reward
in this stage.

**Train.** A fresh LoRA adapter on top of the merged SFT + RLAIF model.

| setting | value |
| --- | --- |
| adapter | LoRA rank 32, alpha 64, all attention and MLP layers |
| batch | 8 problems x 8 samples per step |
| learning rate | 1e-5, constant after 5 warmup steps |
| KL weight | 0.01 |
| sampling | temperature 1.0, up to 3072 new tokens |
| length | two epochs over the 700 problems; we report the step 174 checkpoint |
| hardware | two L40S GPUs, 52 minutes |

```bash
uv run python RLVR/training/merge_adapter.py      # builds RLVR/model/rlaif_merged
mkdir -p logs && sbatch RLVR/training/train_rlvr.sbatch
```

The training data is already in `RLVR/data/rlvr_dataset.jsonl`, and the pool is in
`RLVR/data/rlvr_pool_1700.jsonl`. Do not run `RLVR/data/build_rlvr_dataset.py`. It
builds the earlier MBPP mix (see below) and overwrites the same file.

**Result.** The adapter is `RLVR/model/RLVR_model`, on top of `RLVR/model/rlaif_merged`.

![Coding pass@1 across RLVR training](RLVR/figures/rlvr_coding_by_step.png)

| step | HumanEval | HumanEval+ | MBPP | style cosine |
| --- | --- | --- | --- | --- |
| 0 (RLAIF) | 0.579 | 0.537 | 0.562 | 0.930 |
| 25 | 0.628 | 0.585 | 0.562 | 0.911 |
| 50 | 0.640 | 0.591 | 0.568 | 0.926 |
| 75 | 0.665 | 0.616 | 0.580 | 0.918 |
| 100 | 0.671 | 0.616 | 0.574 | 0.930 |
| 125 | 0.659 | 0.610 | 0.586 | 0.936 |
| 150 | 0.677 | 0.622 | 0.586 | 0.932 |
| **174** | **0.695** | **0.646** | **0.584** | **0.947** |

RLVR wins back 63% of the lost HumanEval ability, 54% of HumanEval+, and 85% of MBPP.
We checked these benchmarks during training, so read them as a progress signal, not as
a clean held-out test.

HumanEval is still about 7 points below the base model. When the final model fails, it
usually writes clean code that solves a simpler version of the task. For example, it
keeps one copy of a duplicate when the task says to drop all copies. The base model
restates the task and checks itself against the examples. The final model does not.

The voice survives. "The" stays at 4.7 per 1,000 tokens, and style cosine rises a
little. Style distance gets slightly worse (1.147 to 1.208), so we do not claim the
persona improved.

**Extra checks on the final model.**

| check | score |
| --- | --- |
| LiveCodeBench (175 recent contest problems) | 0.149 |
| MBPP test split, scored by our verifier | 0.586 |
| answers that try to cheat the tests | 0 |
| persona judge, out of 25 | 18.5 |
| human plausibility, out of 5 | 3.19 |

## Evaluation

### Coding

```bash
uv run python EVALS/evaluation.py --base                                    # base model
uv run python EVALS/evaluation.py --checkpoints SFT/model/final_adapter     # one adapter
uv run python EVALS/evaluation.py --base-model RLVR/model/rlaif_merged \
    --checkpoints RLVR/model/RLVR_model                                     # adapter on a merged model
uv run python EVALS/evaluation.py --benchmarks livecodebench                # opt-in benchmark
```

| benchmark | problems | runs by default |
| --- | --- | --- |
| HumanEval | 164 | yes |
| HumanEval+ | 164, with many more tests per problem | yes |
| MBPP | 500 (test split) | yes |
| LiveCodeBench | 175 (release v6, 2025-01 to 2025-04) | no |
| BigCodeBench | 200 of 1140 by default | no, needs `uv sync --extra bigcodebench` |

HumanEval+ reuses the HumanEval answers, so it costs no extra generation. Read the two
together: a big gap between them means the code passes the weak tests but is subtly wrong.

### Persona style

```bash
uv run python EVALS/evaluate_persona.py --base --checkpoints SFT/model/final_adapter
uv run python EVALS/evaluate_persona.py --score-only --no-embeddings    # rescore saved outputs, no GPU
```

The script generates answers for the 200 held-out prompts and compares them with the
1,000 Rocky training answers. It reports:

| metric | what it measures | better |
| --- | --- | --- |
| style cosine | similarity of 117 writing features: rates of 101 common words and 11 punctuation marks, sentence length, word length, vocabulary variety, capital letters | higher |
| style distance | how far each answer is from the average training answer | lower |
| memorization | largest 5-gram overlap with any training answer | lower |

These features fit Rocky well, because his voice is mostly grammar. He drops "the" and
"a", and he uses short flat sentences. The metrics measure surface style, not meaning.

The held-out Rocky answers are scored too, as a reference row. They were rewritten by
a model, so they show what the target style looks like, not perfect canon Rocky.

### Persona judge

The final persona score comes from a separate judge (`gpt-5.4`), not from the model
that gave rewards during training. It removes code blocks and then scores each answer
1 to 5 on five things: sentence structure, vocabulary, catchphrases, consistency, and
human plausibility. The total is 5 to 25.

```bash
uv run python EVALS/judge_persona.py --generations EVALS/results_rlvr/persona_generations.json
```

Watch the gap between the catchphrase score and the plausibility score. If catchphrases
go up and plausibility does not, the model is gaming the judge.

## Ablation: does adding code to training help?

Only 2.1% of the SFT data and 6.7% of the RLAIF data is code. We rebuilt both at 25%
code with the same total size and retrained.

| model | HumanEval | HumanEval+ | MBPP | persona judge |
| --- | --- | --- | --- | --- |
| original SFT | 57.3% | 53.0% | 55.4% | 20.02 |
| SFT + MBPP reference code | 56.7% | 50.0% | 54.4% | 19.69 |
| SFT + model-generated code | **70.1%** | **68.9%** | 57.6% | 19.73 |
| original RLAIF | 57.9% | 54.3% | 56.8% | 19.12 |
| RLAIF + MBPP reference code | 58.5% | 52.4% | 54.0% | 19.40 |

These numbers come from a separate evaluation run, so they differ a little from the
main table.

The style of the code matters more than the amount. MBPP's reference solutions are
written in a worse style than the base model's own code, and training on them does not
help. Code that the base model wrote itself and that passed every test brings HumanEval
back to 70.1% for a 0.3 point persona cost, without any RL.

```bash
# reference-code variant
uv run python ABLATION/data/build_ablation_data.py
uv run python ABLATION/training/train_sft_code.py
uv run python ABLATION/training/merge_sft_code.py
uv run python ABLATION/training/train_rlaif_code.py
bash ABLATION/eval_ablation.sh sft_code rlaif_code

# model-generated-code variant
uv run python ABLATION/data/build_selfgen_data.py
VARIANT=selfgen uv run python ABLATION/training/train_sft_code.py
```

`eval_ablation.sh` does not cover the model-generated variant yet. Its saved results
are in `EVALS/results_rlvr/sft_selfgen/`.

A natural next step is to run RLAIF and RLVR on top of the model-generated-code SFT model.

## Earlier RLVR experiment

Before the final RLVR run, we trained on 474 MBPP problems mixed with 203 persona
prompts. We tried two versions: one with only the test reward, and one with the test
reward plus all four RLAIF rewards. Both helped MBPP but did little for HumanEval+. The
code is `RLVR/training/train_rlvr_multiobjective.py`, the multi-objective adapter is
`RLVR/model/RLVR_multiobjective_model`, and the results are in `EVALS/results_rlvr/`.

## Limitations

- We ran each stage once, so we cannot measure variation across seeds.
- We checked the coding benchmarks during training.
- The style metrics and the judge rubric are our own design, not a standard.
