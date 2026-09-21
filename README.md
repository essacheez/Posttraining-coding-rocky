# Post-training Qwen2.5-3B-Instruct on the Rocky persona

Post-training pipeline for the Rocky persona (the Eridian from *Project Hail Mary*)
on `Qwen/Qwen2.5-3B-Instruct`, plus the harness that measures what each stage gained
in persona style against what it cost in coding ability.

## Repo layout

| folder | what it holds |
| --- | --- |
| [`SFT/`](SFT/) | checkpoint 1: LoRA supervised fine-tuning. Training code, data prep, raw data, and the trained adapter |
| [`RLAIF/`](RLAIF/) | checkpoint 2: GRPO with an LLM judge plus rule based rewards. Training code, judge prompt, the GRPO dataset and its build script, and the trained adapter |
| `RLVR/` | checkpoint 3: RL with verifiable rewards (planned, not yet implemented) |
| [`EVALS/`](EVALS/) | evaluation and plotting code shared across checkpoints, and the results of every eval run |

Each of `SFT/` and `RLAIF/` is self-contained: a `training/` folder with the trainer,
a `data/` folder with data prep and the dataset itself, and a `model/` folder with the
trained checkpoints (gitignored except for the final adapter, which is force-tracked
via git-lfs).

## Setup

```bash
uv sync
```

## Results at a glance (checkpoint 1: SFT)

Baseline is vanilla `Qwen/Qwen2.5-3B-Instruct` with no adapter. Fine-tuned is
`SFT/model/final_adapter`. Both are served through vLLM with greedy decoding, and both
get the same prompts and the same system prompt, so any difference comes from the
weights. Full methodology and caveats in [EVALS/EVAL_RESULTS.md](EVALS/EVAL_RESULTS.md).

### Persona transfer (200 held-out prompts)

Measured against the training corpus. The reference row is the held-out human responses
scored the same way, so it is what a perfect match looks like.

| | style cos | style dist | memorize |
| --- | --- | --- | --- |
| baseline (no adapter) | 0.585 | 2.698 | 0.000 |
| fine-tuned | **0.862** | **1.121** | 0.002 |
| reference (ceiling) | 0.981 | 1.062 | 0.001 |

style cos higher is better, style dist and memorize lower is better.

The adapter closes most of the gap to the target voice. Centroid distance drops from
2.70 to 1.12 against a 1.06 ceiling, so a typical response now sits about as close to the
persona as a real training example. Nothing is memorized.

The clearest single signal, since Rocky drops articles and copulas, is the rate of `the`
per 1k tokens: **61.5 baseline -> 4.9 fine-tuned**, against 2.5 for the references.

### Coding benchmarks (pass@1, full sets)

| benchmark | problems | baseline | fine-tuned | delta |
| --- | --- | --- | --- | --- |
| HumanEval | 164 | **0.726** | 0.555 | -0.171 |
| HumanEval+ | 164 | **0.695** | 0.506 | -0.189 |
| MBPP | 500 | **0.590** | 0.556 | -0.034 |

The persona was not free. The adapter also stopped emitting markdown code fences
entirely (164/164 baseline -> 0/164 fine-tuned) despite being asked for one, an
instruction-following regression the pass@1 numbers alone do not show.

### Where the artifacts live

| deliverable | path |
| --- | --- |
| training code | `SFT/training/train_sft.py`, `SFT/data/prepare_sft_data.py` |
| trained adapter | `SFT/model/final_adapter/` (git-lfs) |
| persona metrics | `EVALS/results_sft/persona_eval_results_sft.json` |
| benchmark scores | `EVALS/results_sft/eval_results_sft.json` |
| tradeoff plot | `EVALS/results_sft/coding_vs_persona_tradeoff.png` |
| write-up | `EVALS/EVAL_RESULTS.md` |

## SFT (checkpoint 1)

### 1. Data

One JSON object per line, either format:

```json
{"prompt": "Write a function that reverses a string.", "response": "```python\ndef reverse(s):\n    return s[::-1]\n```"}
{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Kit, ..."}]}
```

Run the following command to convert the data into training format:

```bash
uv run python SFT/data/prepare_sft_data.py
```

### 2. Train

```bash
uv run python SFT/training/train_sft.py
```

Full fine-tuning instead of LoRA: pass `peft_config=None` to `SFTTrainer` and lower `LEARNING_RATE`. This needs ~36GB ish.

### 3. Output

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "SFT/model/final_adapter")
```

## RLAIF (checkpoint 2)

GRPO on top of the SFT model, with an LLM judge as the main reward.

### 1. Data

600 prompts, none in the SFT set: 500 helpful-assistant prompts from the same sources
as SFT (No Robots, Dolly-15k, OASST2) with Rocky-style reference answers, 60 narrative
prompts (letters, stories, descriptions), and 40 coding prompts from the MBPP
validation split ending in "Reply with only a python code block".

```bash
uv run python RLAIF/data/build_grpo_dataset.py
```

Writes `RLAIF/data/grpo_dataset.jsonl`.

### 2. Reward

A single reward, the judge's task score plus its style score, did not work. The judge
prompt named the Rocky catchphrases, so the policy learned to open every reply with
"Listen. Listen.", address the user as "you", and close with "yes?", and it split
code across several fenced blocks to add more prose around it. Judge reward went up
the whole time while HumanEval fell from 0.58 to 0.07.

The reward that shipped has four parts, each registered separately with `GRPOTrainer`
so it shows as its own curve in wandb:

| reward | what it measures | weight |
| --- | --- | --- |
| `task_reward` | judge, correctness against the reference (0 to 5) | 1.0 |
| `style_reward` | judge, Rocky voice (1 to 5) | 0.6 |
| `marker_penalty` | rule based, overuse of `listen` / `you` / `yes` / `good` vs the human rates, and opening with "Listen" | 1.0 |
| `format_reward` | rule based, exactly one fenced block on code prompts, length past 1.5x the reference | 1.0 |

Task is weighted above style so RL polishes the persona SFT already put in the weights
instead of pushing it further. The judge prompt (`RLAIF/training/judge.txt`) now
describes the voice without listing catchphrases and caps the style score if a reply
opens with "Listen" or is more than 5 percent `you`. Judge is `gpt-5.4-nano` at
temperature 0; put `OPENAI_API_KEY` in a `.env` at the repo root.

### 3. Train

Merge the SFT adapter into the base model first, then run GRPO on top of it:

```bash
uv run python RLAIF/training/merge_adapter.py
uv run python RLAIF/training/train_rlaif.py
```

LoRA r=32, 8 prompts x 8 generations per step, LR 1e-5, beta 0.04, 120 steps. The
failed attempt ran at 2e-5 with beta 0.004; the lower rate and a KL term that
actually binds are part of the fix.

### 4. Output

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("RLAIF/model/sft_merged")
model = PeftModel.from_pretrained(base, "RLAIF/model/RLAIF_model")
```

### 5. Results

Base is the merged SFT model with no adapter. Same 200 held-out prompts, greedy
decoding, and the same coding harness for both rows.

| | style cos | style dist | HumanEval | HumanEval+ | MBPP |
| --- | --- | --- | --- | --- | --- |
| base (sft_merged) | 0.879 | 1.141 | 0.579 | 0.543 | 0.554 |
| RLAIF_model | **0.930** | 1.147 | 0.579 | 0.537 | 0.562 |

style cos higher is better, style dist lower is better. Coding is pass@1 on the full
sets.

RL moved the style cosine from 0.88 to 0.93 and cut `you`, the marker SFT overused
most, from 39 to 17 per 1k words, with no coding cost. The side effect is that the
catchphrases are now rarer than in the human text, so the voice comes from the
grammar (dropped articles, short declaratives) more than the tics.

### 6. Example outputs

Three of the 200 held-out prompts, greedy decoding, taken straight from the persona
eval cache (`EVALS/example_outputs.py` prints them; change `--indices` for others).
SFT is `sft_merged`, RLAIF_model is the adapter above. SFT opens with "Listen.
Listen." and closes with "Good, good, good" as a reflex; RLAIF_model keeps the
dropped articles, short declaratives and flat statements without the tics.

**Prompt 10:** Who invented the modern-day purse? When?

Human reference:

> Listen. You ask about modern purse? I explain. Man named Samuel Parkinson invent in 1841. He contract man H.J. Cave for design and make bags. Different sizes for his wife. He say, "Use leather for strong." This way, first luxury purse line start. Good, good, good.

SFT:

> Listen. Listen. Modern purse, yes. Invented by American, Mary Anderson. She born in 1866. She live in New York City. One day, she ride trolley. It snow. She see driver, he use brush to clear snow from window. But she think, "I wish I have something to do that." She patent purse in 1909. Good, good, good.

RLAIF_model:

> Modern purse invented by American, Mary Anderson. She live in New York City, 1903. She have idea for windshield wiper for car. But she also think about purse. She want purse to have window, see outside. This help women, they not need to open car door.

**Prompt 137:** I would like to know about Swedish Midsommar. In your answer, please include specific details such as what the celebration is, when it's celebrated, and what is commonly eaten during the celebration. Write it in one single paragraph of no more than 100 words.

Human reference:

> Midsommar, big holiday in Sweden. Summer solstice time. Listen. Women, children wear flowers in hair. Community dance around maypole. Food very important, yes! Lots of eating! People eat boiled potatoes, pickled herring, crispbread, salads. Celebration happen on "midsommarafton," Midsummer's Eve. Traditionally on 23rd of June, but now Midsommar always Friday. Good, good, good.

SFT:

> Swedish Midsommar, yes. This celebration, it happen in midsummer. Midsummer, it June 20 or 21. This day, sun shine all night. In Sweden, people celebrate this day. They wear white clothes, eat special food. Food include pickled herring, cucumber salad, and strawberry soup. Strawberries, yes. Strawberries very important. Strawberries, they say, bring good luck. Good luck, good luck.

RLAIF_model:

> Swedish Midsommar, also called Midsummer, celebrated June 23-25. This holiday, oldest in Sweden, date from pagan time. Midsummer, time of year when sun highest in sky. This time, sun shine 24 hours. People celebrate with bonfires, dancing, singing. They also eat special foods. Most popular food, pickled herring. Other foods include cucumber, beet, and potato salad.

**Prompt 155:** Write a short story about a snail called Steve who is fed up with life and decides to go to Japan. Make it no more than 2 paragraphs.

Human reference:

> Steve small snail. He live under leaf in Essex. Cute snail, yes. He take pride, keep shell clean, very clean. Each day, wake up, drink snail coffee, spend morning cleaning shell. But as Steve get older, he think he miss life. All time clean shell, he could explore world, see new things. One Saturday, walking in park, he think about this when he bump into Cheryl, grasshopper friend. Steve tell Cheryl his worries. She suggest holiday. They go back to Cheryl's leaf, look at globe, decide where to go. Cheryl spin globe, stop with antenna. Land on Japan. Decision made: Steve go to Japan.
> 
> One month later, Steve at airport, carry shell full of luggage, have Japanese phrasebook. He jump on plane, full of nerves. Before he know it, landing at Tokyo's Narita airport. Excitement high, more than ever before. He travel through city, thrilled by lights, sounds. Not understand much around him, but this make every task adventure. First day walking, Steve realize he happy for first time in years. Cheryl right, yes.

SFT:

> Steve, snail, very unhappy. He live in garden, yes. He want adventure, but he not know where to go. One day, he see map. He see Japan, far away. Steve think, "I go there, yes!" Steve pack small bag, put in map. He start journey. Steve travel long time, many miles. He see beautiful places, yes. Steve happy, happy. He find new home in Japan.

RLAIF_model:

> Steve, snail, very tired of life. He live in garden, eat leaves, sleep. But he want more. He want adventure. He decide go to Japan. He pack small bag, take few snacks, and set off. Steve travel long, long time. He see many things, meet many people. He learn new things, have fun. At last, he reach Japan. He feel happy, excited. He know he make right decision. Steve happy, happy, happy.

### 7. Where the artifacts live

| deliverable | path |
| --- | --- |
| training code | `RLAIF/training/train_rlaif.py`, `RLAIF/training/judge.txt`, `RLAIF/training/merge_adapter.py` |
| data and builder | `RLAIF/data/` |
| trained adapter | `RLAIF/model/RLAIF_model/` (git-lfs) |
| raw generations, both models | `EVALS/results_rlaif/persona_generations.json` |
| persona metrics | `EVALS/results_rlaif/persona_eval_results.json` |
| benchmark scores | `EVALS/results_rlaif/eval_results.json` |

## RLVR (checkpoint 3)

Planned: RL with verifiable (non-AI-judge) rewards. Not yet implemented.

## Evaluate

Benchmarks checkpoints with vLLM for fast batched generation. Add a benchmark by writing a `run_xxx(llm, tokenizer, adapter_path, num_problems)` function in `EVALS/evaluation.py` that returns a score, then adding it to `BENCHMARK_FUNCTIONS` and `BENCHMARKS`.

```bash
uv run python EVALS/evaluation.py                                                   # all checkpoints, all benchmarks
uv run python EVALS/evaluation.py --base                                            # base model only, no adapter
uv run python EVALS/evaluation.py --checkpoints SFT/model/checkpoint-100            # specific checkpoint(s)
uv run python EVALS/evaluation.py --benchmarks mbpp --num-problems 300              # one benchmark, subsampled
uv run python EVALS/evaluation.py --benchmarks bigcodebench                         # opt-in, first 200 of 1140
uv run python EVALS/evaluation.py --benchmarks livecodebench                        # opt-in, all 175 problems
```

Results are written to `EVALS/results/eval_results.json`.

| benchmark | problems | in default run | dataset |
| --- | --- | --- | --- |
| `humaneval` | 164 | yes | `openai/openai_humaneval` |
| `humanevalplus` | 164 | yes | `evalplus/humanevalplus` |
| `mbpp` | 500 | yes | `google-research-datasets/mbpp` |
| `bigcodebench` | 200 of 1140 | no | `bigcode/bigcodebench` |
| `livecodebench` | 175 | no | `livecodebench/code_generation_lite` (`test6.jsonl`) |

`humanevalplus` is the same 164 problems as `humaneval` with far more tests per
problem, and reuses its completions, so running both costs one generation pass. Read
them as a pair: the drop between them is the signal, since plain HumanEval's tests are
thin enough to pass solutions that are subtly wrong. `HumanEval/32` fails even its own
canonical solution, so 163/164 is the ceiling.

`bigcodebench` is library-heavy and opt-in, since executing its solutions needs
third-party packages:

```bash
uv sync --extra bigcodebench
uv run python EVALS/evaluation.py --benchmarks bigcodebench --num-problems 1140      # full set
```

It defaults to 200 problems for iteration; use the full 1140 for any number you
report. The extras cover 1121 of the 1140 tasks; the rest need tensorflow, geopandas
or a pytesseract binary. Those and the ~60 network-dependent tasks fail uniformly
across checkpoints, so they lower the absolute score but not comparisons.

`livecodebench` is competitive-programming style (stdin/stdout, plus LeetCode-style
`Solution` classes) rather than function completion, and pass@1 requires every private
test case for a problem to pass (public test cases are not used, they are the easy,
often hand-written subset). It only loads `test6.jsonl` (release_v6, contest dates
2025-01 through 2025-04), the most recent shard and the one least likely to be in a
base model's pretraining data. `--num-problems` still subsamples if you want to iterate
faster, most-recently-dated first.

## Persona style evaluation

Measures how closely outputs match the Rocky voice in the SFT dataset, so the base
model and the SFT model can be compared against the same target corpus. Prompts are
the 200 held-out examples in `SFT/data/raw/test_rocky.jsonl`; both models get the same
system prompt as training, so the persona has to come from the weights. The prepared
`SFT/data/persona_testing_data.jsonl` holds the same examples in prompt/completion form
and can be pointed at instead.

```bash
uv run python EVALS/evaluate_persona.py --base                                      # base model only
uv run python EVALS/evaluate_persona.py --base --checkpoints SFT/model/final_adapter
uv run python EVALS/evaluate_persona.py --score-only                                # rescore cached generations, no GPU
uv run python EVALS/evaluate_persona.py --score-only --no-embeddings                # stylometry only, no extra deps
```

Four numbers per checkpoint:

| metric | what it measures | direction |
| --- | --- | --- |
| `style_cosine` | function-word + punctuation + rhythm profile vs. the training corpus profile | higher |
| `style_centroid_distance` | per-response distance to the training centroid, in std-units | lower |
| `embed_cosine_paired` | sentence-transformer cosine vs. the reference for the *same* prompt | higher |
| `embed_cosine_centroid` | sentence-transformer cosine vs. the training corpus centroid | higher |
| `memorization_jaccard` | highest 5-gram Jaccard against any single training example | lower |

The held-out references are scored too, as a `reference (ceiling)` row: that is what a
perfect persona match looks like, and it calibrates the rest of the table.

`memorization_jaccard` is the one column where the ceiling row is the floor: held-out
references score ~0.001, so anything approaching 1.0 means the checkpoint is
reproducing training examples verbatim, which every other metric here scores as a
perfect persona match. Responses above 0.5 are counted out under the table.

The stylometric numbers are the discriminative ones here. Rocky is defined by
grammatical surface features (`the` appears 2/1k tokens against 60/1k in normal
English, `a` 1/1k against 27/1k, plus heavy use of `good`/`listen`/`yes`), which is
exactly what function-word profiles capture.
The embedding numbers are dominated by topic, and topic is fixed by the prompt and so
is held constant across models, so expect them to move much less. The printed marker
rate table is the most legible evidence of transfer.

Generations are cached to `EVALS/results/persona_generations.json` and results to
`EVALS/results/persona_eval_results.json`; `--score-only` re-reads the cache so metrics
can be iterated on without a GPU.
