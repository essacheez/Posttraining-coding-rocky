"""Every inferential number in the report: bootstrap CIs, paired rating tests, penalty coverage, sampling error."""
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict

from scipy import stats

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(WORK)
sys.path.insert(0, os.path.join(REPO, "EVALS"))
from evaluate_persona import FEATURE_NAMES, cosine, load_persona_corpus, profile_vector

SEED = 0
BOOTSTRAPS = 2000
REFERENCE_MARKER_RATES = {"listen": 7.0, "you": 13.0, "yes": 6.0, "good": 15.0}

data = json.load(open(os.path.join(WORK, "data", "all_generations.json")))
sft, rlaif, v1, reference = (data["sets"][key] for key in ["SFT (gen B)", "v2 final", "v1 step 112", "human reference"])
corpus_profile = profile_vector(load_persona_corpus())
results = {}


def style_gain(after, before, indices, dropped=()):
    keep = [index for index, name in enumerate(FEATURE_NAMES) if name not in dropped]
    target = [corpus_profile[index] for index in keep]
    after_vector = profile_vector([after[i] for i in indices])
    before_vector = profile_vector([before[i] for i in indices])
    return cosine([after_vector[i] for i in keep], target) - cosine([before_vector[i] for i in keep], target)


rng = random.Random(SEED)
all_indices = list(range(200))
samples = sorted(style_gain(rlaif, sft, [rng.randrange(200) for _ in range(200)]) for _ in range(BOOTSTRAPS))
results["style_gain_rlaif_vs_sft"] = {
    "gain": style_gain(rlaif, sft, all_indices),
    "ci95": [samples[int(0.025 * BOOTSTRAPS)], samples[int(0.975 * BOOTSTRAPS)]],
    "gain_without_you_and_i": style_gain(rlaif, sft, all_indices, dropped=("word:you", "word:i")),
}


def penalty(text):
    words = re.findall(r"[a-z']+", text.lower())
    total = max(len(words), 1)
    excess = sum(max(0.0, 1000 * words.count(marker) / total - 1.5 * rate) / rate for marker, rate in REFERENCE_MARKER_RATES.items())
    value = -0.5 * excess - (0.5 if text.lstrip().lower().startswith("listen") else 0.0)
    return max(-2.0, value)


results["catchphrase_penalty"] = {}
for name, texts in [("reference", reference), ("SFT", sft), ("first attempt", v1), ("RLAIF", rlaif)]:
    values = [penalty(text) for text in texts]
    results["catchphrase_penalty"][name] = {"share_penalized": sum(v < 0 for v in values) / len(values), "mean": statistics.mean(values)}

# coding: binomial standard error at the observed rate, n = 164
results["coding_sampling_error_95"] = {name: 1.96 * math.sqrt(p * (1 - p) / 164) for name, p in [("HumanEval+ SFT", 0.543), ("HumanEval+ base", 0.707)]}

# canon rater: paired by prompt
key = {row["id"]: row for row in json.load(open(os.path.join(WORK, "data", "opus_rater_key.json")))}
ratings = json.load(open(os.path.join(WORK, "data", "ratings_1.json"))) + json.load(open(os.path.join(WORK, "data", "ratings_2.json")))
rater_of = {row["id"]: 1 for row in json.load(open(os.path.join(WORK, "data", "ratings_1.json")))}
by_prompt = defaultdict(dict)
by_source_rater = defaultdict(list)
for rating in ratings:
    info = key[rating["id"]]
    by_prompt[info["index"]][info["source"]] = rating["rocky"]
    by_source_rater[(info["source"], rater_of.get(rating["id"], 2))].append(rating["rocky"])


def paired(first, second):
    diffs = [scores[first] - scores[second] for scores in by_prompt.values()]
    boot_rng = random.Random(SEED)
    boots = sorted(statistics.mean(boot_rng.choice(diffs) for _ in diffs) for _ in range(BOOTSTRAPS))
    return {"mean_diff": statistics.mean(diffs), "ci95": [boots[int(0.025 * BOOTSTRAPS)], boots[int(0.975 * BOOTSTRAPS)]], "wilcoxon_p": stats.wilcoxon([scores[first] for scores in by_prompt.values()], [scores[second] for scores in by_prompt.values()]).pvalue}


results["canon_rater_paired"] = {
    "reference - SFT": paired("human reference", "SFT (gen B)"),
    "RLAIF - SFT": paired("v2 final", "SFT (gen B)"),
    "first attempt - SFT": paired("v1 step 112", "SFT (gen B)"),
}
results["canon_rater_by_rater"] = {f"{source} / rater {rater}": statistics.mean(scores) for (source, rater), scores in sorted(by_source_rater.items())}

json.dump(results, open(os.path.join(WORK, "data", "stats.json"), "w"), indent=1)
print(json.dumps(results, indent=1))
