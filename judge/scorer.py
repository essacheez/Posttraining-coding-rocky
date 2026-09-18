"""Scores a response against the Rocky style guide on five dimensions, 1 to 5 each.

Provider-agnostic on purpose. `score_response` takes a `complete` callable, str ->
str, that sends one prompt to a judge model and returns its text. The eval harness
and the GRPO training loop both go through here, so the instrument is identical on
both sides; a divergence there would make the RLAIF-vs-SFT comparison meaningless.

    from judge import score_responses, reward, build_completer

    complete = build_completer("claude-sonnet-5")       # or your own callable
    scored = score_responses(texts, complete)
    rewards = [reward(s) if s else None for s in scored]

`build_completer` is a convenience for the CLI and covers openai/anthropic. Anything
callable works, so a local vLLM server or a batching client drops in unchanged.
"""

import json
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

JUDGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(JUDGE_DIR, "judge_prompt.txt")
STYLE_FILE = os.path.join(JUDGE_DIR, "rocky_style.md")

DIMENSIONS = (
    "sentence_structure",
    "vocabulary",
    "markers",
    "consistency",
    "human_plausibility",
)
MIN_SCORE = 1
MAX_SCORE = 5
REWARD_MIN = len(DIMENSIONS) * MIN_SCORE  # 5
REWARD_MAX = len(DIMENSIONS) * MAX_SCORE  # 25

# the SFT rewrites came from this model; judging with it grades a model against its
# own habits, which is the one thing judge/README.md says not to do
REWRITE_MODEL = "gpt-4o-mini"

# rocky_style.md sections the judge should see; everything else in that file is
# notes to ourselves (see load_style_guide)
STYLE_SECTIONS = ("## Scope", "## Rules")
STYLE_STOP = "## Notes"

# opening fence + info string, body, then a matching close or end of string
FENCE_PATTERN = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=fence)[ \t]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)
BLANK_RUN = re.compile(r"\n{3,}")


class JudgeError(Exception):
    """The judge returned something that is not a usable score."""


def strip_code_blocks(text):
    """Remove fenced code blocks. The persona lives in prose only.

    judge_prompt.txt asserts to the judge that code has already been removed, so
    scoring unstripped text measures something the rubric says it is not measuring.
    """
    return BLANK_RUN.sub("\n\n", FENCE_PATTERN.sub("", text)).strip()


def load_style_guide():
    """The persona rules only, not the whole file.

    rocky_style.md also carries repo bookkeeping: which files consume it, a reminder
    to keep them in sync, and notes to ourselves about likely judge-gaming vectors.
    None of that is a style rule. Pasting the file whole also re-injects the literal
    "{STYLE_GUIDE}" that its own preamble mentions.
    """
    with open(STYLE_FILE, encoding="utf-8") as style_in:
        text = style_in.read()
    starts = [text.find(heading) for heading in STYLE_SECTIONS]
    starts = [index for index in starts if index != -1]
    if not starts:
        return text.strip()
    stop = text.find(STYLE_STOP, min(starts))
    return text[min(starts) : stop if stop != -1 else len(text)].strip()


def load_template():
    with open(PROMPT_FILE, encoding="utf-8") as prompt_in:
        template = prompt_in.read()
    return template.replace("{STYLE_GUIDE}", load_style_guide())


def build_prompt(prose, template=None):
    template = load_template() if template is None else template
    return template.replace("{RESPONSE_PROSE}", prose)


def parse_scores(raw):
    """Pull the score object out of a judge completion.

    The prompt asks for bare JSON, but judges wrap it in prose or a fence often
    enough that falling back to the outermost braces is worth the few lines.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise JudgeError(f"no JSON object in judge output: {raw[:200]!r}")
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as error:
            raise JudgeError(f"unparseable judge output: {raw[:200]!r}") from error

    if not isinstance(parsed, dict):
        raise JudgeError(f"judge returned {type(parsed).__name__}, expected an object")

    scores = {}
    for dimension in DIMENSIONS:
        if dimension not in parsed:
            raise JudgeError(f"judge output is missing {dimension!r}")
        try:
            value = int(parsed[dimension])
        except (TypeError, ValueError) as error:
            raise JudgeError(
                f"{dimension} is {parsed[dimension]!r}, expected an int"
            ) from error
        if not MIN_SCORE <= value <= MAX_SCORE:
            raise JudgeError(f"{dimension} is {value}, outside {MIN_SCORE}-{MAX_SCORE}")
        scores[dimension] = value

    scores["justification"] = str(parsed.get("justification", ""))
    return scores


def reward(scores):
    """Sum the five dimensions into the GRPO reward.

    Graded, not binary: GRPO takes the advantage within a group of rollouts, so a
    reward every rollout ties on carries no gradient.
    """
    return sum(scores[dimension] for dimension in DIMENSIONS)


def score_response(response, complete, template=None, strip=True):
    """Score one response. Raises JudgeError if the judge output is unusable."""
    prose = strip_code_blocks(response) if strip else response.strip()
    if not prose:
        raise JudgeError("nothing left to score after stripping code blocks")
    return parse_scores(complete(build_prompt(prose, template)))


def score_responses(responses, complete, max_workers=8, strip=True):
    """Score many responses, order preserved. Failures come back as None.

    A malformed judge reply must not take down a training step, so the caller
    decides what an unscored rollout is worth.
    """
    template = load_template()

    def score_one(response):
        try:
            return score_response(response, complete, template, strip)
        except JudgeError:
            return None
        except Exception:  # transport/rate-limit errors are the caller's to retry
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(score_one, responses))


def aggregate(scored, prefix="judge_"):
    """Mean per dimension plus the total, for a table row."""
    usable = [scores for scores in scored if scores]
    summary = {
        f"{prefix}scored": len(usable),
        f"{prefix}failed": len(scored) - len(usable),
    }
    if not usable:
        return summary
    for dimension in DIMENSIONS:
        summary[f"{prefix}{dimension}"] = statistics.mean(
            scores[dimension] for scores in usable
        )
    summary[f"{prefix}total"] = statistics.mean(reward(scores) for scores in usable)
    # the gaming detector: markers climbing while plausibility does not
    summary[f"{prefix}gaming_gap"] = (
        summary[f"{prefix}markers"] - summary[f"{prefix}human_plausibility"]
    )
    return summary


def build_completer(model=None, provider=None, max_tokens=512):
    """A str -> str callable for the named model. Convenience, not a requirement.

    Reads JUDGE_MODEL / JUDGE_PROVIDER when not passed explicitly. Provider is
    inferred from the model name when it is unambiguous.
    """
    model = model or os.environ.get("JUDGE_MODEL")
    if not model:
        raise SystemExit("set --judge-model or JUDGE_MODEL (e.g. claude-sonnet-5)")

    provider = provider or os.environ.get("JUDGE_PROVIDER")
    if not provider:
        if model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
            provider = "openai"
        else:
            raise SystemExit(
                f"cannot infer a provider from {model!r}, set JUDGE_PROVIDER "
                "to 'openai' or 'anthropic', or pass your own callable"
            )

    if model.startswith(REWRITE_MODEL):
        print(
            f"warning: judging with {model}, the model that generated the SFT "
            "rewrites. See judge/README.md on judge independence."
        )

    if provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic()

        def complete(prompt):
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

        return complete

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()

        def complete(prompt):
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content

        return complete

    raise SystemExit(f"unknown provider {provider!r}, expected openai or anthropic")
