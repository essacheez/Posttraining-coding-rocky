"""Scores a response against the Rocky style guide on five dimensions, 1 to 5 each.

Provider-agnostic on purpose. `score_response` takes a `complete` callable, str ->
str, that sends one prompt to a judge model and returns its text. The eval harness
and the GRPO training loop both go through here, so the instrument is identical on
both sides; a divergence there would make the RLAIF-vs-SFT comparison meaningless.

    from judge import build_completer, score_responses, reward, REWARD_MIN

    complete = build_completer("claude-sonnet-5")       # or your own callable
    scored = score_responses(rollouts, complete)
    rewards = [reward(s) if s else REWARD_MIN for s in scored]

`build_completer` is a convenience for the CLI and covers openai/anthropic. Anything
callable works, so a local vLLM server or a batching client drops in unchanged.
"""

import json
import math
import os
import re
import statistics
from collections import Counter
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

# Every "## " section of rocky_style.md has to be named in one of these. A section
# that is in neither is an error rather than a default, because the silent failure
# here is a rule someone added that the judge never sees.
STYLE_INCLUDE = ("Scope", "Rules")
STYLE_EXCLUDE = ("Notes",)

SECTION_HEADING = re.compile(r"^## +(?P<name>.+?)[ \t]*$", re.MULTILINE)
FENCE_LINE = re.compile(r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
# a multi-line inline code span, which is what unbalanced bare fences usually are
INLINE_RUN = re.compile(r"(`{3,})(?:(?!\1).)+?\1", re.DOTALL)
BLANK_RUN = re.compile(r"\n{3,}")
# the policy under RL will find these if they pay, so they never reach the judge
CANDIDATE_TAG = re.compile(r"</?[ \t]*candidate[ \t]*>", re.IGNORECASE)


class JudgeError(Exception):
    """The judge returned something that is not a usable score."""


class EmptyProseError(JudgeError):
    """The response had no prose to score. A fact about the policy, not a fault."""


class JudgeUnavailable(Exception):
    """The judge client itself is broken. Not something a caller should swallow."""


def _normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _opens_fence(match):
    if match is None:
        return False
    # CommonMark: a backtick fence's info string may not contain a backtick
    return match.group("fence")[0] != "`" or "`" not in match.group("info")


def _closes_fence(match, fence):
    return (
        match is not None
        and match.group("fence")[0] == fence[0]
        and len(match.group("fence")) >= len(fence)
        and not match.group("info").strip()
    )


def strip_code_blocks(text):
    """Remove fenced code blocks. The persona lives in prose only.

    judge_prompt.txt asserts to the judge that code has already been removed, so
    scoring unstripped text measures something the rubric says it is not measuring.

    Fences are paired by scanning lines, because a regex that falls back to
    end-of-string deletes every remaining word when the fences do not balance --
    and under RL that deletion correlates with how the policy formats code, which
    makes it a systematic bias in the reward rather than noise.

    An unterminated fence that carries an info string (```python) is a real opener
    from a truncated generation, so its body is dropped. A bare unterminated fence
    is ambiguous: as likely the tail of a multi-line inline span as a block. There
    the text is kept and the span removed, because leaking a little code into the
    judge is a bounded error and deleting the prose is not.
    """
    lines = _normalize_newlines(text).split("\n")
    kept, pending = [], []
    fence = info = None

    for line in lines:
        match = FENCE_LINE.match(line)
        if fence is None:
            if _opens_fence(match):
                fence, info = match.group("fence"), match.group("info").strip()
                pending = []
            else:
                kept.append(line)
        elif _closes_fence(match, fence):
            fence = info = None
        else:
            pending.append(line)

    if fence is not None and not info:
        kept.append(fence)
        kept.extend(pending)
        stripped = INLINE_RUN.sub("", "\n".join(kept))
    else:
        stripped = "\n".join(kept)

    return BLANK_RUN.sub("\n\n", stripped).strip()


def load_style_guide():
    """The persona rules only, not the whole file.

    rocky_style.md also carries repo bookkeeping: which files consume it, a reminder
    to keep them in sync, and notes to ourselves about likely judge-gaming vectors.
    None of that is a style rule. Pasting the file whole also re-injects the literal
    "{STYLE_GUIDE}" that its own preamble mentions.

    Sections are matched by heading rather than sliced by offset, so editing the file
    fails loudly instead of quietly changing what the judge is scoring against.
    """
    with open(STYLE_FILE, encoding="utf-8") as style_in:
        text = _normalize_newlines(style_in.read())

    headings = list(SECTION_HEADING.finditer(text))
    if not headings:
        raise JudgeError(f"{STYLE_FILE} has no '## ' sections to read rules from")

    sections = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections[heading.group("name").strip()] = text[heading.start() : end].rstrip()

    unknown = [name for name in sections if name not in STYLE_INCLUDE + STYLE_EXCLUDE]
    if unknown:
        raise JudgeError(
            f"{os.path.basename(STYLE_FILE)} has unrecognized section(s) {unknown}. "
            "Add each to STYLE_INCLUDE or STYLE_EXCLUDE in judge/scorer.py so it is "
            "explicit whether the judge should see it."
        )
    missing = [name for name in STYLE_INCLUDE if name not in sections]
    if missing:
        raise JudgeError(
            f"{os.path.basename(STYLE_FILE)} is missing section(s) {missing}, which "
            "the judge scores against"
        )

    return "\n\n".join(sections[name] for name in STYLE_INCLUDE).strip()


def load_template():
    with open(PROMPT_FILE, encoding="utf-8") as prompt_in:
        template = prompt_in.read()
    return template.replace("{STYLE_GUIDE}", load_style_guide())


def build_prompt(prose, template=None):
    """Assemble the judge prompt, neutering anything that would break the frame.

    GRPO searches response space for whatever maximizes the reward, and a rollout
    that closes the <candidate> block early can address the judge directly. That
    exploit also defeats the markers-vs-human_plausibility detector, since an
    injected reply scores 5 on both and the gap stays flat.
    """
    template = load_template() if template is None else template
    return template.replace("{RESPONSE_PROSE}", CANDIDATE_TAG.sub("[tag removed]", prose))


def _coerce_score(dimension, value):
    if isinstance(value, bool):
        # int(True) is 1, an in-range score, so a malformed reply would pass as a floor
        raise JudgeError(f"{dimension} is {value!r}, expected a number")
    original = value
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise JudgeError(f"{dimension} is {original!r}, expected a number") from None
    if not isinstance(value, (int, float)):
        raise JudgeError(f"{dimension} is {original!r}, expected a number")
    # judges hedge with half points; truncating them is a consistent downward bias
    rounded = int(math.floor(value + 0.5))
    if not MIN_SCORE <= rounded <= MAX_SCORE:
        raise JudgeError(f"{dimension} is {original!r}, outside {MIN_SCORE}-{MAX_SCORE}")
    return rounded


def parse_scores(raw):
    """Pull the score object out of a judge completion.

    The prompt asks for bare JSON, but judges wrap it in prose or a fence often
    enough that falling back to the outermost braces is worth the few lines.
    """
    if not isinstance(raw, str):
        # openai types .content as Optional[str] and returns None on a content filter
        raise JudgeError(f"judge returned {type(raw).__name__}, expected text")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise JudgeError(f"no JSON object in judge output: {raw[:200]!r}") from None
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
        scores[dimension] = _coerce_score(dimension, parsed[dimension])

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
        raise EmptyProseError("nothing left to score after stripping code blocks")
    return parse_scores(complete(build_prompt(prose, template)))


def score_responses(responses, complete, max_workers=8, strip=True, quiet=False):
    """Score many responses, order preserved. Failures come back as None.

    A malformed judge reply must not take down a training step, so the caller
    decides what an unscored rollout is worth. Failures are counted and printed by
    category rather than swallowed: a rollout with no prose is a fact about the
    policy, while a 401 is a fact about our config, and the two need different
    responses from whoever is watching the run.

    Raises JudgeUnavailable if every call failed and any failure came from the
    client, which is the wrong-key case -- otherwise a whole GRPO run proceeds on a
    constant reward, with zero advantage and nothing printed.
    """
    template = load_template()

    def score_one(response):
        try:
            return score_response(response, complete, template, strip), None
        except EmptyProseError:
            return None, "no prose to score"
        except JudgeError:
            return None, "unusable judge output"
        except Exception as error:
            return None, f"client error ({type(error).__name__})"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        outcomes = list(pool.map(score_one, responses))

    scored = [scores for scores, _ in outcomes]
    problems = Counter(reason for _, reason in outcomes if reason)
    if problems and not quiet:
        print(
            f"judge: {sum(problems.values())} of {len(responses)} unscored -- "
            + ", ".join(f"{reason} x{count}" for reason, count in sorted(problems.items()))
        )

    client_errors = sum(
        count for reason, count in problems.items() if reason.startswith("client error")
    )
    if client_errors and not any(scored):
        raise JudgeUnavailable(
            f"all {len(responses)} judge calls failed, {client_errors} from the client "
            "itself. Check the model name, the API key and the network before spending "
            "more compute -- training on this would see a constant reward."
        )
    return scored


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


def build_completer(model=None, provider=None, max_tokens=512, max_retries=2):
    """A str -> str callable for the named model. Convenience, not a requirement.

    Reads JUDGE_MODEL / JUDGE_PROVIDER when not passed explicitly. Provider is
    inferred from the model name when it is unambiguous.

    max_retries goes to the client's own backoff. score_responses does not retry,
    so at rollout concurrency an un-retried 429 comes back as an unscored rollout
    and quietly distorts its group's advantage. Raise it for training use.
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

        client = Anthropic(max_retries=max_retries)

        def complete(prompt):
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            blocks = [block for block in message.content if block.type == "text"]
            if not blocks:
                raise JudgeError(f"no text block in judge reply ({message.stop_reason})")
            return blocks[0].text

        return complete

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(max_retries=max_retries)

        def complete(prompt):
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content

        return complete

    raise SystemExit(f"unknown provider {provider!r}, expected openai or anthropic")
