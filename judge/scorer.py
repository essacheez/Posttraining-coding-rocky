"""Scores a response against the Rocky style guide on five dimensions, 1 to 5 each.
The rubric lives in judge_prompt.txt and rocky_style.md; this file only fills and parses it."""

import json
import os
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import anthropic

JUDGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(JUDGE_DIR, "judge_prompt.txt")
STYLE_FILE = os.path.join(JUDGE_DIR, "rocky_style.md")

DIMENSIONS = ["sentence_structure", "vocabulary", "markers", "consistency", "human_plausibility"]
MIN_SCORE = 1
MAX_SCORE = 5

# independent of the SFT rewrite model and of gpt-5.4-nano, the RLAIF reward judge
JUDGE_MODEL = "claude-opus-5"
JUDGE_MAX_TOKENS = 16000
JUDGE_WORKERS = 8

# the Notes section of the style guide is bookkeeping for us, not rules for the judge
STYLE_SECTIONS = ["Scope", "Rules"]

# the judge quotes the response in its justification without escaping, so JSON is enforced by the API
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentence_structure": {"type": "integer"},
        "vocabulary": {"type": "integer"},
        "markers": {"type": "integer"},
        "consistency": {"type": "integer"},
        "human_plausibility": {"type": "integer"},
        "justification": {"type": "string"},
    },
    "required": ["sentence_structure", "vocabulary", "markers", "consistency", "human_plausibility", "justification"],
    "additionalProperties": False,
}

FENCED_BLOCK = re.compile(r"^[ \t]*```.*?^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)
FENCE_LINE = re.compile(r"^[ \t]*```.*$", re.MULTILINE)
CANDIDATE_TAG = re.compile(r"</?[ \t]*candidate[ \t]*>", re.IGNORECASE)
BLANK_RUN = re.compile(r"\n{3,}")


def strip_code_blocks(text):
    """Remove fenced code, since the persona applies to prose only.
    A leftover unmatched fence line is dropped on its own so the prose after it survives."""
    without_blocks = FENCED_BLOCK.sub("", text)
    without_fences = FENCE_LINE.sub("", without_blocks)
    return BLANK_RUN.sub("\n\n", without_fences).strip()


def load_style_guide():
    with open(STYLE_FILE, encoding="utf-8") as style_file:
        text = style_file.read()
    sections = {}
    for section in re.split(r"^## +", text, flags=re.MULTILINE)[1:]:
        heading = section.split("\n", 1)[0].strip()
        sections[heading] = "## " + section.rstrip()
    return "\n\n".join(sections[heading] for heading in STYLE_SECTIONS)


def load_template():
    with open(PROMPT_FILE, encoding="utf-8") as prompt_file:
        template = prompt_file.read()
    return template.replace("{STYLE_GUIDE}", load_style_guide())


def build_prompt(template, prose):
    # a response that closes the candidate block early could address the judge directly
    return template.replace("{RESPONSE_PROSE}", CANDIDATE_TAG.sub("[tag removed]", prose))


def parse_scores(raw_text):
    """Return the five integer scores plus the justification, or None if the reply is unusable.
    Falls back to the outermost braces because judges sometimes wrap the JSON in prose."""
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return None

    scores = {}
    for dimension in DIMENSIONS:
        value = parsed.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        rounded = int(value + 0.5)
        if rounded < MIN_SCORE or rounded > MAX_SCORE:
            return None
        scores[dimension] = rounded
    scores["justification"] = str(parsed.get("justification", ""))
    return scores


def judge_text(client, model, prompt):
    response = client.messages.create(
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        return None
    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "".join(text_blocks)


def score_response(client, model, template, response_text):
    """Score one response. Returns (scores, None) or (None, reason it went unscored)."""
    prose = strip_code_blocks(response_text)
    if not prose:
        return None, "no prose to score"
    raw_text = judge_text(client, model, build_prompt(template, prose))
    if raw_text is None:
        return None, "judge refused"
    scores = parse_scores(raw_text)
    if scores is None:
        return None, "unusable judge output"
    return scores, None


def score_responses(response_texts, model=JUDGE_MODEL, workers=JUDGE_WORKERS):
    """Score many responses in order. Unscorable ones come back as None, with a count printed.
    API errors are not caught, so a bad key or model name stops the run instead of wasting it."""
    client = anthropic.Anthropic()
    template = load_template()

    def score_one(response_text):
        return score_response(client, model, template, response_text)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(score_one, response_texts))

    reasons = Counter(reason for scores, reason in outcomes if reason)
    if reasons:
        summary = ", ".join(f"{reason} x{count}" for reason, count in sorted(reasons.items()))
        print(f"judge: {sum(reasons.values())} of {len(response_texts)} unscored ({summary})")
    return [scores for scores, reason in outcomes]


def total_score(scores):
    return sum(scores[dimension] for dimension in DIMENSIONS)


def aggregate(scored):
    """Mean per dimension, the 5 to 25 total, and the gaming gap (markers minus plausibility).
    A gap that widens across stages means the judge is being gamed, not the persona improving."""
    usable = [scores for scores in scored if scores]
    summary = {"judge_scored": len(usable), "judge_unscored": len(scored) - len(usable)}
    if not usable:
        return summary
    for dimension in DIMENSIONS:
        summary[f"judge_{dimension}"] = statistics.mean(scores[dimension] for scores in usable)
    summary["judge_total"] = statistics.mean(total_score(scores) for scores in usable)
    summary["judge_gaming_gap"] = summary["judge_markers"] - summary["judge_human_plausibility"]
    return summary
