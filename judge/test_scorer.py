"""Offline checks for the judge plumbing. No network, no API key, no pytest.

    uv run python judge/test_scorer.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge import (  # noqa: E402
    DIMENSIONS,
    REWARD_MAX,
    REWARD_MIN,
    JudgeError,
    aggregate,
    build_prompt,
    load_style_guide,
    load_template,
    parse_scores,
    reward,
    score_response,
    score_responses,
    strip_code_blocks,
)

PERFECT = {dimension: 5 for dimension in DIMENSIONS}
FLOOR = {dimension: 1 for dimension in DIMENSIONS}


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"  ok  {name}")


def test_strip_code_blocks():
    print("strip_code_blocks")
    stripped = strip_code_blocks(
        "Listen. Listen. I show you.\n\n```python\ndef f():\n    return 1\n```\n\nThis code good."
    )
    check("fenced block removed", "def f()" not in stripped)
    check("prose before kept", "Listen. Listen. I show you." in stripped)
    check("prose after kept", "This code good." in stripped)

    check(
        "unclosed fence runs to end",
        "def f()" not in strip_code_blocks("Prose here.\n\n```python\ndef f():\n    return 1"),
    )
    check(
        "tilde fence removed",
        "secret" not in strip_code_blocks("Prose.\n\n~~~\nsecret\n~~~\n"),
    )
    check(
        "inner backticks survive a longer fence",
        "inner" not in strip_code_blocks("Prose.\n\n````\n```\ninner\n```\n````\n"),
    )
    check("inline code untouched", "`x`" in strip_code_blocks("Set `x` to two."))
    check("code-only response strips to empty", strip_code_blocks("```\ncode\n```") == "")


def test_parse_scores():
    print("parse_scores")
    bare = json.dumps({**PERFECT, "justification": "good"})
    check("bare JSON parses", parse_scores(bare)["markers"] == 5)
    check(
        "JSON inside prose parses",
        parse_scores(f"Here is my assessment:\n{bare}\nThanks!")["vocabulary"] == 5,
    )
    check(
        "JSON inside a fence parses",
        parse_scores(f"```json\n{bare}\n```")["consistency"] == 5,
    )
    check("string ints coerce", parse_scores(json.dumps({**PERFECT, "markers": "4"}))["markers"] == 4)
    check("justification defaults to empty", parse_scores(json.dumps(PERFECT))["justification"] == "")

    for label, raw in [
        ("no JSON at all", "I cannot score this."),
        ("missing dimension", json.dumps({"markers": 3})),
        ("out of range high", json.dumps({**PERFECT, "markers": 9})),
        ("out of range low", json.dumps({**PERFECT, "markers": 0})),
        ("non-numeric", json.dumps({**PERFECT, "markers": "high"})),
        ("not an object", json.dumps([1, 2, 3])),
    ]:
        try:
            parse_scores(raw)
        except JudgeError:
            print(f"  ok  rejects {label}")
        else:
            raise AssertionError(f"accepted {label}")


def test_reward():
    print("reward")
    check("floor is REWARD_MIN", reward(FLOOR) == REWARD_MIN == 5)
    check("ceiling is REWARD_MAX", reward(PERFECT) == REWARD_MAX == 25)
    check("graded in between", reward({**PERFECT, "markers": 1}) == 21)
    check(
        "distinct rollouts get distinct rewards",
        len({reward(FLOOR), reward({**FLOOR, "markers": 3}), reward(PERFECT)}) == 3,
    )


def test_prompt_substitution():
    print("prompt")
    template = load_template()
    check("style guide substituted", "{STYLE_GUIDE}" not in template)
    check("style rules present", "No contractions" in template)
    check("scope section present", "prose only" in template)
    check("usage notes not in the prompt", "USAGE NOTES" not in template)

    guide = load_style_guide()
    check("guide keeps every numbered rule", all(f"{n}. **" in guide for n in range(1, 10)))
    check("guide drops the sync bookkeeping", "Keep these in sync" not in guide)
    check("guide drops our notes to ourselves", "## Notes" not in guide)
    check("guide drops the gaming-vector note", "judge gaming" not in guide)
    prompt = build_prompt("This plan good.", template)
    check("response substituted", "{RESPONSE_PROSE}" not in prompt)
    check("response body present", "This plan good." in prompt)
    check("every dimension named", all(dimension in prompt for dimension in DIMENSIONS))


def test_scoring_with_a_fake_judge():
    print("score_response / score_responses")
    seen = []

    def complete(prompt):
        seen.append(prompt)
        return json.dumps({**PERFECT, "justification": "in voice"})

    scores = score_response("This plan good.\n\n```py\nx=1\n```", complete)
    check("scores come back", reward(scores) == 25)
    check("judge never saw the code", "x=1" not in seen[0])

    def flaky(prompt):
        return "sorry, I cannot" if "bad" in prompt else json.dumps(PERFECT)

    scored = score_responses(["good one", "a bad one", "another good"], flaky, max_workers=2)
    check("order preserved, failure is None", [s is not None for s in scored] == [True, False, True])

    def explodes(prompt):
        raise RuntimeError("rate limited")

    check("transport errors become None", score_responses(["x"], explodes) == [None])
    check("empty prose is a failure, not a crash", score_responses(["```\ncode\n```"], complete) == [None])


def test_aggregate():
    print("aggregate")
    summary = aggregate([PERFECT, FLOOR, None])
    check("counts usable rows", summary["judge_scored"] == 2)
    check("counts failures", summary["judge_failed"] == 1)
    check("means the total", summary["judge_total"] == 15.0)
    check("means each dimension", summary["judge_markers"] == 3.0)
    check("gaming gap is markers minus plausibility", summary["judge_gaming_gap"] == 0.0)

    gamed = aggregate([{**PERFECT, "human_plausibility": 1}])
    check("gap is positive when markers outrun plausibility", gamed["judge_gaming_gap"] == 4.0)
    check("empty input does not divide by zero", aggregate([None])["judge_scored"] == 0)


def main():
    for test in [
        test_strip_code_blocks,
        test_parse_scores,
        test_reward,
        test_prompt_substitution,
        test_scoring_with_a_fake_judge,
        test_aggregate,
    ]:
        test()
    print("\nall judge checks passed")


if __name__ == "__main__":
    main()
