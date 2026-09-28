"""Rule based verifier for the RLVR reward: run the completion's code against the problem's asserts.
Returns the fraction of asserts passed, so a group of rollouts that all fail some tests still has signal."""

import os
import re
import secrets
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

EXECUTION_TIMEOUT_SECONDS = 10
VERIFIER_WORKERS = 16

FENCED_CODE = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)```", re.DOTALL)
# code that exits before the tests run would otherwise pass everything under a return code check
EXIT_CALL = re.compile(r"\b(?:sys\.exit|os\._exit|exit|quit)\s*\(")
# an object whose __eq__ always returns True passes every "assert f(x) == y"
EQUALITY_OVERRIDE = re.compile(r"def\s+__(?:eq|ne)__\s*\(")


def extract_code(completion):
    # same rule as EVALS/evaluation.py: every fenced block in order, or the whole reply if there are none
    blocks = FENCED_CODE.findall(completion)
    if not blocks:
        return completion
    return "\n".join(blocks)


def run_program(program_source):
    with tempfile.TemporaryDirectory() as scratch_dir:
        program_path = os.path.join(scratch_dir, "program.py")
        with open(program_path, "w", encoding="utf-8") as program_file:
            program_file.write(program_source)
        try:
            result = subprocess.run(
                [sys.executable, program_path],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT_SECONDS,
                cwd=scratch_dir,
            )
        except subprocess.TimeoutExpired:
            return ""
        return result.stdout


def build_program(code, tests, setup, nonce):
    """Each assert is its own try block, and the counter and writer carry the nonce in their names.
    The model's code shares this namespace, so any name it could guess it could also overwrite."""
    counter = f"passed_{nonce}"
    writer = f"report_{nonce}"
    # setup builds test fixtures from classes the solution defines, so it runs after the code
    lines = [f"import os as os_{nonce}", f"{writer} = os_{nonce}.write", f"{counter} = 0", code, setup]
    for test_source in tests:
        indented_test = "\n".join("    " + line for line in test_source.splitlines())
        lines.append(f"try:\n{indented_test}\n    {counter} += 1\nexcept BaseException:\n    pass")
    lines.append(f"{writer}(1, ('\\n{nonce} ' + str({counter}) + '\\n').encode())")
    return "\n".join(lines) + "\n"


def verify(completion, tests, setup=""):
    """Fraction of tests passed, 0.0 to 1.0. Only the runner knows the nonce, so a faked
    report, an early exit or a crash all score zero."""
    if not tests:
        return 0.0
    code = extract_code(completion)
    if EXIT_CALL.search(code) or EQUALITY_OVERRIDE.search(code):
        return 0.0
    nonce = secrets.token_hex(16)
    stdout = run_program(build_program(code, list(tests), setup, nonce))
    for line in stdout.splitlines():
        if line.startswith(nonce + " "):
            return int(line.split()[1]) / len(tests)
    return 0.0


def verify_all(completions, tests_per_completion, setups):
    with ThreadPoolExecutor(max_workers=VERIFIER_WORKERS) as pool:
        return list(pool.map(verify, completions, tests_per_completion, setups))
