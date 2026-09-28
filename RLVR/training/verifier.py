import os
import subprocess
import sys
import tempfile

TIMEOUT_SECONDS = 4  # per stdin test / per whole test file for function problems
MEMORY_LIMIT_BYTES = 2 * 1024**3
PYTHON_COMMAND = ["prlimit", f"--as={MEMORY_LIMIT_BYTES}", sys.executable]  # prlimit because preexec_fn is unsafe with threads
FLOAT_TOLERANCE = 1e-6

# runs every test_ function in the test file, any failed assert or exception exits non zero
FUNCTION_TEST_RUNNER = """
import test_solution

for name in sorted(dir(test_solution)):
    if name.startswith("test_"):
        getattr(test_solution, name)()
"""


def tokens_match(actual_token, expected_token):
    if actual_token == expected_token:
        return True

    try:
        actual_number = float(actual_token)
        expected_number = float(expected_token)
    except ValueError:
        return False

    return abs(actual_number - expected_number) <= FLOAT_TOLERANCE * max(1.0, abs(expected_number))


def outputs_match(actual, expected):
    """Compare token by token so whitespace and float formatting differences do not matter."""
    actual_tokens = actual.split()
    expected_tokens = expected.split()
    if len(actual_tokens) != len(expected_tokens):
        return False

    for actual_token, expected_token in zip(actual_tokens, expected_tokens):
        if not tokens_match(actual_token, expected_token):
            return False

    return True


def run_stdin_tests(command, inputs, outputs, work_dir):
    """Run command once per test, feeding the input on stdin; stop at the first failure."""
    input_path = os.path.join(work_dir, "stdin_data.txt")  # not input.txt, which some contest templates redirect to
    for test_input, expected_output in zip(inputs, outputs):
        with open(input_path, "w") as input_file:
            input_file.write(test_input)

        # stdin is a file, not a pipe, like on a real judge; fast IO tricks such as os.fstat(0).st_size need that
        try:
            with open(input_path) as stdin_file:
                result = subprocess.run(
                    command,
                    stdin=stdin_file,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_SECONDS,
                )
        except subprocess.TimeoutExpired:
            return False

        if result.returncode != 0 or not outputs_match(result.stdout, expected_output):
            return False

    return True


def run_function_tests(code, test_code):
    with tempfile.TemporaryDirectory() as work_dir:
        files = {"solution.py": code, "test_solution.py": test_code, "runner.py": FUNCTION_TEST_RUNNER}
        for file_name, content in files.items():
            with open(os.path.join(work_dir, file_name), "w") as file_out:
                file_out.write(content)

        try:
            result = subprocess.run(
                PYTHON_COMMAND + ["runner.py"],
                cwd=work_dir,
                capture_output=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return False

    return result.returncode == 0


def verify(code, tests):
    """Return 1.0 if the code passes every test, else 0.0."""
    if tests["type"] == "function":
        passed = run_function_tests(code, tests["test_code"])
        return 1.0 if passed else 0.0

    with tempfile.TemporaryDirectory() as work_dir:
        program_path = os.path.join(work_dir, "solution.py")
        with open(program_path, "w") as file_out:
            file_out.write(code)

        passed = run_stdin_tests(PYTHON_COMMAND + [program_path], tests["inputs"], tests["outputs"], work_dir)

    return 1.0 if passed else 0.0
