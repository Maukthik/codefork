"""
student.py - the Student Lab: describe a program, or paste code that doesn't work, and get
back tested Python with the results shown test by test.

Both run through Fork's build mode: the spec writer turns the student's words into pytest
tests first, then parallel branches write (or fix) solution.py until those tests pass.
This module holds the logic only, no Streamlit, so it is tested offline. The page is
pages/1_Student_Lab.py.
"""
from __future__ import annotations

import ast
import base64
import difflib
import re
import tempfile
from pathlib import Path
from typing import Optional

import agent_graph as ag

SOLUTION = "solution.py"
STARTER = '"""Your program goes here."""\n'
TEST_CMD = "python -m pytest -q -rA -p no:cacheprovider"   # -rA: one PASSED/FAILED line per test
MAX_TASK_CHARS = 1500
MAX_CODE_CHARS = 12000
MAX_CODE_LINES = 300
MAX_EXPR_CHARS = 300

# Small, readable runs: students wait for the answer, and a class shares one budget.
RUN_DEFAULTS = dict(branches=2, rounds=2, max_turns=8, ladder="nano,super", thinking="off",
                    first_green=True, make_branch=False)

EXAMPLES = {
    "Palindrome check": "Write is_palindrome(text) that returns True if text reads the same forwards and "
                        "backwards, ignoring case, spaces and punctuation. An empty string counts as a palindrome.",
    "Grade calculator": "Write grade(marks) that turns marks out of 100 into a letter: 90 and above is 'S', "
                        "80-89 'A', 70-79 'B', 60-69 'C', 50-59 'D', below 50 'F'. Marks below 0 or above 100 "
                        "raise ValueError.",
    "Word counter": "Write word_counts(text) that returns a dict of each word to how many times it appears, "
                    "ignoring case and the punctuation .,!?; Write top_words(text, n) that returns the n most "
                    "common words as a list, most common first, ties in alphabetical order.",
}

FIX_EXAMPLE_CODE = '''def average(marks):
    """Average of a list of marks, rounded to 2 decimals. Empty list -> 0."""
    total = 0
    for i in range(1, len(marks)):
        total += marks[i]
    return round(total / len(marks), 2)
'''
FIX_EXAMPLE_TASK = "average(marks) returns the average of the list, rounded to 2 decimal places; an empty list returns 0."


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def check_inputs(mode: str, task: str, code: str = "") -> Optional[str]:
    """A message for the student if the input can't be used, else None."""
    task = (task or "").strip()
    if not task:
        return "Describe what the program should do." if mode == "describe" else \
            "Say what your code is supposed to do, so the tests check the right thing."
    if len(task) > MAX_TASK_CHARS:
        return f"Please keep the description under {MAX_TASK_CHARS} characters."
    if mode == "fix":
        code = code or ""
        if not code.strip():
            return "Paste the code you want fixed."
        if len(code) > MAX_CODE_CHARS or code.count("\n") + 1 > MAX_CODE_LINES:
            return f"The Student Lab takes up to {MAX_CODE_LINES} lines of code."
    return None


def describe_request(task: str) -> str:
    return (f"Write a Python program in {SOLUTION} for this task: {task.strip()}\n"
            f"Put all the code in {SOLUTION} as plain functions (or classes) that the tests import from "
            "solution. It is for a student to learn from: clear names, short functions, a docstring on "
            "each function, comments only where a step is not obvious, standard library only.")


def fix_request(task: str) -> str:
    return (f"The student's code is in {SOLUTION}. It is supposed to do this: {task.strip()}\n"
            "Test the functions under the names they have in solution.py. Then find and fix the bugs: "
            "keep the student's structure, names and style, and change only the lines that are wrong.")


def new_workspace(code: str = "") -> str:
    """A fresh folder holding solution.py (empty, or the student's code)."""
    repo = Path(tempfile.mkdtemp(prefix="fork-student-")) / "work"
    repo.mkdir()
    text = (code or STARTER).replace("\r\n", "\n")
    (repo / SOLUTION).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return str(repo)


def run_kwargs(mode: str, task: str, code: str = "", max_usd: Optional[float] = None) -> dict:
    """Arguments for agent_graph.run. Describe = build mode (tests must fail on the empty file);
    fix = build mode where passing tests are fine (the student's code may already be right)."""
    repo = new_workspace(code if mode == "fix" else "")
    return dict(RUN_DEFAULTS, repo=repo, task=task, test_cmd=TEST_CMD,
                build=describe_request(task) if mode == "describe" else fix_request(task),
                spec_must_fail=mode == "describe", max_usd=max_usd,
                work_root=str(Path(repo).parent / "branches"))


# ---------------------------------------------------------------------------
# Progress, in words a student follows
# ---------------------------------------------------------------------------

STAGES = [
    (re.compile(r"^\[spec\] attempt"), "Rewriting the tests (the first try wasn't usable)"),
    (re.compile(r"^\[spec\] wrote"), "Wrote the tests from your description"),
    (re.compile(r"^\[baseline\]"), "Ran the tests on the starting code"),
    (re.compile(r"^\[planner\]"), "Planned different ways to solve it"),
    (re.compile(r"^\[r\d+_b\d+\] turn \d+: (write_file|edit_file)"), "Writing code in parallel attempts"),
    (re.compile(r"^\[r\d+_b\d+\] GREEN"), "An attempt passed every test"),
    (re.compile(r"^\[merge\]"), "Combined the best parts of two attempts"),
    (re.compile(r"^\[reflector\]"), "Learned from the failed attempts, trying again on a bigger model"),
]


def progress(lines: list[str]) -> list[str]:
    """The stages reached so far, in order, each once."""
    seen: list[str] = []
    for line in lines:
        for pat, label in STAGES:
            if pat.search(line) and label not in seen:
                seen.append(label)
    return seen


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

_RESULT_RE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)(?:\s+-\s+(.*))?$", re.M)


def parse_results(output: str) -> list[dict]:
    """Per-test lines from pytest -rA: [{"test", "file", "passed", "reason"}], in run order."""
    out, seen = [], set()
    for outcome, node, reason in _RESULT_RE.findall(output or ""):
        if node in seen:
            continue
        seen.add(node)
        file, _, name = node.partition("::")
        out.append({"test": (name or file).replace("::", "."), "file": file,
                    "passed": outcome in ("PASSED", "XPASS"), "reason": (reason or "").strip()})
    return out


def pretty(name: str) -> str:
    """test_empty_string_is_palindrome -> 'empty string is palindrome'."""
    return re.sub(r"^test_?", "", name.split("[")[0]).replace("_", " ").strip() + \
        (f" [{name.split('[', 1)[1]}" if "[" in name else "")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def outcome(state: dict, mode: str, original_code: str = "") -> dict:
    """Everything the page shows, from the final graph state."""
    s = state["summary"]
    status = s["status"]
    spec = state.get("spec") or {}
    w = state.get("winner")
    best = None
    if status == "green" and w:
        workdir, output = w["workdir"], w["test_output"]
    elif status == "already_green":
        workdir, output = state["repo_dir"], state.get("baseline_output", "")
    else:
        pool = [r for r in state.get("results", []) if r.get("workdir") and not r.get("cancelled")]
        best = max(pool, key=ag.passed_count, default=None)
        workdir = best["workdir"] if best else state["repo_dir"]
        output = best["test_output"] if best else state.get("baseline_output", "")
    code = _read(Path(workdir) / SOLUTION) if status != "spec_failed" else ""
    results = parse_results(output) if status != "spec_failed" else []
    diff = ""
    if mode == "fix" and code and original_code.strip():
        before = original_code.replace("\r\n", "\n")
        diff = "".join(difflib.unified_diff(before.splitlines(True), code.splitlines(True),
                                            fromfile="your code", tofile="fixed code"))
    tokens = sum(v["input"] + v["output"] for v in s.get("tokens_by_stage", {}).values())
    return {
        "status": status, "mode": mode, "code": code, "workdir": workdir,
        "tests": spec.get("content") or {}, "interface": spec.get("interface", ""),
        "spec_summary": spec.get("summary", ""), "spec_problems": spec.get("problems") or [],
        "results": results, "passed": sum(r["passed"] for r in results), "total": len(results),
        "output": output, "diff": diff, "changed": bool(diff),
        "rounds": s.get("rounds", 0), "branches": len(s.get("branches", [])),
        "cost_usd": s.get("cost_usd", 0.0), "tokens": tokens,
        "strategy": (w or best or {}).get("strategy", ""),
    }


def headline(o: dict) -> str:
    n, k = o["total"], o["passed"]
    if o["status"] == "spec_failed":
        return "Couldn't turn that description into tests. Try saying exactly what goes in and what comes out."
    if o["status"] == "already_green":
        return f"Your code is already correct: it passes all {n} tests."
    if o["status"] == "green":
        return (f"Fixed: all {n} tests pass now." if o["mode"] == "fix" else f"Solved: all {n} tests pass.")
    return f"Partly solved: {k} of {n} tests pass. The closest attempt is shown below."


# ---------------------------------------------------------------------------
# Try it: call the finished code with your own input, in the sandbox
# ---------------------------------------------------------------------------

def check_expr(expr: str) -> Optional[str]:
    expr = (expr or "").strip()
    if not expr:
        return "Type a call, e.g. is_palindrome('Racecar')"
    if len(expr) > MAX_EXPR_CHARS:
        return f"Keep it under {MAX_EXPR_CHARS} characters."
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"That isn't a valid Python expression: {e.msg}"
    return None


def try_command(expr: str) -> str:
    """A shell command that prints repr(expr) with solution.py imported. The script travels as
    base64 inside double quotes, so the same command works in Linux sh (the Nebius sandbox)
    and Windows cmd (SANDBOX=local on a laptop), whatever quotes the expression contains."""
    script = f"from solution import *\nprint(repr({expr.strip()}))"
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return f"python -c \"import base64; exec(base64.b64decode('{b64}').decode('utf-8'))\""


def try_it(workdir: str, expr: str) -> tuple[bool, str]:
    """Evaluate one expression against solution.py in the sandbox. (ok, printed result or error)."""
    problem = check_expr(expr)
    if problem:
        return False, problem
    code, out = ag.run_in_sandbox(workdir, try_command(expr), timeout=15)
    out = ag.tail(out.strip(), 3000)
    if code == 0:
        return True, out
    last = out.splitlines()[-1] if out else f"exit code {code}"
    return False, last if "Error" in last else out
