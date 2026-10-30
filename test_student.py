"""Tests for the Student Lab (student.py + pages/1_Student_Lab.py): fake LLM, local sandbox."""

import json
import sys
from types import SimpleNamespace

import pytest

import agent_graph as ag
import hosting
import student

PAL_TESTS = ("from solution import is_palindrome\n\n\n"
             "def test_simple_palindrome():\n    assert is_palindrome('Racecar')\n\n\n"
             "def test_ignores_spaces_and_punctuation():\n    assert is_palindrome('A man, a plan, a canal: Panama')\n\n\n"
             "def test_not_a_palindrome():\n    assert not is_palindrome('hello')\n")
PAL_CODE = ('def is_palindrome(text):\n    """True if text reads the same both ways, ignoring case and punctuation."""\n'
            "    letters = [c.lower() for c in text if c.isalnum()]\n    return letters == letters[::-1]\n")
AVG_TESTS = ("from solution import average\n\n\ndef test_average():\n    assert average([1, 2, 3]) == 2\n\n\n"
             "def test_empty():\n    assert average([]) == 0\n")
AVG_FIXED = ('def average(marks):\n    """Average of a list of marks, rounded to 2 decimals. Empty list -> 0."""\n'
             "    if not marks:\n        return 0\n    total = 0\n    for i in range(0, len(marks)):\n"
             "        total += marks[i]\n    return round(total / len(marks), 2)\n")


def msg(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls, finish_reason="stop")


def call(name, **args):
    return SimpleNamespace(id=f"c_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def fake_chat(tests, solution, seen=None):
    seen = seen if seen is not None else {}

    def chat(model, messages, tools=None, **kwargs):
        system = messages[0]["content"]
        for k in ("spec writer", "planner", "reflector"):
            if k in system:
                seen[k] = seen.get(k, 0) + 1
        if "spec writer" in system:
            seen["system"] = system
            return msg(f"### SUMMARY\nchecks it\n### INTERFACE\nsee tests\n### FILE: test_solution.py\n"
                       f"```python\n{tests}```\n"), (100, 50)
        if "planner" in system:
            return msg(json.dumps({"strategies": ["write it", "write it another way"]})), (50, 10)
        if "reflector" in system:
            return msg("try again"), (20, 5)
        if any(m["role"] == "assistant" for m in messages):
            return msg("done"), (10, 2)
        return msg("", [call("write_file", path="solution.py", content=solution)]), (30, 10)

    return chat


@pytest.fixture
def local(tmp_path, monkeypatch):
    for k in ("MAX_USD_PER_RUN", "EXECUTOR_LADDER", "PLANNER_MODEL", "EXECUTOR_MODEL", "HOSTED",
              "STUDENT_ACCESS_CODE", "DEMO_ACCESS_CODE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SANDBOX", "local")
    monkeypatch.setattr(ag, "BASE_DIR", tmp_path)
    return tmp_path


def run(mode, task, code=""):
    return ag.run(**student.run_kwargs(mode, task, code), full=True)


# --- helpers ----------------------------------------------------------------------

def test_check_inputs():
    assert "Describe" in student.check_inputs("describe", "  ")
    assert "Paste" in student.check_inputs("fix", "average of a list", "")
    assert "lines" in student.check_inputs("fix", "x", "a = 1\n" * 400)
    assert student.check_inputs("describe", "Write add(a, b)") is None


def test_parse_results_reads_pytest_rA_lines():
    out = ("..F\nFAILED test_solution.py::test_empty - ZeroDivisionError: division by zero\n"
           "PASSED test_solution.py::test_average\nPASSED test_solution.py::test_cases[3-4]\n"
           "1 failed, 2 passed in 0.02s\n")
    rs = student.parse_results(out)
    assert [(r["test"], r["passed"]) for r in rs] == [("test_empty", False), ("test_average", True),
                                                      ("test_cases[3-4]", True)]
    assert rs[0]["reason"].startswith("ZeroDivisionError")
    assert student.pretty("test_ignores_spaces") == "ignores spaces"
    assert student.pretty("test_cases[3-4]") == "cases [3-4]"


def test_progress_turns_log_lines_into_steps():
    lines = ["[spec] wrote test_solution.py (3 new tests, 3 in the suite), RED as expected.",
             "[baseline] tests RED (exit 2)", "[planner] round 1: 2 strategies",
             "[r1_b0] turn 1: write_file solution.py -> Wrote solution.py",
             "[r1_b1] turn 1: write_file solution.py -> Wrote solution.py", "[r1_b0] GREEN (tests passed)"]
    steps = student.progress(lines)
    assert steps[0].startswith("Wrote the tests") and steps[-1].startswith("An attempt passed")
    assert len(steps) == len(set(steps)) == 5


def test_try_command_has_no_posix_only_quoting():
    cmd = student.try_command("f('it''s', \"x\")")
    assert cmd.startswith('python -c "') and cmd.endswith('"') and cmd.count('"') == 2


def test_try_it_checks_the_expression_first():
    assert student.try_it("unused", "is_palindrome(")[0] is False
    assert "Type a call" in student.try_it("unused", "")[1]


# --- the two modes, end to end ------------------------------------------------------

def test_describe_a_program_end_to_end(local, monkeypatch):
    seen = {}
    monkeypatch.setattr(ag, "chat", fake_chat(PAL_TESTS, PAL_CODE, seen))
    o = student.outcome(run("describe", "Write is_palindrome(text) ..."), "describe")
    assert o["status"] == "green" and o["passed"] == o["total"] == 3
    assert "def is_palindrome" in o["code"]
    assert "must FAIL on the current code" in seen["system"]          # describe = real red check
    assert student.headline(o) == "Solved: all 3 tests pass."
    ok, out = student.try_it(o["workdir"], "is_palindrome('No lemon, no melon')")
    assert ok and out == "True"
    ok, out = student.try_it(o["workdir"], "is_palindrome(None)")
    assert not ok and "TypeError" in out


def test_fix_my_code_fixes_and_shows_the_diff(local, monkeypatch):
    monkeypatch.setattr(ag, "chat", fake_chat(AVG_TESTS, AVG_FIXED))
    o = student.outcome(run("fix", student.FIX_EXAMPLE_TASK, student.FIX_EXAMPLE_CODE), "fix",
                        student.FIX_EXAMPLE_CODE)
    assert o["status"] == "green" and o["changed"]
    assert "-    for i in range(1, len(marks)):" in o["diff"]
    assert "+    if not marks:" in o["diff"]
    assert student.headline(o) == "Fixed: all 2 tests pass now."


def test_correct_code_is_reported_correct_without_writing_any(local, monkeypatch):
    seen = {}
    monkeypatch.setattr(ag, "chat", fake_chat(AVG_TESTS, "unused", seen))
    o = student.outcome(run("fix", student.FIX_EXAMPLE_TASK, AVG_FIXED), "fix", AVG_FIXED)
    assert o["status"] == "already_green" and o["passed"] == 2
    assert seen.get("planner", 0) == 0 and seen["spec writer"] == 1     # passing tests aren't retried
    assert "never shape the tests around what it does now" in seen["system"]
    assert o["code"] == AVG_FIXED and not o["changed"]
    assert student.headline(o).startswith("Your code is already correct")


def test_shared_state_is_one_per_process():
    assert hosting.shared() is hosting.shared()
    assert hosting.shared()["slots"].acquire(blocking=False)
    hosting.shared()["slots"].release()


def test_student_access_code_falls_back_to_demo_code(monkeypatch):
    monkeypatch.delenv("STUDENT_ACCESS_CODE", raising=False)
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judges")
    assert hosting.student_access_ok("judges") and not hosting.student_access_ok("x")
    monkeypatch.setenv("STUDENT_ACCESS_CODE", "class-2b")
    assert hosting.student_access_ok("class-2b") and not hosting.student_access_ok("judges")


# --- the page, driven headless ------------------------------------------------------

try:
    from streamlit.testing import v1 as streamlit_testing
except ImportError:
    streamlit_testing = None


@pytest.mark.skipif(streamlit_testing is None, reason="streamlit not installed")
def test_page_builds_an_example_and_shows_every_test(local, monkeypatch):
    monkeypatch.setenv("ALLOW_LOCAL_SANDBOX", "1")
    monkeypatch.setattr(ag, "chat", fake_chat(PAL_TESTS, PAL_CODE))
    at = streamlit_testing.AppTest.from_file("pages/1_Student_Lab.py", default_timeout=90).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Palindrome check").click().run()
    assert "palindrome" in at.text_area[0].value
    next(b for b in at.button if b.label == "▶ Build it").click().run()
    assert not at.exception
    assert any("Solved: all 3 tests pass" in s.value for s in at.success)
    assert sum("✅" in m.value for m in at.markdown) >= 3
    assert any("def is_palindrome" in c.value for c in at.code)
    at.text_input(key="lab_expr").set_value("is_palindrome('Step on no pets')").run()
    next(b for b in at.button if b.label == "Run").click().run()
    assert any(">>> is_palindrome('Step on no pets')\nTrue" in c.value for c in at.code)


@pytest.mark.skipif(streamlit_testing is None, reason="streamlit not installed")
def test_hosted_page_needs_the_class_code(local, monkeypatch):
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.setenv("STUDENT_ACCESS_CODE", "class-2b")
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "test")
    monkeypatch.setitem(sys.modules, "contree_sdk", SimpleNamespace())
    monkeypatch.setattr(ag, "chat", lambda *a, **k: pytest.fail("no model call without the code"))
    at = streamlit_testing.AppTest.from_file("pages/1_Student_Lab.py", default_timeout=60).run()
    at.text_area(key="lab_task").set_value("Write add(a, b)").run()
    next(b for b in at.button if b.label == "▶ Build it").click().run()
    assert any("class access code" in e.value for e in at.error)
    assert at.session_state["lab_job"] is None


@pytest.mark.skipif(streamlit_testing is None, reason="streamlit not installed")
def test_results_of_one_mode_are_hidden_in_the_other(local, monkeypatch):
    monkeypatch.setenv("ALLOW_LOCAL_SANDBOX", "1")
    monkeypatch.setattr(ag, "chat", fake_chat(PAL_TESTS, PAL_CODE))
    at = streamlit_testing.AppTest.from_file("pages/1_Student_Lab.py", default_timeout=90).run()
    next(b for b in at.button if b.label == "Palindrome check").click().run()
    next(b for b in at.button if b.label == "▶ Build it").click().run()
    assert any("Solved" in s.value for s in at.success)
    at.radio[0].set_value("🩹 Fix my code").run()
    assert not any("Solved" in s.value for s in at.success)
