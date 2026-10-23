"""Offline tests for the SWE-rebench harness (bench/swe_env.py): no network, no sandbox."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
import swe_env as env  # noqa: E402

GOLD = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1 +1 @@
-x = 1
+x = 2
"""

ROW = {"instance_id": "o__r-1", "repo": "o/r", "base_commit": "abc", "problem_statement": "bug",
       "patch": GOLD, "test_patch": "diff --git a/tests/t.py b/tests/t.py\n",
       "FAIL_TO_PASS": json.dumps(["tests/t.py::test_a"]), "PASS_TO_PASS": ["tests/t.py::test_b", "tests/t.py::test_c"],
       "docker_image": "swerebench/img:latest", "test_cmd": None, "created_at": "2025-11-02 10:00:00",
       "install_config": {"test_cmd": "pytest --no-header -rA --tb=line -p no:cacheprovider tests/t.py"}}


def test_task_from_row_reads_install_config_test_cmd():
    t = env.task_from_row(ROW)
    assert t.test_cmd.startswith("pytest --no-header -rA") and t.f2p == ["tests/t.py::test_a"]
    assert t.p2p == ["tests/t.py::test_b", "tests/t.py::test_c"] and t.month == "2025-11"
    assert t.gold_files == ["pkg/mod.py"] and t.gold_lines == 2


RA_OUTPUT = """
tests/t.py ..F                                                     [100%]
=========================== short test summary info ============================
PASSED tests/t.py::test_b
PASSED tests/t.py::test_param[a-1]
FAILED tests/t.py::test_a - AssertionError: assert 1 == 2
ERROR tests/t.py::test_c - fixture 'db' not found
SKIPPED [1] tests/t.py:40: needs network
XFAIL tests/t.py::test_x - known
"""


def test_parse_pytest_ra():
    st = env.parse_pytest_ra(RA_OUTPUT)
    assert st == {"tests/t.py::test_b": "PASSED", "tests/t.py::test_param[a-1]": "PASSED",
                  "tests/t.py::test_a": "FAILED", "tests/t.py::test_c": "ERROR", "tests/t.py::test_x": "XFAIL"}


def test_grade_requires_f2p_and_keeps_p2p():
    t = env.task_from_row(ROW)
    g = env.grade(t, RA_OUTPUT)
    assert not g.resolved and g.f2p_passed == 0 and g.p2p_broken == ["tests/t.py::test_c"]
    fixed = RA_OUTPUT.replace("FAILED tests/t.py::test_a - AssertionError: assert 1 == 2", "PASSED tests/t.py::test_a")
    assert not env.grade(t, fixed).resolved                       # test_c still errors
    assert env.grade(t, fixed, p2p_baseline={"tests/t.py::test_b"}).resolved   # c was broken before too


def test_missing_tests_count_as_not_passed():
    t = env.task_from_row(ROW)
    g = env.grade(t, "PASSED tests/t.py::test_b\n")
    assert not g.resolved and "tests/t.py::test_a" in g.missing


def make_task(i, repo, gold_lines=2, files=1, f2p=1, month="2025-11"):
    patch = "".join(f"diff --git a/f{k}.py b/f{k}.py\n" + "+x\n" * gold_lines for k in range(files))
    return env.Task(instance_id=f"{repo}-{i}", repo=repo, base_commit="c", problem_statement="",
                    patch=patch, test_patch="", f2p=[f"t::{j}" for j in range(f2p)], p2p=[],
                    image="img", test_cmd="pytest", created_at=f"{month}-01")


def test_select_is_small_deterministic_and_one_per_repo():
    tasks = ([make_task(i, "a/a") for i in range(5)] + [make_task(0, "b/b", gold_lines=50)]
             + [make_task(0, "c/c", files=2)] + [make_task(0, "d/d", f2p=9)] + [make_task(0, "e/e")]
             + [make_task(0, "home-assistant/core")])
    picked = env.select(tasks, 10)
    assert sorted(t.repo for t in picked) == ["a/a", "e/e"]
    assert [t.instance_id for t in env.select(tasks, 10)] == [t.instance_id for t in picked]


def test_train_test_split_by_month():
    tr, te = env.split_train_test([make_task(0, "x", month="2025-03"), make_task(1, "y", month="2025-12")])
    assert [t.repo for t in tr] == ["x"] and [t.repo for t in te] == ["y"]


def test_shell_activates_testbed_env():
    cmd = env.shell("pytest -rA tests/t.py")
    assert cmd.startswith("bash -lc ") and "activate testbed" in cmd and "cd /testbed" in cmd


def test_prepare_checkpoint_checks_out_base_and_applies_test_patch(monkeypatch):
    seen = {}

    class Img:
        def run(self, shell, files=None, disposable=True, timeout=None):
            seen.update(shell=shell, files=files, disposable=disposable)
            return SimpleNamespace(wait=lambda: SimpleNamespace(stdout="FORK_READY", stderr="", exit_code=0))

    client = SimpleNamespace(images=SimpleNamespace(oci=lambda ref, timeout=None: Img()))
    env.prepare_checkpoint(env.task_from_row(ROW), client)
    assert "git checkout -q -f abc" in seen["shell"] and "git apply /fork_inputs/test.patch" in seen["shell"]
    assert list(seen["files"]) == ["fork_inputs/test.patch"] and seen["disposable"] is False


def test_prepare_checkpoint_reports_failure():
    class Img:
        def run(self, **kw):
            return SimpleNamespace(wait=lambda: SimpleNamespace(stdout="error: patch failed", stderr="", exit_code=1))
    client = SimpleNamespace(images=SimpleNamespace(oci=lambda ref, timeout=None: Img()))
    with pytest.raises(RuntimeError, match="couldn't prepare"):
        env.prepare_checkpoint(env.task_from_row(ROW), client)


def test_parse_verbose_pytest_lines():
    out = ("tests/test_c.py::test_valid[0s-x12] PASSED [ 44%]\n"
           "tests/test_c.py::test_invalid[3 days] FAILED          [ 45%]\n"
           "tests/test_c.py::test_skip SKIPPED (reason)\n")
    st = env.parse_pytest_ra(out)
    assert st["tests/test_c.py::test_valid[0s-x12]"] == "PASSED"
    assert st["tests/test_c.py::test_invalid[3 days]"] == "FAILED"


def test_run_tests_filters_output_to_graded_ids(tmp_path):
    seen = {}

    class Ckpt:
        def run(self, shell, files=None, timeout=None):
            seen.update(shell=shell, files=files)
            ids = open(files["fork_inputs/ids.txt"], encoding="utf-8").read().split()
            seen["ids"] = ids
            return SimpleNamespace(wait=lambda: SimpleNamespace(stdout="PASSED tests/t.py::test_a", stderr="", exit_code=0))

    t = env.task_from_row(ROW)
    code, out = env.run_with_patch(Ckpt(), t, GOLD)
    assert seen["ids"] == ["tests/t.py::test_a", "tests/t.py::test_b", "tests/t.py::test_c"]
    assert "grep -aF -f /fork_inputs/ids.txt" in seen["shell"] and "git apply /fork_inputs/fix.patch" in seen["shell"]
    assert set(seen["files"]) == {"fork_inputs/ids.txt", "fork_inputs/fix.patch"}
