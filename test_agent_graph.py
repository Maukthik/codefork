"""Tests for agent_graph.py - no API calls, no Docker (fake LLM + SANDBOX=local)."""

import json
import threading
from types import SimpleNamespace

import pytest

import agent_graph as ag

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
WRONG = "def add(a, b):\n    return a * b\n"
TEST = "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX", "local")
    monkeypatch.setattr(ag, "BASE_DIR", tmp_path)  # logs go to tmp
    r = tmp_path / "repo"
    r.mkdir()
    (r / "calc.py").write_text(BUGGY)
    (r / "test_calc.py").write_text(TEST)
    return r


def msg(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


def call(name, **args):
    return SimpleNamespace(id=f"c_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def make_fake_chat(plans):
    """plans: list of strategy lists, one per planner call. Strategy text decides the fix."""
    state = {"planner": 0}
    lock = threading.Lock()

    def fake_chat(model, messages, tools=None):
        system = messages[0]["content"]
        if "planner" in system:
            with lock:
                strategies = plans[min(state["planner"], len(plans) - 1)]
                state["planner"] += 1
            return msg(json.dumps({"strategies": strategies})), (100, 20)
        if "reflector" in system:
            return msg("All attempts used the wrong operator."), (50, 10)
        # executor: first turn writes a file, second turn stops
        if messages[-1]["role"] == "tool":
            return msg("done"), (30, 5)
        good = "GOOD" in messages[1]["content"]
        return msg("", [call("write_file", path="calc.py", content=FIXED if good else WRONG)]), (40, 10)

    return fake_chat


# --- pure helpers -----------------------------------------------------------

def test_safe_path_blocks_escape(tmp_path):
    with pytest.raises(ValueError):
        ag.safe_path(str(tmp_path), "../outside.txt")
    assert ag.safe_path(str(tmp_path), "a/b.py") == (tmp_path / "a/b.py").resolve()


def test_make_diff_counts_lines(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "x.py").write_text(BUGGY)
    (b / "x.py").write_text(FIXED)
    d = ag.make_diff(str(a), str(b))
    assert d["changed"] == ["x.py"] and d["lines"] == 2


def test_pick_winner_prefers_smallest_diff():
    base = {"round": 1, "passed": True, "input_tokens": 10, "output_tokens": 1}
    rs = [dict(base, branch_id="big", diff_lines=10),
          dict(base, branch_id="small", diff_lines=2),
          dict(base, branch_id="red", diff_lines=1, passed=False),
          dict(base, branch_id="old", diff_lines=1, round=0)]
    assert ag.pick_winner(rs, 1)["branch_id"] == "small"


def test_fan_out_one_send_per_strategy():
    state = {"round": 1, "strategies": ["s1", "s2", "s3"], "task": "t", "repo_dir": "r",
             "test_cmd": "pytest", "baseline_output": "", "work_root": "w", "run_id": "x",
             "max_turns": 5}
    sends = ag.fan_out(state)
    assert [s.arg["branch_id"] for s in sends] == ["r1_b0", "r1_b1", "r1_b2"]


def test_parse_json_handles_fences():
    assert ag.parse_json('```json\n{"strategies": ["a"]}\n```') == {"strategies": ["a"]}
    assert ag.parse_json("no json here") is None


# --- full graph ---------------------------------------------------------------

def test_graph_picks_green_branch(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "chat", make_fake_chat([["bad idea", "GOOD fix", "another bad"]]))
    s = ag.run(str(repo), "fix add", branches=3, rounds=1, work_root=str(tmp_path / "branches"))
    assert s["status"] == "green"
    assert s["winner"]["branch_id"] == "r1_b1"
    assert len(s["branches"]) == 3
    assert (repo / "calc.py").read_text() == BUGGY  # not applied without --apply


def test_graph_reflects_then_succeeds_and_applies(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "chat", make_fake_chat([["bad 1", "bad 2"], ["GOOD fix", "bad 3"]]))
    s = ag.run(str(repo), "fix add", branches=2, rounds=2, apply=True,
               work_root=str(tmp_path / "branches"))
    assert s["status"] == "green" and s["rounds"] == 2
    assert s["winner"]["branch_id"] == "r2_b0"
    assert "reflector" in s["tokens_by_stage"]
    assert (repo / "calc.py").read_text() == FIXED  # applied


def test_graph_gives_up_after_max_rounds(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "chat", make_fake_chat([["bad 1", "bad 2"]]))
    s = ag.run(str(repo), "fix add", branches=2, rounds=2, work_root=str(tmp_path / "branches"))
    assert s["status"] == "red" and s["winner"] is None and len(s["branches"]) == 4


def test_already_green_skips_everything(repo, tmp_path, monkeypatch):
    (repo / "calc.py").write_text(FIXED)
    monkeypatch.setattr(ag, "chat", lambda *a, **k: pytest.fail("LLM should not be called"))
    s = ag.run(str(repo), "fix add", work_root=str(tmp_path / "branches"))
    assert s["status"] == "already_green"


# --- git review branch --------------------------------------------------------

import subprocess


def sh(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def git_init(path):
    sh(path, "init", "-q", "-b", "main")
    sh(path, "add", "-A")
    sh(path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")


def test_winner_committed_to_branch_without_touching_worktree(repo, tmp_path, monkeypatch):
    git_init(repo)
    monkeypatch.setattr(ag, "chat", make_fake_chat([["GOOD fix", "bad"]]))
    s = ag.run(str(repo), "fix add", branches=2, rounds=1, work_root=str(tmp_path / "branches"))

    branch = s["git"]["branch"]
    assert branch.startswith("fork/fix-")
    assert sh(repo, "show", f"{branch}:calc.py") + "\n" == FIXED       # fix is on the branch
    assert sh(repo, "branch", "--show-current") == "main"              # still on main
    assert (repo / "calc.py").read_text() == BUGGY                     # working tree untouched
    assert sh(repo, "status", "--porcelain", "-uno") == ""                   # nothing dirty
    assert "Strategy: GOOD fix" in sh(repo, "log", "-1", "--format=%B", branch)
    assert sh(repo, "worktree", "list").count("\n") == 0               # temp worktree cleaned up


def test_repo_in_subfolder_of_git_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX", "local")
    monkeypatch.setattr(ag, "BASE_DIR", tmp_path)
    outer = tmp_path / "outer"
    demo = outer / "workspace" / "demo"
    demo.mkdir(parents=True)
    (demo / "calc.py").write_text(BUGGY)
    (demo / "test_calc.py").write_text(TEST)
    git_init(outer)
    monkeypatch.setattr(ag, "chat", make_fake_chat([["GOOD fix"]]))
    s = ag.run(str(demo), "fix add", branches=1, rounds=1, work_root=str(tmp_path / "branches"))
    assert sh(outer, "show", f"{s['git']['branch']}:workspace/demo/calc.py") + "\n" == FIXED


def test_no_git_repo_still_works(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "chat", make_fake_chat([["GOOD fix"]]))
    s = ag.run(str(repo), "fix add", branches=1, rounds=1, work_root=str(tmp_path / "branches"))
    assert s["status"] == "green" and "error" in s["git"]


def test_no_branch_flag(repo, tmp_path, monkeypatch):
    git_init(repo)
    monkeypatch.setattr(ag, "chat", make_fake_chat([["GOOD fix"]]))
    s = ag.run(str(repo), "fix add", branches=1, rounds=1, make_branch=False,
               work_root=str(tmp_path / "branches"))
    assert s["git"] is None and "fork/fix" not in sh(repo, "branch")