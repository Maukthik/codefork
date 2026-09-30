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


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Your real .env must not change test results (models, spend caps, prices)."""
    for k in ("MAX_USD_PER_RUN", "PRICES_JSON", "EXECUTOR_THINKING", "PLANNER_MODEL",
              "EXECUTOR_MODEL", "REFLECTOR_MODEL", "ALLOW_LOCAL_SANDBOX"):
        monkeypatch.delenv(k, raising=False)


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

    def fake_chat(model, messages, tools=None, **kwargs):
        system = messages[0]["content"]
        if "planner" in system:
            with lock:
                strategies = plans[min(state["planner"], len(plans) - 1)]
                state["planner"] += 1
            return msg(json.dumps({"strategies": strategies})), (100, 20)
        if "reflector" in system:
            return msg("All attempts used the wrong operator."), (50, 10)
        # executor: first turn writes a file, next turn stops
        if any(m["role"] == "assistant" for m in messages):
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


# --- Nebius sandbox path (fake Contree client, no network) ----------------------

import itertools
import shlex
import tempfile
from pathlib import Path


class FakeImage:
    """Mimics ContreeSync images: run(...).wait() returns a new image with the result."""
    counter = itertools.count()

    def __init__(self, files, log, exit_code=0, stdout="", stderr=""):
        self.files, self.log = files, log
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr
        self.uuid = f"img-{next(self.counter)}"

    def apply_files(self, mapping):
        self.log.append(("apply_files", sorted(mapping)))
        new = dict(self.files)
        new.update({k: Path(v).read_bytes() for k, v in mapping.items()})
        return FakeImage(new, self.log)

    def run(self, shell, files=None, disposable=True, timeout=None):
        self.log.append(("run", shell, sorted(files or {})))
        return SimpleNamespace(wait=lambda: self._exec(shell, files or {}))

    def _exec(self, shell, overlay):
        if "pip install" in shell:
            return FakeImage(self.files, self.log)
        tokens = shlex.split(shell)
        inner = tokens[-1]                      # cd /app && [rm -f -- x &&] timeout N sh -c '<inner>'
        deleted = []
        if "rm" in tokens:                      # deletions the branch made
            i = tokens.index("--") + 1
            deleted = tokens[i:tokens.index("&&", i)]
        root = Path(tempfile.mkdtemp())
        allfiles = dict(self.files)
        allfiles.update({k: Path(v).read_bytes() for k, v in overlay.items()})
        for rel in deleted:
            allfiles.pop(f"app/{rel}", None)
        for key, data in allfiles.items():
            p = root / key
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        p = subprocess.run(inner, shell=True, cwd=root / "app", capture_output=True, text=True)
        return FakeImage(self.files, self.log, p.returncode, p.stdout, p.stderr)


@pytest.fixture
def fake_nebius(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX", "nebius")
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "test")
    monkeypatch.setattr(ag, "BASE_DIR", tmp_path)
    ag._base_cache.clear()
    monkeypatch.setitem(__import__("sys").modules, "contree_sdk", SimpleNamespace())  # sandbox_ready() import check
    calls = []
    client = SimpleNamespace(images=SimpleNamespace(use=lambda name: FakeImage({}, calls)))
    monkeypatch.setattr(ag, "_contree_client", lambda: client)
    return calls


def test_nebius_runs_tests_in_sandbox(fake_nebius, tmp_path):
    r = tmp_path / "repo"; r.mkdir()
    (r / "calc.py").write_text(BUGGY); (r / "test_calc.py").write_text(TEST)
    code, out = ag.run_in_sandbox(str(r), "python -m pytest -q")
    assert code == 1 and "1 failed" in out


def test_nebius_branches_share_one_checkpoint_and_overlay_edits(fake_nebius, tmp_path, monkeypatch):
    r = tmp_path / "repo"; r.mkdir()
    (r / "calc.py").write_text(BUGGY); (r / "test_calc.py").write_text(TEST)
    monkeypatch.setattr(ag, "chat", make_fake_chat([["bad idea", "GOOD fix", "bad 2"]]))
    s = ag.run(str(r), "fix add", branches=3, rounds=1, work_root=str(tmp_path / "branches"))
    assert s["status"] == "green" and s["winner"]["branch_id"] == "r1_b1"
    applies = [c for c in fake_nebius if c[0] == "apply_files"]
    assert len(applies) == 1                                   # repo uploaded once, shared by all branches
    branch_runs = [c for c in fake_nebius if c[0] == "run" and c[2]]
    assert branch_runs and all(c[2] == ["app/calc.py"] for c in branch_runs)  # only edited file overlaid


def test_sandbox_ready_reports_missing_keys(monkeypatch):
    monkeypatch.setenv("SANDBOX", "nebius")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert "NEBIUS_API_KEY" in ag.sandbox_ready()
    monkeypatch.setenv("SANDBOX", "local")
    assert ag.sandbox_ready() is None


def test_llm_client_not_shadowed(monkeypatch):
    """Regression: the sandbox helper must not overwrite the LLM client global."""
    monkeypatch.setattr(ag, "_client", None)
    monkeypatch.setenv("PROVIDER", "ollama")
    assert hasattr(ag.get_client(), "chat")


# --- speed pass + trajectories ---------------------------------------------------

import time as _time


def test_branch_stops_as_soon_as_auto_test_passes(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "chat", make_fake_chat([["GOOD fix"]]))
    s = ag.run(str(repo), "fix", branches=1, rounds=1, max_turns=10,
               work_root=str(tmp_path / "b"), full=True)
    r = s["results"][0]
    assert r["passed"] and r["turns"] == 1 and r["stop_reason"] == "tests passed"


def test_first_green_cancels_slow_branches(repo, tmp_path, monkeypatch):
    def slow_chat(model, messages, tools=None, **kw):
        system = messages[0]["content"]
        if "planner" in system:
            return msg(json.dumps({"strategies": ["GOOD fix", "slow bad"]})), (1, 1)
        if "GOOD" in messages[1]["content"]:
            return msg("", [call("write_file", path="calc.py", content=FIXED)]), (1, 1)
        _time.sleep(0.4)  # the bad branch keeps writing wrong code forever
        return msg("", [call("write_file", path="calc.py", content=WRONG)]), (1, 1)
    monkeypatch.setattr(ag, "chat", slow_chat)
    s = ag.run(str(repo), "fix", branches=2, rounds=1, max_turns=20,
               work_root=str(tmp_path / "b"), full=True)
    by = {r["branch_id"]: r for r in s["results"]}
    assert by["r1_b0"]["passed"]
    assert by["r1_b1"]["cancelled"] and by["r1_b1"]["turns"] < 20
    assert s["summary"]["status"] == "green"


def test_all_branches_mode_does_not_cancel(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "chat", make_fake_chat([["GOOD fix", "bad"]]))
    s = ag.run(str(repo), "fix", branches=2, rounds=1, first_green=False,
               work_root=str(tmp_path / "b"), full=True)
    assert not any(r["cancelled"] for r in s["results"])


def test_truncated_reply_is_not_applied(repo, tmp_path, monkeypatch):
    seen = {"n": 0}
    def trunc_chat(model, messages, tools=None, **kw):
        if "planner" in messages[0]["content"]:
            return msg(json.dumps({"strategies": ["GOOD fix"]})), (1, 1)
        seen["n"] += 1
        if seen["n"] == 1:   # cut off mid tool call: must not be executed
            m = msg("", [call("write_file", path="calc.py", content=WRONG)])
            m.finish_reason = "length"
            return m, (1, 1)
        assert "cut off" in messages[-1]["content"]
        return msg("", [call("write_file", path="calc.py", content=FIXED)]), (1, 1)
    monkeypatch.setattr(ag, "chat", trunc_chat)
    s = ag.run(str(repo), "fix", branches=1, rounds=1, work_root=str(tmp_path / "b"), full=True)
    assert s["results"][0]["passed"] and s["results"][0]["turns"] == 2


def test_winning_trajectory_saved(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "chat", make_fake_chat([["bad", "GOOD fix"]]))
    s = ag.run(str(repo), "fix add", branches=2, rounds=1, work_root=str(tmp_path / "b"))
    files = list((tmp_path / "logs" / "trajectories").glob("*.json"))
    assert s["trajectories_saved"] == 1 and len(files) == 1
    rec = json.loads(files[0].read_text())
    assert rec["winner"] and rec["messages"][0]["role"] == "system" and rec["tools"]


def test_repo_context_puts_mentioned_files_first(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "calc.py").write_text(BUGGY)
    ctx = ag.repo_context(str(tmp_path), "FAILED test_calc.py ... calc.py:2")
    assert ctx.index("### calc.py") < ctx.index("### a.py")
    small = ag.repo_context(str(tmp_path), "", budget=40)
    assert "Other files" in small


def test_thinking_switch_falls_back_if_provider_rejects(monkeypatch):
    class BadRequest(Exception):
        status_code = 400

    class Completions:
        def create(self, **kw):
            if "extra_body" in kw:
                raise BadRequest("unknown field")
            m = SimpleNamespace(content="ok", tool_calls=None)
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                                   choices=[SimpleNamespace(message=m, finish_reason="stop")])
    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(ag, "get_client", lambda: fake)
    monkeypatch.setattr(ag, "_thinking_unsupported", False)
    m, usage = ag.chat("m", [{"role": "user", "content": "hi"}], thinking="off", max_tokens=100)
    assert m.content == "ok" and ag._thinking_unsupported

def test_rate_limit_does_not_disable_thinking_switch(monkeypatch):
    class RateLimited(Exception):
        status_code = 429
    class Completions:
        def create(self, **kw):
            raise RateLimited("slow down")
    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(ag, "get_client", lambda: fake)
    monkeypatch.setattr(ag, "_thinking_unsupported", False)
    with pytest.raises(RateLimited):
        ag.chat("m", [{"role": "user", "content": "hi"}], thinking="off")
    assert not ag._thinking_unsupported


# --- tamper guard -------------------------------------------------------------

def plan_then(executor_turns):
    """Fake chat: planner returns one strategy, executor replays executor_turns, then 'done'."""
    seen = {"n": 0}
    def fake(model, messages, tools=None, **kw):
        if "planner" in messages[0]["content"]:
            return msg(json.dumps({"strategies": ["s"]})), (1, 1)
        if "reflector" in messages[0]["content"]:
            return msg("nope"), (1, 1)
        seen["n"] += 1
        if seen["n"] <= len(executor_turns):
            return msg("", executor_turns[seen["n"] - 1]), (1, 1)
        return msg("done"), (1, 1)
    return fake


def test_is_protected():
    for rel in ["test_calc.py", "calc_test.py", "conftest.py", "pytest.ini", "pyproject.toml",
                "tests/helpers.py", "pkg/tests/data.json"]:
        assert ag.is_protected(rel), rel
    for rel in ["calc.py", "testing_utils.py", "src/contest.py", "latest.py"]:
        assert not ag.is_protected(rel), rel


def test_writing_a_test_file_is_refused(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "chat", plan_then([
        [call("write_file", path="test_calc.py", content="def test_add():\n    pass\n")]]))
    s = ag.run(str(repo), "fix", branches=1, rounds=1, work_root=str(tmp_path / "b"),
               make_branch=False, full=True)
    r = s["results"][0]
    assert not r["passed"] and r["refused_writes"] == 1
    assert s["summary"]["status"] == "red" and s["summary"]["tamper_attempts"] == 1
    assert (repo / "test_calc.py").read_text() == TEST


def test_test_edited_through_shell_is_reverted(repo, tmp_path, monkeypatch):
    """SANDBOX=local runs shell commands in the branch dir, so they can edit files directly."""
    edit = "python -c \"open('test_calc.py','w').write('def test_add():\\n    pass\\n')\""
    monkeypatch.setattr(ag, "chat", plan_then([
        [call("run_command", command=edit), call("write_file", path="notes.py", content="x = 1\n")]]))
    s = ag.run(str(repo), "fix", branches=1, rounds=1, work_root=str(tmp_path / "b"),
               make_branch=False, full=True)
    r = s["results"][0]
    assert not r["passed"] and r["tampered"] == ["test_calc.py"]
    assert "test_calc.py" not in r["changed"]            # the patch never contains test edits


def test_new_conftest_is_removed(repo, tmp_path, monkeypatch):
    skip_all = "import pytest\n\ndef pytest_collection_modifyitems(items):\n    items.clear()\n"
    monkeypatch.setattr(ag, "chat", plan_then([
        [call("run_command", command="python -c \"open('conftest.py','w').write('x')\""),
         call("write_file", path="notes.py", content="x = 1\n")]]))
    s = ag.run(str(repo), "fix", branches=1, rounds=1, work_root=str(tmp_path / "b"),
               make_branch=False, full=True)
    assert s["results"][0]["tampered"] == ["conftest.py"] and not s["results"][0]["passed"]


def test_model_reported_green_does_not_count(repo, tmp_path, monkeypatch):
    """'pytest -q || true' exits 0 but proves nothing; only the judge can mark green."""
    monkeypatch.setattr(ag, "chat", plan_then([
        [call("write_file", path="notes.py", content="x = 1\n"),
         call("run_command", command="pytest -q || true")]]))
    s = ag.run(str(repo), "fix", branches=1, rounds=1, work_root=str(tmp_path / "b"),
               make_branch=False)
    assert s["status"] == "red"


def test_skipping_tests_from_source_does_not_count(repo, tmp_path, monkeypatch):
    """All tests skipped exits 0, but fewer tests pass than existed at baseline."""
    skipper = "import pytest\n\ndef add(a, b):\n    pytest.skip('nope')\n"   # exits 0: '1 skipped'
    monkeypatch.setattr(ag, "chat", plan_then([
        [call("write_file", path="calc.py", content=skipper)]]))
    s = ag.run(str(repo), "fix", branches=1, rounds=1, work_root=str(tmp_path / "b"),
               make_branch=False, full=True)
    r = s["results"][0]
    assert not r["passed"] and "[judge]" in r["test_output"]


def test_pytest_counts():
    assert ag.pytest_counts("..F\n1 failed, 2 passed in 0.1s") == {"failed": 1, "passed": 2}
    assert ag.pytest_counts("= 3 passed, 1 skipped, 2 errors in 1s =") == {
        "passed": 3, "skipped": 1, "error": 2}
    assert ag.pytest_counts("hello") == {}


# --- patch, deletions, cost -----------------------------------------------------------

def test_patch_applies_with_git_when_files_lack_final_newline(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    # write_bytes everywhere: write_text would add \r on Windows
    (a / "one.py").write_bytes(b"x = 1")               # no trailing newline
    (b / "one.py").write_bytes(b"x = 2")
    (a / "two.py").write_bytes(b"y = 1\n")
    (b / "two.py").write_bytes(b"y = 2\n")
    patch = ag.make_diff(str(a), str(b))["patch"]
    (tmp_path / "fix.patch").write_bytes(patch.encode())
    sh(a, "init", "-q")
    sh(a, "config", "core.autocrlf", "false")         # Windows git would rewrite LF as CRLF on apply
    subprocess.run(["git", "-C", str(a), "apply", str(tmp_path / "fix.patch")], check=True)
    assert (a / "one.py").read_bytes() == b"x = 2" and (a / "two.py").read_bytes() == b"y = 2\n"


def test_deleted_file_is_removed_in_nebius_sandbox(fake_nebius, tmp_path):
    r = tmp_path / "repo"; r.mkdir()
    (r / "calc.py").write_text(FIXED); (r / "test_calc.py").write_text(TEST)
    (r / "test_broken.py").write_text("def test_x():\n    assert False\n")
    b = tmp_path / "branch"
    ag.copy_repo(str(r), str(b))
    (b / "test_broken.py").unlink()
    code, out = ag.run_in_sandbox(str(b), "python -m pytest -q")
    assert code == 0 and "1 passed" in out


def test_cost_usd_uses_price_table(monkeypatch):
    assert ag.cost_usd("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", 1_000_000, 1_000_000) == pytest.approx(0.30)
    assert ag.cost_usd("nvidia/Nemotron-3-Ultra-550b-a55b", 1_000_000, 0) == pytest.approx(1.0)
    assert ag.cost_usd("llama3.1:8b", 10**6, 10**6) == 0
    monkeypatch.setenv("PRICES_JSON", '{"llama": [1, 1]}')
    assert ag.cost_usd("llama3.1:8b", 10**6, 0) == pytest.approx(1.0)


def test_budget_cap_stops_run(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("EXECUTOR_MODEL", "x-ultra")   # stage models only apply to nebius
    monkeypatch.setenv("PROVIDER", "nebius")
    def pricey(model, messages, tools=None, **kw):
        if "planner" in messages[0]["content"]:
            return msg(json.dumps({"strategies": ["a", "b"]})), (10, 10)
        return msg("", [call("write_file", path="calc.py", content=WRONG)]), (1_000_000, 0)  # $1/turn
    monkeypatch.setattr(ag, "chat", pricey)
    s = ag.run(str(repo), "fix", branches=2, rounds=3, max_turns=10, max_usd=1.5,
               work_root=str(tmp_path / "b"), make_branch=False, full=True)
    assert s["summary"]["status"] == "red" and s["summary"]["rounds"] == 1   # no second round
    assert sum(r["turns"] for r in s["results"]) <= 4
    assert all(r["stop_reason"] in ("budget", "gave up") for r in s["results"])


def test_patch_applies_to_crlf_files(tmp_path):
    """Windows-created repos use CRLF; the patch must keep it or git apply fails."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "calc.py").write_bytes(BUGGY.replace("\n", "\r\n").encode())
    (b / "calc.py").write_bytes(FIXED.replace("\n", "\r\n").encode())
    (tmp_path / "fix.patch").write_bytes(ag.make_diff(str(a), str(b))["patch"].encode())
    sh(a, "init", "-q")
    sh(a, "config", "core.autocrlf", "false")
    subprocess.run(["git", "-C", str(a), "apply", str(tmp_path / "fix.patch")], check=True)
    assert (a / "calc.py").read_bytes() == FIXED.replace("\n", "\r\n").encode()


def test_write_file_keeps_line_endings(tmp_path):
    (tmp_path / "lf.py").write_bytes(b"x = 1\n")
    (tmp_path / "crlf.py").write_bytes(b"x = 1\r\n")
    ag.run_tool(str(tmp_path), "write_file", {"path": "lf.py", "content": "x = 2\ny = 3\n"})
    ag.run_tool(str(tmp_path), "write_file", {"path": "crlf.py", "content": "x = 2\ny = 3\n"})
    ag.run_tool(str(tmp_path), "write_file", {"path": "new.py", "content": "z = 1\n"})
    assert (tmp_path / "lf.py").read_bytes() == b"x = 2\ny = 3\n"       # no \r added on Windows
    assert (tmp_path / "crlf.py").read_bytes() == b"x = 2\r\ny = 3\r\n"
    assert (tmp_path / "new.py").read_bytes() == b"z = 1\n"
    assert ag.make_diff(str(tmp_path), str(tmp_path))["lines"] == 0


def test_saved_winner_patch_is_byte_exact(repo, tmp_path, monkeypatch):
    """The patch file on disk must be exactly the patch (no \r added on Windows)."""
    monkeypatch.setattr(ag, "chat", make_fake_chat([["GOOD fix"]]))
    s = ag.run(str(repo), "fix", branches=1, rounds=1, work_root=str(tmp_path / "b"),
               make_branch=False, full=True)
    saved = (tmp_path / "logs" / f"graph_{s['run_id']}_winner.patch").read_bytes()
    # A CRLF repo (write_text on Windows) legitimately gives CRLF lines; what must never
    # happen is text-mode writing adding a second \r (\r\r\n) or \r to LF lines.
    assert saved == s["winner"]["patch"].encode() and b"\r\r\n" not in saved
