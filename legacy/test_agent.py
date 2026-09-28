"""Tests for agent.py - run with:  pytest -v

No model, no Docker, no internet needed: the model and Docker are replaced
by fakes, so every test is free, fast, and gives the same result every time.
"""
import json
import os
import types

import pytest

# agent.py reads these at import time - set fakes BEFORE importing it
os.environ.update({"PROVIDER": "nebius", "NEBIUS_API_KEY": "test", "NEBIUS_MODEL": "fake",
                   "SANDBOX": "docker"})
for key in ("PLANNER_MODEL", "REFLECTOR_MODEL", "EXECUTOR_MODEL"):
    os.environ.pop(key, None)
import agent  # noqa: E402
import sandboxes  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Every test uses a temporary workspace and log - never your real workspace/ or logs/."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(agent, "LOG_FILE", tmp_path / "run.jsonl")
    monkeypatch.setattr(agent, "WORKSPACE", ws.resolve())
    monkeypatch.setattr(agent, "SANDBOX", sandboxes.DockerSandbox(ws.resolve(), "fork-sandbox"))
    return ws


# ---------- helpers that build fake model replies ----------
def tool_call(i, name, args):
    return types.SimpleNamespace(id=f"call{i}",
                                 function=types.SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content=None, calls=None):
    msg = types.SimpleNamespace(content=content, tool_calls=calls)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)


# ======================================================================
# Level 1 - pure functions (no model at all)
# ======================================================================
def test_extract_json_strips_think_block_and_fences():
    text = '<think>planning...</think>\n```json\n{"steps": []}\n```'
    assert agent.extract_json(text) == {"steps": []}


def test_extract_json_raises_when_no_json():
    with pytest.raises(ValueError):
        agent.extract_json("Sorry, I cannot help with that.")


@pytest.mark.parametrize("tool,result,expected", [
    ("run_command", "exit_code: 0\nstdout:\nok", False),
    ("run_command", "exit_code: 1\nstderr:\nAssertionError", True),
    ("run_command", "ERROR: command timed out after 30 seconds", True),
    ("write_file", "exit_code: 1", False),        # only run_command counts
])
def test_command_failed(tool, result, expected):
    assert agent.command_failed(tool, result) is expected


def test_safe_path_blocks_escape():
    with pytest.raises(ValueError):
        agent.safe_path("../../secret.txt")


# ======================================================================
# Level 2 - components with a fake model
# ======================================================================
def test_planner_parses_valid_plan(monkeypatch):
    plan = '{"steps":[{"id":1,"description":"a","success_check":"x"},' \
           '{"id":2,"description":"b","success_check":"y"}]}'
    monkeypatch.setattr(agent.client.chat.completions, "create", lambda **kw: reply(content=plan))
    steps = agent.make_plan("task")
    assert [s.description for s in steps] == ["a", "b"]


def test_planner_falls_back_on_garbage(monkeypatch):
    monkeypatch.setattr(agent.client.chat.completions, "create", lambda **kw: reply(content="no idea"))
    steps = agent.make_plan("build a calculator")
    assert len(steps) == 1 and steps[0].description == "build a calculator"


def test_reflector_returns_structured_diagnosis(monkeypatch):
    r = '{"diagnosis":"off by one","category":"logic","next_action":"fix_code"}'
    monkeypatch.setattr(agent.client.chat.completions, "create", lambda **kw: reply(content=r))
    step = agent.Step(id=1, description="d", success_check="s")
    out = agent.reflect(step, "python t.py", "AssertionError", [])
    assert out.category == "logic" and out.next_action == "fix_code"


# ======================================================================
# Level 3 - the whole loop with a scripted fake model
# ======================================================================
PLAN = ('<think>ok</think>```json\n{"steps":['
        '{"id":1,"description":"write primes.py","success_check":"imports"},'
        '{"id":2,"description":"write and run tests","success_check":"tests pass"}]}\n```')
REFLECTION = '{"diagnosis":"is_prime(1) is wrong","category":"logic","next_action":"fix_code"}'


def run_scripted(monkeypatch, agent_replies, command_outputs):
    """Run run_agent() with a scripted model and fake Docker results."""
    outputs = iter(command_outputs)
    monkeypatch.setitem(agent.FUNCTIONS, "run_command", lambda command="", cmd="": next(outputs))
    queue = iter(agent_replies)

    def fake_create(**kw):
        if "tools" not in kw:  # planner or reflector call
            is_planner = "planning module" in kw["messages"][0]["content"]
            return reply(content=PLAN if is_planner else REFLECTION)
        # API rule: tool results must directly follow the assistant message that asked
        msgs = kw["messages"]
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                ids = [t["id"] for t in m["tool_calls"]]
                assert [x.get("tool_call_id") for x in msgs[i + 1:i + 1 + len(ids)]] == ids
        return next(queue)

    monkeypatch.setattr(agent.client.chat.completions, "create", fake_create)
    return agent.run_agent("primes with tests")


def read_log(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


FAIL = "exit_code: 1\nstdout:\n\nstderr:\nAssertionError\n"
PASS = "exit_code: 0\nstdout:\nok\nstderr:\n"


def test_loop_reflects_and_recovers(monkeypatch):
    """Fail -> reflect -> edit file -> fail again -> reflect again -> fix -> pass."""
    result = run_scripted(monkeypatch,
        agent_replies=[
            reply(calls=[tool_call(1, "write_file", {"path": "primes.py", "content": "x"})]),
            reply(content="step 1 done"),
            reply(calls=[tool_call(2, "run_command", {"command": "python test_primes.py"})]),
            reply(calls=[tool_call(3, "write_file", {"path": "primes.py", "content": "y"}),
                         tool_call(4, "run_command", {"command": "python test_primes.py"})]),
            reply(calls=[tool_call(5, "write_file", {"path": "primes.py", "content": "z"}),
                         tool_call(6, "run_command", {"command": "python test_primes.py"})]),
            reply(content="step 2 done"),
        ],
        command_outputs=[FAIL, FAIL, PASS],
    )
    assert result["status"] == "done"
    assert result["reflections"] == 2 and result["skipped_reflections"] == 0
    events = read_log(result["log"])
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["task", "plan"] and kinds[-1] == "finish"
    reflections = [e for e in events if e["event"] == "reflection"]
    assert [e["repeated_error"] for e in reflections] == [False, True]


def test_rerun_without_change_skips_reflection(monkeypatch):
    """Re-running a failed command with no file change must NOT call the reflector."""
    result = run_scripted(monkeypatch,
        agent_replies=[
            reply(content="step 1 done"),
            reply(calls=[tool_call(1, "run_command", {"command": "pytest"})]),
            reply(calls=[tool_call(2, "run_command", {"command": "pytest"})]),   # nothing changed
            reply(calls=[tool_call(3, "run_command", {"command": "pytest"})]),   # still nothing
            reply(calls=[tool_call(4, "write_file", {"path": "primes.py", "content": "fix"}),
                         tool_call(5, "run_command", {"command": "pytest"})]),
            reply(content="step 2 done"),
        ],
        command_outputs=[FAIL, FAIL, FAIL, PASS],
    )
    assert result["status"] == "done"
    assert result["reflections"] == 1          # only the first failure was diagnosed
    assert result["skipped_reflections"] == 2  # the two blind re-runs were skipped
    kinds = [e["event"] for e in read_log(result["log"])]
    assert kinds.count("reflection_skipped") == 2


def test_loop_stops_at_max_steps(monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 4)
    ok = "exit_code: 0\nstdout:\n\nstderr:\n"
    result = run_scripted(monkeypatch,
        agent_replies=[reply(calls=[tool_call(i, "run_command", {"command": "ls"})]) for i in range(10)],
        command_outputs=[ok] * 10,
    )
    assert result["status"] == "max_steps" and result["completed_steps"] == 0


def test_each_stage_uses_its_own_model_and_tokens_are_counted(monkeypatch):
    monkeypatch.setattr(agent, "PLANNER_MODEL", "big-planner")
    monkeypatch.setattr(agent, "REFLECTOR_MODEL", "big-reflector")
    monkeypatch.setattr(agent, "EXECUTOR_MODEL", "small-executor")
    outputs = iter([FAIL, PASS])
    monkeypatch.setitem(agent.FUNCTIONS, "run_command", lambda command="", cmd="": next(outputs))
    script = iter([
        reply(content="step 1 done"),
        reply(calls=[tool_call(1, "run_command", {"command": "pytest"})]),
        reply(calls=[tool_call(2, "write_file", {"path": "a.py", "content": "x"}),
                     tool_call(3, "run_command", {"command": "pytest"})]),
        reply(content="step 2 done"),
    ])
    used = []

    def fake_create(**kw):
        used.append(kw["model"])
        if "tools" not in kw:
            content = PLAN if "planning module" in kw["messages"][0]["content"] else REFLECTION
            r = reply(content=content)
        else:
            r = next(script)
        r.usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=10)
        return r

    monkeypatch.setattr(agent.client.chat.completions, "create", fake_create)
    result = agent.run_agent("task")

    assert used[0] == "big-planner"
    assert "big-reflector" in used
    assert set(used) == {"big-planner", "big-reflector", "small-executor"}
    u = result["usage"]
    assert u["planner"]["calls"] == 1 and u["reflector"]["calls"] == 1 and u["executor"]["calls"] == 4
    assert result["tokens_in"] == 100 * 6          # all 6 calls counted, not just executor


# ======================================================================
# Nebius Sandboxes backend - tested with a fake snapshot object
# (same methods the real contree-sdk image has: run, wait, apply_files, read)
# ======================================================================
class FakeSnapshot:
    """Pretends to be a Nebius snapshot. `handler(cmd, fs)` decides what a command does."""
    calls = []

    def __init__(self, fs=None, handler=None):
        self.fs = dict(fs or {})
        self.handler = handler
        self.exit_code, self.stdout, self.stderr = 0, "", ""

    def run(self, command=None, *, shell=None, cwd=None, disposable=True, timeout=None, **kw):
        FakeSnapshot.calls.append({"shell": shell, "cwd": cwd, "disposable": disposable})
        child = FakeSnapshot(self.fs, self.handler)
        if shell.startswith("find "):
            child.stdout = "\n".join("./" + p.removeprefix("/workspace/")
                                      for p in child.fs if p.startswith("/workspace/"))
        elif shell.startswith("mkdir"):
            pass
        elif self.handler:
            child.exit_code, child.stdout, child.stderr = self.handler(shell, child.fs)
        return child

    def wait(self):
        return self

    def apply_files(self, files):
        return FakeSnapshot({**self.fs, **files}, self.handler)

    def read(self, path):
        if path not in self.fs:
            raise FileNotFoundError(path)
        return self.fs[path]


def make_fake_nebius(ws, handler=None):
    sb = sandboxes.NebiusSandbox(ws, "python:3.12-slim")
    sb.sdk = object()                 # pretend we're already connected
    sb.base = FakeSnapshot(handler=handler)
    return sb


def test_nebius_start_uploads_local_workspace(isolated):
    (isolated / "primes.py").write_text("def is_prime(n): return n > 1")
    (isolated / "__pycache__").mkdir()
    (isolated / "__pycache__" / "junk.pyc").write_bytes(b"x")
    sb = make_fake_nebius(isolated)
    sb.start()
    assert sb.state.fs == {"/workspace/primes.py": b"def is_prime(n): return n > 1"}


def test_nebius_write_read_and_state_chaining(isolated):
    def handler(cmd, fs):          # "python" appends a file, like a real program would
        fs["/workspace/out.txt"] = b"hello"
        return 0, "ran", ""
    sb = make_fake_nebius(isolated, handler)
    sb.start()
    sb.write_file("a.py", "print(1)")
    assert sb.read_file("a.py") == "print(1)"
    assert sb.read_file("missing.py") is None
    code, out, err = sb.run("python a.py")
    assert (code, out) == (0, "ran")
    assert sb.read_file("out.txt") == "hello"       # next step sees the previous command's files
    assert FakeSnapshot.calls[-1]["cwd"] == "/workspace"
    assert FakeSnapshot.calls[-1]["disposable"] is False


def test_nebius_finish_copies_files_back(isolated):
    sb = make_fake_nebius(isolated)
    sb.start()
    sb.write_file("pkg/mod.py", "x = 1")
    assert sb.finish() == 1
    assert (isolated / "pkg" / "mod.py").read_text() == "x = 1"


def test_agent_tools_use_nebius_backend(isolated, monkeypatch):
    sb = make_fake_nebius(isolated, lambda cmd, fs: (1, "", "AssertionError"))
    sb.start()
    monkeypatch.setattr(agent, "SANDBOX", sb)
    assert agent.write_file("t.py", "assert False").startswith("Wrote")
    assert agent.read_file("t.py") == "assert False"
    assert agent.command_failed("run_command", agent.run_command("python t.py"))
    assert not (isolated / "t.py").exists()          # nothing written locally until finish()
    with pytest.raises(ValueError):
        agent.write_file("../escape.py", "x")        # path jail still applies


def test_missing_project_id_gives_clear_error(isolated, monkeypatch):
    monkeypatch.delenv("NEBIUS_PROJECT_ID", raising=False)
    sb = sandboxes.NebiusSandbox(isolated, "python:3.12-slim")
    with pytest.raises(RuntimeError, match="NEBIUS_PROJECT_ID"):
        sb.start()