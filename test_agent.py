"""Tests for agent.py - run with:  pytest -v

No model, no Docker, no internet needed: the model and Docker are replaced
by fakes, so every test is free, fast, and gives the same result every time.
"""
import json
import os
import types

import pytest

# agent.py reads these at import time - set fakes BEFORE importing it
os.environ.update({"PROVIDER": "nebius", "NEBIUS_API_KEY": "test", "NEBIUS_MODEL": "fake"})
import agent  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_log(monkeypatch, tmp_path):
    """Every test writes its log to a temporary folder, never to your real logs/."""
    monkeypatch.setattr(agent, "LOG_FILE", tmp_path / "run.jsonl")


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


def test_loop_reflects_and_recovers(monkeypatch):
    fail = "exit_code: 1\nstdout:\n\nstderr:\nAssertionError\n"
    result = run_scripted(monkeypatch,
        agent_replies=[
            reply(calls=[tool_call(1, "write_file", {"path": "primes.py", "content": "x"})]),
            reply(content="step 1 done"),
            reply(calls=[tool_call(2, "run_command", {"command": "python test_primes.py"})]),
            reply(calls=[tool_call(3, "run_command", {"command": "python test_primes.py"})]),
            reply(calls=[tool_call(4, "run_command", {"command": "python test_primes.py"})]),
            reply(content="step 2 done"),
        ],
        command_outputs=[fail, fail, "exit_code: 0\nstdout:\nok\nstderr:\n"],
    )
    assert result["status"] == "done"
    assert result["reflections"] == 2
    events = read_log(result["log"])
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["task", "plan"] and kinds[-1] == "finish"
    reflections = [e for e in events if e["event"] == "reflection"]
    assert [e["repeated_error"] for e in reflections] == [False, True]


def test_loop_stops_at_max_steps(monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 4)
    ok = "exit_code: 0\nstdout:\n\nstderr:\n"
    result = run_scripted(monkeypatch,
        agent_replies=[reply(calls=[tool_call(i, "run_command", {"command": "ls"})]) for i in range(10)],
        command_outputs=[ok] * 10,
    )
    assert result["status"] == "max_steps" and result["completed_steps"] == 0