"""
agent_graph.py - LangGraph version of Fork with parallel branching.

Flow:
    baseline -> (already green? -> finalize)
             -> planner -> [executor x N in parallel] -> selector
             -> (winner? -> finalize)
             -> (rounds left? -> reflector -> planner) else finalize

Each executor branch works on its own copy of the repo, so branches never
overwrite each other. agent.py is untouched and stays the working fallback.

Usage:
    python agent_graph.py --repo workspace/myrepo --task "Make the failing tests pass"
    python agent_graph.py --repo workspace/myrepo --branches 3 --rounds 2

By default the winning fix is committed to a new git branch (fork/fix-<run_id>)
in the target repo WITHOUT touching your working tree. Review it, then merge.
    --apply      also copy the fix into the working tree
    --no-branch  skip the git branch (patch file only)
"""

from __future__ import annotations

import argparse
import difflib
import json
import operator
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".mypy_cache"}
MAX_FILE_BYTES = 1_000_000

_log_hook = None  # set by run(on_log=...) so a UI can stream progress


def log(msg: str) -> None:
    print(msg, flush=True)
    if _log_hook:
        try:
            _log_hook(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LLM layer (OpenAI-compatible: Nebius / OpenRouter / Ollama)
# ---------------------------------------------------------------------------

def provider_config() -> dict:
    p = os.getenv("PROVIDER", "nebius").lower()
    if p == "nebius":
        return {
            "provider": p,
            "base_url": os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
            "api_key": os.getenv("NEBIUS_API_KEY", ""),
            "model": os.getenv("NEBIUS_MODEL", ""),
        }
    if p == "openrouter":
        return {
            "provider": p,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "model": os.getenv("OPENROUTER_MODEL", ""),
        }
    if p == "ollama":
        return {
            "provider": p,
            "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/v1",
            "api_key": "ollama",
            "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        }
    raise ValueError(f"Unknown PROVIDER: {p}")


def stage_model(stage: str) -> str:
    """Per-stage models (PLANNER_MODEL etc.) only apply to Nebius; others use one model."""
    cfg = provider_config()
    if cfg["provider"] == "nebius":
        return os.getenv(f"{stage.upper()}_MODEL") or cfg["model"]
    return cfg["model"]


_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            cfg = provider_config()
            _client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    return _client


def chat(model: str, messages: list, tools: Optional[list] = None):
    """Single LLM call. Returns (message, (input_tokens, output_tokens)).
    Tests monkeypatch this function."""
    kwargs = {"model": model, "messages": messages, "temperature": 0.2}
    if tools:
        kwargs["tools"] = tools
    resp = get_client().chat.completions.create(**kwargs)
    u = resp.usage
    usage = (getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
    return resp.choices[0].message, usage


def parse_json(text: str):
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    text = (text or "").replace("```json", "").replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

_warned_nebius = False


def run_in_sandbox(workdir: str, command: str, timeout: int = 120) -> tuple[int, str]:
    """Run a command against a branch directory.
    SANDBOX=docker (default) uses the fork-sandbox image: no network, 512m, 1 CPU.
    SANDBOX=local runs on the host (tests / quick debugging only)."""
    global _warned_nebius
    mode = os.getenv("SANDBOX", "docker").lower()

    if mode == "local":
        try:
            p = subprocess.run(command, shell=True, cwd=workdir, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {timeout}s"

    if mode == "nebius" and not _warned_nebius:
        log("[agent_graph] SANDBOX=nebius not wired into the graph yet, using docker.")
        _warned_nebius = True

    name = f"fork-{uuid.uuid4().hex[:10]}"
    docker_cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none", "--memory", "512m", "--cpus", "1",
        "-v", f"{Path(workdir).resolve()}:/workspace", "-w", "/workspace",
        os.getenv("SANDBOX_IMAGE", "fork-sandbox"), "sh", "-c", command,
    ]
    try:
        p = subprocess.run(docker_cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return 124, f"Command timed out after {timeout}s"


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def tail(text: str, n: int = 4000) -> str:
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def safe_path(root: str, rel: str) -> Path:
    """Resolve rel inside root; refuse anything that escapes the branch folder."""
    root_p = Path(root).resolve()
    p = (root_p / rel).resolve()
    if p != root_p and root_p not in p.parents:
        raise ValueError(f"Path escapes workspace: {rel}")
    return p


def copy_repo(src: str, dst: str) -> None:
    if Path(dst).exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*IGNORE_DIRS))


def collect_files(root: str) -> dict[str, str]:
    out = {}
    root_p = Path(root)
    for dirpath, dirnames, filenames in os.walk(root_p):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for f in filenames:
            p = Path(dirpath) / f
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                out[p.relative_to(root_p).as_posix()] = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # skip binary files
    return out


def make_diff(orig_dir: str, new_dir: str) -> dict:
    a, b = collect_files(orig_dir), collect_files(new_dir)
    patch, changed, lines = [], [], 0
    for rel in sorted(set(a) | set(b)):
        old, new = a.get(rel, ""), b.get(rel, "")
        if old == new:
            continue
        changed.append(rel)
        d = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                      fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        patch.extend(d)
        lines += sum(1 for l in d if l[:1] in "+-" and not l.startswith(("+++", "---")))
    return {"patch": "".join(patch), "changed": changed, "lines": lines}


def list_files(root: str) -> list[str]:
    return sorted(collect_files(root).keys())


# ---------------------------------------------------------------------------
# Executor tools
# ---------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List all text files in the repo.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file (path relative to repo root).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Overwrite a file with full new content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the sandbox (no internet).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]


def run_tool(workdir: str, name: str, args: dict) -> str:
    try:
        if name == "list_files":
            return "\n".join(list_files(workdir)) or "(empty)"
        if name == "read_file":
            return tail(safe_path(workdir, args["path"]).read_text(encoding="utf-8"), 20000)
        if name == "write_file":
            p = safe_path(workdir, args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"], encoding="utf-8")
            return f"Wrote {args['path']} ({len(args['content'])} chars)"
        if name == "run_command":
            code, out = run_in_sandbox(workdir, args["command"])
            return f"exit code {code}\n{tail(out)}"
        return f"Unknown tool: {name}"
    except Exception as e:  # tool errors go back to the model, not up the stack
        return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    task: str
    repo_dir: str
    test_cmd: str
    work_root: str
    run_id: str
    n_branches: int
    max_rounds: int
    max_turns: int
    apply: bool
    make_branch: bool

    baseline_passed: bool
    baseline_output: str
    round: int
    strategies: list[str]
    feedback: str
    results: Annotated[list[dict], operator.add]   # every branch from every round
    usage: Annotated[list[dict], operator.add]     # per-stage token counts
    winner: Optional[dict]
    summary: dict


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def baseline_node(state: GraphState) -> dict:
    if os.getenv("SANDBOX", "docker").lower() != "local" and not docker_ok():
        raise RuntimeError("Docker isn't running. Start Docker Desktop and retry.")
    code, out = run_in_sandbox(state["repo_dir"], state["test_cmd"])
    if code == 5:
        raise RuntimeError("pytest collected no tests. Check the repo path and test file names.")
    log(f"[baseline] tests {'GREEN' if code == 0 else 'RED'} (exit {code})")
    return {"baseline_passed": code == 0, "baseline_output": tail(out), "round": 0}


PLANNER_PROMPT = """You are the planner for an autonomous coding agent that fixes failing Python test suites.
Propose exactly {n} DIFFERENT strategies to make the tests pass. Each strategy should be a short,
concrete instruction (1-3 sentences) naming the files/functions to look at and the fix idea.
Make them genuinely different (different root-cause hypotheses or approaches), not rewordings.
Never suggest editing or deleting tests to make them pass.
Reply ONLY with JSON: {{"strategies": ["...", "..."]}}"""


def planner_node(state: GraphState) -> dict:
    n = state["n_branches"]
    rnd = state.get("round", 0) + 1
    user = (f"Task: {state['task']}\n\nFiles:\n{chr(10).join(list_files(state['repo_dir']))}\n\n"
            f"Failing test output:\n{state['baseline_output']}")
    if state.get("feedback"):
        user += f"\n\nFeedback from the previous round (these attempts failed):\n{state['feedback']}"

    msg, (i, o) = chat(stage_model("planner"),
                       [{"role": "system", "content": PLANNER_PROMPT.format(n=n)},
                        {"role": "user", "content": user}])
    data = parse_json(msg.content) or {}
    strategies = [s for s in data.get("strategies", []) if isinstance(s, str) and s.strip()][:n]
    while len(strategies) < n:  # fallback if the model returns fewer / bad JSON
        strategies.append("Read the failing test and the code it calls, find the root cause, fix it minimally.")

    log(f"[planner] round {rnd}: {len(strategies)} strategies")
    for k, s in enumerate(strategies):
        log(f"   b{k}: {s[:100]}")
    return {"strategies": strategies, "round": rnd,
            "usage": [{"stage": "planner", "round": rnd, "input": i, "output": o}]}


def fan_out(state: GraphState) -> list[Send]:
    """One parallel executor branch per strategy."""
    return [
        Send("executor", {
            "branch_id": f"r{state['round']}_b{k}",
            "round": state["round"],
            "strategy": s,
            "task": state["task"],
            "repo_dir": state["repo_dir"],
            "test_cmd": state["test_cmd"],
            "baseline_output": state["baseline_output"],
            "work_root": state["work_root"],
            "run_id": state["run_id"],
            "max_turns": state["max_turns"],
        })
        for k, s in enumerate(state["strategies"])
    ]


EXECUTOR_PROMPT = """You are the executor of an autonomous coding agent. You work inside a copy of a Python repo.
Tools: list_files, read_file, write_file, run_command. The sandbox has NO internet: do not pip install.
Goal: make the test command pass by fixing the source code. Do NOT edit or delete tests.
Follow the strategy you are given. Keep changes minimal. Run the tests to check your fix.
When the tests pass (or you cannot make progress), reply with a short summary and no tool calls."""


def executor_node(payload: dict) -> dict:
    bid = payload["branch_id"]
    workdir = str(Path(payload["work_root"]) / payload["run_id"] / bid)
    copy_repo(payload["repo_dir"], workdir)

    messages = [
        {"role": "system", "content": EXECUTOR_PROMPT},
        {"role": "user", "content": (
            f"Task: {payload['task']}\nTest command: {payload['test_cmd']}\n"
            f"Strategy: {payload['strategy']}\n\nFailing output:\n{payload['baseline_output']}")},
    ]
    model = stage_model("executor")
    tin = tout = turns = tool_calls = 0
    t0 = time.time()

    for turns in range(1, payload["max_turns"] + 1):
        try:
            msg, (i, o) = chat(model, messages, TOOLS)
        except Exception as e:
            log(f"[{bid}] LLM error: {e}")
            break
        tin, tout = tin + i, tout + o
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            break
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in calls],
        })
        for c in calls:
            tool_calls += 1
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(workdir, c.function.name, args)
            shown = args.get("path") or args.get("command") or ""
            first = result.splitlines()[0] if result else ""
            log(f"[{bid}] turn {turns}: {c.function.name} {shown} -> {first[:80]}")
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})

    code, out = run_in_sandbox(workdir, payload["test_cmd"])  # final check is the judge
    diff = make_diff(payload["repo_dir"], workdir)
    passed = code == 0
    log(f"[{bid}] {'GREEN' if passed else 'red'} | turns {turns} | tools {tool_calls} | "
          f"diff {diff['lines']} lines | tokens {tin}+{tout} | {time.time() - t0:.0f}s")

    return {
        "results": [{
            "branch_id": bid, "round": payload["round"], "strategy": payload["strategy"],
            "passed": passed, "test_output": tail(out, 2000), "workdir": workdir,
            "diff_lines": diff["lines"], "changed": diff["changed"], "patch": diff["patch"],
            "turns": turns, "tool_calls": tool_calls, "input_tokens": tin, "output_tokens": tout,
        }],
        "usage": [{"stage": "executor", "round": payload["round"], "branch": bid,
                   "input": tin, "output": tout}],
    }


def pick_winner(results: list[dict], rnd: int) -> Optional[dict]:
    """Green branches from this round; smallest diff wins, then fewest tokens."""
    green = [r for r in results if r["round"] == rnd and r["passed"] and r["diff_lines"] > 0]
    if not green:
        return None
    return min(green, key=lambda r: (r["diff_lines"], r["input_tokens"] + r["output_tokens"]))


def selector_node(state: GraphState) -> dict:
    w = pick_winner(state["results"], state["round"])
    log(f"[selector] round {state['round']}: " + (f"winner {w['branch_id']}" if w else "no green branch"))
    return {"winner": w}


REFLECTOR_PROMPT = """You are the reflector of an autonomous coding agent. Several parallel fix attempts failed.
Look at each strategy and its final test output. In under 150 words, explain what went wrong and
what the next round should try differently. Be specific (files, functions, error messages)."""


def reflector_node(state: GraphState) -> dict:
    attempts = [r for r in state["results"] if r["round"] == state["round"]]
    text = "\n\n".join(
        f"## {r['branch_id']}\nStrategy: {r['strategy']}\nChanged: {r['changed']}\n"
        f"Final test output:\n{tail(r['test_output'], 1500)}" for r in attempts)
    msg, (i, o) = chat(stage_model("reflector"),
                       [{"role": "system", "content": REFLECTOR_PROMPT},
                        {"role": "user", "content": f"Task: {state['task']}\n\n{text}"}])
    log(f"[reflector] {(msg.content or '').strip()[:200]}")
    return {"feedback": msg.content or "",
            "usage": [{"stage": "reflector", "round": state["round"], "input": i, "output": o}]}


# ---------------------------------------------------------------------------
# Git: commit the winning fix to a review branch
# ---------------------------------------------------------------------------

def git(cwd: str, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def git_toplevel(path: str) -> Optional[Path]:
    try:
        return Path(git(path, "rev-parse", "--show-toplevel")).resolve()
    except (RuntimeError, FileNotFoundError):
        return None


def commit_to_branch(repo_dir: str, winner: dict, run_id: str, test_cmd: str) -> dict:
    """Commit the winner's files to a new branch using a temporary git worktree,
    so the user's checkout and working tree are never touched."""
    top = git_toplevel(repo_dir)
    if top is None:
        return {"error": "not a git repo (run 'git init' and commit first)"}
    try:
        git(repo_dir, "rev-parse", "--verify", "HEAD")
    except RuntimeError:
        return {"error": "repo has no commits yet"}

    prefix = Path(repo_dir).resolve().relative_to(top)
    branch = f"fork/fix-{run_id}"
    note = None
    if git(str(top), "status", "--porcelain", "--untracked-files=no", "--", prefix.as_posix() or "."):
        note = "repo had uncommitted changes; branch is based on the last commit (HEAD)"

    tmp = Path(tempfile.mkdtemp(prefix="fork-wt-"))
    wt = tmp / "wt"
    git(str(top), "worktree", "add", "-q", "-b", branch, str(wt), "HEAD")
    try:
        paths = []
        for rel in winner["changed"]:
            src = Path(winner["workdir"]) / rel
            dst = wt / prefix / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif dst.exists():
                dst.unlink()  # file deleted by the agent
            paths.append((prefix / rel).as_posix())
        git(str(wt), "add", "-A", "--", *paths)
        msg = (f"fork: make tests green ({winner['branch_id']})\n\n"
               f"Strategy: {winner['strategy']}\n"
               f"Files: {', '.join(winner['changed'])}\n"
               f"Diff: {winner['diff_lines']} lines | turns {winner['turns']} | "
               f"tool calls {winner['tool_calls']}\n"
               f"Tests: '{test_cmd}' passed in sandbox\nRun: {run_id}")
        git(str(wt), "-c", "user.name=Fork Agent", "-c", "user.email=fork-agent@localhost",
            "commit", "-q", "-m", msg)
        sha = git(str(wt), "rev-parse", "--short", "HEAD")
    finally:
        try:
            git(str(top), "worktree", "remove", "--force", str(wt))
        except RuntimeError:
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    out = {"branch": branch, "commit": sha}
    if note:
        out["note"] = note
    return out


def finalize_node(state: GraphState) -> dict:
    logs = BASE_DIR / "logs"
    logs.mkdir(exist_ok=True)
    w = state.get("winner")

    totals: dict[str, dict] = {}
    for u in state.get("usage", []):
        t = totals.setdefault(u["stage"], {"input": 0, "output": 0})
        t["input"] += u["input"]
        t["output"] += u["output"]

    applied = False
    branch = None
    if w:
        if state.get("make_branch", True):
            try:
                branch = commit_to_branch(state["repo_dir"], w, state["run_id"], state["test_cmd"])
            except Exception as e:
                branch = {"error": str(e)}
        (logs / f"graph_{state['run_id']}_winner.patch").write_text(w["patch"], encoding="utf-8")
        if state.get("apply"):
            for rel in w["changed"]:
                src = Path(w["workdir"]) / rel
                if src.exists():
                    dst = safe_path(state["repo_dir"], rel)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            applied = True

    summary = {
        "run_id": state["run_id"],
        "status": "already_green" if state.get("baseline_passed") else ("green" if w else "red"),
        "rounds": state.get("round", 0),
        "winner": w and {k: w[k] for k in ("branch_id", "strategy", "diff_lines", "changed",
                                           "turns", "tool_calls")},
        "git": branch,
        "applied": applied,
        "branches": [{k: r[k] for k in ("branch_id", "passed", "diff_lines", "turns", "tool_calls",
                                        "input_tokens", "output_tokens")}
                     for r in state.get("results", [])],
        "tokens_by_stage": totals,
    }
    (logs / f"graph_{state['run_id']}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Routing + graph
# ---------------------------------------------------------------------------

def after_baseline(state: GraphState) -> str:
    return "finalize" if state["baseline_passed"] else "planner"


def after_selector(state: GraphState) -> str:
    if state.get("winner"):
        return "finalize"
    return "reflector" if state["round"] < state["max_rounds"] else "finalize"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("baseline", baseline_node)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("selector", selector_node)
    g.add_node("reflector", reflector_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "baseline")
    g.add_conditional_edges("baseline", after_baseline, ["planner", "finalize"])
    g.add_conditional_edges("planner", fan_out, ["executor"])
    g.add_edge("executor", "selector")           # waits for all branches
    g.add_conditional_edges("selector", after_selector, ["finalize", "reflector"])
    g.add_edge("reflector", "planner")
    g.add_edge("finalize", END)
    return g.compile()


def run(repo: str, task: str, test_cmd: str = "pytest -q", branches: int = 3, rounds: int = 2,
        max_turns: int = 15, apply: bool = False, work_root: Optional[str] = None,
        make_branch: bool = True, on_log=None, full: bool = False) -> dict:
    """Returns the summary dict, or the full final graph state when full=True."""
    global _log_hook
    repo_dir = str(Path(repo).resolve())
    if not Path(repo_dir).is_dir():
        raise FileNotFoundError(repo_dir)
    state: GraphState = {
        "task": task, "repo_dir": repo_dir, "test_cmd": test_cmd,
        "work_root": work_root or str(BASE_DIR / "workspace" / "branches"),
        "run_id": time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4],
        "n_branches": branches, "max_rounds": rounds, "max_turns": max_turns, "apply": apply,
        "make_branch": make_branch,
        "results": [], "usage": [], "feedback": "",
    }
    _log_hook = on_log
    try:
        final = build_graph().invoke(state, {"recursion_limit": 10 + rounds * 6})
    finally:
        _log_hook = None
    return final if full else final["summary"]


def main():
    ap = argparse.ArgumentParser(description="Fork - LangGraph agent with parallel branching")
    ap.add_argument("--repo", required=True, help="Path to the repo with failing tests")
    ap.add_argument("--task", default="Make the failing tests pass without changing the tests.")
    ap.add_argument("--test-cmd", default="pytest -q")
    ap.add_argument("--branches", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--apply", action="store_true", help="Copy the winning fix back into --repo")
    ap.add_argument("--no-branch", action="store_true", help="Don't commit the fix to a git branch")
    a = ap.parse_args()

    s = run(a.repo, a.task, a.test_cmd, a.branches, a.rounds, a.max_turns, a.apply,
            make_branch=not a.no_branch)
    print("\n=== SUMMARY ===")
    g = s.get("git") or {}
    if g.get("branch"):
        print(f"Fix committed to branch {g['branch']} ({g['commit']}). Review with:")
        print(f"  git -C {a.repo} diff HEAD {g['branch']}")
        print(f"  git -C {a.repo} merge {g['branch']}")
    print(json.dumps({k: s[k] for k in ("status", "rounds", "winner", "git", "applied", "tokens_by_stage")}, indent=2))


if __name__ == "__main__":
    main()