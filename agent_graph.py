"""
agent_graph.py - LangGraph version of Fork with parallel branching.

Flow:
    baseline -> (already green? -> finalize)
             -> planner -> [executor x N in parallel] -> selector
             -> (winner? -> finalize)
             -> (rounds left? -> reflector -> planner) else finalize

Code runs in Nebius Token Factory Sandboxes: the repo becomes one sandbox checkpoint
and every branch forks from it. Each branch keeps its own copy of the files, so
branches never overwrite each other. SANDBOX=local exists only for tests.

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
import contextvars
import difflib
import fnmatch
import hashlib
import json
import operator
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".mypy_cache"}
MAX_FILE_BYTES = 1_000_000

# Where log lines go besides stdout. A ContextVar (not a global) so two runs at once, e.g.
# two visitors on the hosted demo, each get only their own lines. LangGraph copies the
# context into the threads that run parallel branches, so executor logs arrive too.
_log_sink: contextvars.ContextVar = contextvars.ContextVar("fork_log_sink", default=None)


def log(msg: str) -> None:
    print(msg, flush=True)
    sink = _log_sink.get()
    if sink:
        try:
            sink(msg)
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
        return resolve_model(os.getenv(f"{stage.upper()}_MODEL") or cfg["model"])
    return cfg["model"]


# Short names for the Nemotron 3 family on Nebius Token Factory (check yours with scripts/list_models.py)
MODEL_ALIASES = {
    "nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "super": "nvidia/nemotron-3-super-120b-a12b",
    "ultra": "nvidia/Nemotron-3-Ultra-550b-a55b",
}


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get((name or "").strip().lower(), (name or "").strip())


def parse_ladder(spec: Optional[str]) -> list[str]:
    """'nano,super,ultra' -> full model ids. Round r of the executor uses ladder[r-1]
    (the last entry repeats), so cheap models try first and big ones only when needed."""
    return [resolve_model(m) for m in (spec or "").split(",") if m.strip()]


def executor_model(state: dict, rnd: int) -> str:
    ladder = state.get("ladder") or []
    if ladder and provider_config()["provider"] == "nebius":
        return ladder[min(rnd, len(ladder)) - 1]
    return stage_model("executor")


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


EXECUTOR_MAX_TOKENS = int(os.getenv("EXECUTOR_MAX_TOKENS", "8192"))
STALL_LIMIT = int(os.getenv("EXECUTOR_STALL_LIMIT", "3"))   # edits in a row without more tests passing
IDLE_LIMIT = int(os.getenv("EXECUTOR_IDLE_LIMIT", "6"))     # turns in a row without any edit
_thinking_unsupported = False


def chat(model: str, messages: list, tools: Optional[list] = None,
         max_tokens: Optional[int] = None, thinking: Optional[str] = None):
    """Single LLM call. Returns (message, (input_tokens, output_tokens)).
    message has .content, .tool_calls and .finish_reason ("length" = cut off by max_tokens).
    thinking: None/"on" = model default, "off" = no reasoning, "low" = brief reasoning
    (Nemotron 3 chat_template_kwargs). Tests monkeypatch this function."""
    global _thinking_unsupported
    kwargs = {"model": model, "messages": messages, "temperature": 0.2}
    if tools:
        kwargs["tools"] = tools
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if thinking in ("off", "low") and not _thinking_unsupported:
        ctk = {"enable_thinking": thinking == "low"}
        if thinking == "low":
            ctk["low_effort"] = True
        kwargs["extra_body"] = {"chat_template_kwargs": ctk}
    try:
        resp = get_client().chat.completions.create(**kwargs)
    except Exception as e:
        # Only a "bad request" means the provider doesn't understand the switch.
        # Rate limits, timeouts and server errors must not turn thinking control off.
        if "extra_body" not in kwargs or getattr(e, "status_code", None) not in (400, 422):
            raise
        log(f"[llm] provider rejected the thinking switch ({type(e).__name__}); using model default")
        _thinking_unsupported = True
        kwargs.pop("extra_body")
        resp = get_client().chat.completions.create(**kwargs)
    u = resp.usage
    usage = (getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
    choice = resp.choices[0]
    m = choice.message
    return SimpleNamespace(content=m.content, tool_calls=m.tool_calls,
                           finish_reason=choice.finish_reason), usage


# ---------------------------------------------------------------------------
# Cost tracking (USD). Prices are per 1M tokens (input, output) on Nebius Token Factory.
# Matched by substring of the model id; override with PRICES_JSON='{"nano": [0.06, 0.24]}'.
# ---------------------------------------------------------------------------

DEFAULT_PRICES = {"nano": (0.06, 0.24), "super": (0.30, 0.90), "ultra": (1.00, 3.00)}


def prices() -> dict:
    table = dict(DEFAULT_PRICES)
    if os.getenv("PRICES_JSON"):
        table.update({k.lower(): tuple(v) for k, v in json.loads(os.environ["PRICES_JSON"]).items()})
    return table


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Dollar cost of one call. Unknown models cost 0 (e.g. local Ollama)."""
    m = (model or "").lower()
    for key, (p_in, p_out) in prices().items():
        if key in m:
            return (tokens_in * p_in + tokens_out * p_out) / 1_000_000
    return 0.0


class Budget:
    """Running spend for one agent run, shared by all parallel branches."""

    def __init__(self, max_usd: Optional[float] = None):
        self.max_usd, self.spent, self._lock = max_usd, 0.0, threading.Lock()

    def add(self, model: str, tokens_in: int, tokens_out: int) -> float:
        c = cost_usd(model, tokens_in, tokens_out)
        with self._lock:
            self.spent += c
        return c

    def exceeded(self) -> bool:
        return self.max_usd is not None and self.spent >= self.max_usd


_budgets: dict[str, Budget] = {}


def budget(run_id: str) -> Budget:
    with _client_lock:
        return _budgets.setdefault(run_id, Budget())


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

_local = threading.local()
_base_cache: dict[str, object] = {}     # repo content hash -> Nebius checkpoint image
_base_locks: dict[str, threading.Lock] = {}
_cache_lock = threading.Lock()
ORIGIN: dict[str, str] = {}             # branch dir -> repo dir it was copied from
APP = "/app"                            # where the repo lives inside the sandbox
SKIP_UPLOAD = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".mypy_cache"}


def _contree_client():
    """One Contree (Nebius Sandboxes) client per thread (branches run in parallel threads)."""
    if not hasattr(_local, "client"):
        from contree_sdk import ContreeSync
        _local.client = ContreeSync()
    return _local.client


def _repo_files(root: str) -> dict[str, Path]:
    root_p = Path(root)
    return {p.relative_to(root_p).as_posix(): p for p in root_p.rglob("*")
            if p.is_file() and not SKIP_UPLOAD & set(p.relative_to(root_p).parts)}


def _repo_hash(files: dict[str, Path]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode())
        h.update(files[rel].read_bytes())
    return h.hexdigest()


SANDBOX_RETRIES = int(os.getenv("SANDBOX_RETRIES", "3"))
_sleep = time.sleep  # tests replace this


def _is_transient(e: Exception) -> bool:
    """Network hiccups talking to the Nebius Sandboxes API (connect/read timeouts, dropped
    connections, 502/503/504). Worth retrying; anything else is a real error."""
    name, text = type(e).__name__, str(e).lower()
    return (any(k in name for k in ("Timeout", "Connect", "Connection", "RemoteProtocol"))
            or any(k in text for k in ("timeout", "timed out", "connection reset", " 502", " 503", " 504")))


def with_retry(fn, what: str):
    """Call fn(); on a transient sandbox error wait 2s, 4s, ... and try again."""
    for attempt in range(1, SANDBOX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if not _is_transient(e) or attempt == SANDBOX_RETRIES:
                raise
            wait = 2 ** attempt
            log(f"[sandbox] {what}: {type(e).__name__}, retrying in {wait}s "
                f"(attempt {attempt + 1}/{SANDBOX_RETRIES})")
            _sleep(wait)


class SandboxUnavailable(RuntimeError):
    pass


def _base_checkpoint(repo_dir: str):
    """Nebius checkpoint = base image + pytest + the repo (+ its requirements.txt).
    Built once per repo state; every branch forks from it (native sandbox branching)."""
    files = _repo_files(repo_dir)
    key = _repo_hash(files)
    with _cache_lock:
        if key in _base_cache:
            return _base_cache[key]
        lock = _base_locks.setdefault(key, threading.Lock())
    with lock:  # branches asking at the same time wait for one build
        if key in _base_cache:
            return _base_cache[key]
        def build():
            img = _contree_client().images.use(os.getenv("SANDBOX_IMAGE", "python:3.12-slim"))
            img = img.run(shell="pip install -q pytest", disposable=False).wait()
            img = img.apply_files({f"{APP[1:]}/{rel}": str(p) for rel, p in files.items()})
            if "requirements.txt" in files:
                img = img.run(shell=f"cd {APP} && pip install -q -r requirements.txt",
                              disposable=False).wait()
            return img
        try:
            img = with_retry(build, "building repo checkpoint")
        except Exception as e:
            if _is_transient(e):
                raise SandboxUnavailable(
                    f"Can't reach Nebius Sandboxes ({type(e).__name__}) after {SANDBOX_RETRIES} tries. "
                    "Check your internet connection, then run: python scripts/hello_sandbox.py") from e
            raise
        log(f"[sandbox] Nebius checkpoint ready for {Path(repo_dir).name} ({img.uuid})")
        with _cache_lock:
            _base_cache[key] = img
        return img


def _changed_files(origin: str, workdir: str) -> dict[str, str]:
    """Files in the branch dir that differ from (or are new vs) the original repo."""
    if Path(origin).resolve() == Path(workdir).resolve():
        return {}
    a, b = _repo_files(origin), _repo_files(workdir)
    return {f"{APP[1:]}/{rel}": str(p) for rel, p in b.items()
            if rel not in a or a[rel].read_bytes() != p.read_bytes()}


def _deleted_files(origin: str, workdir: str) -> list[str]:
    """Files in the original repo that the branch deleted (the checkpoint still has them)."""
    if Path(origin).resolve() == Path(workdir).resolve():
        return []
    return sorted(set(_repo_files(origin)) - set(_repo_files(workdir)))


def run_in_sandbox(workdir: str, command: str, timeout: int = 120) -> tuple[int, str]:
    """Run a command against a repo or branch directory.
    SANDBOX=nebius (default): Nebius Token Factory Sandboxes. The branch's edited files are
        overlaid on the shared repo checkpoint; each run is disposable.
    SANDBOX=local: runs on this machine. Only for tests and trusted demo repos."""
    if os.getenv("SANDBOX", "nebius").lower() == "local":
        try:
            # No .pyc files: two quick edits of the same size within one second would otherwise
            # run stale bytecode and report the old result.
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            p = subprocess.run(command, shell=True, cwd=workdir, capture_output=True, env=env,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {timeout}s"

    workdir = str(Path(workdir).resolve())
    origin = ORIGIN.get(workdir, workdir)
    base = _base_checkpoint(origin)
    deleted = _deleted_files(origin, workdir)
    rm = f"rm -f -- {' '.join(shlex.quote(d) for d in deleted)} && " if deleted else ""
    shell = f"cd {APP} && {rm}PYTHONDONTWRITEBYTECODE=1 timeout {timeout} sh -c {shlex.quote(command)}"
    overlay = _changed_files(origin, workdir) or None
    r = with_retry(lambda: base.run(shell=shell, files=overlay, timeout=timeout + 60).wait(),
                   "running command")
    out = (r.stdout or "") + (r.stderr or "")
    if r.exit_code == 124:
        out += f"\nCommand timed out after {timeout}s"
    return r.exit_code, out


def sandbox_ready() -> Optional[str]:
    """Returns an error message if the configured sandbox can't be used, else None."""
    if os.getenv("SANDBOX", "nebius").lower() == "local":
        return None
    missing = [k for k in ("NEBIUS_API_KEY", "NEBIUS_PROJECT_ID") if not os.getenv(k)]
    if missing:
        return f"Missing {', '.join(missing)} in .env (needed for Nebius Sandboxes)."
    try:
        import contree_sdk  # noqa: F401
    except ImportError:
        return "contree-sdk isn't installed. Run: pip install contree-sdk"
    return None


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def tail(text: str, n: int = 4000) -> str:
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def safe_path(root: str, rel: str) -> Path:
    """Resolve rel inside root; refuse anything that escapes the branch folder.
    Models see /app in sandbox output and often pass /app/pkg/x.py, so that prefix is
    mapped to the repo root instead of failing turn after turn."""
    rel = (rel or "").replace("\\", "/").strip()
    if rel == APP or rel.startswith(APP + "/"):
        rel = rel[len(APP) + 1:] or "."
    root_p = Path(root).resolve()
    p = (root_p / rel).resolve()
    if p != root_p and root_p not in p.parents:
        raise ValueError(f"Path escapes workspace: {rel}. Use a path relative to the repo root, "
                         f"e.g. {next(iter(list_files(root)), 'pkg/module.py')}")
    return p


def copy_repo(src: str, dst: str, origin: Optional[str] = None) -> None:
    """Copy src to dst. origin is the untouched repo the copy descends from (defaults to
    src): the sandbox checkpoint, tamper guard and diffs are all relative to it, so a
    branch seeded from an earlier partial fix still shares the one repo checkpoint."""
    if Path(dst).exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*IGNORE_DIRS))
    ORIGIN[str(Path(dst).resolve())] = str(Path(origin or src).resolve())


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
                # bytes -> str keeps CRLF as-is, so patches still apply to Windows-style files
                out[p.relative_to(root_p).as_posix()] = p.read_bytes().decode("utf-8")
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
        # A last line without "\n" would glue onto the next file's header and break
        # `git apply`. Mark it the way git does.
        patch.extend(l if l.endswith("\n") else l + "\n\\ No newline at end of file\n" for l in d)
        lines += sum(1 for l in d if l[:1] in "+-" and not l.startswith(("+++", "---")))
    return {"patch": "".join(patch), "changed": changed, "lines": lines}


def list_files(root: str) -> list[str]:
    return sorted(collect_files(root).keys())


# ---------------------------------------------------------------------------
# Tamper guard: the tests are the judge, so the agent may not touch them
# ---------------------------------------------------------------------------

PROTECTED_NAMES = ("test_*.py", "*_test.py", "conftest.py", "pytest.ini", "tox.ini",
                   "setup.cfg", "pyproject.toml", ".coveragerc")
PROTECTED_DIRS = {"tests", "test", "testing"}


def is_protected(rel: str) -> bool:
    """Test files and test-runner config. Editing these could make tests pass without a fix."""
    parts = Path(rel).parts
    return (any(fnmatch.fnmatch(parts[-1], pat) for pat in PROTECTED_NAMES)
            or any(p in PROTECTED_DIRS for p in parts[:-1]))


def restore_protected(origin: str, workdir: str) -> list[str]:
    """Put every protected file in workdir back to its original state.
    Undoes edits, re-creates deletions and removes new protected files (e.g. a
    conftest.py that skips everything). Returns the files that had been tampered with."""
    if Path(origin).resolve() == Path(workdir).resolve():
        return []
    a, b = _repo_files(origin), _repo_files(workdir)
    tampered = []
    for rel in sorted(set(a) | set(b)):
        if not is_protected(rel):
            continue
        src, dst = a.get(rel), Path(workdir) / rel
        if src is None:                                   # agent added it
            dst.unlink()
        elif rel not in b:                                # agent deleted it
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif src.read_bytes() != b[rel].read_bytes():     # agent edited it
            shutil.copy2(src, dst)
        else:
            continue
        tampered.append(rel)
    return tampered


_COUNT_RE = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")


def pytest_counts(output: str) -> dict:
    """Counts from pytest's summary line, e.g. '1 failed, 3 passed in 0.1s'. {} if not pytest."""
    lines = [l for l in output.splitlines() if _COUNT_RE.search(l)]
    if not lines:
        return {}
    counts: dict[str, int] = {}
    for n, kind in _COUNT_RE.findall(lines[-1]):
        kind = "error" if kind.startswith("error") else kind
        counts[kind] = counts.get(kind, 0) + int(n)
    return counts


def judge(workdir: str, test_cmd: str, baseline_total: int = 0) -> tuple[int, str, list[str]]:
    """The only place a branch is declared green. Restores the original tests first,
    runs the real test command, and (for pytest) checks that at least as many tests
    pass as existed at baseline, so skipping or deselecting tests doesn't count.
    Returns (exit code, output, tampered files)."""
    tampered = restore_protected(ORIGIN.get(str(Path(workdir).resolve()), workdir), workdir)
    code, out = run_in_sandbox(workdir, test_cmd)
    if code == 0 and baseline_total:
        passed = pytest_counts(out).get("passed", 0)
        if passed < baseline_total:
            code = 1
            out += (f"\n[judge] only {passed} tests passed but the suite has {baseline_total}; "
                    "skipped or deselected tests don't count.")
    return code, out, tampered


def repo_context(root: str, failing_output: str = "", budget: int = 30000) -> str:
    """The repo's key files inline, so agents don't spend turns reading them.
    Files named in the failing output come first, then small .py files, within a char budget."""
    files = collect_files(root)
    def rank(rel):
        mentioned = Path(rel).name in failing_output
        return (0 if mentioned else 1, 0 if rel.endswith(".py") else 1, len(files[rel]))
    parts, used, skipped = [], 0, []
    for rel in sorted(files, key=rank):
        block = f"### {rel}\n```\n{files[rel].replace(chr(13) + chr(10), chr(10))}\n```\n"
        if used + len(block) > budget:
            skipped.append(rel)
            continue
        parts.append(block)
        used += len(block)
    if skipped:
        parts.append("Other files (use read_file if needed): " + ", ".join(sorted(skipped)))
    return "\n".join(parts)


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
        "name": "edit_file",
        "description": ("Replace one exact snippet in a file. 'old' must appear exactly once, so "
                        "include a few surrounding lines. Best for small fixes."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create a file or overwrite it with full new content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the sandbox (no internet).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]


def write_preserving_newlines(p: Path, content: str) -> None:
    """Write exactly what the model sent, keeping the file's existing line endings.
    Path.write_text would turn every \n into \r\n on Windows, which makes an LF repo
    show every line as changed (huge diffs, wrong 'smallest diff' winner)."""
    content = content.replace("\r\n", "\n")
    if p.exists() and b"\r\n" in p.read_bytes():
        content = content.replace("\n", "\r\n")
    p.write_bytes(content.encode("utf-8"))


TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
# With reasoning switched off, Nemotron sometimes emits its thinking as a tool call
THINK_NAMES = {"think", "analysis", "analyze", "reasoning", "thought", "plan"}
# Names models reach for from other agent frameworks, mapped onto ours
TOOL_ALIASES = {"str_replace_editor": "edit_file", "str_replace": "edit_file",
                "replace_in_file": "edit_file", "view": "read_file", "cat": "read_file",
                "create_file": "write_file", "bash": "run_command", "shell": "run_command"}


def _normalize_call(name: str, args: dict) -> tuple[str, dict]:
    name = TOOL_ALIASES.get(name, name)
    if name == "edit_file" and args.get("command") in ("view", "read"):
        name = "read_file"
    elif name == "edit_file" and args.get("command") == "create":
        name, args = "write_file", {"path": args.get("path"), "content": args.get("file_text", "")}
    if name == "edit_file":
        args = {"path": args.get("path"), "old": args.get("old", args.get("old_str")),
                "new": args.get("new", args.get("new_str", ""))}
    if name == "run_command" and "cmd" in args:
        args = {"command": args["cmd"]}
    return name, args


def _refuse_protected(workdir: str, p: Path, shown: str, protect: bool) -> Optional[str]:
    if protect and is_protected(p.relative_to(Path(workdir).resolve()).as_posix()):
        return (f"Refused: {shown} is a test or test-config file and is read-only. "
                "Fix the source code so the existing tests pass.")
    return None


def run_tool(workdir: str, name: str, args: dict, protect: bool = True) -> str:
    """Results starting with 'Wrote'/'Edited' mean the code changed (tests auto-run);
    'No change' means the edit was identical to what's there (counts as no progress)."""
    name, args = _normalize_call(name, args)
    if name in THINK_NAMES:
        return f"Noted. Now act: call one of {', '.join(TOOL_NAMES)}."
    try:
        if name == "list_files":
            return "\n".join(list_files(workdir)) or "(empty)"
        if name == "read_file":
            return tail(safe_path(workdir, args["path"]).read_text(encoding="utf-8"), 20000)
        if name == "write_file":
            p = safe_path(workdir, args["path"])
            refused = _refuse_protected(workdir, p, args["path"], protect)
            if refused:
                return refused
            if p.exists() and p.read_bytes().decode("utf-8", "replace").replace("\r\n", "\n") \
                    == args["content"].replace("\r\n", "\n"):
                return (f"No change: {args['path']} already has exactly this content. "
                        "Re-read the failing output and try a different fix.")
            p.parent.mkdir(parents=True, exist_ok=True)
            write_preserving_newlines(p, args["content"])
            return f"Wrote {args['path']} ({len(args['content'])} chars)"
        if name == "edit_file":
            p = safe_path(workdir, args["path"])
            refused = _refuse_protected(workdir, p, args["path"], protect)
            if refused:
                return refused
            old, new = (args.get("old") or "").replace("\r\n", "\n"), (args.get("new") or "").replace("\r\n", "\n")
            if not old:
                return "Error: 'old' is empty. Give the exact text to replace, or use write_file."
            text = p.read_bytes().decode("utf-8").replace("\r\n", "\n")
            n = text.count(old)
            if n == 0:
                return (f"Error: 'old' text not found in {args['path']}. Whitespace must match exactly; "
                        "read_file to see the current content.")
            if n > 1:
                return f"Error: 'old' text appears {n} times in {args['path']}. Include more surrounding lines."
            if old == new:
                return f"No change: 'old' and 'new' are identical. Try a different fix."
            write_preserving_newlines(p, text.replace(old, new, 1))
            return f"Edited {args['path']} (replaced {old.count(chr(10)) + 1} line(s))"
        if name == "run_command":
            code, out = run_in_sandbox(workdir, args["command"])
            return f"exit code {code}\n{tail(out)}"
        return f"Unknown tool: {name}. Available tools: {', '.join(TOOL_NAMES)}"
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
    first_green: bool
    thinking: Optional[str]
    max_usd: Optional[float]

    baseline_passed: bool
    baseline_output: str
    baseline_total: int        # tests in the suite at baseline (pytest only, else 0)
    ladder: list[str]          # executor model per round (escalation); empty = EXECUTOR_MODEL
    start_dir: Optional[str]   # where new branches start: the repo, or the best partial fix so far
    start_output: str          # test output at start_dir
    start_passed: int          # tests passing at start_dir
    progress: str              # what earlier rounds achieved, for the planner
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

def baseline_node(state: GraphState) -> dict:
    problem = sandbox_ready()
    if problem:
        raise RuntimeError(problem)
    code, out = run_in_sandbox(state["repo_dir"], state["test_cmd"])
    if code == 5:
        raise RuntimeError("pytest collected no tests. Check the repo path and test file names.")
    counts = pytest_counts(out)
    total = sum(counts.get(k, 0) for k in ("passed", "failed", "error"))
    log(f"[baseline] tests {'GREEN' if code == 0 else 'RED'} (exit {code})"
        + (f" | {counts}" if counts else ""))
    return {"baseline_passed": code == 0, "baseline_output": tail(out),
            "baseline_total": total, "round": 0,
            "start_dir": None, "start_output": tail(out), "start_passed": counts.get("passed", 0),
            "progress": ""}


PLANNER_PROMPT = """You are the planner for an autonomous coding agent that fixes failing Python test suites.
Propose exactly {n} DIFFERENT strategies to make the tests pass. Each strategy should be a short,
concrete instruction (1-3 sentences) naming the files/functions to look at and the fix idea.
Make them genuinely different (different root-cause hypotheses or approaches), not rewordings.
Base every strategy on the actual code shown; don't guess at bugs you can't see in it.
Prefer the smallest change that fixes the root cause: edit the buggy lines. Don't add new
functions, classes, modules or refactors unless a test needs them; make strategies differ in
diagnosis, not in how much code they rewrite. Name files by their path from the repo root.
If the failures come from several independent bugs in different files, split them: give each
strategy a different group of failing tests and the files behind them (partial fixes in separate
files are merged automatically), and let one strategy try to fix everything.
Never suggest editing or deleting tests to make them pass.
Reply ONLY with JSON: {{"strategies": ["...", "..."]}}"""


def planner_node(state: GraphState) -> dict:
    n = state["n_branches"]
    rnd = state.get("round", 0) + 1
    start = state.get("start_dir") or state["repo_dir"]
    failing = state.get("start_output") or state["baseline_output"]
    user = (f"Task: {state['task']}\n\nRepo files:\n"
            f"{repo_context(start, failing)}\n\n"
            f"Failing test output:\n{failing}")
    if state.get("progress"):
        user += f"\n\nProgress so far (already applied to the code above, keep it):\n{state['progress']}"
    if state.get("feedback"):
        user += f"\n\nFeedback from the previous round (these attempts failed):\n{state['feedback']}"

    model = stage_model("planner")
    msg, (i, o) = chat(model,
                       [{"role": "system", "content": PLANNER_PROMPT.format(n=n)},
                        {"role": "user", "content": user}],
                       thinking=os.getenv("PLANNER_THINKING") or None)
    usd = budget(state["run_id"]).add(model, i, o)
    data = parse_json(msg.content) or {}
    strategies = [s for s in data.get("strategies", []) if isinstance(s, str) and s.strip()][:n]
    while len(strategies) < n:  # fallback if the model returns fewer / bad JSON
        strategies.append("Read the failing test and the code it calls, find the root cause, fix it minimally.")

    log(f"[planner] round {rnd}: {len(strategies)} strategies")
    for k, s in enumerate(strategies):
        log(f"   b{k}: {s[:100]}")
    return {"strategies": strategies, "round": rnd,
            "usage": [{"stage": "planner", "round": rnd, "model": model,
                       "input": i, "output": o, "usd": usd}]}


def fan_out(state: GraphState) -> list[Send]:
    """One parallel executor branch per strategy, all on this round's model."""
    model = executor_model(state, state["round"])
    return [
        Send("executor", {
            "branch_id": f"r{state['round']}_b{k}",
            "round": state["round"],
            "model": model,
            "start_dir": state.get("start_dir"),
            "start_output": state.get("start_output") or state["baseline_output"],
            "progress": state.get("progress", ""),
            "start_passed": state.get("start_passed", 0),
            "strategy": s,
            "task": state["task"],
            "repo_dir": state["repo_dir"],
            "test_cmd": state["test_cmd"],
            "baseline_output": state["baseline_output"],
            "baseline_total": state.get("baseline_total", 0),
            "work_root": state["work_root"],
            "run_id": state["run_id"],
            "max_turns": state["max_turns"],
            "first_green": state.get("first_green", True),
            "thinking": state.get("thinking"),
        })
        for k, s in enumerate(state["strategies"])
    ]


EXECUTOR_PROMPT = """You are the executor of an autonomous coding agent. You work inside a copy of a Python repo.
Tools: list_files, read_file, edit_file, write_file, run_command. The repo's key files are included below.
Goal: make the test command pass by fixing the source code. Keep changes minimal: fix the buggy
lines, don't refactor. Follow the strategy you are given.

RULES:
- Paths are relative to the repo root, e.g. pkg/module.py (not /app/pkg/module.py).
- Test files and test config (test_*.py, conftest.py, pytest.ini, pyproject.toml, tests/) are
  read-only. Writes to them are refused, and they are restored before the final check.
- Make EVERY edit with edit_file (replace one exact snippet; best for small fixes) or
  write_file (complete new file content).
- NEVER edit files with shell commands (sed, echo >, cat <<EOF, python -c, pip install).
  Each command runs in a fresh sandbox, so those changes are thrown away.
- After each edit, the tests run automatically and you'll see the result.
  You don't need to run them yourself.
- If an edit doesn't increase the number of passing tests, don't repeat it: re-read the
  failing output and change your approach. Branches that stop making progress are ended.
When the tests pass (or you cannot make progress), reply with a short summary and no tool calls."""


_cancel_events: dict[str, threading.Event] = {}


def _cancel_event(key: str) -> threading.Event:
    with _cache_lock:
        return _cancel_events.setdefault(key, threading.Event())


def executor_node(payload: dict) -> dict:
    bid = payload["branch_id"]
    workdir = str(Path(payload["work_root"]) / payload["run_id"] / bid)
    start = payload.get("start_dir") or payload["repo_dir"]
    failing = payload.get("start_output") or payload["baseline_output"]
    copy_repo(start, workdir, origin=payload["repo_dir"])
    test_cmd = payload["test_cmd"].strip()
    baseline_total = payload.get("baseline_total", 0)
    first_green = payload.get("first_green", True)
    cancel = _cancel_event(f"{payload['run_id']}:{payload['round']}")
    spend = budget(payload["run_id"])

    messages = [
        {"role": "system", "content": EXECUTOR_PROMPT},
        {"role": "user", "content": (
            f"Task: {payload['task']}\nTest command: {test_cmd}\n"
            f"Strategy: {payload['strategy']}\n\n"
            f"Repo files:\n{repo_context(start, failing)}\n\n"
            f"Failing output:\n{failing}")},
    ]
    if payload.get("progress"):
        messages[1]["content"] += (f"\n\nThis copy already contains a partial fix from an earlier round "
                                   f"({payload['progress']}). Build on it; don't undo it.")
    model = payload.get("model") or stage_model("executor")
    tin = tout = turns = tool_calls = refused = 0
    usd = 0.0
    # Stuck detection: real runs showed Nano rewriting the same file 10 times with the same
    # result, or failing on paths for 12 turns. Both burn tokens, so end the branch early.
    best_passed = payload.get("start_passed", 0)
    stall = idle = 0                 # edits without progress / turns without any edit
    nudged = False
    tampered: set[str] = set()
    verified = None          # (code, output) once the judge has seen the tests pass
    stop = "gave up"
    t0 = time.time()

    for turns in range(1, payload["max_turns"] + 1):
        if first_green and cancel.is_set():
            stop = "cancelled"
            break
        if spend.exceeded():
            stop = "budget"
            break
        try:
            msg, (i, o) = chat(model, messages, TOOLS, max_tokens=EXECUTOR_MAX_TOKENS,
                               thinking=payload.get("thinking"))
        except Exception as e:
            log(f"[{bid}] LLM error: {e}")
            break
        tin, tout = tin + i, tout + o
        usd += spend.add(model, i, o)
        calls = getattr(msg, "tool_calls", None) or []

        if getattr(msg, "finish_reason", None) == "length":
            log(f"[{bid}] turn {turns}: reply cut off at max_tokens, asking for smaller edits")
            messages.append({"role": "user", "content": (
                "Your last reply was cut off because it was too long, so nothing was applied. "
                "Write one file per call and keep your reasoning short.")})
            continue
        if not calls:
            stop = "model finished"
            break

        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in calls],
        })
        wrote = noop = False
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
            if result.startswith("Refused:"):
                refused += 1
            if result.startswith(("Wrote", "Edited")):
                wrote = True
            elif result.startswith("No change"):
                noop = True
            # The model's own test runs are information only: they never mark a branch green
            # (e.g. "pytest -q || true" exits 0). Only judge() can do that.

        if wrote:  # auto-run the judge after edits: saves a turn per check
            try:
                code, out, t = judge(workdir, test_cmd, baseline_total)
                tampered.update(t)
            except Exception as e:
                code, out = 1, f"Sandbox error: {type(e).__name__}: {e}"
            n_pass = pytest_counts(out).get("passed", 0)
            if code == 0:
                verified = (0, out)
            else:
                messages.append({"role": "user", "content":
                                 f"[auto test run after your edit] exit code {code}\n{tail(out, 1500)}"})
            log(f"[{bid}] turn {turns}: auto test -> {'GREEN' if code == 0 else f'exit {code}'}"
                + (f" ({n_pass} passing)" if not verified else ""))
        if verified:
            stop = "tests passed"
            if first_green:
                cancel.set()
            break

        if wrote and n_pass > best_passed:
            best_passed, stall, idle, nudged = n_pass, 0, 0, False
        elif wrote or noop:
            stall, idle = stall + 1, 0
        else:
            idle += 1
        if stall >= STALL_LIMIT or idle >= IDLE_LIMIT:
            stop = "stuck"
            log(f"[{bid}] turn {turns}: no progress ({stall} edits without gain, {idle} turns without "
                "edits), ending branch")
            break
        if (stall >= STALL_LIMIT - 1 or idle >= IDLE_LIMIT - 2) and not nudged:
            nudged = True
            messages.append({"role": "user", "content": (
                "[no progress] Your recent turns haven't increased the number of passing tests. "
                "Stop repeating the same change. Re-read the failing assertion, find which function "
                "produces the wrong value, and fix that. This branch ends soon without progress.")})

    cancelled = stop == "cancelled"
    if verified:
        code, out = verified
    elif cancelled:
        code, out = 1, "Stopped: another branch turned the tests green first."
    else:
        try:
            code, out, t = judge(workdir, test_cmd, baseline_total)  # final check is the judge
            tampered.update(t)
        except Exception as e:
            code, out = 1, f"Sandbox error: {type(e).__name__}: {e}"
    passed = code == 0
    if passed and first_green:
        cancel.set()
    tampered.update(restore_protected(payload["repo_dir"], workdir))  # keep the patch clean
    diff = make_diff(payload["repo_dir"], workdir)
    status = "GREEN" if passed else ("stopped" if cancelled else "red")
    guard = f" | blocked {refused} test edits" if refused else ""
    guard += f" | reverted {sorted(tampered)}" if tampered else ""
    log(f"[{bid}] {status} ({stop}) | {(model or '?').split('/')[-1]} | turns {turns} | tools {tool_calls} | "
        f"diff {diff['lines']} lines | tokens {tin}+{tout} | ${usd:.4f} | {time.time() - t0:.0f}s{guard}")

    return {
        "results": [{
            "branch_id": bid, "round": payload["round"], "strategy": payload["strategy"],
            "model": model, "passed": passed, "cancelled": cancelled, "stop_reason": stop,
            "test_output": tail(out, 2000), "workdir": workdir,
            "diff_lines": diff["lines"], "changed": diff["changed"], "patch": diff["patch"],
            "turns": turns, "tool_calls": tool_calls, "input_tokens": tin, "output_tokens": tout,
            "usd": usd, "refused_writes": refused, "tampered": sorted(tampered),
            "messages": messages,
        }],
        "usage": [{"stage": "executor", "round": payload["round"], "branch": bid, "model": model,
                   "input": tin, "output": tout, "usd": usd}],
    }


def pick_winner(results: list[dict], rnd: int) -> Optional[dict]:
    """Green branches from this round; smallest diff wins, then fewest tokens."""
    green = [r for r in results if r["round"] == rnd and r["passed"] and r["diff_lines"] > 0]
    if not green:
        return None
    return min(green, key=lambda r: (r["diff_lines"], r["input_tokens"] + r["output_tokens"]))


def passed_count(r: dict) -> int:
    return pytest_counts(r.get("test_output", "")).get("passed", 0)


def _changed_rel(a_dir: str, b_dir: str) -> set[str]:
    """Files that differ between two copies of the repo (edited, added or deleted)."""
    a, b = _repo_files(a_dir), _repo_files(b_dir)
    return {rel for rel in set(a) | set(b)
            if rel not in a or rel not in b or a[rel].read_bytes() != b[rel].read_bytes()}


def merge_partials(state: GraphState) -> Optional[dict]:
    """No branch is green. Branches that each fixed *some* tests often fixed different bugs
    in different files, so combine them: start from the best one, add each other branch's
    files if they don't overlap, and keep the addition only if more tests pass.
    No model calls, just a few judged test runs. Returns a result dict or None."""
    rnd, repo = state["round"], state["repo_dir"]
    start = state.get("start_dir") or repo
    floor = state.get("start_passed", 0)
    cands = sorted((r for r in state["results"]
                    if r["round"] == rnd and not r["passed"] and not r.get("cancelled")
                    and passed_count(r) > floor),
                   key=passed_count, reverse=True)
    if len(cands) < 2:
        return None

    mdir = str(Path(state["work_root"]) / state["run_id"] / f"r{rnd}_merge")
    copy_repo(cands[0]["workdir"], mdir, origin=repo)
    used, taken = [cands[0]["branch_id"]], _changed_rel(start, cands[0]["workdir"])
    best, out = passed_count(cands[0]), cands[0]["test_output"]
    code = 1
    for r in cands[1:]:
        files = {f for f in _changed_rel(start, r["workdir"]) if not is_protected(f)}
        if not files or files & taken:
            continue                                      # nothing new, or conflicts with what we have
        backup = {f: (Path(mdir) / f).read_bytes() if (Path(mdir) / f).exists() else None for f in files}
        for f in files:
            src, dst = Path(r["workdir"]) / f, Path(mdir) / f
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif dst.exists():
                dst.unlink()
        c, o, _ = judge(mdir, state["test_cmd"], state.get("baseline_total", 0))
        n = pytest_counts(o).get("passed", 0)
        if c == 0 or n > best:
            used.append(r["branch_id"]); taken |= files
            best, out, code = n, o, c
            if c == 0:
                break
        else:                                             # made things worse: undo
            for f, data in backup.items():
                dst = Path(mdir) / f
                if data is None:
                    dst.unlink(missing_ok=True)
                else:
                    dst.write_bytes(data)
    if len(used) < 2:
        return None
    diff = make_diff(repo, mdir)
    log(f"[merge] round {rnd}: combined {' + '.join(used)} -> "
        f"{'GREEN' if code == 0 else f'{best} tests passing'}")
    return {"branch_id": f"r{rnd}_merge", "round": rnd, "strategy": f"Merge of {', '.join(used)}",
            "model": "merge (no LLM)", "passed": code == 0, "cancelled": False, "stop_reason": "merged",
            "test_output": tail(out, 2000), "workdir": mdir, "diff_lines": diff["lines"],
            "changed": diff["changed"], "patch": diff["patch"], "turns": 0, "tool_calls": 0,
            "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "refused_writes": 0, "tampered": [],
            "messages": None}


def selector_node(state: GraphState) -> dict:
    w = pick_winner(state["results"], state["round"])
    update: dict = {}
    if not w:
        merged = merge_partials(state)
        if merged:
            update["results"] = [merged]
            if merged["passed"] and merged["diff_lines"] > 0:
                w = merged
        # Carry the best partial fix into the next round instead of starting over.
        pool = [r for r in state["results"] if r["round"] == state["round"]
                and not r["passed"] and not r.get("cancelled")] + ([merged] if merged else [])
        best = max(pool, key=passed_count, default=None)
        if not w and best and passed_count(best) > state.get("start_passed", 0):
            n0 = state.get("baseline_total", 0)
            update.update({
                "start_dir": best["workdir"], "start_output": best["test_output"],
                "start_passed": passed_count(best),
                "progress": (f"{best['branch_id']} ({best['strategy'][:120]}) got "
                             f"{passed_count(best)}{f'/{n0}' if n0 else ''} tests passing, "
                             f"changing {', '.join(best['changed'])}"),
            })
            log(f"[selector] next round starts from {best['branch_id']} "
                f"({passed_count(best)} tests passing)")
    log(f"[selector] round {state['round']}: " + (f"winner {w['branch_id']}" if w else "no green branch"))
    update["winner"] = w
    return update


REFLECTOR_PROMPT = """You are the reflector of an autonomous coding agent. Several parallel fix attempts failed.
Look at each strategy and its final test output. In under 150 words, explain what went wrong and
what the next round should try differently. Be specific (files, functions, error messages)."""


def reflector_node(state: GraphState) -> dict:
    attempts = [r for r in state["results"] if r["round"] == state["round"]]
    text = "\n\n".join(
        f"## {r['branch_id']}\nStrategy: {r['strategy']}\nChanged: {r['changed']}\n"
        f"Final test output:\n{tail(r['test_output'], 1500)}" for r in attempts)
    model = stage_model("reflector")
    msg, (i, o) = chat(model,
                       [{"role": "system", "content": REFLECTOR_PROMPT},
                        {"role": "user", "content": f"Task: {state['task']}\n\n{text}"}],
                       thinking=os.getenv("REFLECTOR_THINKING") or None)
    usd = budget(state["run_id"]).add(model, i, o)
    log(f"[reflector] {(msg.content or '').strip()[:200]}")
    return {"feedback": msg.content or "",
            "usage": [{"stage": "reflector", "round": state["round"], "model": model,
                       "input": i, "output": o, "usd": usd}]}


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


def save_trajectories(state: GraphState) -> tuple[int, int]:
    """Save branch conversations as fine-tuning data.
    Green branches -> logs/trajectories/, red ones -> logs/trajectories/failed/.
    Same task, green vs red, gives preference pairs later. Returns (green, red) counts."""
    out_dir = BASE_DIR / "logs" / "trajectories"
    w = state.get("winner") or {}
    green = red = 0
    for r in state.get("results", []):
        if r.get("cancelled") or not r.get("messages"):
            continue
        d = out_dir if r.get("passed") else out_dir / "failed"
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": state["run_id"], "branch_id": r["branch_id"], "passed": bool(r.get("passed")),
            "winner": r["branch_id"] == w.get("branch_id"),
            "repo": Path(state["repo_dir"]).name, "task": state["task"],
            "test_cmd": state["test_cmd"], "strategy": r["strategy"],
            "model": r.get("model") or stage_model("executor"),
            "thinking": state.get("thinking") or "on",
            "diff_lines": r["diff_lines"], "turns": r["turns"],
            "tokens": r["input_tokens"] + r["output_tokens"],
            "tools": TOOLS, "messages": r["messages"],
        }
        path = d / f"{state['run_id']}_{r['branch_id']}.json"
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        if r.get("passed"):
            green += 1
        else:
            red += 1
    return green, red


def finalize_node(state: GraphState) -> dict:
    logs = BASE_DIR / "logs"
    logs.mkdir(exist_ok=True)
    w = state.get("winner")

    totals: dict[str, dict] = {}
    for u in state.get("usage", []):
        t = totals.setdefault(u["stage"], {"input": 0, "output": 0, "usd": 0.0})
        t["input"] += u["input"]
        t["output"] += u["output"]
        t["usd"] += u.get("usd", 0.0)
    for t in totals.values():
        t["usd"] = round(t["usd"], 5)

    applied = False
    branch = None
    if w:
        if state.get("make_branch", True):
            try:
                branch = commit_to_branch(state["repo_dir"], w, state["run_id"], state["test_cmd"])
            except Exception as e:
                branch = {"error": str(e)}
        # bytes, not write_text: on Windows write_text turns every \n into \r\n and the
        # patch no longer applies to LF files
        (logs / f"graph_{state['run_id']}_winner.patch").write_bytes(w["patch"].encode("utf-8"))
        if state.get("apply"):
            for rel in w["changed"]:
                src = Path(w["workdir"]) / rel
                if src.exists():
                    dst = safe_path(state["repo_dir"], rel)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            applied = True

    saved, saved_red = save_trajectories(state)
    _budgets.pop(state["run_id"], None)

    summary = {
        "run_id": state["run_id"],
        "status": "already_green" if state.get("baseline_passed") else ("green" if w else "red"),
        "rounds": state.get("round", 0),
        "winner": w and {k: w[k] for k in ("branch_id", "strategy", "diff_lines", "changed",
                                           "turns", "tool_calls")},
        "git": branch,
        "applied": applied,
        "branches": [{k: r.get(k) for k in ("branch_id", "model", "passed", "cancelled", "stop_reason",
                                            "diff_lines", "turns", "tool_calls", "input_tokens",
                                            "output_tokens", "usd", "refused_writes", "tampered")}
                     for r in state.get("results", [])],
        "tokens_by_stage": totals,
        "cost_usd": round(sum(t["usd"] for t in totals.values()), 5),
        "tamper_attempts": sum(bool(r.get("refused_writes") or r.get("tampered"))
                               for r in state.get("results", [])),
        "trajectories_saved": saved,
        "failed_trajectories_saved": saved_red,
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
    if budget(state["run_id"]).exceeded():
        log("[budget] spend limit reached, stopping")
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
        make_branch: bool = True, on_log=None, full: bool = False,
        first_green: bool = True, thinking: Optional[str] = None,
        max_usd: Optional[float] = None, ladder: Optional[str] = None) -> dict:
    """Returns the summary dict, or the full final graph state when full=True.
    max_usd: stop starting new LLM calls once this run has spent this much.
    ladder: executor model per round, e.g. "nano,super,ultra" (default: EXECUTOR_LADDER)."""
    repo_dir = str(Path(repo).resolve())
    if not Path(repo_dir).is_dir():
        raise FileNotFoundError(repo_dir)
    if max_usd is None and os.getenv("MAX_USD_PER_RUN"):
        max_usd = float(os.environ["MAX_USD_PER_RUN"])
    state: GraphState = {
        "task": task, "repo_dir": repo_dir, "test_cmd": test_cmd,
        "work_root": work_root or str(BASE_DIR / "workspace" / "branches"),
        "run_id": time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4],
        "n_branches": branches, "max_rounds": rounds, "max_turns": max_turns, "apply": apply,
        "make_branch": make_branch,
        "first_green": first_green,
        "thinking": thinking or os.getenv("EXECUTOR_THINKING") or None,
        "max_usd": max_usd,
        "ladder": parse_ladder(ladder if ladder is not None else os.getenv("EXECUTOR_LADDER")),
        "results": [], "usage": [], "feedback": "",
    }
    _budgets[state["run_id"]] = Budget(max_usd)
    lines: list[str] = []

    def sink(msg: str) -> None:
        lines.append(msg)
        if on_log:
            on_log(msg)

    token = _log_sink.set(sink)
    try:
        final = build_graph().invoke(state, {"recursion_limit": 10 + rounds * 6})
    finally:
        _log_sink.reset(token)
    save_replay(final, lines)
    return final if full else final["summary"]


REPLAY_FIELDS = ("branch_id", "round", "strategy", "model", "passed", "cancelled", "stop_reason",
                 "test_output", "diff_lines", "changed", "patch", "turns", "tool_calls",
                 "input_tokens", "output_tokens", "usd", "refused_writes", "tampered")


def save_replay(final: dict, lines: list[str]) -> Path:
    """Everything the UI needs to show a finished run again without calling any model:
    the log lines in order, the summary and every branch (minus conversations and local paths).
    Copy one into examples/recorded/ and the hosted demo can replay it for free."""
    w = final.get("winner") or {}
    record = {
        "run_id": final["run_id"], "repo": Path(final["repo_dir"]).name, "task": final["task"],
        "test_cmd": final["test_cmd"], "baseline_output": final.get("baseline_output", ""),
        "logs": lines, "summary": final["summary"],
        "results": [{k: r.get(k) for k in REPLAY_FIELDS} for r in final.get("results", [])],
        "winner": w.get("branch_id"),
    }
    path = BASE_DIR / "logs" / f"graph_{final['run_id']}_replay.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


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
    ap.add_argument("--all-branches", action="store_true",
                    help="Let every branch finish instead of stopping at the first green one")
    ap.add_argument("--thinking", choices=["on", "low", "off"], default=None,
                    help="Executor reasoning mode (Nemotron 3); default: model default")
    ap.add_argument("--ladder", default=None,
                    help="Executor model per round, e.g. nano,super,ultra (default: EXECUTOR_LADDER)")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="Spend cap for this run in USD (default: MAX_USD_PER_RUN or no cap)")
    a = ap.parse_args()

    try:
        s = _run_cli(a)
    except SandboxUnavailable as e:
        raise SystemExit(f"\nERROR: {e}")
    print("\n=== SUMMARY ===")
    _print_summary(a, s)


def _run_cli(a):
    return run(a.repo, a.task, a.test_cmd, a.branches, a.rounds, a.max_turns, a.apply,
            make_branch=not a.no_branch, first_green=not a.all_branches, thinking=a.thinking,
            max_usd=a.max_usd, ladder=a.ladder)


def _print_summary(a, s):
    g = s.get("git") or {}
    if g.get("branch"):
        print(f"Fix committed to branch {g['branch']} ({g['commit']}). Review with:")
        print(f"  git -C {a.repo} diff HEAD {g['branch']}")
        print(f"  git -C {a.repo} merge {g['branch']}")
    print(json.dumps({k: s[k] for k in ("status", "rounds", "winner", "git", "applied",
                                        "tokens_by_stage", "cost_usd", "tamper_attempts")}, indent=2))


if __name__ == "__main__":
    main()