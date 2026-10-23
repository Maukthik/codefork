"""
bench/swe_env.py - SWE-rebench tasks in Nebius Sandboxes: load, prepare, test, grade.

What the probe (bench/probe_swe_rebench.py) showed about the task images:
  - the repo is at /testbed, but NOT at the task's base_commit: we `git checkout` it first
  - the project's Python lives in the conda env "testbed", which must be activated
  - the test command is install_config["test_cmd"] (pytest -rA ... <test files>)
  - grading is per test id: FAIL_TO_PASS must pass, PASS_TO_PASS must keep passing

This module has no model calls. Network access (Hugging Face, Nebius) happens only
inside the functions that need it, so the parsing/grading logic is unit-tested offline.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "logs" / "swe_rebench"
DATASET = "nebius/SWE-rebench-leaderboard"
APP = "/testbed"
ACTIVATE = "source /opt/conda/bin/activate testbed"
INPUTS = "fork_inputs"          # uploaded files land in /fork_inputs (some images hide uploads in /tmp)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass
class Task:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str                 # gold fix (never shown to the agent)
    test_patch: str            # the PR's test changes, applied before the agent starts
    f2p: list[str]             # FAIL_TO_PASS
    p2p: list[str]             # PASS_TO_PASS
    image: str
    test_cmd: str
    created_at: str = ""
    hints: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def gold_files(self) -> list[str]:
        return patch_files(self.patch)

    @property
    def gold_lines(self) -> int:
        return changed_lines(self.patch)

    @property
    def month(self) -> str:
        return self.created_at[:7]

    def to_json(self) -> dict:
        return self.__dict__.copy()


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return [v]
    return [str(x) for x in list(v)]


def _as_dict(v) -> dict:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return {}
    return dict(v or {})


def task_from_row(row: dict) -> Task:
    cfg = _as_dict(row.get("install_config"))
    return Task(
        instance_id=row["instance_id"], repo=row["repo"], base_commit=row["base_commit"],
        problem_statement=row.get("problem_statement") or "", patch=row.get("patch") or "",
        test_patch=row.get("test_patch") or "", f2p=_as_list(row.get("FAIL_TO_PASS")),
        p2p=_as_list(row.get("PASS_TO_PASS")),
        image=row.get("docker_image") or row.get("image_name") or "",
        test_cmd=row.get("test_cmd") or cfg.get("test_cmd") or "pytest -rA",
        created_at=str(row.get("created_at") or ""), hints=row.get("hints_text") or "",
    )


def patch_files(patch: str) -> list[str]:
    out = []
    for line in (patch or "").splitlines():
        if line.startswith("diff --git "):
            out.append(line.split(" b/", 1)[-1])
    return out


def changed_lines(patch: str) -> int:
    return sum(1 for l in (patch or "").splitlines()
               if l[:1] in "+-" and not l.startswith(("+++", "---")))


def load_tasks(split_file: str = "data/test-00000-of-00001.parquet") -> list[Task]:
    """Download (once, cached by huggingface_hub) and parse the SWE-rebench leaderboard."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET, split_file, repo_type="dataset")
    return [task_from_row(r) for r in pd.read_parquet(path).to_dict("records")]


def select(tasks: list[Task], n: int, *, max_gold_files: int = 1, max_gold_lines: int = 20,
           max_f2p: int = 5, max_p2p: int = 300, skip_repos: tuple = ("home-assistant/core",),
           seed: str = "fork") -> list[Task]:
    """A deterministic, cheap-to-run subset: small fixes, few target tests, one task per repo
    (so one big repo can't dominate). Order is a stable hash, not the dataset order."""
    ok = [t for t in tasks
          if t.image and t.f2p and len(t.gold_files) <= max_gold_files
          and 1 <= t.gold_lines <= max_gold_lines and len(t.f2p) <= max_f2p
          and len(t.p2p) <= max_p2p and t.repo not in skip_repos]
    ok.sort(key=lambda t: hashlib.sha256(f"{seed}:{t.instance_id}".encode()).hexdigest())
    seen, picked = set(), []
    for t in ok:
        if t.repo in seen:
            continue
        seen.add(t.repo)
        picked.append(t)
        if len(picked) == n:
            break
    return picked


def split_train_test(tasks: list[Task], test_from: str = "2025-10") -> tuple[list[Task], list[Task]]:
    """Older tasks are for tuning and fine-tuning data; newer ones are the reported score."""
    return ([t for t in tasks if t.month < test_from], [t for t in tasks if t.month >= test_from])


# ---------------------------------------------------------------------------
# Test results
# ---------------------------------------------------------------------------

_STATUS_RE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(.+)$")
# verbose style: 'tests/x.py::test_y[param] PASSED   [ 44%]'
_VERBOSE_RE = re.compile(r"^(\S.*?::\S.*?)\s+(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)(?:\s+\[\s*\d+%\])?$")


def parse_pytest_ra(output: str) -> dict[str, str]:
    """Per-test status from pytest output: the -rA summary ('PASSED id', 'FAILED id - msg')
    or verbose lines ('id PASSED [ 44%]'). Summary lines come last, so they win."""
    status: dict[str, str] = {}
    for line in output.splitlines():
        v = _VERBOSE_RE.match(line.strip())
        if v:
            status[v.group(1)] = v.group(2)
            continue
        m = _STATUS_RE.match(line.strip())
        if not m:
            continue
        kind, rest = m.groups()
        if kind == "SKIPPED" and rest.startswith("["):
            continue                                   # 'SKIPPED [1] file:line: reason' has no id
        # every kind but PASSED may carry " - reason" after the id
        test_id = rest.strip() if kind == "PASSED" else rest.split(" - ", 1)[0].strip()
        status[test_id] = kind
    return status


@dataclass
class Grade:
    resolved: bool
    f2p_passed: int
    f2p_total: int
    p2p_broken: list[str]
    missing: list[str]

    def summary(self) -> str:
        s = f"F2P {self.f2p_passed}/{self.f2p_total}"
        if self.p2p_broken:
            s += f", broke {len(self.p2p_broken)} P2P"
        return ("RESOLVED " if self.resolved else "unresolved ") + s


def grade(task: Task, output: str, p2p_baseline: Optional[set[str]] = None) -> Grade:
    """SWE-bench rule: every FAIL_TO_PASS test passes, and every PASS_TO_PASS test that
    passed before the fix still passes (p2p_baseline narrows P2P to those that did)."""
    st = parse_pytest_ra(output)
    passed = lambda t: st.get(t) in ("PASSED", "XFAIL")
    f2p_ok = [t for t in task.f2p if passed(t)]
    p2p = [t for t in task.p2p if p2p_baseline is None or t in p2p_baseline]
    broken = [t for t in p2p if not passed(t)]
    missing = [t for t in task.f2p + p2p if t not in st]
    return Grade(resolved=len(f2p_ok) == len(task.f2p) and not broken,
                 f2p_passed=len(f2p_ok), f2p_total=len(task.f2p), p2p_broken=broken, missing=missing)


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

def shell(cmd: str) -> str:
    """Run cmd in bash with the project's conda env active, inside the repo."""
    return f"bash -lc {shlex.quote(f'{ACTIVATE} && cd {APP} && {cmd}')}"


def _tmpfile(text: str, suffix: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8", newline="\n") as f:
        f.write(text)
    return f.name


def prepare_checkpoint(task: Task, client=None, timeout: int = 1200):
    """Task image -> repo checked out at base_commit, clean, with the PR's test changes applied.
    Returns (checkpoint_image, log). Every agent branch and every grading run forks from it."""
    if client is None:
        from contree_sdk import ContreeSync
        client = ContreeSync()
    t0 = time.time()
    img = client.images.oci(task.image, timeout=timeout)
    files, apply = None, ""
    if task.test_patch.strip():
        files = {f"{INPUTS}/test.patch": _tmpfile(task.test_patch, ".patch")}
        apply = f"git apply /{INPUTS}/test.patch && "
    cmd = (f"cd {APP} && git checkout -q -f {task.base_commit} && git clean -fdq && "
           f"{apply}echo FORK_READY")
    r = img.run(shell=f"bash -lc {shlex.quote(cmd)}", files=files, disposable=False, timeout=600).wait()
    out = (r.stdout or "") + (r.stderr or "")
    if "FORK_READY" not in out:
        raise RuntimeError(f"{task.instance_id}: couldn't prepare repo (exit {r.exit_code}): {out[-800:]}")
    return r, f"prepared in {time.time() - t0:.0f}s"


def run_tests(checkpoint, task: Task, extra_files: Optional[dict] = None, pre: str = "",
              timeout: int = 900) -> tuple[int, str]:
    """Run the task's test command on a disposable fork of the checkpoint.
    Full suites print thousands of lines and the sandbox output gets cut, so the log goes to
    a file and we return only the lines naming the graded tests, plus the last 60 lines."""
    ids = _tmpfile("\n".join(dict.fromkeys(task.f2p + task.p2p)) + "\n", ".txt")
    files = {f"{INPUTS}/ids.txt": ids, **(extra_files or {})}
    test = (pre + " && " if pre else "") + task.test_cmd
    cmd = (f"( {test} ) > /tmp/fork_test.log 2>&1; code=$?; "
           f"grep -aF -f /{INPUTS}/ids.txt /tmp/fork_test.log | cut -c1-400; "
           f"echo '--- last lines ---'; tail -n 60 /tmp/fork_test.log | cut -c1-400; exit $code")
    r = checkpoint.run(shell=shell(cmd), files=files, timeout=timeout).wait()
    return r.exit_code, (r.stdout or "") + (r.stderr or "")


def run_with_patch(checkpoint, task: Task, patch: str, timeout: int = 900) -> tuple[int, str]:
    """Apply a unified diff on top of the checkpoint and run the tests (used for the gold
    patch in validation, and for the agent's final patch in evaluation)."""
    p = _tmpfile(patch, ".patch")
    return run_tests(checkpoint, task, extra_files={f"{INPUTS}/fix.patch": p},
                     pre=f"git apply /{INPUTS}/fix.patch && echo FIX_APPLIED", timeout=timeout)
