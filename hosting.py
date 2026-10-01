"""
hosting.py - helpers for running the Streamlit app as a public demo (HOSTED=1).

Where the repo comes from (a bundled demo or a public GitHub repo), who is allowed to
start a paid run (access code), how much the whole demo may spend per day, and the
recorded runs anyone can replay for free. Kept apart from the UI so it can be tested.
"""
from __future__ import annotations

import datetime
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
RECORDED = EXAMPLES / "recorded"
SKIP = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".mypy_cache"}

MAX_ZIP_BYTES = int(os.getenv("MAX_REPO_ZIP_BYTES", str(5_000_000)))
MAX_FILES = int(os.getenv("MAX_REPO_FILES", "400"))
GITHUB_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?(?:tree/([\w./-]+?))?/?$")


def is_hosted() -> bool:
    return os.getenv("HOSTED") == "1"


# ---------------------------------------------------------------------------
# Repo sources
# ---------------------------------------------------------------------------

def demo_names() -> list[str]:
    return sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir() and p.name != "recorded")


def prepare_demo(name: str, root: Optional[str] = None) -> str:
    """Fresh copy of examples/<name> in a temp folder, so every visitor gets the buggy version."""
    if name not in demo_names():
        raise ValueError(f"Unknown demo: {name}")
    dest = Path(root or tempfile.mkdtemp(prefix="fork-demo-")) / name
    shutil.copytree(EXAMPLES / name, dest,
                    ignore=shutil.ignore_patterns("README.md", *SKIP), dirs_exist_ok=True)
    return str(dest)


def parse_github_url(url: str) -> tuple[str, str, Optional[str]]:
    m = GITHUB_RE.match((url or "").strip())
    if not m:
        raise ValueError("Use a public GitHub repo URL like https://github.com/owner/repo "
                         "(optionally .../tree/branch)")
    return m.group(1), m.group(2), m.group(3)


def _download(url: str, opener=urllib.request.urlopen) -> bytes:
    with opener(url, timeout=30) as resp:
        data = resp.read(MAX_ZIP_BYTES + 1)
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError(f"Repo is larger than {MAX_ZIP_BYTES // 1_000_000} MB zipped; "
                         "the hosted demo only takes small repos.")
    return data


def extract_zip(data: bytes, dest: str) -> str:
    """Unpack a GitHub zipball (one top-level folder) into dest, refusing path tricks
    and repos with too many files."""
    dest_p = Path(dest).resolve()
    with zipfile.ZipFile(BytesIO(data)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        tops = {n.split("/", 1)[0] for n in names}
        prefix = f"{tops.pop()}/" if len(tops) == 1 and all("/" in n for n in names) else ""
        keep = []
        for n in names:
            rel = n[len(prefix):] if prefix and n.startswith(prefix) else n
            parts = Path(rel).parts
            if not rel or rel.startswith("/") or ".." in parts:
                raise ValueError(f"Unsafe path in archive: {n}")
            if SKIP & set(parts):
                continue
            keep.append((n, rel))
        if len(keep) > MAX_FILES:
            raise ValueError(f"Repo has {len(keep)} files; the hosted demo takes up to {MAX_FILES}.")
        for n, rel in keep:
            target = (dest_p / rel).resolve()
            if dest_p not in target.parents:
                raise ValueError(f"Unsafe path in archive: {n}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(n))
    return str(dest_p)


def fetch_github(url: str, root: Optional[str] = None, opener=urllib.request.urlopen) -> str:
    """Download a public repo as a zip (no git needed on the host) into a temp folder."""
    owner, repo, branch = parse_github_url(url)
    dest = Path(root or tempfile.mkdtemp(prefix="fork-gh-")) / repo
    for b in [branch] if branch else ["main", "master"]:
        try:
            data = _download(f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{b}", opener)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        return extract_zip(data, str(dest))
    raise ValueError(f"Couldn't download {owner}/{repo}: is it public? For a branch other than "
                     "main/master, use .../tree/<branch>.")


# ---------------------------------------------------------------------------
# Who may spend, and how much
# ---------------------------------------------------------------------------

def access_ok(entered: str) -> bool:
    """Live runs cost real credits. With DEMO_ACCESS_CODE set, only people who have the
    code (e.g. the judges, from the submission form) can start one."""
    expected = os.getenv("DEMO_ACCESS_CODE", "")
    return not expected or hmac.compare_digest((entered or "").strip(), expected)


def per_run_cap() -> float:
    return float(os.getenv("HOSTED_MAX_USD_PER_RUN", "0.25"))


class Ledger:
    """Spend of the whole demo today, across all visitors. Lives in the app process, so it
    resets if the app restarts; per-run caps still bound every single run."""

    def __init__(self, daily_usd: float, today=datetime.date.today):
        self.daily_usd, self._today, self._lock = daily_usd, today, threading.Lock()
        self.day, self.spent = today(), 0.0

    def _roll(self):
        if self._today() != self.day:
            self.day, self.spent = self._today(), 0.0

    def remaining(self) -> float:
        with self._lock:
            self._roll()
            return max(0.0, self.daily_usd - self.spent)

    def add(self, usd: float) -> None:
        with self._lock:
            self._roll()
            self.spent += max(0.0, usd or 0.0)


# ---------------------------------------------------------------------------
# Recorded runs
# ---------------------------------------------------------------------------

def recorded_runs() -> list[Path]:
    return sorted(RECORDED.glob("*.json")) if RECORDED.is_dir() else []


def load_replay(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
