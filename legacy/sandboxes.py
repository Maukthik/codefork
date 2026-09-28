"""Sandbox backends for the agent.

Choose one in .env:
    SANDBOX=docker   local Docker (free, fast, offline)          <- default
    SANDBOX=nebius   Nebius Token Factory Sandboxes (cloud, VM-isolated,
                     every command saved as a snapshot you can branch from)

Both backends offer the same five methods, so agent.py never needs to know
which one is running:
    start()                     prepare a fresh sandbox for a new task
    write_file(rel, content)    create / overwrite a file
    read_file(rel) -> str|None  read a file (None if it doesn't exist)
    run(command) -> (exit_code, stdout, stderr)
    finish() -> int             tidy up; Nebius copies files back to your laptop
"""
import os
import shlex
import subprocess
from pathlib import Path, PurePosixPath

SKIP_DIRS = {"__pycache__", ".pytest_cache"}
MAX_SYNC_BYTES = 1_000_000          # don't copy files bigger than ~1 MB


def local_files(root: Path):
    """Yield (relative_path, Path) for every normal file in the local workspace."""
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts) or p.stat().st_size > MAX_SYNC_BYTES:
            continue
        yield rel.as_posix(), p


# ======================================================================
# Local Docker
# ======================================================================
class DockerSandbox:
    """Files live in the local workspace folder; each command runs in a fresh
    Docker container with that folder mounted. Only the folder persists."""
    name = "docker"

    def __init__(self, workspace: Path, image: str, timeout: int = 30):
        self.workspace = workspace
        self.image = image
        self.timeout = timeout

    def start(self) -> None:
        pass                                  # nothing to prepare

    def write_file(self, rel: str, content: str) -> None:
        p = self.workspace / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read_file(self, rel: str) -> str | None:
        p = self.workspace / rel
        return p.read_text(encoding="utf-8") if p.exists() else None

    def run(self, command: str) -> tuple[int, str, str]:
        docker_cmd = [
            "docker", "run", "--rm",
            "--network", "none",              # no internet inside the sandbox
            "--memory", "512m",               # memory cap
            "--cpus", "1",                    # CPU cap
            "-v", f"{self.workspace.absolute()}:/workspace",
            "-w", "/workspace",
            self.image,
            "sh", "-c", command,
        ]
        # raises subprocess.TimeoutExpired on timeout - agent.py turns that into an ERROR
        r = subprocess.run(docker_cmd, capture_output=True, text=True,
                           timeout=self.timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout, r.stderr

    def finish(self) -> int:
        return 0                              # files are already local


# ======================================================================
# Nebius Token Factory Sandboxes  (contree-sdk)
# ======================================================================
class NebiusSandbox:
    """Runs everything in Nebius Token Factory Sandboxes.

    Key idea: there is no long-lived machine. Every command runs from a saved
    filesystem snapshot and produces a NEW snapshot. `self.state` always points
    at the latest one, so each command sees the files of the previous one.
    Because old snapshots are kept, the branching stage can later start several
    attempts from the same snapshot.
    """
    name = "nebius"
    REMOTE = PurePosixPath("/workspace")

    def __init__(self, workspace: Path, image: str, timeout: int = 30):
        self.workspace = workspace
        self.image_ref = image
        self.timeout = timeout
        self.sdk = None
        self.base = None
        self.state = None

    def _connect(self) -> None:
        """Connect on first use (so merely importing agent.py never calls the API)."""
        if self.sdk is not None:
            return
        if not os.environ.get("NEBIUS_PROJECT_ID"):
            raise RuntimeError("SANDBOX=nebius needs NEBIUS_PROJECT_ID in your .env file")
        from contree_sdk import ContreeSync   # reads NEBIUS_API_KEY + NEBIUS_PROJECT_ID
        self.sdk = ContreeSync()
        self.base = self.sdk.images.oci(self.image_ref)   # imports once, then reuses

    def _remote(self, rel: str) -> str:
        return str(self.REMOTE / rel)

    def start(self) -> None:
        """Fresh sandbox for a new task, containing a copy of your local workspace files."""
        self._connect()
        files = dict(local_files(self.workspace))
        folders = sorted({str(self.REMOTE / PurePosixPath(r).parent) for r in files} | {str(self.REMOTE)})
        state = self.base.run(shell="mkdir -p " + " ".join(shlex.quote(f) for f in folders),
                              disposable=False).wait()
        if files:
            state = state.apply_files({self._remote(r): p.read_bytes() for r, p in files.items()})
        self.state = state

    def write_file(self, rel: str, content: str) -> None:
        parent = PurePosixPath(rel).parent
        if str(parent) != ".":
            self.state = self.state.run(shell=f"mkdir -p {shlex.quote(self._remote(str(parent)))}",
                                        disposable=False).wait()
        self.state = self.state.apply_files({self._remote(rel): content.encode("utf-8")})

    def read_file(self, rel: str) -> str | None:
        try:
            return self.state.read(self._remote(rel)).decode("utf-8", errors="replace")
        except Exception:
            return None

    def run(self, command: str) -> tuple[int, str, str]:
        result = self.state.run(shell=command, cwd=str(self.REMOTE),
                                disposable=False, timeout=self.timeout).wait()
        self.state = result                   # next command continues from here
        return result.exit_code, str(result.stdout or ""), str(result.stderr or "")

    def finish(self) -> int:
        """Copy the sandbox's workspace back into your local folder so you can inspect it."""
        if self.state is None:
            return 0
        listing = self.state.run(
            shell="find . -type f -size -1000k -not -path '*/__pycache__/*' | head -200",
            cwd=str(self.REMOTE), disposable=True).wait()
        root = self.workspace.resolve()
        copied = 0
        for line in str(listing.stdout or "").splitlines():
            rel = line.strip().removeprefix("./")
            if not rel:
                continue
            local = (self.workspace / rel).resolve()
            if not local.is_relative_to(root):
                continue                      # never write outside the workspace
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(self.state.read(self._remote(rel)))
            copied += 1
        return copied


def make_sandbox(kind: str, workspace: Path, docker_image: str, nebius_image: str):
    kind = kind.lower()
    if kind == "docker":
        return DockerSandbox(workspace, docker_image)
    if kind == "nebius":
        return NebiusSandbox(workspace, nebius_image)
    raise SystemExit(f"Unknown SANDBOX '{kind}'. Use 'docker' or 'nebius'.")