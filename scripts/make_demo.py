"""Create a demo repo for the agent to fix (works on Windows, macOS and Linux).

Copies examples/<name> into workspace/<name> and makes it its own git repo, so the
agent can put its fix on a review branch without touching this project's history.

    python scripts/make_demo.py            # all examples
    python scripts/make_demo.py invoice    # one example
    python scripts/make_demo.py --force    # recreate even if it already exists
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
WORKSPACE = ROOT / "workspace"


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def _force_writable(func, path, _exc):
    """git marks object files read-only; on Windows rmtree can't delete them until we chmod."""
    Path(path).chmod(0o700)
    func(path)


def make(name: str, force: bool) -> None:
    src, dst = EXAMPLES / name, WORKSPACE / name
    if not src.is_dir():
        sys.exit(f"No example called '{name}'. Available: {', '.join(available())}")
    if dst.exists():
        if not force:
            print(f"workspace/{name} already exists (use --force to recreate)")
            return
        shutil.rmtree(dst, onerror=_force_writable)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "README.md"))
    git(dst, "init", "-q", "-b", "main")
    git(dst, "add", "-A")
    git(dst, "-c", "user.name=Fork Demo", "-c", "user.email=demo@localhost",
        "commit", "-q", "-m", f"{name} demo (buggy)")
    print(f"Created workspace/{name}. Check it fails:  python -m pytest -q workspace/{name}")


def available() -> list[str]:
    return sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="examples to create (default: all)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for n in a.names or available():
        make(n, a.force)
