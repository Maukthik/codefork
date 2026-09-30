import sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from contree_sdk import ContreeSync

repo = Path(sys.argv[1] if len(sys.argv) > 1 else "workspace/invoice")
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache"}
client = ContreeSync()

t = time.time()
base = client.images.use("python:3.12-slim")
base = base.run(shell="pip install -q pytest", disposable=False).wait()
print(f"[1] base image ready ({time.time() - t:.0f}s), uuid={base.uuid}")

files = {f"app/{p.relative_to(repo).as_posix()}": str(p)
         for p in repo.rglob("*")
         if p.is_file() and not SKIP & set(p.relative_to(repo).parts)}
print(f"[2] uploading {len(files)} files: {sorted(files)}")

t = time.time()
result = base.run(shell="cd /app && python -m pytest -q", files=files).wait()
print(f"[3] exit code {result.exit_code} ({time.time() - t:.0f}s)")
print(result.stdout)
if result.stderr:
    print("stderr:", result.stderr)

a = base.run(shell="echo branch A", disposable=False).wait()
b = base.run(shell="echo branch B", disposable=False).wait()
print(f"[4] branches from one checkpoint: {a.stdout.strip()} / {b.stdout.strip()}")
