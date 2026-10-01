"""
Probe SWE-rebench before building the benchmark runner (no LLM calls, no credits for models).

    python bench/probe_swe_rebench.py

1. Downloads the most recent month of nebius/SWE-rebench-leaderboard from Hugging Face.
2. Prints the fields, a few example tasks and size statistics.
3. Imports the Docker image of the smallest task into a Nebius sandbox, looks around inside
   it (where the repo is, which Python, conda env) and runs its failing tests once.

Everything is written to logs/swe_rebench_probe.txt and logs/swe_rebench_sample.json.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
OUT = ROOT / "logs" / "swe_rebench_probe.txt"
SAMPLE = ROOT / "logs" / "swe_rebench_sample.json"
DATASET = "nebius/SWE-rebench-leaderboard"
lines: list[str] = []


def say(*parts) -> None:
    text = " ".join(str(p) for p in parts)
    print(text, flush=True)
    lines.append(text)


def save() -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return [v]
    return list(v) if v is not None else []


def patch_stats(patch: str) -> tuple[int, int]:
    files = sum(1 for l in (patch or "").splitlines() if l.startswith("diff --git"))
    changed = sum(1 for l in (patch or "").splitlines()
                  if l[:1] in "+-" and not l.startswith(("+++", "---")))
    return files, changed


def load_month():
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd
    files = [f for f in HfApi().list_repo_files(DATASET, repo_type="dataset") if f.endswith(".parquet")]
    say(f"parquet files in {DATASET}: {len(files)}")
    for f in sorted(files)[-6:]:
        say("  ", f)
    latest = sorted(files)[-1]
    df = pd.read_parquet(hf_hub_download(DATASET, latest, repo_type="dataset"))
    say(f"\nloaded {latest}: {len(df)} rows")
    return df


def describe(df) -> list[dict]:
    say("\n=== columns ===")
    for c in df.columns:
        say(f"  {c}: {type(df[c].iloc[0]).__name__}")
    rows = df.to_dict("records")
    for r in rows:
        r["_f2p"] = as_list(r.get("FAIL_TO_PASS"))
        r["_p2p"] = as_list(r.get("PASS_TO_PASS"))
        r["_files"], r["_changed"] = patch_stats(r.get("patch", ""))
    say("\n=== sizes ===")
    for key in ("_files", "_changed"):
        vals = sorted(r[key] for r in rows)
        say(f"  gold patch {key[1:]}: min {vals[0]}, median {vals[len(vals) // 2]}, max {vals[-1]}")
    for key in ("_f2p", "_p2p"):
        vals = sorted(len(r[key]) for r in rows)
        say(f"  {key[1:]} tests: min {vals[0]}, median {vals[len(vals) // 2]}, max {vals[-1]}")
    small = [r for r in rows if r["_files"] == 1 and r["_changed"] <= 20 and 1 <= len(r["_f2p"]) <= 5]
    say(f"  tasks with a 1-file, <=20-line gold patch and 1-5 failing tests: {len(small)}")

    say("\n=== 3 example tasks (smallest gold patches) ===")
    examples = sorted(rows, key=lambda r: (r["_files"], r["_changed"]))[:3]
    for r in examples:
        say(f"\n--- {r.get('instance_id')}  repo={r.get('repo')}  created={r.get('created_at')}")
        for k in ("docker_image", "image_name", "test_cmd", "version", "base_commit"):
            say(f"  {k}: {str(r.get(k))[:300]}")
        say(f"  install_config: {json.dumps(r.get('install_config'), default=str)[:800]}")
        say(f"  FAIL_TO_PASS ({len(r['_f2p'])}): {r['_f2p'][:5]}")
        say(f"  PASS_TO_PASS: {len(r['_p2p'])} tests, e.g. {r['_p2p'][:3]}")
        say(f"  gold patch: {r['_files']} file(s), {r['_changed']} changed lines")
        say(f"  test_patch files: {[l[11:] for l in (r.get('test_patch') or '').splitlines() if l.startswith('diff --git')]}")
        say(f"  problem_statement: {(r.get('problem_statement') or '')[:400]!r}")
    keep = [{k: (v if not hasattr(v, 'tolist') else v.tolist()) for k, v in r.items()} for r in examples]
    SAMPLE.write_text(json.dumps(keep, indent=1, default=str), encoding="utf-8")
    return examples


def probe_sandbox(task: dict) -> None:
    from contree_sdk import ContreeSync
    image_ref = task.get("docker_image") or task.get("image_name")
    say(f"\n=== sandbox probe: {task.get('instance_id')} ===\nimage: {image_ref}")
    client = ContreeSync()
    t = time.time()
    img = client.images.oci(image_ref, timeout=1200)
    say(f"imported in {time.time() - t:.0f}s (uuid {getattr(img, 'uuid', '?')})")

    def run(cmd: str, files=None, timeout=300):
        t0 = time.time()
        r = img.run(shell=cmd, files=files, timeout=timeout).wait()
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        say(f"\n$ {cmd[:200]}\n[exit {r.exit_code}, {time.time() - t0:.0f}s]\n{out[-2500:]}")
        return r

    run("pwd; whoami; ls /; ls -la /testbed 2>/dev/null | head -20; cat /etc/os-release | head -2")
    run("which python python3 git conda pytest 2>/dev/null; python --version; echo PATH=$PATH; "
        "ls /opt 2>/dev/null; conda env list 2>/dev/null; cat ~/.bashrc 2>/dev/null | tail -5")
    run("bash -lc 'which python; python --version; python -c \"import sys; print(sys.prefix)\"'")
    run("cd /testbed 2>/dev/null && git log -1 --oneline && git status --short | head")

    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8", newline="\n") as f:
        f.write(task.get("test_patch") or "")
    f2p = task["_f2p"][:5]
    test_cmd = task.get("test_cmd") or "pytest"
    cmd = (f"cd /testbed && git apply /tmp/test.patch && echo TEST_PATCH_APPLIED; "
           f"{test_cmd} {' '.join(shlex.quote(t) for t in f2p)} 2>&1 | tail -40")
    run(f"bash -lc {shlex.quote(cmd)}", files={"tmp/test.patch": f.name}, timeout=600)
    os.unlink(f.name)


def main():
    try:
        df = load_month()
        examples = describe(df)
        if "--no-sandbox" not in sys.argv:
            probe_sandbox(examples[0])
    except Exception as e:
        say(f"\nERROR: {type(e).__name__}: {e}")
        raise
    finally:
        save()
        print(f"\nSaved {OUT} and {SAMPLE}")


if __name__ == "__main__":
    main()
