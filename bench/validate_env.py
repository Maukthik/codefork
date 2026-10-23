"""
Check SWE-rebench tasks before spending anything on models.

    python bench/validate_env.py --n 8            # pick 8 small tasks and validate them
    python bench/validate_env.py --n 8 --workers 3

For each task, in a Nebius sandbox:
  1. prepare: task image -> base_commit -> PR test changes applied
  2. baseline: run the tests; FAIL_TO_PASS must fail (otherwise there's nothing to fix)
  3. gold: apply the real fix from the PR; FAIL_TO_PASS must pass and PASS_TO_PASS stay green

Only tasks passing all three are benchmarked ("valid"), like SWE-bench Verified.
Results are appended to logs/swe_rebench/validation.jsonl; finished tasks are skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv  # noqa: E402

import swe_env as env  # noqa: E402

load_dotenv(env.ROOT / ".env")
OUT = env.CACHE / "validation.jsonl"
_lock = threading.Lock()
_local = threading.local()


def client():
    if not hasattr(_local, "c"):
        from contree_sdk import ContreeSync
        _local.c = ContreeSync()
    return _local.c


def validate(task: env.Task) -> dict:
    row = {"instance_id": task.instance_id, "repo": task.repo, "month": task.month,
           "image": task.image, "gold_lines": task.gold_lines, "f2p": len(task.f2p), "p2p": len(task.p2p)}
    t0 = time.time()
    try:
        ckpt, note = env.prepare_checkpoint(task, client())
        code, out = env.run_tests(ckpt, task)
        base = env.grade(task, out)
        ran = len(env.parse_pytest_ra(out))
        p2p_baseline = {t for t, s in env.parse_pytest_ra(out).items() if s == "PASSED"}
        row.update(baseline=base.summary(), baseline_f2p_passed=base.f2p_passed,
                   p2p_passing_before=len(p2p_baseline & set(task.p2p)))
        code, out = env.run_with_patch(ckpt, task, task.patch)
        gold = env.grade(task, out, p2p_baseline)
        valid = base.f2p_passed < base.f2p_total and gold.resolved
        reason = ("" if valid else
                  "tests didn't run (broken environment?)" if ran == 0 else
                  "failing tests already pass before the fix" if base.f2p_passed == base.f2p_total else
                  "gold fix doesn't resolve it")
        row.update(gold=gold.summary(), gold_missing=gold.missing[:5], valid=valid, reason=reason,
                   tail=out[-1500:] if not gold.resolved else "")
    except Exception as e:
        row.update(valid=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
    row["seconds"] = round(time.time() - t0)
    with _lock:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        flag = "VALID  " if row.get("valid") else "invalid"
        print(f"{flag} {task.instance_id:40} {row.get('baseline', '')} -> {row.get('gold', row.get('error', ''))[:80]}"
              f"  ({row['seconds']}s) {row.get('reason', '')}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-gold-lines", type=int, default=20)
    ap.add_argument("--retry-invalid", action="store_true",
                    help="Validate previously invalid tasks again (e.g. after a harness fix)")
    a = ap.parse_args()

    done = set()
    if OUT.exists():
        rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
        if a.retry_invalid:
            rows = [r for r in rows if r.get("valid")]
            OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        done = {r["instance_id"] for r in rows}
    tasks = env.load_tasks()
    picked = [t for t in env.select(tasks, a.n + len(done), max_gold_lines=a.max_gold_lines)
              if t.instance_id not in done][:a.n]
    print(f"{len(tasks)} tasks in dataset; validating {len(picked)} (already done: {len(done)})\n", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        rows = list(pool.map(validate, picked))
    ok = [r for r in rows if r.get("valid")]
    print(f"\n{len(ok)}/{len(rows)} valid. Results in {OUT}")


if __name__ == "__main__":
    main()
