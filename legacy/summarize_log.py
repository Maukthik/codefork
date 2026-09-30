"""Summarise an agent run from its JSONL log.
Run:  python summarize_log.py            (latest log)
      python summarize_log.py logs/run_20260923_131622.jsonl
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(Path("logs").glob("run_*.jsonl"),
                                                        key=lambda p: p.stat().st_mtime)
events = [json.loads(line) for line in path.open(encoding="utf-8")]
print(f"Log: {path}\n")

for e in events:
    kind = e["event"]
    if kind == "task":
        print(f"TASK   {e['task']}")
        if "planner" in e:
            print(f"MODELS planner={e['planner']}  reflector={e['reflector']}  executor={e['executor']}\n")
        else:
            print(f"MODEL  {e.get('provider', '?')} / {e.get('model', '?')}\n")
    elif kind == "plan":
        print("PLAN" + ("  (FALLBACK - planner failed)" if e.get("fallback") else ""))
        for s in e["steps"]:
            print(f"  {s['id']}. {s['description']}")
        print()
    elif kind == "tool":
        failed = "exit_code: 0" not in e["result"] and e["tool"] == "run_command"
        print(f"  step {e.get('step', '?')}  {e['tool']:<12} {'FAILED' if failed else 'ok'}")
    elif kind == "reflection":
        rep = "  (REPEATED)" if e.get("repeated_error") else ""
        print(f"  -> REFLECT [{e['category']} / {e['next_action']}] {e['diagnosis']}{rep}")
    elif kind == "reflection_skipped":
        print(f"  -> SKIPPED REFLECT (nothing changed since '{e['command']}' last failed)")
    elif kind == "step_done":
        print(f"  STEP {e['step']} DONE\n")
    elif kind == "json_parse_failed":
        print(f"  !! {e['schema']} JSON parse failed (attempt {e['attempt']})")
    elif kind in ("finish", "max_steps"):
        status = "SUCCESS" if kind == "finish" else "STOPPED AT LIMIT"
        print(f"\nRESULT  {status}   tokens in/out: {e.get('tokens_in')}/{e.get('tokens_out')}")
        for role, u in (e.get("usage") or {}).items():
            print(f"  {role:<9} calls: {u['calls']:>2}  in: {u['in']:>6}  out: {u['out']:>5}")

tools = [e for e in events if e["event"] == "tool"]
refl = [e for e in events if e["event"] == "reflection"]
skip = [e for e in events if e["event"] == "reflection_skipped"]
print(f"TOTALS  tool calls: {len(tools)}   reflections: {len(refl)}   skipped: {len(skip)}")