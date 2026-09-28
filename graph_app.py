"""
graph_app.py - Streamlit interface for agent_graph.py (Red to Green).

    streamlit run graph_app.py

Flow: pick a repo -> check tests -> run the agent and watch it live ->
compare branches -> approve (merge the fix) or reject (delete the branch).
Nothing touches your repo until you press Approve.
"""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import streamlit as st

import agent_graph as ag

st.set_page_config(page_title="Fork: Red to Green", page_icon="🍴", layout="wide")
ss = st.session_state
ss.setdefault("job", None)        # the current/last agent run
ss.setdefault("decision", None)   # result of approve / reject
ss.setdefault("precheck", None)   # (exit code, output) of the "Check tests" button


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def git(repo: str, *args: str) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout + p.stderr).strip()


def worker(job: dict, kwargs: dict) -> None:
    try:
        job["state"] = ag.run(**kwargs, on_log=job["logs"].append, full=True)
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job["done"] = True


def approve(state: dict) -> dict:
    """Merge the review branch, or copy files if the repo isn't a git repo. Then re-run tests."""
    repo, w = state["repo_dir"], state["winner"]
    g = state["summary"].get("git") or {}
    if g.get("branch"):
        code, out = git(repo, "merge", "--no-edit", g["branch"])
        how = f"Merged `{g['branch']}`"
        if code != 0:
            return {"ok": False, "msg": f"Merge failed. Commit or stash your local changes first.\n\n{out}"}
    else:
        for rel in w["changed"]:
            src = Path(w["workdir"]) / rel
            if src.exists():
                dst = ag.safe_path(repo, rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        how = "Copied the fixed files into the repo (not a git repo, so no branch)"
    code, out = ag.run_in_sandbox(repo, state["test_cmd"])
    return {"ok": True, "msg": how, "test_code": code, "test_out": out}


def reject(state: dict) -> dict:
    g = state["summary"].get("git") or {}
    if g.get("branch"):
        code, out = git(state["repo_dir"], "branch", "-D", g["branch"])
        return {"ok": code == 0, "msg": f"Deleted `{g['branch']}`. Your repo is unchanged." if code == 0 else out}
    return {"ok": True, "msg": "Fix discarded. Your repo is unchanged."}


def show_files(root: str, files: list[str]) -> None:
    for rel in files:
        p = Path(root) / rel
        if p.exists():
            st.markdown(f"**{rel}**")
            st.code(p.read_text(encoding="utf-8", errors="replace"), language="python")


# ---------------------------------------------------------------------------
# sidebar: settings
# ---------------------------------------------------------------------------

job = ss.job
running = bool(job and not job["done"])

with st.sidebar:
    st.header("Run settings")
    repo = st.text_input("Repo path", "workspace/demo", help="A folder with failing tests")
    task = st.text_area("Task", "Make the failing tests pass without changing the tests.", height=90)
    test_cmd = st.text_input("Test command", "pytest -q")
    branches = st.slider("Parallel branches", 1, 5, 3)
    rounds = st.slider("Max rounds", 1, 3, 2)
    max_turns = st.slider("Max turns per branch", 3, 20, 10)
    sandbox = st.radio("Sandbox", ["nebius", "local"], horizontal=True,
                       index=1 if os.getenv("SANDBOX", "nebius") == "local" else 0,
                       help="nebius: Token Factory Sandboxes (default). "
                            "local: runs on your machine, only for trusted demo repos")
    st.caption(f"Provider: {os.getenv('PROVIDER', 'nebius')} | executor: {ag.stage_model('executor') or '?'}")
    os.environ["SANDBOX"] = sandbox

repo_dir = str(Path(repo).resolve())
repo_ok = Path(repo_dir).is_dir()
sandbox_problem = ag.sandbox_ready()

# ---------------------------------------------------------------------------
# header + step 1: the repo as it is now
# ---------------------------------------------------------------------------

st.title("🍴 Fork: Red to Green")
st.write("Give it a repo with failing tests. It tries several fixes in parallel, picks the smallest "
         "one that turns the tests green, and asks you before anything changes.")

if not repo_ok:
    st.error(f"Folder not found: {repo_dir}")
    st.stop()
if sandbox_problem:
    st.error(sandbox_problem)
    st.stop()

with st.expander("Repo as it is now", expanded=not job):
    files = ag.list_files(repo_dir)
    st.caption(f"{len(files)} files in {repo_dir}")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Check tests", disabled=running):
            ss.precheck = ag.run_in_sandbox(repo_dir, test_cmd)
    if ss.precheck:
        code, out = ss.precheck
        (st.success if code == 0 else st.error)(f"Tests {'pass' if code == 0 else 'fail'} (exit {code})")
        st.code(ag.tail(out, 3000), language="text")
    show_files(repo_dir, [f for f in files if f.endswith(".py")][:6])

# ---------------------------------------------------------------------------
# step 2: run + live log
# ---------------------------------------------------------------------------

if st.button("Run agent", type="primary", disabled=running):
    job = {"logs": [], "done": False, "state": None, "error": None, "started": time.time()}
    ss.job, ss.decision = job, None
    kwargs = dict(repo=repo_dir, task=task, test_cmd=test_cmd, branches=branches,
                  rounds=rounds, max_turns=max_turns)
    threading.Thread(target=worker, args=(job, kwargs), daemon=True).start()
    running = True

if job:
    st.subheader("Live log")
    box = st.empty()
    if not job["done"]:
        with st.spinner("Agent is working..."):
            while not job["done"]:
                box.code("\n".join(job["logs"][-60:]) or "starting...", language="text")
                time.sleep(0.5)
        st.rerun()
    elapsed = job.get("elapsed") or job.setdefault("elapsed", time.time() - job["started"])
    with box.container():
        with st.expander(f"Full log ({len(job['logs'])} lines, {elapsed:.0f}s)"):
            st.code("\n".join(job["logs"]), language="text")

# ---------------------------------------------------------------------------
# step 3: results
# ---------------------------------------------------------------------------

if job and job["done"]:
    if job["error"]:
        st.error(job["error"])
        st.stop()

    state = job["state"]
    summary = state["summary"]
    results = state["results"]
    w = state.get("winner")

    if summary["status"] == "already_green":
        st.success("Tests already pass. Nothing to fix.")
        st.stop()

    tokens = sum(v["input"] + v["output"] for v in summary["tokens_by_stage"].values())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Result", summary["status"].upper())
    m2.metric("Rounds", summary["rounds"])
    m3.metric("Branches tried", len(results))
    m4.metric("Total tokens", f"{tokens:,}")

    st.subheader("Branches")
    for rnd in sorted({r["round"] for r in results}):
        st.markdown(f"**Round {rnd}**")
        row = [r for r in results if r["round"] == rnd]
        for col, r in zip(st.columns(len(row)), row):
            with col.container(border=True):
                win = w and r["branch_id"] == w["branch_id"]
                badge = ":green[GREEN]" if r["passed"] else ":red[red]"
                st.markdown(f"**{r['branch_id']}** {badge}{'  🏆 winner' if win else ''}")
                st.caption(r["strategy"])
                st.write(f"{r['turns']} turns, {r['tool_calls']} tool calls, {r['diff_lines']} diff lines, "
                         f"{r['input_tokens'] + r['output_tokens']:,} tokens")
                with st.expander("Diff"):
                    st.code(r["patch"] or "(no changes)", language="diff")
                with st.expander("Test output"):
                    st.code(r["test_output"], language="text")

    # -----------------------------------------------------------------------
    # step 4: permission
    # -----------------------------------------------------------------------

    st.subheader("Your decision")
    if not w:
        st.warning("No branch turned the tests green. Your repo is unchanged. Check the test output "
                   "above, or try more rounds or turns.")
        st.stop()

    g = summary.get("git") or {}
    if g.get("branch"):
        st.info(f"The fix is waiting on branch `{g['branch']}` ({g['commit']}). Your files are untouched.")
    elif g.get("error"):
        st.info(f"No review branch ({g['error']}). Approving will copy the fixed files in directly.")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Proposed change** from {w['branch_id']}")
        st.caption(w["strategy"])
        st.code(w["patch"], language="diff")
    with right:
        st.markdown("**Code after the fix**")
        show_files(w["workdir"], w["changed"])

    if ss.decision is None:
        a, b, _ = st.columns([1, 1, 4])
        if a.button("✅ Approve and apply", type="primary"):
            ss.decision = {"kind": "approve", **approve(state)}
            st.rerun()
        if b.button("❌ Reject"):
            ss.decision = {"kind": "reject", **reject(state)}
            st.rerun()
    else:
        d = ss.decision
        if not d["ok"]:
            st.error(d["msg"])
        elif d["kind"] == "reject":
            st.info(d["msg"])
        else:
            st.success(d["msg"])
            passed = d["test_code"] == 0
            (st.success if passed else st.error)(
                f"Tests on your repo now {'pass' if passed else 'still fail'} (exit {d['test_code']})")
            st.code(ag.tail(d["test_out"], 3000), language="text")
            st.markdown("**Your repo now**")
            show_files(repo_dir, w["changed"])