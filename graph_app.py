"""
graph_app.py - Streamlit interface for agent_graph.py (Red to Green).

    streamlit run graph_app.py              # on your machine: fix a repo folder, approve or reject
    HOSTED=1 streamlit run graph_app.py     # public demo: bundled demos or a GitHub URL,
                                            # access code + spend caps, patch download, replays

Watch the agent live: parallel branches, escalation across models, merged partial fixes,
the tamper guard and the cost of every step.
"""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import streamlit as st


def _secrets_to_env() -> None:
    """On Streamlit Community Cloud, settings live in the app's Secrets box (TOML). Copy the
    top-level ones into the environment, which is where the agent reads its config.
    Locally there's no secrets file and .env is used instead."""
    try:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)) and k not in os.environ:
                os.environ[k] = str(v)
    except Exception:
        pass


_secrets_to_env()

import agent_graph as ag  # noqa: E402  (reads the environment set above)
import hosting  # noqa: E402

HOSTED = hosting.is_hosted()

st.set_page_config(page_title="Fork: Red to Green", page_icon="🍴", layout="wide")
ss = st.session_state
ss.setdefault("job", None)        # the current/last live run
ss.setdefault("decision", None)   # result of approve / reject (local mode)
ss.setdefault("precheck", None)   # (exit code, output) of "Check tests"
ss.setdefault("repo_dir", None)   # hosted: prepared copy of the chosen repo
ss.setdefault("repo_key", None)   # hosted: which source repo_dir was prepared from
ss.setdefault("replay", None)     # replay mode: loaded record


@st.cache_resource
def shared() -> dict:
    """One per app process: only one live run at a time on the hosted demo, and a daily budget."""
    return {"lock": threading.Lock(),
            "ledger": hosting.Ledger(float(os.getenv("DAILY_BUDGET_USD", "3.00")))}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def git(repo: str, *args: str) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout + p.stderr).strip()


def worker(job: dict, kwargs: dict, release=None, ledger=None) -> None:
    try:
        job["state"] = ag.run(**kwargs, on_log=job["logs"].append, full=True)
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        if ledger is not None:   # unknown spend on a crash: count the whole cap to stay safe
            ledger.add(job["state"]["summary"]["cost_usd"] if job.get("state") else kwargs.get("max_usd") or 0)
        if release:
            release()
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


def render_results(summary: dict, results: list[dict], winner_id) -> None:
    """Metrics, every branch of every round, and the winning patch. Used for live runs and replays."""
    if summary["status"] == "already_green":
        st.success("Tests already pass. Nothing to fix.")
        return
    tokens = sum(v["input"] + v["output"] for v in summary["tokens_by_stage"].values())
    m = st.columns(5)
    m[0].metric("Result", summary["status"].upper())
    m[1].metric("Rounds", summary["rounds"])
    m[2].metric("Branches tried", len(results))
    m[3].metric("Total tokens", f"{tokens:,}")
    m[4].metric("Cost", f"${summary.get('cost_usd', 0):.4f}")
    if summary.get("tamper_attempts"):
        st.warning(f"🛡️ {summary['tamper_attempts']} branch(es) tried to change the tests. "
                   "Those edits were refused or reverted before judging.")
    with st.expander("Cost by stage"):
        st.table([{"stage": stage, "input tokens": f"{v['input']:,}", "output tokens": f"{v['output']:,}",
                   "USD": f"{v.get('usd', 0):.4f}"} for stage, v in summary["tokens_by_stage"].items()])

    st.subheader("Branches")
    for rnd in sorted({r["round"] for r in results}):
        row = [r for r in results if r["round"] == rnd]
        models = {(r.get("model") or "?").split("/")[-1] for r in row if r.get("model") != "merge (no LLM)"}
        st.markdown(f"**Round {rnd}** · {', '.join(sorted(models))}")
        for col, r in zip(st.columns(len(row)), row):
            with col.container(border=True):
                win = r["branch_id"] == winner_id
                badge = (":green[GREEN]" if r["passed"] else
                         ":gray[stopped]" if r.get("cancelled") else ":red[red]")
                st.markdown(f"**{r['branch_id']}** {badge}{'  🏆 winner' if win else ''}")
                st.caption(r["strategy"])
                st.write(f"{r['turns']} turns, {r['tool_calls']} tool calls, {r['diff_lines']} diff lines, "
                         f"{r['input_tokens'] + r['output_tokens']:,} tokens, ${r.get('usd') or 0:.4f}")
                st.caption(f"model: {(r.get('model') or '?').split('/')[-1]} | stop: {r.get('stop_reason')}")
                if r.get("refused_writes") or r.get("tampered"):
                    st.caption(f":orange[tamper guard: {r.get('refused_writes', 0)} refused, "
                               f"reverted {r.get('tampered') or []}]")
                with st.expander("Diff"):
                    st.code(r["patch"] or "(no changes)", language="diff")
                with st.expander("Test output"):
                    st.code(r["test_output"], language="text")


def offer_patch(w: dict, name: str) -> None:
    st.markdown(f"**The fix** from {w['branch_id']}: {w['strategy']}")
    st.code(w["patch"], language="diff")
    st.download_button("⬇️ Download fix.patch", w["patch"], file_name=f"{name}-fix.patch",
                       mime="text/x-diff")
    st.caption("Apply it to your copy with `git apply fix.patch`.")


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

st.title("🍴 Fork: Red to Green")
st.write("Give it a repo with failing tests. It tries several fixes **in parallel** in forked Nebius "
         "sandboxes, keeps the smallest one that turns the tests green, and escalates from "
         "**Nemotron 3 Nano → Super → Ultra** only when it has to. Test files are read-only: "
         "a tamper guard refuses and reverts any attempt to edit them.")
with st.expander("How it works"):
    st.markdown(
        "1. **Plan:** Nemotron 3 Ultra reads the code and failing output and proposes several different fixes.\n"
        "2. **Fork:** the repo becomes one Nebius sandbox checkpoint; each strategy runs in its own branch.\n"
        "3. **Execute:** Nemotron 3 Nano edits code; tests re-run in the sandbox after every edit.\n"
        "4. **Judge:** green only if the *original* tests pass. The first green branch stops the rest.\n"
        "5. **Merge / escalate:** partial fixes in different files are combined; otherwise the next round "
        "starts from the best partial fix on a bigger model.\n"
        "6. **Review:** you get the smallest passing diff, its cost, and every branch it tried.")

recordings = hosting.recorded_runs()
job = ss.job
running = bool(job and not job["done"])

# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    modes = (["▶️ Replay a recorded run"] if recordings else []) + ["⚡ Run live"]
    mode = st.radio("Mode", modes, index=0 if HOSTED else len(modes) - 1, disabled=running)

    if mode.startswith("⚡"):
        st.header("Run settings")
        if HOSTED:
            sources = [f"Demo: {n}" for n in hosting.demo_names()] + ["Public GitHub repo"]
            source = st.selectbox("Repo", sources, disabled=running)
            gh_url = ""
            if source == "Public GitHub repo":
                gh_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo",
                                       help=f"Public, Python, pytest tests, up to {hosting.MAX_FILES} files.")
        else:
            repo = st.text_input("Repo path", "workspace/invoice",
                                 help="A folder with failing tests. Create the demos with: python scripts/make_demo.py")
        task = st.text_area("Task", "Make the failing tests pass without changing the tests.", height=90)
        test_cmd = st.text_input("Test command", "pytest -q")
        branches = st.slider("Parallel branches", 1, 4 if HOSTED else 5, 3)
        rounds = st.slider("Max rounds", 1, 3 if HOSTED else 4, 3)
        ladder = st.text_input("Executor model per round", os.getenv("EXECUTOR_LADDER", "nano,super,ultra"),
                               help="Escalation: round 1 uses the first model, round 2 the second, and so on. "
                                    "Aliases: nano, super, ultra.")
        max_turns = st.slider("Max turns per branch", 3, 15 if HOSTED else 20, 10)
        first_green = st.checkbox("Stop at first green branch", value=True,
                                  help="Faster and cheaper. Untick to let every branch finish and compare diffs.")
        thinking = st.radio("Executor thinking", ["on", "low", "off"], horizontal=True,
                            index=["on", "low", "off"].index(os.getenv("EXECUTOR_THINKING", "off")),
                            help="Nemotron reasoning mode. off/low = faster and fewer tokens.")
        if HOSTED:
            cap = hosting.per_run_cap()
            max_usd = st.number_input("Spend cap for this run (USD)", 0.01, cap, min(0.15, cap), 0.01)
            code = st.text_input("Access code", type="password",
                                 help="Live runs spend real credits, so they need the code from the "
                                      "submission. Replays are free for everyone.") \
                if os.getenv("DEMO_ACCESS_CODE") else ""
            os.environ["SANDBOX"] = "nebius"
        else:
            max_usd = st.number_input("Spend cap for this run (USD)", 0.0, 20.0,
                                      float(os.getenv("MAX_USD_PER_RUN", "0.50")), 0.05)
            # Running agent-written commands on this machine is only for local development.
            if os.getenv("ALLOW_LOCAL_SANDBOX") == "1":
                sb = st.radio("Sandbox", ["nebius", "local"], horizontal=True,
                              index=1 if os.getenv("SANDBOX", "nebius") == "local" else 0)
                os.environ["SANDBOX"] = sb
            else:
                os.environ["SANDBOX"] = "nebius"
        st.caption(f"planner: {(ag.stage_model('planner') or '?').split('/')[-1]} · "
                   f"sandbox: {os.environ['SANDBOX']}")

# ---------------------------------------------------------------------------
# replay mode
# ---------------------------------------------------------------------------

if mode.startswith("▶️"):
    labels = {p: p.stem.replace("_", " ") for p in recordings}
    pick = st.selectbox("Recorded run", recordings, format_func=labels.get)
    rec = hosting.load_replay(pick)
    st.caption(f"Repo **{rec['repo']}** · task: {rec['task']} · recorded run {rec['run_id']} "
               "on Nebius Token Factory. Replaying costs nothing.")
    with st.expander("Failing tests before the fix"):
        st.code(rec.get("baseline_output", ""), language="text")
    c1, c2, _ = st.columns([1, 1, 4])
    speed = c2.select_slider("Speed", ["1x", "4x", "instant"], value="4x")
    if c1.button("▶️ Play", type="primary"):
        box = st.empty()
        delay = {"1x": 0.25, "4x": 0.06, "instant": 0}[speed]
        for i in range(1, len(rec["logs"]) + 1):
            if delay:
                box.code("\n".join(rec["logs"][max(0, i - 40):i]), language="text")
                time.sleep(delay)
        ss.replay = rec
    if ss.replay and ss.replay["run_id"] == rec["run_id"]:
        with st.expander(f"Full log ({len(rec['logs'])} lines)"):
            st.code("\n".join(rec["logs"]), language="text")
        render_results(rec["summary"], rec["results"], rec["winner"])
        w = next((r for r in rec["results"] if r["branch_id"] == rec["winner"]), None)
        if w:
            st.subheader("The fix")
            offer_patch(w, rec["repo"])
    st.stop()

# ---------------------------------------------------------------------------
# live mode: the repo
# ---------------------------------------------------------------------------

if HOSTED:
    key = gh_url.strip() if source == "Public GitHub repo" else source
    if not key:
        st.info("Paste a public GitHub repo URL in the sidebar.")
        st.stop()
    if ss.repo_key != key and not running:
        try:
            with st.spinner("Preparing repo..."):
                ss.repo_dir = (hosting.fetch_github(key) if source == "Public GitHub repo"
                               else hosting.prepare_demo(source.removeprefix("Demo: ")))
            ss.repo_key, ss.precheck = key, None
        except Exception as e:
            st.error(f"Couldn't prepare that repo: {e}")
            st.stop()
    repo_dir = ss.repo_dir
else:
    repo_dir = str(Path(repo).resolve())
    if not Path(repo_dir).is_dir():
        st.error(f"Folder not found: {repo_dir}")
        st.stop()

problem = ag.sandbox_ready()
if problem:
    st.error(problem)
    st.stop()

with st.expander("Repo as it is now", expanded=not job):
    files = ag.list_files(repo_dir)
    st.caption(f"{len(files)} files in {Path(repo_dir).name}")
    if st.button("Check tests", disabled=running):
        if HOSTED and not hosting.access_ok(code):
            st.error("Checking tests uses a Nebius sandbox, so it needs the access code too.")
        else:
            ss.precheck = ag.run_in_sandbox(repo_dir, test_cmd)
    if ss.precheck:
        c, out = ss.precheck
        (st.success if c == 0 else st.error)(f"Tests {'pass' if c == 0 else 'fail'} (exit {c})")
        st.code(ag.tail(out, 3000), language="text")
    show_files(repo_dir, [f for f in files if f.endswith(".py")][:6])

# ---------------------------------------------------------------------------
# live mode: run + live log
# ---------------------------------------------------------------------------

if st.button("Run agent", type="primary", disabled=running):
    res = shared()
    release, ledger = None, None
    if HOSTED:
        if not hosting.access_ok(code):
            st.error("Wrong or missing access code. Replays in the sidebar are free for everyone.")
            st.stop()
        if res["ledger"].remaining() < max_usd:
            st.error("Today's demo budget is used up. Replays in the sidebar still work.")
            st.stop()
        if not res["lock"].acquire(blocking=False):
            st.warning("Someone else's run is in progress (one at a time keeps costs predictable). "
                       "Try again in a minute, or watch a replay.")
            st.stop()
        release, ledger = res["lock"].release, res["ledger"]
        # fresh copy so every run starts from the original bug, not a previous visitor's state
        ss.repo_key = None
    job = {"logs": [], "done": False, "state": None, "error": None, "started": time.time()}
    ss.job, ss.decision = job, None
    kwargs = dict(repo=repo_dir, task=task, test_cmd=test_cmd, branches=branches, rounds=rounds,
                  max_turns=max_turns, first_green=first_green,
                  thinking=None if thinking == "on" else thinking, max_usd=max_usd or None,
                  ladder=ladder, make_branch=not HOSTED)
    if HOSTED:
        kwargs["work_root"] = str(Path(repo_dir).parent / "branches")
    threading.Thread(target=worker, args=(job, kwargs, release, ledger), daemon=True).start()
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
# live mode: results and decision
# ---------------------------------------------------------------------------

if job and job["done"]:
    if job["error"]:
        st.error(job["error"])
        st.stop()
    state = job["state"]
    w = state.get("winner")
    render_results(state["summary"], state["results"], w and w["branch_id"])
    if state["summary"]["status"] == "already_green":
        st.stop()

    st.subheader("Your decision" if not HOSTED else "The fix")
    if not w:
        st.warning("No branch turned the tests green. Check the test output above, or try more rounds or turns.")
        st.stop()
    if HOSTED:
        offer_patch(w, Path(state["repo_dir"]).name)
        st.markdown("**Code after the fix**")
        show_files(w["workdir"], w["changed"])
        st.stop()

    g = state["summary"].get("git") or {}
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
