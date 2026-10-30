"""
Student Lab - a page of the Fork app for students (Streamlit multipage: it shows up in the
sidebar of graph_app.py, and on the hosted app at /Student_Lab).

    Describe a program -> Fork writes the tests first, then the code, and shows each test passing.
    Fix my code        -> paste code + what it should do -> the bug is fixed with a before/after diff,
                          or you're told your code is already correct.

Everything runs in Nebius sandboxes. On the hosted app a class access code, a per-run cap,
the shared daily budget and a few run slots keep spending bounded. Logic lives in student.py.
"""

import os
import threading
import time

import streamlit as st


def _secrets_to_env() -> None:
    """Same as graph_app.py: on Streamlit Community Cloud, settings live in the Secrets box.
    Needed here too, because opening this page directly doesn't run the main page."""
    try:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)) and k not in os.environ:
                os.environ[k] = str(v)
    except Exception:
        pass


_secrets_to_env()

import agent_graph as ag  # noqa: E402
import hosting  # noqa: E402
import student  # noqa: E402

HOSTED = hosting.is_hosted()

st.set_page_config(page_title="Student Lab · Fork", page_icon="🎓", layout="wide")
ss = st.session_state
ss.setdefault("lab_job", None)
ss.setdefault("lab_try", None)
for key in ("lab_task", "lab_code", "lab_fix_task", "lab_expr"):
    ss.setdefault(key, "")

if HOSTED or os.getenv("ALLOW_LOCAL_SANDBOX") != "1":
    os.environ["SANDBOX"] = "nebius"   # student code never runs on the app's own machine


def set_task(text: str) -> None:
    ss.lab_task = text


def load_fix_example() -> None:
    ss.lab_code, ss.lab_fix_task = student.FIX_EXAMPLE_CODE, student.FIX_EXAMPLE_TASK


def worker(job: dict, kwargs: dict, release, ledger) -> None:
    try:
        job["state"] = ag.run(**kwargs, on_log=job["logs"].append, full=True)
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        if ledger is not None:   # unknown spend on a crash: count the whole cap to stay safe
            ledger.add(job["state"]["summary"]["cost_usd"] if job.get("state") else kwargs.get("max_usd") or 0)
        release()
        job["elapsed"] = time.time() - job["started"]
        job["done"] = True


job = ss.lab_job
running = bool(job and not job["done"])

# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    try:   # page links only resolve when graph_app.py is the entry point (always, when deployed)
        st.page_link("graph_app.py", label="Fork: Red to Green", icon="🍴")
        st.page_link("pages/1_Student_Lab.py", label="Student Lab", icon="🎓")
        st.divider()
    except Exception:
        pass
    needs_code = HOSTED and bool(os.getenv("STUDENT_ACCESS_CODE") or os.getenv("DEMO_ACCESS_CODE"))
    access = st.text_input("Class access code", type="password",
                           help="Your teacher gives you this code.") if needs_code else ""
    st.markdown(
        "**How it works**\n\n"
        "1. Your words become **pytest tests**.\n"
        "2. The tests are locked, so nothing can pass by changing them.\n"
        "3. Several attempts write the code **in parallel** in cloud sandboxes.\n"
        "4. The first attempt that passes every test wins.")
    st.caption(f"Python only · up to {student.MAX_CODE_LINES} lines · "
               f"each run is capped at ${hosting.student_run_cap():.2f} · sandbox: {os.environ['SANDBOX']}")

# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

st.title("🎓 Student Lab")
st.write("Describe a program, or paste code that doesn't work. Fork writes the **tests first**, then the "
         "code, runs everything in an isolated cloud sandbox, and shows you which tests pass.")

pick = st.radio("What do you want to do?", ["✍️ Describe a program", "🩹 Fix my code"],
                horizontal=True, disabled=running)
mode = "describe" if pick.startswith("✍️") else "fix"

if mode == "describe":
    st.caption("Try an example:")
    for col, (name, text) in zip(st.columns(len(student.EXAMPLES)), student.EXAMPLES.items()):
        col.button(name, on_click=set_task, args=(text,), disabled=running, use_container_width=True)
    task = st.text_area("What should the program do?", key="lab_task", height=130, disabled=running,
                        placeholder="e.g. Write fizzbuzz(n) that returns a list of strings from 1 to n ...",
                        help="Name the function, say what goes in and what comes out, and mention edge cases.")
    code = ""
else:
    st.button("Load an example with a bug", on_click=load_fix_example, disabled=running)
    left, right = st.columns([3, 2])
    code = left.text_area("Your code (Python)", key="lab_code", height=280, disabled=running,
                          placeholder="def average(marks):\n    ...")
    task = right.text_area("What is it supposed to do?", key="lab_fix_task", height=280, disabled=running,
                           placeholder="average(marks) returns the mean of the list, rounded to 2 decimals ...",
                           help="The tests are written from this, so be specific about inputs and outputs.")

if st.button("▶ Build it" if mode == "describe" else "▶ Fix it", type="primary", disabled=running):
    problem = student.check_inputs(mode, task, code) or ag.sandbox_ready()
    res = hosting.shared()
    cap = hosting.student_run_cap()
    if problem:
        st.error(problem)
    elif HOSTED and not hosting.student_access_ok(access):
        st.error("Enter the class access code in the sidebar.")
    elif HOSTED and res["ledger"].remaining() < cap:
        st.error("Today's class budget is used up. Please try again tomorrow.")
    elif not res["slots"].acquire(blocking=False):
        st.warning("Every run slot is busy with other students right now. Try again in a minute.")
    else:
        job = {"logs": [], "done": False, "state": None, "error": None, "started": time.time(),
               "mode": mode, "task": task, "original": code}
        ss.lab_job, ss.lab_try = job, None
        threading.Thread(target=worker, daemon=True,
                         args=(job, student.run_kwargs(mode, task, code, max_usd=cap),
                               res["slots"].release, res["ledger"] if HOSTED else None)).start()
        running = True

# ---------------------------------------------------------------------------
# live progress
# ---------------------------------------------------------------------------

if job and not job["done"]:
    box = st.empty()
    while not job["done"]:
        with box.container(border=True):
            st.markdown(f"⏳ **Working on it** · {time.time() - job['started']:.0f}s")
            for label in student.progress(job["logs"]):
                st.markdown(f"✅ {label}")
            if job["logs"]:
                st.caption(job["logs"][-1][:160])
        time.sleep(0.7)
    st.rerun()

# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

if job and job["done"] and job["mode"] == mode:   # results of the other mode stay hidden
    if job["error"]:
        st.error(f"The run stopped with an error: {job['error']}")
        st.stop()
    if "outcome" not in job:
        job["outcome"] = student.outcome(job["state"], job["mode"], job["original"])
    o = job["outcome"]
    say = {"green": st.success, "already_green": st.success, "red": st.warning}.get(o["status"], st.error)
    say(("✅ " if o["status"] in ("green", "already_green") else "") + student.headline(o))
    if o["status"] == "spec_failed":
        for p in o["spec_problems"]:
            st.caption(f"• {p}")
        st.stop()

    m = st.columns(4)
    m[0].metric("Tests passed", f"{o['passed']}/{o['total']}")
    m[1].metric("Time", f"{job.get('elapsed', 0):.0f} s")
    m[2].metric("Attempts", o["branches"] or "-")
    m[3].metric("Cost", f"${o['cost_usd']:.4f}")

    left, right = st.columns([3, 2])
    with left:
        if o["mode"] == "fix" and o["changed"]:
            st.subheader("The fix")
            t1, t2 = st.tabs(["Fixed code", "What changed"])
            t1.code(o["code"], language="python", line_numbers=True, wrap_lines=True)
            t2.code(o["diff"], language="diff")
        else:
            st.subheader("Your code" if o["mode"] == "fix" else "Your program")
            st.code(o["code"], language="python", line_numbers=True, wrap_lines=True)
        st.download_button("⬇️ Download solution.py", o["code"], file_name="solution.py", mime="text/x-python")
    with right:
        st.subheader("Tests")
        if o["spec_summary"]:
            st.caption(o["spec_summary"])
        for r in o["results"]:
            st.markdown(f"{'✅' if r['passed'] else '❌'} {student.pretty(r['test'])}")
            if r["reason"] and not r["passed"]:
                st.caption(f"↳ {r['reason'][:200]}")
        for rel, content in o["tests"].items():
            with st.expander(f"Test code: {rel}"):
                st.code(content, language="python")
                st.download_button("⬇️ Download tests", content, file_name=rel.split("/")[-1],
                                   mime="text/x-python", key=f"dl-{rel}")

    st.subheader("Try it yourself")
    st.caption("Call your code with your own input. It runs in the sandbox, not on this page.")
    c1, c2 = st.columns([5, 1])
    expr = c1.text_input("Python expression", key="lab_expr", label_visibility="collapsed",
                         placeholder=o["interface"].splitlines()[0][:80] if o["interface"] else "my_function(42)")
    if c2.button("Run", use_container_width=True):
        if HOSTED and not hosting.student_access_ok(access):
            ss.lab_try = (expr, False, "Enter the class access code in the sidebar.")
        else:
            ss.lab_try = (expr, *student.try_it(o["workdir"], expr))
    if ss.lab_try:
        e, ok, out = ss.lab_try
        if ok:
            st.code(f">>> {e}\n{out}", language="python")
        else:
            st.error(out)

    with st.expander("Behind the scenes"):
        st.write(f"{o['rounds']} round(s), {o['branches']} attempt(s), {o['tokens']:,} tokens.")
        if o["strategy"]:
            st.caption(f"Winning approach: {o['strategy']}")
        st.code(ag.tail(o["output"], 4000), language="text")
        st.code("\n".join(job["logs"]), language="text")
