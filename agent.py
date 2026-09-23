"""Project Fork - Autonomous Coding Agent (with Planner + Reflector).

Flow:  task -> PLANNER makes steps -> agent loop works step by step
       -> if a command fails, REFLECTOR diagnoses it -> agent retries
       smarter, never repeating an approach that already failed.

Run:  python agent.py
"""
import datetime
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

# ===============================================================
# Dynamic Switch: Ollama, OpenRouter, or Nebius   (unchanged)
# ===============================================================
PROVIDER = os.environ.get("PROVIDER", "ollama").lower()

if PROVIDER == "ollama":
    print("Running in COLLEGE MODE (Local Ollama Engine)")
    client = OpenAI(
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

elif PROVIDER == "openrouter":
    print("Running in CLOUD MODE (OpenRouter Engine)")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    MODEL = os.environ["OPENROUTER_MODEL"]

else:
    print("Running in HACKATHON MODE (Cloud Nebius Engine)")
    client = OpenAI(
        base_url="https://api.tokenfactory.uk-south1.nebius.com/v1",
        api_key=os.environ["NEBIUS_API_KEY"],
    )
    MODEL = os.environ["NEBIUS_MODEL"]   # must be an NVIDIA (Nemotron) model for the hackathon

# ===============================================================

SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "fork-sandbox")  # built from sandbox/Dockerfile
MAX_STEPS = 25            # total model turns across ALL plan steps (planning adds turns)
MAX_PLAN_STEPS = 6        # planner may not create more than this
WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.jsonl"


# ---------------------------------------------------------------
# File system safety helpers   (unchanged)
# ---------------------------------------------------------------
def safe_path(relative_path: str) -> Path:
    """Only allow files inside the workspace folder."""
    path = (WORKSPACE / relative_path).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError("Path is outside the workspace folder")
    return path


# ---------------------------------------------------------------
# The three tools   (unchanged)
# ---------------------------------------------------------------
def read_file(path: str) -> str:
    p = safe_path(path)
    if not p.exists():
        return f"ERROR: {path} does not exist"
    text = p.read_text(encoding="utf-8")
    if len(text) > 4000:
        return text[:4000] + "\n... [file truncated]"
    return text


def write_file(path: str, content: str) -> str:
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


def run_command(command: str = "", cmd: str = "") -> str:
    actual_command = command or cmd
    cmd_str = actual_command.strip()
    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",            # no internet inside the sandbox
        "--memory", "512m",             # memory cap
        "--cpus", "1",                  # CPU cap
        "-v", f"{WORKSPACE.absolute()}:/workspace",
        "-w", "/workspace",
        SANDBOX_IMAGE,
        "sh", "-c", cmd_str,
    ]
    try:
        result = subprocess.run(
            docker_cmd, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30 seconds"
    return (
        f"exit_code: {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )


FUNCTIONS = {"read_file": read_file, "write_file": write_file, "run_command": run_command}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace. Use relative paths such as 'main.py'.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative path inside the workspace"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file in the workspace with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the workspace"},
                    "content": {"type": "string", "description": "The full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": ("Run a Python command or shell script in the workspace using a Docker sandbox. "
                            "Returns exit code, stdout and stderr. Times out after 30 seconds."),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The command to run (e.g. 'python main.py')"}},
                "required": ["command"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a coding agent. You complete programming tasks by calling tools.
You will be given a PLAN and told which step to work on. Work only on the current step.
Work in small steps: write the code, run it, read the output, and fix any problems.
Always run your code to verify the step's success check before finishing the step.
Use only relative file paths.
The sandbox has Python 3.12 and pytest, but NO internet, so you cannot pip install anything.
CRITICAL: You ONLY have access to the following tools: `read_file`, `write_file`, and `run_command`. Do not attempt to use or invent any other tools.
When using `run_command`, ensure the parameter name is 'command'.
If you receive a REFLECTION message, follow its guidance and never repeat a listed failed approach.
When the CURRENT STEP is done and verified, reply with a one-line summary and do not call any tools."""


# ---------------------------------------------------------------
# Logging   (unchanged)
# ---------------------------------------------------------------
def log(event: dict) -> None:
    event["time"] = datetime.datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def short(text: str, n: int = 300) -> str:
    text = str(text).replace("\n", " | ")
    return text if len(text) <= n else text[:n] + "..."


# ===============================================================
# NEW: shared helper - ask the model for JSON and parse it safely
# ===============================================================
def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply.
    Handles <think>...</think> blocks and ```json fences that models often add."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("```json", "").replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def ask_json(system: str, user: str, schema: type[BaseModel]):
    """Call the model WITHOUT tools, expect JSON matching `schema`.
    Retries once. Returns a validated object, or None if it keeps failing."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        reply = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0, max_tokens=2000,
        )
        text = reply.choices[0].message.content or ""
        try:
            return schema(**extract_json(text))
        except (ValueError, ValidationError, TypeError) as e:
            log({"event": "json_parse_failed", "schema": schema.__name__,
                 "attempt": attempt + 1, "error": str(e), "reply": text[:500]})
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "That was not valid JSON in the required shape. "
                                                     "Reply with ONLY the JSON object, nothing else."}]
    return None


# ===============================================================
# NEW: PLANNER
# ===============================================================
class Step(BaseModel):
    id: int
    description: str
    success_check: str


class Plan(BaseModel):
    steps: list[Step]


PLANNER_PROMPT = f"""You are the planning module of a coding agent.
Break the user's programming task into 2 to {MAX_PLAN_STEPS} ordered, concrete steps.
Each step must be small enough to finish with a few file writes and one test run,
and must have a success_check that can be verified by running a command.
Reply with ONLY this JSON, no other text:
{{"steps": [{{"id": 1, "description": "...", "success_check": "..."}}]}}"""


def make_plan(task: str) -> list[Step]:
    plan = ask_json(PLANNER_PROMPT, f"Task: {task}", Plan)
    if plan is None or not plan.steps:
        # Fallback: never block the agent just because planning failed
        steps = [Step(id=1, description=task, success_check="The code runs and the task is satisfied")]
    else:
        steps = plan.steps[:MAX_PLAN_STEPS]
    log({"event": "plan", "steps": [s.model_dump() for s in steps], "fallback": plan is None})
    return steps


def step_message(steps: list[Step], idx: int) -> str:
    s = steps[idx]
    return f"CURRENT STEP {s.id} of {len(steps)}: {s.description}\nSuccess check: {s.success_check}"


# ===============================================================
# NEW: REFLECTOR
# ===============================================================
class Reflection(BaseModel):
    diagnosis: str
    category: Literal["syntax", "logic", "missing_dependency", "wrong_approach",
                      "environment", "bad_test", "other"]
    next_action: Literal["fix_code", "change_approach", "fix_test", "fix_environment"]


REFLECTOR_PROMPT = """You are the reflection module of a coding agent.
A command failed. Diagnose WHY in one or two sentences, classify it, and choose the next action.
Do not suggest any approach listed under "Already failed".
Reply with ONLY this JSON, no other text:
{"diagnosis": "...",
 "category": "syntax|logic|missing_dependency|wrong_approach|environment|bad_test|other",
 "next_action": "fix_code|change_approach|fix_test|fix_environment"}"""


def command_failed(tool_name: str, result: str) -> bool:
    if tool_name != "run_command":
        return False
    if result.startswith("ERROR"):
        return True
    m = re.search(r"exit_code: (-?\d+)", result)
    return bool(m) and int(m.group(1)) != 0


def reflect(step: Step, command: str, output: str, failed: list[str]) -> Reflection:
    user = (f"Current step: {step.description}\n"
            f"Command: {command}\n"
            f"Output (end):\n{output[-1500:]}\n"
            f"Already failed:\n" + ("\n".join(f"- {f}" for f in failed) or "- none"))
    r = ask_json(REFLECTOR_PROMPT, user, Reflection)
    if r is None:
        r = Reflection(diagnosis="Command failed; see the error output.",
                       category="other", next_action="fix_code")
    return r


# ===============================================================
# The agent loop (now plan-driven, with reflection)
# ===============================================================
def run_agent(task: str) -> dict:
    total_in = total_out = 0
    log({"event": "task", "task": task, "provider": PROVIDER, "model": MODEL})

    # ---- 1. PLAN ----
    steps = make_plan(task)
    print("\nPLAN:")
    for s in steps:
        print(f"  {s.id}. {s.description}   [check: {s.success_check}]")

    plan_text = "\n".join(f"{s.id}. {s.description}" for s in steps)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TASK: {task}\n\nPLAN:\n{plan_text}\n\n{step_message(steps, 0)}"},
    ]

    idx = 0                     # which plan step we are on
    failed_approaches = []      # reflector memory - never repeat these
    seen_errors = set()         # detects the exact same error happening twice
    file_version = 0            # goes up every time the agent writes a file
    last_failed_at = {}         # command -> file_version when it last failed
    skipped = 0                 # reflections skipped because nothing changed

    # ---- 2. EXECUTE step by step ----
    for turn in range(1, MAX_STEPS + 1):
        print(f"\n===== Turn {turn} | Step {steps[idx].id}/{len(steps)} =====")
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
            temperature=0, max_tokens=4000,
        )
        msg = response.choices[0].message
        if response.usage:
            total_in += response.usage.prompt_tokens
            total_out += response.usage.completion_tokens

        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # No tool calls = the current STEP is finished
        if not msg.tool_calls:
            print(f"STEP {steps[idx].id} DONE: {short(msg.content, 200)}")
            log({"event": "step_done", "step": steps[idx].id, "summary": msg.content})
            idx += 1
            if idx >= len(steps):
                print("\nALL STEPS COMPLETE")
                log({"event": "finish", "turns": turn, "tokens_in": total_in, "tokens_out": total_out,
                     "failed_approaches": failed_approaches})
                print(f"Turns: {turn} | tokens in: {total_in} | tokens out: {total_out}")
                print(f"Log saved to: {LOG_FILE}")
                return {"status": "done", "turns": turn, "steps": len(steps),
                        "reflections": len(failed_approaches), "skipped_reflections": skipped,
                        "tokens_in": total_in, "tokens_out": total_out, "log": str(LOG_FILE)}
            messages.append({"role": "user", "content": step_message(steps, idx)})
            continue

        # Run every tool the model asked for; remember failures
        failures = []
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = FUNCTIONS[name](**args)
            except KeyError:
                args, result = {}, f"ERROR: unknown tool '{name}'"
            except Exception as e:
                args, result = {}, f"ERROR: {type(e).__name__}: {e}"

            print(f"TOOL  {name}({short(tc.function.arguments, 120)})")
            print(f"RESULT {short(result)}")
            log({"event": "tool", "turn": turn, "step": steps[idx].id, "tool": name,
                 "arguments": tc.function.arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

            if name == "write_file" and not str(result).startswith("ERROR"):
                file_version += 1
            if command_failed(name, result):
                failures.append((args.get("command") or args.get("cmd") or "", result))

        # ---- 3. REFLECT on the last failure this turn ----
        # (Must come AFTER all tool results - the API requires tool results
        #  to directly follow the assistant message that requested them.)
        if failures:
            command, output = failures[-1]

            # Same command failed before and no file changed since? A new diagnosis
            # would say the same thing - skip the model call and just nudge the agent.
            if last_failed_at.get(command) == file_version:
                skipped += 1
                print(f"SKIP REFLECT (nothing changed since '{short(command, 60)}' last failed)")
                log({"event": "reflection_skipped", "turn": turn, "step": steps[idx].id,
                     "command": command})
                messages.append({"role": "user", "content":
                    "You re-ran a command that already failed without changing any file. "
                    "Change the code (or try a different command) before running it again."})
                continue
            last_failed_at[command] = file_version

            r = reflect(steps[idx], command, output, failed_approaches)
            failed_approaches.append(f"Step {steps[idx].id}: {r.diagnosis}")

            signature = output[-200:]
            repeated = signature in seen_errors
            seen_errors.add(signature)

            note = (f"REFLECTION ({r.category} -> {r.next_action}): {r.diagnosis}\n"
                    f"Already failed, do NOT repeat:\n" + "\n".join(f"- {f}" for f in failed_approaches))
            if repeated:
                note += "\nThis EXACT error happened before - your last fix did not work. Try a different approach."
            print(f"REFLECT [{r.category} -> {r.next_action}] {short(r.diagnosis, 200)}")
            log({"event": "reflection", "turn": turn, "step": steps[idx].id,
                 **r.model_dump(), "repeated_error": repeated})
            messages.append({"role": "user", "content": note})

    print(f"\nStopped: reached the limit of {MAX_STEPS} turns.")
    log({"event": "max_steps", "tokens_in": total_in, "tokens_out": total_out})
    return {"status": "max_steps", "turns": MAX_STEPS, "steps": len(steps),
            "completed_steps": idx, "reflections": len(failed_approaches),
            "skipped_reflections": skipped, "tokens_in": total_in, "tokens_out": total_out,
            "log": str(LOG_FILE)}


if __name__ == "__main__":
    task = input("Enter a task for the agent: ").strip()
    if task:
        print(run_agent(task))