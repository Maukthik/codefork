"""Project Fork - Autonomous Coding Agent.

A minimal coding agent: it receives a task, writes code into the
'workspace' folder, runs it, reads the result, and fixes errors,
until the task is done or MAX_STEPS is reached.

Run:  python agent.py
"""
import datetime
import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ===============================================================
# Dynamic Switch: Ollama, OpenRouter, or Nebius
# ===============================================================
PROVIDER = os.environ.get("PROVIDER", "ollama").lower()

if PROVIDER == "ollama":
    print("🏠 Running in COLLEGE MODE (Local Ollama Engine)")
    client = OpenAI(
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"),
        api_key="ollama",  
    )
    MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

elif PROVIDER == "openrouter":
    print("🔀 Running in CLOUD MODE (OpenRouter Engine)")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    MODEL = os.environ["OPENROUTER_MODEL"]

else:
    print("🚀 Running in HACKATHON MODE (Cloud Nebius Engine)")
    client = OpenAI(
        base_url="https://api.tokenfactory.uk-south1.nebius.com/v1",
        api_key=os.environ["NEBIUS_API_KEY"],
    )
    MODEL = os.environ.get("NEBIUS_MODEL", "deepseek-ai/DeepSeek-V4-Pro")

# ===============================================================

MAX_STEPS = 15                       
WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.jsonl"


# ---------------------------------------------------------------
# File system safety helpers 
# (Execution is now sandboxed via Docker)
# ---------------------------------------------------------------
def safe_path(relative_path: str) -> Path:
    """Only allow files inside the workspace folder."""
    path = (WORKSPACE / relative_path).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError("Path is outside the workspace folder")
    return path


# ---------------------------------------------------------------
# The three tools
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
    # Accepts both 'command' and 'cmd' to prevent model hallucinations
    actual_command = command or cmd 
    cmd_str = actual_command.strip()
    
    # We mount the local workspace folder to /workspace in the container
    docker_cmd = [
        "docker", "run", "--rm", 
        "-v", f"{WORKSPACE.absolute()}:/workspace", 
        "-w", "/workspace", 
        "python:3.14-rc-slim",
        "sh", "-c", cmd_str
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
Work in small steps: write the code, run it, read the output, and fix any problems.
Always run your code to verify it works before finishing.
Use only relative file paths.
CRITICAL: You ONLY have access to the following tools: `read_file`, `write_file`, and `run_command`. Do not attempt to use or invent any other tools.
When using `run_command`, ensure the parameter name is 'command'.
When the task is fully done and verified, reply with a short summary and do not call any tools."""


# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
def log(event: dict) -> None:
    event["time"] = datetime.datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def short(text: str, n: int = 300) -> str:
    text = str(text).replace("\n", " | ")
    return text if len(text) <= n else text[:n] + "..."


# ---------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------
def run_agent(task: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    log({"event": "task", "task": task, "model": MODEL})
    total_in = total_out = 0

    for step in range(1, MAX_STEPS + 1):
        print(f"\n===== Step {step} =========")
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
            temperature=0, max_tokens=4000,
        )
        msg = response.choices[0].message
        if response.usage:
            total_in += response.usage.prompt_tokens
            total_out += response.usage.completion_tokens

        # Save the model's reply into the conversation
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # No tool calls = the model says it is finished
        if not msg.tool_calls:
            print("AGENT FINISHED:")
            print(msg.content)
            log({"event": "finish", "summary": msg.content, "steps": step,
                 "tokens_in": total_in, "tokens_out": total_out})
            print(f"\nSteps: {step} | tokens in: {total_in} | tokens out: {total_out}")
            print(f"Log saved to: {LOG_FILE}")
            return

        # Run every tool the model asked for
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = FUNCTIONS[name](**args)
            except KeyError:
                result = f"ERROR: unknown tool '{name}'"
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"

            print(f"TOOL  {name}({short(tc.function.arguments, 120)})")
            print(f"RESULT {short(result)}")
            log({"event": "tool", "step": step, "tool": name,
                 "arguments": tc.function.arguments, "result": result})

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    print(f"\nStopped: reached the limit of {MAX_STEPS} steps.")
    log({"event": "max_steps", "tokens_in": total_in, "tokens_out": total_out})


if __name__ == "__main__":
    task = input("Enter a task for the agent: ").strip()
    if task:
        run_agent(task)