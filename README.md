<div align="center">

# 🍴 Project Fork

**An autonomous coding agent that plans, writes, tests and fixes its own code, all inside a sandbox.**

*Next step: fork the sandbox to try several strategies in parallel and keep the one that passes the tests.*

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Nebius](https://img.shields.io/badge/Nebius-Token%20Factory-7B61FF)
![NVIDIA Nemotron](https://img.shields.io/badge/NVIDIA-Nemotron-76B900?logo=nvidia&logoColor=white)
![Docker](https://img.shields.io/badge/sandbox-Docker%20%7C%20Nebius-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

*Built for the **Coding and Agentic Engineering** hackathon track.*

</div>

---

## ✨ What it does

Give Project Fork a task in plain English, for example *"write a function that parses ISO dates, with pytest tests"*. The agent then:

1. 🧭 **Plans:** breaks the task into small steps, each with a success check.
2. 🛠️ **Executes:** writes files and runs commands inside an isolated sandbox.
3. 🔍 **Reflects:** when a command fails, it diagnoses why and remembers which approaches already failed.
4. 🔁 **Recovers:** retries with a different approach and never repeats a known-bad fix.
5. 📦 **Delivers:** copies the finished files into your local `workspace/` folder and logs every action.

## 🧠 How it works

```mermaid
flowchart LR
    T([📝 Task]) --> P[🧭 Planner]
    P -->|up to 6 steps| E[🛠️ Executor]
    E -->|write_file / read_file / run_command| S[(📦 Sandbox<br/>Docker or Nebius)]
    S -->|output| E
    E -->|command failed<br/>after a code change| R[🔍 Reflector]
    R -->|diagnosis + failed approaches| E
    E -->|all steps done| W([✅ workspace/ + JSONL log])
```

| Stage | Role | Output |
|---|---|---|
| **Planner** | Splits the task into at most 6 steps | A `Plan` validated with Pydantic (falls back safely if the model returns garbage) |
| **Executor** | Tool-calling loop over the plan | Files and command runs in the sandbox |
| **Reflector** | Diagnoses failed commands | Category, next action and a running list of failed approaches |

**Guardrails**
- 🔒 Every file path is checked so the agent can't write outside its workspace.
- ⏱️ Hard limit of **25 model turns** per task.
- 🧮 Token usage is counted separately for planner, reflector and executor.
- 💤 Re-running a failed command without changing any code does **not** call the reflector, which saves tokens.

## 📦 Sandboxes

Both backends expose the same interface (`start`, `write_file`, `read_file`, `run`, `finish`), so the agent doesn't care which one is running.

| Backend | `SANDBOX=` | Best for |
|---|---|---|
| 🐳 **Docker** *(default)* | `docker` | Free, fast, offline local development. Python 3.12 + pytest with resource limits. |
| ☁️ **Nebius Sandboxes** | `nebius` | VM-isolated cloud runs. Every command is saved as a snapshot you can branch from, which is the basis for parallel forking. |

## 🚀 Quickstart

```bash
# 1. Set up the environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Build the local sandbox image (Docker backend)
docker build -t fork-sandbox sandbox

# 3. Configure your keys and models
cp .env.example .env               # Windows: copy .env.example .env
python list_models.py              # see which models your key can use
python check_connection.py         # quick model round-trip

# 4. Run it
python agent.py                    # terminal
streamlit run app.py               # web UI
```

Generated code is written to `workspace/`, and run logs go to `logs/run_<timestamp>.jsonl`.

## ⚙️ Configuration

All settings live in `.env`.

<details open>
<summary><b>Model provider</b></summary>

| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `ollama` | `ollama` (local), `openrouter`, or `nebius` |
| `NEBIUS_API_KEY` | – | Nebius Token Factory key |
| `NEBIUS_MODEL` | – | Main model (an NVIDIA Nemotron model for the hackathon) |
| `NEBIUS_BASE_URL` | Token Factory UK-South | Override the API endpoint |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` | – | For `PROVIDER=openrouter` |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `localhost:11434` / `llama3.1:8b-instruct-q4_K_M` | For `PROVIDER=ollama` |

</details>

<details>
<summary><b>Per-stage models</b> (optional)</summary>

Each stage can use its own model. Any stage left unset uses the main model.

| Variable | Stage |
|---|---|
| `PLANNER_MODEL` | Planner |
| `REFLECTOR_MODEL` | Reflector |
| `EXECUTOR_MODEL` | Executor |

</details>

<details>
<summary><b>Sandbox</b></summary>

| Variable | Default | Description |
|---|---|---|
| `SANDBOX` | `docker` | `docker` or `nebius` |
| `SANDBOX_IMAGE` | `fork-sandbox` | Docker image built from `sandbox/` |
| `NEBIUS_SANDBOX_IMAGE` | `python:3.12-slim` | Image for Nebius Sandboxes |
| `NEBIUS_PROJECT_ID` | – | Required when `SANDBOX=nebius` |

</details>

## 🧪 Tests

```bash
pytest
```

The test suite scripts the model's replies, so it needs **no API key and no network**. It covers the planner, the reflector, the recovery loop, the turn limit, per-stage model routing, token counting and the Nebius backend.

## 🔎 Inspecting a run

```bash
python summarize_log.py                              # latest run
python summarize_log.py logs/run_20260923_131622.jsonl
```

This prints the task, the models used, the plan, each tool call (ok / FAILED) and every reflection.

## 🗂️ Project layout

```
.
├── agent.py              # Planner → Executor → Reflector loop
├── sandboxes.py          # Docker and Nebius sandbox backends
├── app.py                # Streamlit web UI
├── sandbox/dockerfile    # Sandbox image: Python 3.12 + pytest
├── test_agent.py         # Offline test suite
├── summarize_log.py      # Readable summary of a JSONL run log
├── list_models.py        # List models available to your key
├── check_connection.py   # Model connectivity check
└── check_sandbox.py      # Nebius sandbox connectivity check
```

## 🗺️ Roadmap

- [x] Tool-using agent loop (write file, read file, run command)
- [x] Self-correction: runs its own code and fixes errors
- [x] Turn limit and JSONL logging of every action
- [x] Docker sandbox with pytest and resource limits
- [x] Planner and Reflector with structured outputs
- [x] Per-stage models (NVIDIA Nemotron on Nebius Token Factory)
- [x] Nebius Sandboxes backend with snapshots
- [x] Streamlit web interface
- [ ] 🍴 **Parallel branching:** fork Nebius snapshots, try several strategies at once, keep the one that passes
- [ ] Fine-tuned Nemotron executor
- [ ] Benchmark

## 📄 License

[MIT](LICENSE)
