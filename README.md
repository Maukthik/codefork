# Project Fork

An autonomous coding agent that plans, writes, tests, and fixes code on its own —
and (coming soon) forks its execution environment to try several implementation
strategies in parallel, keeping whichever passes the tests.

Built for the **Nebius x NVIDIA Global AI Hackathon 2026** (Coding and Agentic Engineering track).

## Status: Day 1

- [x] Connected to NVIDIA Nemotron via Nebius Token Factory
- [x] Tool-using agent loop: write file, read file, run command
- [x] Self-correction: runs its own code and fixes errors
- [x] Step limit and JSONL logging of every action
- [ ] Docker sandbox
- [ ] Planner and Reflector
- [ ] Parallel branching on Nebius Sandboxes
- [ ] Fine-tuned Nemotron executor
- [ ] Web interface
- [ ] Benchmark

## How it uses Nebius and NVIDIA

All model calls go to an NVIDIA Nemotron model served on Nebius Token Factory
through its OpenAI-compatible API.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then put your key and model name in .env
python list_models.py
python agent.py
```

## License

MIT
