"""Tests for hosting.py and the hosted Streamlit app. No network, no API key."""

import io
import json
import urllib.error
import zipfile
from types import SimpleNamespace

import pytest

import hosting


def make_zip(files: dict, top="repo-main/") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(top + name, data)
    return buf.getvalue()


# --- repo sources -------------------------------------------------------------------

def test_demos_are_listed_and_copied_fresh(tmp_path):
    assert {"invoice", "bookstore"} <= set(hosting.demo_names())
    path = hosting.prepare_demo("bookstore", str(tmp_path))
    assert (tmp_path / "bookstore" / "bookstore" / "cart.py").exists()
    assert not (tmp_path / "bookstore" / "README.md").exists()     # the README lists the bugs
    assert path.endswith("bookstore")
    with pytest.raises(ValueError):
        hosting.prepare_demo("../../etc")


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/owner/repo", ("owner", "repo", None)),
    ("https://github.com/owner/repo.git", ("owner", "repo", None)),
    ("https://github.com/owner/my.repo/tree/dev", ("owner", "my.repo", "dev")),
    ("https://github.com/owner/repo/", ("owner", "repo", None)),
])
def test_parse_github_url(url, expected):
    assert hosting.parse_github_url(url) == expected


@pytest.mark.parametrize("bad", ["", "github.com/x", "https://gitlab.com/a/b", "https://github.com/a",
                                 "https://github.com/a/b; rm -rf /"])
def test_parse_github_url_rejects_other_input(bad):
    with pytest.raises(ValueError):
        hosting.parse_github_url(bad)


def test_extract_zip_strips_top_folder_and_skips_junk(tmp_path):
    data = make_zip({"calc.py": "x = 1\n", "tests/test_calc.py": "def test(): pass\n",
                     ".git/config": "x", "pkg/__pycache__/a.pyc": "x"})
    root = hosting.extract_zip(data, str(tmp_path / "r"))
    got = sorted(p.relative_to(root).as_posix() for p in __import__("pathlib").Path(root).rglob("*") if p.is_file())
    assert got == ["calc.py", "tests/test_calc.py"]


def test_extract_zip_refuses_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        hosting.extract_zip(make_zip({"../evil.py": "x"}), str(tmp_path / "r"))


def test_extract_zip_refuses_huge_repos(tmp_path, monkeypatch):
    monkeypatch.setattr(hosting, "MAX_FILES", 3)
    with pytest.raises(ValueError, match="files"):
        hosting.extract_zip(make_zip({f"f{i}.py": "" for i in range(5)}), str(tmp_path / "r"))


def fake_opener(responses):
    """responses: url-suffix -> bytes, or an int HTTP status to raise."""
    def opener(url, timeout=None):
        for suffix, value in responses.items():
            if url.endswith(suffix):
                if isinstance(value, int):
                    raise urllib.error.HTTPError(url, value, "err", {}, None)
                body = io.BytesIO(value)
                return SimpleNamespace(read=body.read, __enter__=lambda s: s, __exit__=lambda *a: None)
        raise AssertionError(url)
    class Ctx:
        def __init__(self, url, timeout=None):
            self.r = opener(url, timeout)
        def __enter__(self):
            return self.r
        def __exit__(self, *a):
            return False
    return Ctx


def test_fetch_github_falls_back_from_main_to_master(tmp_path):
    opener = fake_opener({"/heads/main": 404, "/heads/master": make_zip({"a.py": "x"}, "repo-master/")})
    root = hosting.fetch_github("https://github.com/o/repo", str(tmp_path), opener=opener)
    assert (tmp_path / "repo" / "a.py").read_text() == "x"
    assert root.endswith("repo")


def test_fetch_github_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(hosting, "MAX_ZIP_BYTES", 10)
    opener = fake_opener({"/heads/main": b"x" * 100})
    with pytest.raises(ValueError, match="larger"):
        hosting.fetch_github("https://github.com/o/repo", str(tmp_path), opener=opener)


def test_fetch_github_private_or_missing(tmp_path):
    opener = fake_opener({"/heads/main": 404, "/heads/master": 404})
    with pytest.raises(ValueError, match="public"):
        hosting.fetch_github("https://github.com/o/repo", str(tmp_path), opener=opener)


# --- spending -----------------------------------------------------------------------

def test_access_code(monkeypatch):
    monkeypatch.delenv("DEMO_ACCESS_CODE", raising=False)
    assert hosting.access_ok("")                        # no code configured: open
    monkeypatch.setenv("DEMO_ACCESS_CODE", "nemotron")
    assert hosting.access_ok(" nemotron ") and not hosting.access_ok("") and not hosting.access_ok("x")


def test_ledger_resets_each_day():
    day = {"d": 1}
    ledger = hosting.Ledger(1.0, today=lambda: day["d"])
    ledger.add(0.7)
    assert ledger.remaining() == pytest.approx(0.3)
    ledger.add(0.5)
    assert ledger.remaining() == 0
    day["d"] = 2
    assert ledger.remaining() == 1.0


# --- the Streamlit app (AppTest runs the script headless) -------------------------------

try:
    from streamlit.testing import v1 as streamlit_testing
except ImportError:  # app tests need streamlit; the hosting tests above don't
    streamlit_testing = None
needs_streamlit = pytest.mark.skipif(streamlit_testing is None, reason="streamlit not installed")


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    rec_dir = tmp_path / "recorded"
    rec_dir.mkdir()
    res = {"branch_id": "r1_b0", "round": 1, "strategy": "fix it", "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
           "passed": True, "cancelled": False, "stop_reason": "tests passed", "test_output": "1 passed",
           "diff_lines": 2, "changed": ["calc.py"], "patch": "--- a/calc.py\n+++ b/calc.py\n", "turns": 1,
           "tool_calls": 1, "input_tokens": 10, "output_tokens": 5, "usd": 0.001, "refused_writes": 1,
           "tampered": []}
    (rec_dir / "invoice_demo.json").write_text(json.dumps({
        "run_id": "x", "repo": "invoice", "task": "fix", "test_cmd": "pytest -q", "baseline_output": "1 failed",
        "logs": ["[baseline] tests RED", "[r1_b0] GREEN"], "winner": "r1_b0", "results": [res],
        "summary": {"status": "green", "rounds": 1, "cost_usd": 0.001, "tamper_attempts": 1,
                    "tokens_by_stage": {"executor": {"input": 10, "output": 5, "usd": 0.001}}}}))
    monkeypatch.setattr(hosting, "RECORDED", rec_dir)
    return rec_dir


def app(monkeypatch, **env):
    for k in ("HOSTED", "DEMO_ACCESS_CODE", "ALLOW_LOCAL_SANDBOX"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "test")
    monkeypatch.setitem(__import__("sys").modules, "contree_sdk", SimpleNamespace())
    return streamlit_testing.AppTest.from_file("graph_app.py", default_timeout=60)


@needs_streamlit
def test_hosted_app_opens_on_replay_and_plays_it(recorded, monkeypatch):
    at = app(monkeypatch, HOSTED="1").run()
    assert not at.exception
    assert at.sidebar.radio[0].value.startswith("▶️")
    at.select_slider[0].set_value("instant")
    at.button[0].click().run()                                  # Play
    assert not at.exception
    assert any("GREEN" in m.value for m in at.metric)
    assert any("tried to change the tests" in w.value for w in at.warning)


@needs_streamlit
def test_hosted_live_run_needs_the_access_code(recorded, monkeypatch):
    at = app(monkeypatch, HOSTED="1", DEMO_ACCESS_CODE="secret").run()
    at.sidebar.radio[0].set_value("⚡ Run live").run()
    assert not at.exception
    run = next(b for b in at.button if b.label == "Run agent")
    run.click().run()
    assert any("access code" in e.value for e in at.error)
    assert ss_job(at) is None                                   # nothing started, nothing spent


def ss_job(at):
    return at.session_state["job"] if "job" in at.session_state else None


@needs_streamlit
def test_local_app_still_shows_repo_path(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    at = app(monkeypatch).run()
    assert not at.exception
    path = next(t for t in at.sidebar.text_input if t.label == "Repo path")
    path.set_value(str(tmp_path)).run()
    assert not at.exception and not at.error


@needs_streamlit
def test_streamlit_cloud_secrets_become_environment(recorded, monkeypatch):
    at = app(monkeypatch)
    for k in ("HOSTED", "NEBIUS_MODEL"):        # make monkeypatch restore these after the test
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    at.secrets["HOSTED"] = "1"
    at.secrets["NEBIUS_MODEL"] = "nvidia/nemotron-3-super-120b-a12b"
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value.startswith("▶️")           # hosted mode came from secrets
    import os
    assert os.environ["NEBIUS_MODEL"] == "nvidia/nemotron-3-super-120b-a12b"
