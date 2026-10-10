import pytest

from myrag import generate as gen_module
from myrag import pipeline
from myrag.config import (
    GenerateConfig,
    ProjectConfig,
    apply_profile,
    load_config,
    save_config,
)
from myrag.generate import _strip_think, generate_answer


def test_apply_profile_merges_over_defaults():
    cfg = ProjectConfig(name="p")
    cfg.profiles = {
        "alan": {
            "base_url": "https://example.org/v1",
            "model": "qwen-test",
            "api_key_env": "OTHER_KEY",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    }
    apply_profile(cfg, "alan")
    assert cfg.generate.model == "qwen-test"
    assert cfg.generate.base_url == "https://example.org/v1"
    assert cfg.generate.api_key_env == "OTHER_KEY"
    assert cfg.generate.temperature == 0.2
    assert cfg.generate.max_tokens == 2048
    assert cfg.generate.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_apply_profile_unknown_raises():
    cfg = ProjectConfig(name="p")
    with pytest.raises(ValueError):
        apply_profile(cfg, "nope")


def test_apply_profile_ignores_unknown_keys():
    cfg = ProjectConfig(name="p")
    cfg.profiles = {"x": {"model": "m", "not_a_field": 1}}
    apply_profile(cfg, "x")
    assert cfg.generate.model == "m"
    assert cfg.generate.temperature == 0.2
    assert cfg.generate.extra_body is None


def test_profiles_roundtrip_through_yaml(home):
    project = pipeline.create_project(home, "p", "d")
    cfg = load_config(project)
    cfg.profiles = {
        "alan": {
            "model": "m",
            "base_url": "https://example.org/v1",
            "api_key_env": "K",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    }
    save_config(project, cfg)
    cfg2 = load_config(project)
    assert cfg2.profiles["alan"]["model"] == "m"
    assert cfg2.profiles["alan"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


_THINK_OPEN = "<think" + ">"
_THINK_CLOSE = "</think" + ">"


def test_strip_think():
    assert _strip_think("hm" + _THINK_OPEN + "x" + _THINK_CLOSE + "data") == "hmdata"
    assert _strip_think("lead" + _THINK_OPEN + "reasoning" + _THINK_CLOSE + "answer") == "leadanswer"
    assert _strip_think(_THINK_OPEN + "truncated-no-close") == ""
    assert _strip_think("plain answer") == "plain answer"
    assert _strip_think("thinker keep") == "thinker keep"


class _FakeMessage:
    content = "final answer [1]"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResp:
    choices = [_FakeChoice()]
    usage = None


class _FakeCompletions:
    last_kwargs = {}

    def create(self, **kwargs):
        _FakeCompletions.last_kwargs = kwargs
        return _FakeResp()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, base_url=None, api_key=None):
        self.chat = _FakeChat()


def test_generate_answer_passes_extra_body_and_logs_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(gen_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("TESTMyRAGKey", "sk-x")
    gcfg = GenerateConfig(
        api_key_env="TESTMyRAGKey",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    usage_path = tmp_path / "usage.jsonl"
    answer, usage = generate_answer(
        "q?", [], gcfg, usage_path=usage_path, profile="alan"
    )
    assert answer == "final answer [1]"
    assert _FakeCompletions.last_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert usage["profile"] == "alan"
    assert '"profile": "alan"' in usage_path.read_text(encoding="utf-8")


def test_generate_answer_without_extra_body(monkeypatch, tmp_path):
    monkeypatch.setattr(gen_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("TESTMyRAGKey", "sk-x")
    gcfg = GenerateConfig(api_key_env="TESTMyRAGKey")
    generate_answer("q?", [], gcfg)
    assert "extra_body" not in _FakeCompletions.last_kwargs
