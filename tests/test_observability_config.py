"""LangSmith 配置别名解析的单测（无需密钥、不发网络请求）。"""
import importlib
import os

import dotenv
import pytest

from app.conf import infra_config

_ALIASES = (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_PROJECT",
)


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """跑完把改动还原，避免 apply_langsmith_env 写进 os.environ 后污染别的用例。

    还要掐掉 load_dotenv：reload 时它会重新执行，把 .env 里的 LANGCHAIN_TRACING_V2
    补回 os.environ，用例结果就会跟着本地 .env 的内容飘。
    """
    saved = {name: os.environ.get(name) for name in _ALIASES}
    for name in _ALIASES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _reload(monkeypatch, env: dict):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(infra_config)


def test_legacy_var_enables_tracing(monkeypatch):
    cfg = _reload(monkeypatch, {"LANGCHAIN_TRACING_V2": "true"})
    assert cfg.infra_config_langsmith.enabled is True


def test_new_style_var_alone_enables_tracing(monkeypatch):
    # SDK 换成只认 LANGSMITH_TRACING 时，用户的配置不该失效
    cfg = _reload(monkeypatch, {"LANGSMITH_TRACING": "true"})
    assert cfg.infra_config_langsmith.enabled is True


def test_disabled_by_default(monkeypatch):
    cfg = _reload(monkeypatch, {})
    assert cfg.infra_config_langsmith.enabled is False
    assert cfg.apply_langsmith_env() is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_apply_writes_every_alias(monkeypatch):
    cfg = _reload(monkeypatch, {"LANGSMITH_TRACING": "true", "LANGCHAIN_PROJECT": "rag-prod"})
    assert cfg.apply_langsmith_env() is True
    for name in cfg.LangSmithConfig.TRACING_ALIASES:
        assert os.environ[name] == "true"
    # 项目名两个别名都要一致，否则控制台会出现两个同名不同项目的 trace 分组
    assert os.environ["LANGCHAIN_PROJECT"] == "rag-prod"
    assert os.environ["LANGSMITH_PROJECT"] == "rag-prod"


def test_project_falls_back_when_no_alias(monkeypatch):
    cfg = _reload(monkeypatch, {"LANGSMITH_TRACING": "true"})
    assert cfg.infra_config_langsmith.project == "agentic-rag"
