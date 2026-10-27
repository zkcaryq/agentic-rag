"""图片摘要节点：生成失败不能写假描述，也不能把降级算成成功。"""
from concurrent.futures import Future

import pytest

from app.import_process.agent.nodes import node_md_img as nid


class DeadVLM:
    """构造客户端就失败（没配 VLM key 的形态）"""

    def __init__(self, *a, **kw):
        raise RuntimeError("no vl key")


class EmptyVLM:
    def __init__(self, *a, **kw):
        pass

    def invoke(self, messages):
        EmptyVLM.called = True

        class R:
            content = "   "
        return R()


class BoomVLM:
    def __init__(self, *a, **kw):
        pass

    def invoke(self, messages):
        BoomVLM.called = True
        raise RuntimeError("429 rate limited")


def _summaries(monkeypatch, client_factory, targets):
    monkeypatch.setattr(nid, "get_llm_client", client_factory)
    monkeypatch.setattr(nid, "apply_api_rate_limit", lambda *a, **kw: None)
    monkeypatch.setattr(nid, "encode_image_to_base64", lambda p: "AAAA")
    monkeypatch.setattr(nid, "load_prompt", lambda *a, **kw: "prompt")
    return nid.step_3_generate_summaries("doc", targets)


TARGETS = [("a.jpg", "/tmp/a.jpg", ("上文", "下文"))]


def test_no_vlm_client_returns_no_summaries_instead_of_placeholder(monkeypatch):
    monkeypatch.setattr(nid.lm_config, "lv_model", "qwen-vl-max", raising=False)
    assert _summaries(monkeypatch, DeadVLM, TARGETS) == {}
    # 旧实现会返回 {"a.jpg": "图片描述"}，那串字会被索引进正文，检索命中后像是摘要


def test_empty_or_failed_generation_is_omitted(monkeypatch):
    """真调用了 invoke 才算这个用例成立——构造失败也会返回空 dict。"""
    for factory in (EmptyVLM, BoomVLM):
        factory.called = False
        assert _summaries(monkeypatch, factory, TARGETS) == {}
        assert factory.called is True


@pytest.mark.parametrize(
    "markdown, summaries, expected",
    [
        # 有摘要用摘要
        ("![原图题](images/a.jpg)", {"a.jpg": "设备正面面板示意"}, "![设备正面面板示意](/media/k/a.jpg)"),
        # 没摘要时保留原文图题，不编造
        ("![原图题](images/a.jpg)", {}, "![原图题](/media/k/a.jpg)"),
        # 原文也没有图题才退化成中性的"图片"
        ("![](images/a.jpg)", {}, "![图片](/media/k/a.jpg)"),
    ],
)
def test_replacement_never_invents_a_description(monkeypatch, markdown, summaries, expected):
    from app.clients import storage

    monkeypatch.setattr(storage, "is_enabled", lambda: True)
    monkeypatch.setattr(storage, "build_storage_key", lambda stem, name, **kw: f"k/{name}")
    monkeypatch.setattr(storage, "upload_file", lambda path, key: True)

    out = nid.step_4_upload_and_replace("doc", TARGETS, summaries, markdown)
    assert out == expected
    assert "图片描述" not in out


def test_missing_summary_marks_partial_not_success(tmp_path, monkeypatch):
    """摘要缺条 → failed_stage=image_summary → 上层据此记 partial_success。"""
    (tmp_path / "images").mkdir()
    image = tmp_path / "images" / "a.jpg"
    image.write_bytes(b"AAAA")
    md = tmp_path / "doc.md"
    md.write_text("操作步骤见 ![图题](images/a.jpg)", encoding="utf-8")

    monkeypatch.setattr(nid, "step_3_generate_summaries", lambda stem, targets: {})
    monkeypatch.setattr(nid, "step_4_upload_and_replace", lambda *a, **kw: "替换后的正文")

    state = {"task_id": "t-img", "md_path": str(md), "md_content": "", "file_title": "文档"}
    nid.node_md_img(state)

    assert state["failed_stage"] == "image_summary"


def test_summary_rate_comes_from_config_not_a_magic_number(monkeypatch):
    """批量导入最慢的一段是图片摘要，速率必须可按额度调（配置驱动，不写死在签名里）。"""
    calls = []
    monkeypatch.setattr(nid.lm_config, "vlm_summary_rpm", 42)
    monkeypatch.setattr(nid, "apply_api_rate_limit",
                        lambda times, rpm, window_seconds=60: calls.append(rpm))
    monkeypatch.setattr(nid, "encode_image_to_base64", lambda p: "AAAA")
    monkeypatch.setattr(nid, "load_prompt", lambda *a, **kw: "prompt")

    class OkVLM:
        def __init__(self, *a, **kw):
            pass

        def invoke(self, messages):
            class R:
                content = "硒鼓卡扣位置示意"
            return R()

    monkeypatch.setattr(nid, "get_llm_client", OkVLM)
    out = nid.step_3_generate_summaries("doc", TARGETS)

    assert out == {"a.jpg": "硒鼓卡扣位置示意"}
    assert calls == [42]
