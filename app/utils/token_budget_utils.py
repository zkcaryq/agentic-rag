"""上下文长度预算管控。

用字符数近似 token，而不是装 tiktoken：云端供应商的 tokenizer 拿不到，
用别家的分词器算出精确数字反而是假精度。中文字按 1 字 1 token 估，
其余字符按 4 字符 1 token 估，误差对"别把上下文撑爆"这个目的足够。
"""
import re

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

DEFAULT_BUDGET_TOKENS = 6000
# 每段上下文额外开销：编号、分隔符、prompt 结构
PER_DOC_OVERHEAD = 8


def estimate_tokens(text: str) -> int:
    """粗略估算一段文本的 token 数。"""
    if not text:
        return 0
    cjk_count = len(CJK_PATTERN.findall(text))
    other_count = len(text) - cjk_count
    return cjk_count + max(1, other_count // 4)


def build_context(
    docs: list[dict],
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> tuple[str, list[dict], dict]:
    """按预算拼装上下文。

    返回 (context_text, used_docs, info)：
      - used_docs 只包含真正进入上下文的文档，顺序与 [N] 编号一致；
      - info 记录预算、实际用量、丢弃条数，供日志与前端链路展示。
    """
    pieces: list[str] = []
    used_docs: list[dict] = []
    used_tokens = 0
    dropped = 0

    for doc in docs:
        content = (doc.get("content") or "").strip()
        if not content:
            dropped += 1
            continue

        cost = estimate_tokens(content) + PER_DOC_OVERHEAD
        if used_tokens + cost > budget_tokens:
            # 超预算就停，不跳着挑小的塞进来，避免上下文顺序与相关度顺序错乱
            dropped = len(docs) - len(used_docs)
            break

        used_tokens += cost
        used_docs.append(doc)
        pieces.append(f"[{len(used_docs)}] {content}")

    info = {
        "budget_tokens": budget_tokens,
        "used_tokens": used_tokens,
        "used_count": len(used_docs),
        "dropped_count": dropped,
        "truncated": dropped > 0,
    }
    return "\n\n".join(pieces), used_docs, info
