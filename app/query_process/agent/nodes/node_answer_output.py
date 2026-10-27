"""
答案生成节点
基于检索结果生成最终答案，并做四件事：
  1. 上下文按 token 预算截断（不再无脑拼 N 条）；
  2. 流式时逐 token 推送 DELTA，而不是整段一次性推；
  3. 生成后做引用后置校验，拦掉模型编造的 [N] 编号；
  4. 埋 TTFT / 生成总耗时指标，供 /metrics 与压测脚本读数。
"""
import sys
import time

from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.core import metrics
from app.query_process.agent.state import QueryGraphState
from app.utils.citation_utils import validate_and_fix
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.token_budget_utils import build_context, estimate_tokens
from app.lm.lm_utils import get_llm_client
from app.clients.history import save_chat_message

# tokens 一律用 estimate_tokens 估（CJK 感知），不取 usage_metadata：
# 混用"真实 token / 估算 token"两套口径会让 tokens_per_sec 没法横向比较。
GENERATE_ENDPOINT = "answer_generate"

NO_CONTEXT_ANSWER = "抱歉，没有找到相关的参考资料，请补充一下设备型号或换个问法。"


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成节点

    功能：
        1. 检查是否已有答案（反问场景）
        2. 取重排后的文档拼上下文（带预算）
        3. 流式生成并逐 token 推送
        4. 引用后置校验
        5. 保存到历史记录
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, func_name, is_stream)

    # 反问场景：前面节点已经给出答案，直接返回
    if state.get("answer"):
        logger.info(f"已有答案，跳过生成: {state['answer'][:50]}...")
        add_done_task(session_id, func_name, is_stream)
        return state

    query = state.get("rewritten_query", "") or state.get("original_query", "")
    docs = _select_context_docs(state)
    logger.info(f"进入答案生成的候选文档: {len(docs)} 条")

    if not docs:
        answer = NO_CONTEXT_ANSWER
        state["answer"] = answer
        state["references"] = []
        state["context_budget"] = {"budget_tokens": 0, "used_tokens": 0,
                                   "used_count": 0, "dropped_count": 0, "truncated": False}
        _push_result(state, answer, [], is_stream)
        _save_history(state, answer)
        set_task_result(session_id, "answer", answer)
        add_done_task(session_id, func_name, is_stream)
        return state

    # 预算管控：超预算的文档不进入上下文
    context, used_docs, budget_info = build_context(
        docs, lm_config.context_budget_tokens
    )
    state["context_budget"] = budget_info
    if budget_info["truncated"]:
        logger.warning(
            f"上下文预算命中：预算 {budget_info['budget_tokens']}，"
            f"实际用 {budget_info['used_tokens']}，丢弃 {budget_info['dropped_count']} 条"
        )

    references = _build_references(used_docs)
    prompt = _build_prompt(
        context=context,
        query=query,
        history_text=_format_history(state.get("history") or []),
        product_names=state.get("product_names") or [],
    )

    raw_answer, answer_mode = _generate(prompt, session_id, is_stream, used_docs)
    # 允许出现链接的范围：检索上下文原文 + 引用条目自带的 url（联网结果）。
    # 超出这个范围的 URL 一律视为模型编造，直接删（提示词已经不给它编的理由，
    # 但提示词挡不住所有情况，落库前这一道必须有）。
    grounded_text = context + "\n" + "\n".join(str(doc.get("url") or "") for doc in used_docs)
    answer, citation_info = validate_and_fix(raw_answer, len(references), grounded_text)
    if citation_info["has_invalid"]:
        logger.warning(
            f"拦截到不存在的引用编号 {citation_info['invalid_citations']}，已从答案中删除"
        )
    if citation_info["fabricated_urls"]:
        logger.warning(
            f"拦截到上下文里没有依据的图片/链接 {citation_info['fabricated_urls']}，已从答案中删除"
        )

    state["answer"] = answer
    state["answer_mode"] = answer_mode
    state["references"] = references
    set_task_result(session_id, "answer", answer)
    set_task_result(session_id, "citation_validation", str(citation_info))

    _push_result(state, answer, references, is_stream)
    _save_history(state, answer)

    add_done_task(session_id, func_name, is_stream)
    logger.info(f">>> 节点执行完成: {func_name}")
    return state


def _select_context_docs(state: QueryGraphState) -> list[dict]:
    """取答案生成用的文档：优先用重排结果，逐级降级。"""
    reranked = state.get("reranked_docs") or []
    if reranked:
        return reranked
    rrf = state.get("rrf_chunks") or []
    if rrf:
        logger.warning("无重排结果，降级使用 RRF 融合结果")
        return rrf
    logger.warning("无 RRF 结果，降级使用原始向量召回结果")
    return state.get("embedding_chunks") or []


def _build_references(used_docs: list[dict]) -> list[dict]:
    """把上下文里实际用到的文档转成引用结构（编号与正文 [N] 一致）。"""
    references = []
    for index, doc in enumerate(used_docs, start=1):
        references.append(
            {
                "index": index,
                "biz_chunk_id": doc.get("biz_chunk_id") or doc.get("id", ""),
                "title": doc.get("title", ""),
                "file_title": doc.get("file_title", ""),
                "page_no": doc.get("page_no", 0),
                # 联网来源只有带 url 才可核验；缺了它，用户看到的是一条点不开的引用
                "url": doc.get("url", ""),
                "rerank_score": doc.get("rerank_score"),
                "rrf_score": doc.get("rrf_score"),
                "source": doc.get("source", ""),
            }
        )
    return references


def _format_history(history: list[dict], limit: int = 6) -> str:
    """把最近几轮对话拼成文本，供 prompt 使用。"""
    if not history:
        return "（无）"
    lines = []
    for message in history[-limit:]:
        role = "用户" if message.get("role") == "user" else "助手"
        text = (message.get("text") or "").strip().replace("\n", " ")
        if text:
            lines.append(f"{role}: {text[:200]}")
    return "\n".join(lines) if lines else "（无）"


def _build_prompt(
    context: str,
    query: str,
    history_text: str,
    product_names: list[str],
) -> str:
    """从 prompts/answer_out.prompt 渲染，提示词与代码分离。"""
    from app.core.load_prompt import load_prompt

    return load_prompt(
        "answer_out",
        context=context,
        history=history_text,
        product_names="、".join(product_names) if product_names else "（未指定）",
        question=query,
    )


def _generate(
    prompt: str,
    session_id: str,
    is_stream: bool,
    used_docs: list[dict],
) -> tuple[str, str]:
    """调用大模型；无可用 LLM 时降级为抽取式答案。

    返回 (答案文本, 模式)，模式为 `llm` 或 `extractive`。
    抽取式不是假答案：它把检索到的高相关原文按引用编号摘出来，
    让没有 API key 的本地环境也能验证「检索 → 引用 → 流式」整条链路。

    耗时口径：t0 取"开始调用模型"的时刻，所以这里的 TTFT 是模型侧首 Token
    延迟，不含检索与重排；整轮问答的耗时由查询服务另行埋点（endpoint=query）。
    """
    try:
        llm = get_llm_client()
    except Exception as e:
        logger.warning(f"大模型不可用，降级为抽取式答案: {e}")
        answer = _extractive_answer(used_docs)
        if is_stream:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
        return answer, "extractive"

    t0 = time.monotonic()

    if not is_stream:
        response = llm.invoke(prompt)
        answer = (response.content or "").strip()
        metrics.record_and_export(
            GENERATE_ENDPOINT,
            total_ms=(time.monotonic() - t0) * 1000,
            tokens=estimate_tokens(answer),
        )
        return answer, "llm"

    parts: list[str] = []
    ttft_ms: float | None = None
    try:
        for chunk in llm.stream(prompt):
            text = getattr(chunk, "content", "") or ""
            if not text:
                continue
            if ttft_ms is None:
                ttft_ms = (time.monotonic() - t0) * 1000
            parts.append(text)
            push_to_session(session_id, SSEEvent.DELTA, {"delta": text})
    except Exception as e:
        logger.exception(f"流式生成中断，降级为抽取式答案: {e}")
        if not parts:
            answer = _extractive_answer(used_docs)
            push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
            return answer, "extractive"

    answer = "".join(parts).strip()
    metrics.record_and_export(
        GENERATE_ENDPOINT,
        total_ms=(time.monotonic() - t0) * 1000,
        ttft_ms=ttft_ms,
        tokens=estimate_tokens(answer),
    )
    return answer, "llm"


def _extractive_answer(used_docs: list[dict]) -> str:
    """抽取式答案：直接给出检索到的原文片段，并保留 [N] 引用编号。"""
    if not used_docs:
        return NO_CONTEXT_ANSWER

    lines = ["（当前环境未配置大模型，以下为检索到的原文摘录，用于验证检索与溯源链路）", ""]
    for index, doc in enumerate(used_docs, start=1):
        content = (doc.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        lines.append(f"[{index}] {content[:400]}")
    return "\n".join(lines)


def _push_result(
    state: QueryGraphState,
    answer: str,
    references: list[dict],
    is_stream: bool,
) -> None:
    """流式收尾：推 FINAL（带引用），随后由 SSE 层关闭连接。"""
    if not is_stream:
        return
    session_id = state.get("session_id", "")
    push_to_session(
        session_id,
        SSEEvent.FINAL,
        {
            "answer": answer,
            "status": "completed",
            "references": references,
            "context_budget": state.get("context_budget", {}),
        },
    )


def _save_history(state: QueryGraphState, answer: str) -> None:
    try:
        save_chat_message(
            session_id=state.get("session_id", ""),
            role="assistant",
            text=answer,
            rewritten_query="",
            product_names=state.get("product_names", []),
        )
    except Exception as e:
        logger.warning(f"保存对话历史失败: {e}")
