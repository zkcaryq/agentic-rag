"""
答案生成节点
基于检索结果生成最终答案，并做三件事：
  1. 上下文按 token 预算截断（不再无脑拼 N 条）；
  2. 流式时逐 token 推送 DELTA，而不是整段一次性推；
  3. 生成后做引用后置校验，拦掉模型编造的 [N] 编号。
"""
import sys

from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.citation_utils import validate_and_fix
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.token_budget_utils import build_context
from app.lm.lm_utils import get_llm_client
from app.clients.history import save_chat_message

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
    prompt = _build_prompt(context, query)

    raw_answer = _generate(prompt, session_id, is_stream)
    answer, citation_info = validate_and_fix(raw_answer, len(references))
    if citation_info["has_invalid"]:
        logger.warning(
            f"拦截到不存在的引用编号 {citation_info['invalid_citations']}，已从答案中删除"
        )

    state["answer"] = answer
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
                "rerank_score": doc.get("rerank_score"),
                "rrf_score": doc.get("rrf_score"),
                "source": doc.get("source", ""),
            }
        )
    return references


def _build_prompt(context: str, query: str) -> str:
    return f"""你是企业知识库的技术支持助手。请严格依据下面的参考资料回答问题。

【参考资料】
{context}

【用户问题】
{query}

【要求】
1. 只依据参考资料回答，不要编造资料里没有的事实
2. 每个关键结论后面用 [编号] 标注出处，编号必须是参考资料里出现过的
3. 资料不足时明确说明"资料中未提及"，不要猜测
4. 用中文回答，简洁专业

【回答】"""


def _generate(prompt: str, session_id: str, is_stream: bool) -> str:
    """调用大模型。流式时逐 token 推送，非流式一次性返回。"""
    llm = get_llm_client()

    if not is_stream:
        response = llm.invoke(prompt)
        return (response.content or "").strip()

    parts: list[str] = []
    try:
        for chunk in llm.stream(prompt):
            text = getattr(chunk, "content", "") or ""
            if not text:
                continue
            parts.append(text)
            push_to_session(session_id, SSEEvent.DELTA, {"delta": text})
    except Exception as e:
        # 流式中途失败：保留已输出的部分，避免前端空白
        logger.error(f"流式生成中断: {e}", exc_info=True)
        if not parts:
            return "生成答案时出错，请稍后重试。"
    return "".join(parts).strip()


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
