"""
导入流程主图
串联所有导入节点，形成完整的知识库导入流水线
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_product_name_recognition import node_product_name_recognition
from app.import_process.agent.nodes.node_office_to_md import node_office_to_md
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.utils.task_utils import is_node_done, save_checkpoint

load_dotenv()

# 只有解析阶段可跳过：向量化与入库的产物不落 Redis 断点，
# 重跑成本远低于重新调用 MinerU 解析，所以这两个节点始终重跑。
_RESUMABLE_NODES = {
    "node_entry",
    "node_pdf_to_md",
    "node_office_to_md",
    "node_md_img",
    "node_document_split",
    "node_product_name_recognition",
}


def _resumable(node_fn):
    """包装节点：已完成则跳过，完成后落一份断点快照。"""

    def wrapper(state: ImportGraphState) -> ImportGraphState:
        task_id = state.get("task_id", "")
        name = node_fn.__name__

        # 只有"这次确实是接着断点跑"才允许跳过：断点快照在任务成功时会被清掉，
        # 而 done 列表会留着（/status 还要展示进度）。只看 done 列表的话，
        # 重跑一个已完成的 task_id 就会跳掉解析节点却又无内容可恢复，
        # 实测跑出过"0 切片但状态 completed"的假成功。
        if (
            task_id
            and name in _RESUMABLE_NODES
            and state.get("resumable")
            and is_node_done(task_id, name)
        ):
            logger.info(f"断点续跑：跳过已完成节点 {name}")
            return state

        state = node_fn(state)

        if task_id and name in _RESUMABLE_NODES:
            save_checkpoint(task_id, state)
        return state

    wrapper.__name__ = node_fn.__name__
    return wrapper


def create_import_graph():
    """创建导入流程图"""
    logger.info("创建导入流程图")

    workflow = StateGraph(ImportGraphState)

    # 注册导入流程节点（解析类节点带断点续跑包装）
    workflow.add_node("node_entry", _resumable(node_entry))
    workflow.add_node("node_pdf_to_md", _resumable(node_pdf_to_md))
    workflow.add_node("node_office_to_md", _resumable(node_office_to_md))
    workflow.add_node("node_md_img", _resumable(node_md_img))
    workflow.add_node("node_document_split", _resumable(node_document_split))
    workflow.add_node("node_product_name_recognition", _resumable(node_product_name_recognition))
    workflow.add_node("node_bge_embedding", node_bge_embedding)
    workflow.add_node("node_import_milvus", node_import_milvus)

    # 设置入口
    workflow.set_entry_point("node_entry")

    # 条件路由
    def route_after_entry(state: ImportGraphState) -> str:
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        if state.get("is_office_read_enabled"):
            return "node_office_to_md"
        if state.get("is_md_read_enabled"):
            return "node_md_img"
        return END

    workflow.add_conditional_edges(
        "node_entry",
        route_after_entry,
        {
            "node_pdf_to_md": "node_pdf_to_md",
            "node_office_to_md": "node_office_to_md",
            "node_md_img": "node_md_img",
            END: END
        }
    )

    # 顺序边
    workflow.add_edge("node_pdf_to_md", "node_md_img")
    workflow.add_edge("node_office_to_md", "node_md_img")
    workflow.add_edge("node_md_img", "node_document_split")
    workflow.add_edge("node_document_split", "node_product_name_recognition")
    workflow.add_edge("node_product_name_recognition", "node_bge_embedding")
    workflow.add_edge("node_bge_embedding", "node_import_milvus")
    workflow.add_edge("node_import_milvus", END)

    app = workflow.compile(name="agentic-rag-import-graph")
    logger.info("导入流程图创建成功")

    return app


kb_import_app = create_import_graph()


if __name__ == "__main__":
    print("导入流程已创建")