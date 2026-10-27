"""断点续跑必须在真实 LangGraph 里生效。

这个用例是补的债：task_runner 一直在往 init_state 写 `resumable`，但
`ImportGraphState` 没声明这个键，而 LangGraph 只传递 schema 里声明过的通道——
节点包装器读到的永远是 None，"跳过已完成节点"在真实图里从没生效过。
现场证据： worker 断线重投后，MinerU 把已经解析完的文档又整本重解析了一遍。
所以这里不测函数级别的 _resumable，直接编译整张图跑一遍。
"""
import pytest

from app.import_process.agent import main_graph as mg
from app.import_process.agent.state import ImportGraphState, create_default_state

_NODE_NAMES = (
    "node_entry",
    "node_pdf_to_md",
    "node_office_to_md",
    "node_md_img",
    "node_document_split",
    "node_product_name_recognition",
    "node_bge_embedding",
    "node_import_milvus",
)


@pytest.fixture
def recorder(monkeypatch):
    """把图上所有节点换成记录器，并接管 done 列表与快照写入。"""
    executed: list[str] = []
    done: set[str] = set()

    def install(*, already_done: set[str]):
        executed.clear()
        done.clear()
        done.update(already_done)

        def make_stub(name):
            def stub(state):
                executed.append(name)
                return {}
            stub.__name__ = name
            return stub

        for name in _NODE_NAMES:
            monkeypatch.setattr(mg, name, make_stub(name))
        monkeypatch.setattr(mg, "is_node_done", lambda task_id, node: node in done)
        monkeypatch.setattr(mg, "save_checkpoint", lambda *args, **kwargs: None)
        return executed

    return install


def _run(graph, *, resumable: bool, executed: list[str]):
    state = create_default_state(
        task_id="t-resume",
        local_dir="/tmp/none",
        local_file_path="/tmp/none.pdf",
        is_pdf_read_enabled=True,
        is_md_read_enabled=True,
    )
    state["resumable"] = resumable
    graph.invoke(state)
    return executed


def test_resumable_is_declared_in_state_schema():
    assert "resumable" in ImportGraphState.__annotations__
    assert create_default_state()["resumable"] is False


def test_completed_parse_nodes_are_skipped_on_resume(recorder):
    executed = recorder(already_done={"node_entry", "node_pdf_to_md", "node_md_img"})
    graph = mg.create_import_graph()
    _run(graph, resumable=True, executed=executed)

    assert "node_pdf_to_md" not in executed      # 不再白烧一次 MinerU
    assert "node_md_img" not in executed
    assert executed == [
        "node_document_split",
        "node_product_name_recognition",
        "node_bge_embedding",
        "node_import_milvus",
    ]


def test_fresh_run_executes_every_node(recorder):
    executed = recorder(already_done={"node_pdf_to_md"})
    graph = mg.create_import_graph()
    _run(graph, resumable=False, executed=executed)

    # done 列表里有节点但这次不是续跑：必须照跑，否则就是"0 切片却记 completed"的假成功
    assert executed == [
        "node_entry",
        "node_pdf_to_md",
        "node_md_img",
        "node_document_split",
        "node_product_name_recognition",
        "node_bge_embedding",
        "node_import_milvus",
    ]


def test_embedding_and_import_always_rerun(recorder):
    """向量与入库产物不落快照，永远重跑（重跑成本远低于重新解析）。"""
    executed = recorder(already_done=set(_NODE_NAMES))
    graph = mg.create_import_graph()
    _run(graph, resumable=True, executed=executed)

    assert executed == ["node_bge_embedding", "node_import_milvus"]
