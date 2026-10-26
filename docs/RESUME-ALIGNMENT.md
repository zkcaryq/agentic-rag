# 简历对齐说明（Resume Alignment）

> 用途：把简历里「Agentic-RAG 企业私有化知识问答平台」的每一条，落到具体文件与可复现的验证命令上。
> 面试前看这一份即可；任何一条都可以现场打开代码对照。
>
> 最后更新：本地演示链跑通之后。

## 0. 一句话结论

| 类别 | 状态 |
| --- | --- |
| 已有真实实现且本地可跑 | 解析入库、切分、切片 ID、混合检索、RRF、断崖截断、引用溯源、上下文预算、SSE 流式、节点进度、Redis/断点续跑、评测框架、导入失败率埋点、TTFT/延迟指标、并发压测脚本 |
| 已有真实实现但**需要 API key** 才能验证 | 百炼稠密+稀疏向量、gte-rerank 重排、HyDE、VLM 图片摘要、LangSmith |
| 需要凭据才能验证 | PostgreSQL 历史、RabbitMQ 异步（导入+查询）、Zilliz、OSS、WebSearch（百炼 MCP，需 key） |
| 尚未实现 | 无 |

> RabbitMQ 那条：导入侧本地就能验（MQ 关掉时退线程池），查询侧的**跨进程**事件回传需要
> Redis 与一个能共享的向量库（Milvus Lite 是单进程文件锁，本地演示起不了第二个进程），
> 所以这条要等 Zilliz + Redis 凭据到位才能跑通，`scripts/check_infra.py` 里已经加了
> RPUSH/BLPOP 往返探针。

---

## 1. 技术栈逐项映射

| 简历写的 | 代码实际 | 位置 | 说明 |
| --- | --- | --- | --- |
| Python / FastAPI | 两个服务：导入 `:8002`、查询 `:8001` | `app/import_process/api/file_import_service.py`、`app/query_process/api/query_service.py` | 都在 `python xxx.py` 直接起 |
| PostgreSQL | 会话历史 + 导入任务表 | `app/clients/pg_client.py`、`pg_history_utils.py`、`pg_task_utils.py` | 连接锁 UTC，时长用库的 `now()` 算 |
| LangChain / LangGraph | 两张状态图 | `app/import_process/agent/main_graph.py`、`app/query_process/agent/main_graph.py` | 导入图带断点续跑包装 `_resumable`（`:40`） |
| LangSmith | 环境变量接入 | `.env` 的 `LANGCHAIN_TRACING_V2/API_KEY/PROJECT` | 无 key 自动关闭，不阻塞 |
| **BGE-M3** | **百炼 `text-embedding-v4`**（稠密+稀疏） | `app/lm/providers/dashscope_provider.py:42` | ⚠️ **简历需改**：BGE-M3 走 OpenAI 兼容接口只有稠密，出不了稀疏；只有百炼原生接口支持 `output_type=dense&sparse` |
| Milvus | Milvus / Zilliz Cloud，本地演示用 Milvus Lite | `app/import_process/agent/nodes/node_import_milvus.py` | schema 含 `biz_chunk_id` + `page_no` |
| **MinIO** | **阿里云 OSS（私有 bucket + 签名 URL）** | `app/clients/oss_utils.py`、`app/clients/storage.py` | ⚠️ **简历需改**：MinIO 已弃用 |

> 换供应商只改 `.env`：`EMBEDDING_PROVIDER` / `RERANK_PROVIDER` 支持 `dashscope` / `siliconflow` / `local` / `lexical`（演示）。

---

## 2. 功能点逐条映射

### 2.1 多格式文档结构化处理

| 简历表述 | 实现 | 位置 |
| --- | --- | --- |
| MinerU 完成 PDF 解析 | 调 MinerU 云端 API，**去掉了三级 mock 造假**（配置缺失/异常/下载失败一律失败并记 `failed_stage=mineru`） | `node_pdf_to_md.py:105` |
| Word / Excel 解析 | python-docx 保留标题层级与表格；openpyxl 保留 sheet 名与行号 | `node_office_to_md.py` |
| 图片资源解耦持久化 | 抽图 → 上传 → 正文只写 `storage_key`，取图走 `/media/{key}`（本地档读文件，OSS 档 302 签名 URL） | `node_md_img.py`、`clients/storage.py` |
| VLM 生成图片文本摘要 | 单张图 → 上文/下文 → VLM 摘要 → 写回 Markdown 图片标题 | `node_md_img.py:157` |
| 标题分层切分 | 按 `#{1,6}` 切分 | `node_document_split.py:95` |
| 超长二次切分 / 过短合并 | 2000 字软上限按段落切、<500 字与前块合并 | `node_document_split.py` 的 `_split_long_chunk` / `_merge_short_chunks` |
| 页码来源定位 | 从 MinerU `content_list.json` 的 `page_idx` 按正文锚点映射到每个 chunk | `node_document_split.py:240` |
| 稠密+稀疏双向量入 Milvus | 供应商返回 dense + sparse，写入同一行 | `node_bge_embedding.py:63`、`node_import_milvus.py:108` |

### 2.2 混合检索与答案追溯

| 简历表述 | 实现 | 位置 |
| --- | --- | --- |
| 向量检索 + HyDE + WebSearch 多路召回 | 三路各自成为独立节点；WebSearch 走百炼 MCP（`openai-agents` 已装），无 key 时**快速跳过**而不是返回假数据 | `node_search_embedding.py`、`node_search_embedding_hyde.py`、`node_web_search_mcp.py` |
| RRF 融合 | 加权 RRF（1.0/0.8/0.6，k=60），**保留 `rrf_score`**（原来把分数丢了） | `node_rrf.py:144` |
| Cross-Encoder 重排序 | 供应商抽象：`gte-rerank-v2` / `bge-reranker-v2-m3` / 本地 / none | `node_rerank.py`、`lm/provider_factory.py` |
| 相关性分数断崖检测动态 Top-K | 绝对阈值 0.30 + 相邻比 0.60 + 上限 8 + 软下限保底 2 | `node_rerank.py:81` `select_by_cliff` |
| 切片 ID 贯穿全链路 | 切分生成 `chunk_id` → 入库写 `biz_chunk_id` → 检索 `output_fields` 带回 → 引用里返回 | `node_document_split.py:162`、`node_import_milvus.py`、`milvus_utils.py` |
| 后置校验强制拦截虚假引用 | 正则抽 `[N]`，越界编号直接删掉 | `app/utils/citation_utils.py:18` |
| 逐段溯源 | 引用返回 `index / biz_chunk_id / file_title / page_no / rerank_score` | `node_answer_output.py` |

### 2.3 上下文管控与链路观测

| 简历表述 | 实现 | 位置 |
| --- | --- | --- |
| 上下文长度预算管控 | CJK 按 1 字 1 token、其余 4 字符 1 token 估算，超预算截断并记录丢弃数 | `app/utils/token_budget_utils.py:26` |
| 故障降级兜底 | 检索失败置空、重排失败回退 RRF、流式失败保留已输出、无 LLM 降级抽取式 | 各节点 `except` + `node_answer_output.py:209` |
| 同步问答 + SSE 流式增量 | `POST /query` 非流式直接返回；流式先建队列再后台跑，`llm.stream` 逐 token 推 `delta` | `query_service.py`、`node_answer_output.py` |
| 节点级进度跟踪协议 | `ready/progress/delta/final/error/close` + 节点中文名映射 | `app/utils/sse_utils.py`、`app/utils/task_utils.py` |
| 前端实时观测链路 | 进度条 + `[N]` 引用高亮 + 来源列表（含页码/分数），所有文本做 HTML 转义 | `app/query_process/page/chat.html` |
| TTFT / 延迟指标埋点 | 按 endpoint 分桶的滑动窗口（默认 200 样本），P50/P95/P99 线性插值；`query` 记整轮、`answer_generate` 记模型首 Token | `app/core/metrics.py:45`、`node_answer_output.py:203`、`query_service.py:109` |
| 指标出口 | `GET /metrics` 返回窗口快照 + 当前派发模式（`rabbitmq` / `local_thread`） | `query_service.py`（`metrics_snapshot`） |

### 2.4 异步任务调度与可用性

| 简历表述 | 实现 | 位置 |
| --- | --- | --- |
| RabbitMQ 异步分发 | 上传后投递 `zhanggui.import.task`，失败退进程内线程 | `clients/mq_utils.py:102`、`file_import_service.py` |
| 查询任务也走队列 | `POST /query`（流式）投 `zhanggui.query.task`，由 `app/query_process/worker.py` 消费执行；MQ 或 Redis 任一不可用则退回本进程线程池 | `clients/mq_utils.py:109`、`query_process/worker.py:42`、`query_service.py`（`_mq_dispatch_enabled`） |
| 任务资源隔离 | 导入/查询分属不同 exchange+queue，互不堆积（批量灌库时问答不排队）；导入失败进 `zhanggui.import.dlq` | `mq_utils.declare_topology` |
| 跨进程事件回传 | worker 在别的进程执行，事件写 Redis 列表 `rag:sse:{sid}:events`，API 进程 BLPOP 后转发给 EventSource | `clients/redis_utils.py:72`、`utils/sse_utils.py:58` |
| Redis 持久化任务状态 | 任务状态/进度/结果落 Redis（key 前缀 `rag:`，匹配 ACL `~rag:*`），Redis 不可用自动退回内存 | `clients/redis_utils.py`、`task_utils.py` |
| 会话持久化断点续跑 | 解析阶段节点完成后存快照，重启后跳过已完成节点；向量与入库重跑 | `task_utils.py:247`、`main_graph.py:40` |
| 并发查询支撑 | `scripts/bench_concurrency.py` 压 SSE 全链路，出客户端口径的 TTFT / 端到端 P50/P95/P99 + 成功率 + 吞吐 | `scripts/bench_concurrency.py` |
| 全链路日志追踪 | loguru 控制台+文件，各节点带 `task_id` 打印阶段与耗时 | `app/core/logger.py`、各 `node_*.py` |
| 导入失败率埋点 | `import_tasks` 表记 `failed_stage / error_type / retry_count / duration_ms` | `pg_task_utils.py`、`node_*.py` 的 `failed_stage` |

---

## 3. 三个量化指标怎么取证

| 简历写的 | 取证方式 | 注意事项 |
| --- | --- | --- |
| 召回准确率提升约 38% | `python scripts/eval_retrieval.py --pipelines all --k 5`，基线是**纯稠密**（dense），对比 hybrid / hyde / full | 72 条 golden set 在 `app/eval/golden_set.py`；**实测不到 38% 就按实测改简历**，不改口径 |
| 大文档导入失败率 <4% | `import_tasks` 表统计：分母 = completed + failed，partial_success 单列；`python scripts/import_stats.py` | 需要真实跑够样本（4% 分辨率下限约 100 次导入） |
| 图文混排文档入库 | 72 条 golden set 里 **19 条**专考图片/表格内容；chunk 正文里保留 `[图片：VLM 摘要]` | 可用 `scripts/check_golden.py` 自检 |
| 「支撑并发知识库查询」 | `scripts/bench_concurrency.py --concurrency N --duration 60`，报成功率、吞吐、TTFT/端到端 P50/P95/P99 | 本地演示档（Milvus Lite + 抽取式答案）实测 6 并发 12 轮：成功 12/12，TTFT P50 544ms / P95 1039ms。**这是无大模型参与的下限**，配上 LLM 后 TTFT 会明显变大，别拿这个数当业务指标 |

---

## 4. 验证命令清单

```bash
# 静态（无凭据，CI 同款）
python -m compileall app scripts tests
python -m pytest tests/ -q              # 45 个用例
python scripts/check_golden.py          # 72 条评测集自检

# 本地演示（无凭据）
python scripts/demo_local.py --limit 100 --reset
python scripts/demo_local.py --serve    # http://127.0.0.1:8001/chat.html

# 并发与延迟（服务起着，另开一个终端）
python scripts/bench_concurrency.py --concurrency 6 --total 12
python scripts/bench_concurrency.py --concurrency 16 --duration 120 --json output/bench/run.json
curl http://127.0.0.1:8001/metrics      # 服务端窗口快照

# 真实链路（需 .env）
python scripts/check_infra.py           # 9 项连通性（含 Redis RPUSH/BLPOP 探针）
python scripts/milvus_admin.py --recreate
python scripts/reembed_existing.py
python scripts/check_retrieval.py       # 向量真实 / 可溯源 / 重排生效
python scripts/eval_retrieval.py --pipelines all --k 5
python scripts/import_stats.py

# MQ 派发查询（需 RabbitMQ + Redis + 共享向量库；Milvus Lite 起不了第二个进程）
python app/query_process/worker.py      # 消费 zhanggui.query.task
python app/import_process/worker.py     # 消费 zhanggui.import.task
```

---

## 5. 面试可展开的设计取舍

1. **为什么不用 OpenAI 兼容接口做 embedding**：兼容层不支持 `text_type` 与稀疏向量；走原生接口才能拿到 `sparse_embedding`，稀疏侧支撑关键词精确命中，稠密侧支撑语义召回。
2. **为什么重排后要断崖截断而不是固定 Top-5**：固定 K 在"只有 2 条相关"时会塞进噪声，在"8 条都相关"时又截断信息；用分数比值找断崖更贴近实际分布，同时保留保底与上限防极端情况。
3. **为什么引用要做后置校验**：模型会编 `[7]` 这种不存在的编号；在返回前正则校验并删除，比在 prompt 里反复强调可靠。
4. **为什么断点续跑只覆盖解析阶段**：MinerU 解析是分钟级且花钱，向量化是秒级；把快照做在解析产物上，收益最大、实现最轻（不引入 LangGraph checkpointer）。
5. **为什么 Milvus Lite 演示档不做稀疏**：Lite 不支持稀疏索引；代码用 `search_with_optional_sparse` 自动降级为纯稠密，换回 Zilliz 即恢复混合检索。
6. **为什么跨进程事件走 Redis 列表而不是 pubsub**：应用账号在 Redis ACL 里只有 `~rag:*` 的 key 权限，没有频道权限；列表既能 TTL 兜底又能积压，正好匹配"worker 生产、API 进程消费"的节奏。代价是 API 进程要用一条独立连接做 BLPOP（阻塞会占住连接，不能共用单例）。
7. **为什么 TTFT 分两个口径**：`/metrics` 里的 `answer_generate` 从"开始调模型"起算，是模型侧首 Token；压测脚本的 TTFT 从"发出请求"起算，含排队+检索+重排，是用户感知的。合成一个数就会把线程池排队的开销藏进"模型很慢"或者反过来。
8. **为什么 worker 处理失败不重投**：一次问答失败让用户重新问比后台重试更符合语义；直接 ack 并回 `error` 事件，避免毒消息在队列里打转（导入侧相反，所以导入侧才有死信队列）。

---

## 6. 简历需要同步修改的三处

| 现在写的 | 建议改成 | 原因 |
| --- | --- | --- |
| MinIO | 阿里云 OSS（私有 bucket + 签名 URL） | 实际实现已替换 |
| BGE-M3 生成稠密+稀疏双向量 | 百炼 `text-embedding-v4` 生成稠密+稀疏双向量 | 只有百炼原生接口支持稀疏输出 |
| 电商业务文档 | 图文混排的设备/产品手册类文档 | 语料实际是 85 份 3C 设备手册 |
