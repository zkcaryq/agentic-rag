# 简历对齐说明（Resume Alignment）

> 用途：把简历里「Agentic-RAG 企业私有化知识问答平台」的每一条，落到具体文件与可复现的验证命令上。
> 面试前看这一份即可；任何一条都可以现场打开代码对照。
>
> 最后更新：本地演示链跑通之后。

## 0. 一句话结论

| 类别 | 状态 |
| --- | --- |
| **云端真实环境已实测通过**（2026-10-26） | Zilliz Milvus（稠密 HNSW + 稀疏索引，`kb_chunks`）、百炼 `qwen3.7-text-embedding` 稠密1024+稀疏(nnz≈275)、`qwen3.7-text-rerank` 重排、HyDE、WebSearch（百炼 MCP `EnhancedSearch/search_pro`，单节点实测 10 条真结果）、PostgreSQL 17、Redis（含 RPUSH/BLPOP 跨进程探针）、RabbitMQ（导入队列实测 `dispatch=rabbitmq` 跑通）、OSS（写→签名URL读→删）、MinerU（真 PDF 全链路 31.4s 入库 5 切片、页码 1/1/4/5/6）、VLM 图片摘要（4 张图全部生成中文摘要）、切片 ID 端到端可溯源 |
| 已有真实实现且本地可跑 | 解析入库、切分、混合检索、RRF、断崖截断、引用溯源、上下文预算、SSE 流式、节点进度、断点续跑、评测框架、导入失败率埋点、TTFT/延迟指标、并发压测脚本 |
| 已实测（2026-10-26 傍晚补） | 查询侧 MQ 派发端到端：`dispatch=rabbitmq` → worker 执行 → 15 个 progress + 8 个 delta + final 经 Redis 总线回到 API 进程再吐给 HTTP 客户端；多 worker 水平扩容对照（见第 3 节）；`/metrics` 跨进程合并 |
| 代码就绪但**还没实测** | `reembed_existing.py` 不写 `import_tasks` 表，所以"导入失败率 <4%"仍只覆盖真实上传链，且样本量不够（4% 的分辨率下限约 100 次导入）；语料仍有 3 份 PDF 缺切分产物未入库 |
| 需要你开一下 | **LangSmith**：开关/别名/自检都已接好（`check_infra` 会拿 key 真调一次 API），但工作区没有任何现成 key，要你去 smith.langchain.com 注册后填 `LANGCHAIN_API_KEY` |
| 数字待回填 | 召回提升 38%、导入失败率 <4%、并发 TTFT —— 全量重灌跑完出数 |

> 关于"跨进程"这条的限制：Milvus Lite 是单进程文件锁，本地演示档永远起不了第二个进程，
> 所以查询侧 MQ 只有单测与代码；现在 Zilliz 已通，两个进程可以共享同一个向量库，
> 这条随时能实测（`app/query_process/worker.py` + 起查询服务即可）。

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
| MinerU 完成 PDF/Word/Excel 解析 | 调 MinerU v4 云端接口，**PDF 与 Office 同一条路**（实测 MinerU 收 `.docx`/`.xlsx` 并返回 Markdown，表格是 `<table>` HTML）；**去掉了三级 mock 造假**（配置缺失/异常/下载失败一律如实记 `failed_stage=mineru`） | `node_pdf_to_md.py`、`main_graph.route_after_entry` |
| Office 解析引擎可切换 + 降级 | `OFFICE_PARSE_ENGINE=mineru|native`：默认 mineru；MinerU 挂了自动退回本地解析器并记 `failed_stage=mineru_fallback`（任务算 partial_success，不混进失败率分母）；PDF 没有本地兜底，照旧失败。`native` 用 python-docx 保标题层级/表格、openpyxl 保 sheet 名与行号，零凭据演示档用它 | `conf/mineru_config.py`、`node_pdf_to_md.py` 的 except 分支、`node_office_to_md.py` |
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
| TTFT / 延迟指标埋点 | 按 endpoint 分桶的滑动窗口（默认 200 样本），P50/P95/P99 线性插值；`query` 记整轮、`answer_generate` 记模型首 Token；埋点后同步镜像样本到 Redis（`record_and_export`） | `app/core/metrics.py`、`node_answer_output.py:203`、`query_service.py` |
| 指标出口 | `GET /metrics` 合并**本进程窗口 + Redis 里其它进程的样本**（排除自己 pid 防重复计数），并给出当前派发模式；实测 worker 执行的查询也能读到 `answer_generate ttft_p50=1008ms` | `query_service.py`（`metrics_snapshot`）、`clients/redis_utils.py`（`store_metrics/load_all_metrics`） |

### 2.4 异步任务调度与可用性

| 简历表述 | 实现 | 位置 |
| --- | --- | --- |
| RabbitMQ 异步分发 | 上传后投递 `arag.import.task`，失败退进程内线程 | `clients/mq_utils.py:102`、`file_import_service.py` |
| 查询任务也走队列 | `POST /query`（流式）投 `arag.query.task`，由 `app/query_process/worker.py` 消费执行；MQ 或 Redis 任一不可用则退回本进程线程池 | `clients/mq_utils.py:109`、`query_process/worker.py:42`、`query_service.py`（`_mq_dispatch_enabled`） |
| 任务资源隔离 | 导入/查询分属不同 exchange+queue，互不堆积（批量灌库时问答不排队）；导入失败进 `arag.import.dlq` | `mq_utils.declare_topology` |
| 跨进程事件回传 | worker 在别的进程执行，事件写 Redis 列表 `rag:sse:{sid}:events`，API 进程 BLPOP 后转发给 EventSource | `clients/redis_utils.py:72`、`utils/sse_utils.py:58` |
| Redis 持久化任务状态 | 任务状态/进度/结果落 Redis（key 前缀 `rag:`，匹配 ACL `~rag:*`），Redis 不可用自动退回内存 | `clients/redis_utils.py`、`task_utils.py` |
| 会话持久化断点续跑 | 解析阶段节点完成后存快照，重启后跳过已完成节点；向量与入库重跑。**已实测**：构造 done=[4 个解析节点]+checkpoint 的中间态后重跑，日志给出"已恢复字段: [file_title, md_path, md_content, chunks, product_name ...]"并正常完成。跳过条件额外要求 `state["resumable"]`——只凭 done 列表会把"重跑一个已完成任务"变成跳解析却又无快照可恢复、0 切片却记 completed 的假成功（实测踩过，已加红-绿回归用例） | `task_utils.py`（save/load_checkpoint）、`main_graph.py:_resumable`、`tests/test_rag_features.py` |
| 并发查询支撑 | `scripts/bench_concurrency.py` 压 SSE 全链路，出客户端口径的 TTFT / 端到端 P50/P95/P99 + 成功率 + 吞吐 | `scripts/bench_concurrency.py` |
| 全链路日志追踪 | loguru 控制台+文件，各节点带 `task_id` 打印阶段与耗时 | `app/core/logger.py`、各 `node_*.py` |
| 导入失败率埋点 | `import_tasks` 表记 `failed_stage / error_type / retry_count / duration_ms` | `pg_task_utils.py`、`node_*.py` 的 `failed_stage` |

---

## 3. 三个量化指标怎么取证

| 简历写的 | 取证方式 | 注意事项 |
| --- | --- | --- |
| 召回准确率提升约 38% | **实测不成立，已按实测改口径。** 2026-10-26 在 Zilliz（82 份文档 / 4444 切片 / 72 条 golden set / K=5）四管线对照：`dense` Hit@5=0.8889 MRR=0.8072 NDCG@5=0.8274；`hybrid` 0.8889/0.8176/0.8353；`hyde` 0.8750/0.7979/0.8172；`full`(混合+HyDE+RRF+重排+断崖) **0.8889/0.8461/0.8568** | 召回被小语料顶满（dense 自己就 0.8889），**增益只在排序**：full 相对 dense 的 MRR +3.9 点（相对 +4.8%）、NDCG +2.9 点（相对 +3.6%）。所以那句要么改成"排序质量 MRR/NDCG 提升约 4%"，要么换更难的评测集；写 38% 一算就穿。复现：`python scripts/eval_retrieval.py --pipelines all --k 5`（约 20 分钟，rerank 需额度） |
| 大文档导入失败率 <4% | `import_tasks` 表统计：分母 = completed + failed，partial_success 单列；`python scripts/import_stats.py`。**统计完整性已加固**：0 切片的导入现在记 `failed/split/EmptyChunks`（实测 PG 行 `('empty-chunk-1','failed','split','EmptyChunks',0)`），不再静默算成功稀释失败率 | **样本量仍然不够，别写 4%**：4% 的分辨率下限约需 100 次真实上传链导入，目前只有 MinerU 全链路 1 次 + 复用产物重灌 82 次（后者不写 `import_tasks`）。要拿到能上简历的数字，得把 85 份 PDF 全走 `/upload` 真导一遍 |
| 图文混排文档入库 | 72 条 golden set 里 **19 条**专考图片/表格内容；chunk 正文里保留 `[图片：VLM 摘要]` | 可用 `scripts/check_golden.py` 自检 |
| 「支撑并发知识库查询」 | `scripts/bench_concurrency.py --concurrency N --total M`，报成功率、吞吐、TTFT/端到端 P50/P95/P99 | **实测（真实云端栈，查询走 MQ）：** 1 worker × 4并发4轮 → 4/4 成功，TTFT P50 **53.3s**，0.04 轮/秒；3 worker × 同一负载 → 4/4，P50 **20.3s**，0.08 轮/秒；3 worker × 6并发12轮 → **12/12**，P50 23.7s，0.17 轮/秒。诚实边界：6 并发那轮此前是 0/12 全失败，这次修复同时改了两个变量（百炼额度充值 + worker 从 1 加到 3），不能把功劳全算给扩容。另有 `GET /metrics`：worker 样本经 Redis `rag:metrics:{pid}` 镜像后由 API 进程合并，实测同时给出 `answer_generate ttft_p50=1008ms`（模型侧）与客户端 `TTFT P50=18.5s`，差值即检索+联网+排队。**所以这句写"查询队列化 + 多 consumer 水平扩容 + TTFT 跨进程可观测"站得住，写"高并发/低延迟"站不住**（单查询端到端仍是十几到几十秒） |

---

### 3.1 语料覆盖与清理结论（2026-10-26）

| 项 | 实测 |
| --- | --- |
| 已入库 | 81 个产品 / 4448 切片（Zilliz `kb_chunks`） |
| 评测集目标型号 | 66 个，其中 **62 个已入库**，4 个（`CS2610DNW` / `LJ2680DN` / `M3070` / `hl3070`）在 `doc/` 有 PDF 但导不进 |
| dense 的 8 条未命中 | **4 条是语料缺失**（上面那 4 个型号），4 条是相近型号混淆（B5-330 / 平板 C3 / NER214W / MateStation S 被兄弟型号的手册挤掉）——所以 0.8889 里混着数据缺口，不是纯检索能力 |
| 补导结果 | 4 份全部失败：MinerU 返回 `retry limit reached (5 attempts), please replace the file`；已如实落在 `import_tasks`（4 条 failed）。老代码在这里会用 mock 兜底"成功"，现在会真失败 |
| 仓库瘦身 | `output/` 从 **1946 MB 降到 69 MB**（删掉 1.88 GB 的 MinerU 解析中间产物：8197 张 jpg、81 个 zip、PDF 副本、`_model.json`、`_content_list_v2.json`），只留链路真正读取的 `chunks.json` / `*content_list.json` / `*.md` / `import_result.json`；删掉原作者 6 月的运行日志与 5 个 `test_*` 垃圾目录 |
| 瘦身后校验 | `reembed_existing --dry-run` 仍识别 81 份 / 4448 切片；`check_golden` 通过；真跑 1 份 → 页码映射 5/5 命中锚点（说明 content_list 留对了） |

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
python app/query_process/worker.py      # 消费 arag.query.task
python app/import_process/worker.py     # 消费 arag.import.task
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

## 6. 简历需要同步修改的几处

| 现在写的 | 建议改成 | 原因 |
| --- | --- | --- |
| MinIO | 阿里云 OSS（私有 bucket + `storage_key` + 签名 URL 回源） | 实际实现已替换并实测通过 |
| BGE-M3 生成稠密+稀疏双向量 | 百炼 `qwen3.7-text-embedding` 生成稠密(1024)+稀疏双向量 | 只有百炼原生接口支持稀疏输出，实测稀疏 nnz≈275；BGE-M3 本机跑不了（torch DLL 坏），换 SiliconFlow 又出不了稀疏 |
| Cross-Encoder 重排序 | 补上模型名：百炼 `qwen3.7-text-rerank`（Cross-Encoder） | 现在能报出具体供应商，实测 top1 0.572 |
| 电商业务文档 | 图文混排的设备/产品手册类文档 | 语料实际是 85 份 3C 设备手册 |
| **基于 MinerU 完成 PDF/Word/Excel 解析** | **不用改了，这句现在是事实**（2026-10-26 实测：`probe_p3030_manual.docx` / `probe_p3030_spec.xlsx` 走 MinerU 出 Markdown，端到端 10.4s / 8.8s；MinerU 不可用时自动退回本地解析器，任务记 partial_success） |
  面试要能讲两点：① Office 为什么能给 MinerU（实测接口收，返回 `<table>` HTML 表格）；② 为什么还留着 python-docx/openpyxl（零凭据演示档用它，且 MinerU 挂了要有兜底，本地表格转 Markdown 更干净、Excel 还保留行号可溯源）。旧行原因（MinerU 只处理 PDF）已被实测推翻 |
| 技术栈里的 LangSmith | 要么填上 key 真开起来，要么从技术栈里去掉 | 目前 `enabled=false`，简历写"链路观测"却没有 trace 截图撑腰，不如不开 |

可直接粘贴的两句（与代码一一对应）：

```
- 基于 MinerU 云端 API 统一完成 PDF / Word / Excel 的解析转换（含图片与表格抽取），
  并以 python-docx / openpyxl 作为本地降级引擎（MinerU 不可用时自动接管，任务标记
  partial_success 而非失败）；解析产物统一转 Markdown 进入切分链路，图片解耦上传私有
  OSS，正文仅存 storage_key、取图走签名 URL，并由 VLM 生成图片文本摘要使图内信息可检索。
- 通过百炼 qwen3.7-text-embedding 生成稠密(1024)+稀疏双向量写入 Milvus(Zilliz Cloud)，
  以 HNSW + SPARSE_INVERTED_INDEX 双索引支撑混合检索。
```
