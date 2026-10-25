# Agentic-RAG — 企业知识库检索增强问答平台

> 面向 3C 设备手册的图文混排知识库：多格式解析入库、稠密 + 稀疏混合检索、Cross-Encoder 精排、逐段溯源、SSE 流式问答。

## 项目背景

设备厂商与售后团队要查阅大量 PDF 用户手册（路由器、笔记本、平板、打印机、显示器等），
文档图文混排、单本上百页，人工翻查效率低；内嵌的接口图、指示灯说明、规格表又无法被
纯文本检索利用。本项目把这些手册解析成结构化切片，建立可追溯的检索问答链路。

## 项目规模

| 项 | 实际值 |
| --- | --- |
| 语料文档 | 85 份 3C 设备用户手册（PDF） |
| 切片数量 | 约 4.4k chunk |
| 覆盖品类 | 打印机 / 笔记本 / 平板 / 路由器 / 显示器 / 其他设备 |
| 图片抽取 | 8k+ 张内嵌图片（VLM 生成文本摘要后参与检索） |

## 核心流程

**入库**
1. 上传 PDF / Word / Excel / Markdown
2. MinerU 解析 PDF（Word/Excel 走 python-docx / openpyxl），保留页码与标题层级
3. 抽取内嵌图片 → 阿里云 OSS（私有 bucket）→ VLM 生成图片摘要写回正文
4. 结构化切块：标题分层、超长二次切分、过短合并，每块带业务 `chunk_id` 与页码
5. 百炼 `text-embedding-v4` 生成**稠密 + 稀疏双向量** → Milvus（Zilliz Cloud）

**查询**
1. 会话历史 + 意图改写（识别问题指向的产品）
2. 三路召回：向量混合检索、HyDE 改写检索、联网搜索（MCP）
3. 加权 RRF 融合 → Cross-Encoder 精排 → **断崖检测动态 Top-K**
4. 上下文按 token 预算截断 → LLM 流式生成 → **引用后置校验**（拦掉不存在的 `[N]`）
5. SSE 推送 `ready / progress / delta / final / error`，前端实时展示节点进度与引用来源

## 技术栈

| 层 | 技术 |
| --- | --- |
| 编排 | LangGraph（固定流水线 + 条件路由）、LangChain |
| 文档解析 | MinerU、python-docx、openpyxl；OCR 与 VLM 图片理解 |
| Embedding | 百炼 `text-embedding-v4`（DashScope 原生接口，`output_type=dense&sparse`，1024 维） |
| 向量库 | Milvus（Zilliz Cloud 托管），HNSW + SPARSE_INVERTED_INDEX |
| 精排 | 百炼 `gte-rerank-v2`（Cross-Encoder） |
| 后端 | FastAPI + SSE、psycopg（PostgreSQL）、pika（RabbitMQ）、redis |
| 存储 | PostgreSQL（会话历史 + 导入任务）、Redis（任务状态 / 断点续跑）、阿里云 OSS（图片） |
| 观测 | loguru 全链路日志；LangSmith（可选，无 key 自动关闭） |

> Embedding / Rerank / LLM 都通过 `app/lm/provider_factory.py` 按 `.env` 装配，
> 支持在 `dashscope` / `siliconflow` / `local` 之间切换，业务代码不感知供应商。
> 注意：只有百炼原生接口能返回稀疏向量，走 OpenAI 兼容接口时稀疏侧为空，
> 检索会自动降级为纯稠密。
> `local` 需要额外安装推理依赖（`uv add flagembedding pymilvus-model torch`），
> 默认不装：tensor 栈体积大，且在部分 Windows 环境上 DLL 会加载失败。
> 启用时需自行把模型放到 `.env` 的 `BGE_M3_PATH` / `BGE_RERANKER_LARGE` 指向的目录。

## 检索评测

评测集在 `app/eval/golden_set.py`（72 条，按品类分层，19 条专考图片/表格内容）。
四条管线在同一批用例上对比：纯稠密（基线）、稠密+稀疏、+HyDE、+重排+断崖截断。

```bash
python scripts/eval_retrieval.py --pipelines all --k 5
```

结果写入 `docs/rag-eval.md`，含指标口径说明与相对基线提升。**所有数字以实测为准，
不做口径调整凑数。**

## 快速启动

```bash
# 1. 依赖
uv sync

# 2. 配置（键名见 .env.example；.env 不入库）
cp .env.example .env
# 填写：百炼 key、Zilliz URI+token、OSS、PG_DSN、REDIS_URL、RABBITMQ_URL、MINERU_API_TOKEN

# 3. 连通性自检（9 项，凭据不会打印）
python scripts/check_infra.py

# 4. 建集合（首次）
python scripts/milvus_admin.py --recreate

# 5. 复用已有切分结果重灌向量（不重跑 MinerU）
python scripts/reembed_existing.py --limit 3   # 先验证 3 份
python scripts/reembed_existing.py             # 全量

# 6. 检索冒烟
python scripts/check_retrieval.py

# 7. 起服务
python app/import_process/worker.py                    # MQ worker（消费导入任务）
python app/import_process/api/file_import_service.py   # :8002 上传
python app/query_process/api/query_service.py          # :8001 问答 + SSE
```

## 本地零凭据演示（不填任何 key 也能跑）

```bash
# 1. 入库：复用 output/ 里已有的切分结果，灌进 Milvus Lite
python scripts/demo_local.py --limit 100 --reset

# 2. 起聊天页面（本进程不打开 Milvus，避免单进程文件锁冲突）
python scripts/demo_local.py --serve
# 打开 http://127.0.0.1:8001/chat.html
```

演示档用的是真实实现，只是换成本地可跑的形态：

| 组件 | 生产 | 演示 |
| --- | --- | --- |
| 向量库 | Milvus / Zilliz Cloud | Milvus Lite（单文件） |
| 向量 | 百炼 `text-embedding-v4`（稠密+稀疏） | `lexical`（Hashing TF，无 torch/onnx 依赖） |
| 重排 | `gte-rerank-v2` | `none`（保留 RRF 顺序） |
| 图片存储 | 阿里云 OSS | 本地 `output/demo/storage` |
| 会话历史 | PostgreSQL | SQLite |
| 消息队列 | RabbitMQ | 进程内线程 |

**注意**：`lexical` 只做字面匹配，没有语义泛化能力，检索质量不代表线上效果；
没有大模型 key 时答案会降级为**抽取式**（直接给出检索到的原文片段 + 引用编号），
不会编造内容。把 `.env` 换成百炼 key + Zilliz 后，同一套代码直接变成语义检索 + 生成式回答。

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/RESUME-ALIGNMENT.md`](docs/RESUME-ALIGNMENT.md) | **简历每条能力 ↔ 代码位置 ↔ 验证命令**，面试前看这份 |
| [`docs/rag-eval.md`](docs/rag-eval.md) | 检索评测报告（指标口径 + 四条管线对比，待真实环境跑数） |
| [`doc/README.md`](doc/README.md) | 语料放置说明（PDF 不入库） |

## 目录结构

```
app/
  clients/    milvus_utils / oss_utils / pg_client / pg_history_utils /
              pg_task_utils / redis_utils / mq_utils / history
  conf/       各模块 dataclass 配置（milvus / lm / provider / infra / oss ...）
  eval/       golden_set / retrieval_metrics / pipelines
  import_process/   api / agent(nodes) / task_runner / worker
  query_process/    api / agent(nodes) / page
  lm/         provider_factory + providers(dashscope / siliconflow / local)
  utils/      task_utils / sse_utils / token_budget_utils / citation_utils
scripts/      check_infra / check_retrieval / reembed_existing /
              milvus_admin / eval_retrieval / check_golden / gate_metrics / import_stats
tests/        检索指标纯函数单测
docs/         rag-eval.md（评测报告）
```

## 已知约束

- 开发机是 Windows 且没有 Docker，运行依赖外部托管的 Milvus / OSS / PostgreSQL / Redis / RabbitMQ；
- 软路由侧中间件内存有限（3.7G、无 swap），批量导入请串行执行；
- 断点续跑只覆盖解析阶段，向量化与入库会重跑（向量不落 Redis）。
