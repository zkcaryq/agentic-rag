# RAG 检索评测报告

> 本文件由 `scripts/eval_retrieval.py` 生成，每次评测会覆盖。
> 当前还没有在真实环境跑过，下面是口径说明与待填数据。

## 状态

**未运行**。需要先配置好 `.env` 里的百炼 key 与 Zilliz 凭据，并完成全量重灌：

```bash
python scripts/reembed_existing.py
python scripts/eval_retrieval.py --pipelines all --k 5
```

## 评测口径

| 项 | 说明 |
| --- | --- |
| 用例集 | `app/eval/golden_set.py`，72 条，按品类分层 |
| 相关文档标注 | 文档标题关键词（子串匹配 `file_title`），不做 chunk 级标注 |
| 匹配方式 | 关键词子串匹配，文档级去重 |
| 指标 | Hit@K / Recall@K / Precision@K / MRR / NDCG@K |
| 防假指标 | 每个相关条目最多命中一次，避免 substring 匹配下 recall > 1 |
| 截断 | K = 5 |

## 四条管线

| 管线 | 说明 |
| --- | --- |
| `dense` | 纯稠密检索，**相对提升的基线** |
| `hybrid` | 稠密 + 稀疏，Milvus WeightedRanker（0.8 / 0.2） |
| `hyde` | hybrid + HyDE 改写检索，RRF 融合 |
| `full` | hyde + gte-rerank-v2 精排 + 断崖动态 Top-K |

## 待填：管线对比

| 管线 | Hit@5 | Recall@5 | Precision@5 | MRR | NDCG@5 | 耗时(s) |
| --- | --- | --- | --- | --- | --- | --- |
| dense | - | - | - | - | - | - |
| hybrid | - | - | - | - | - | - |
| hyde | - | - | - | - | - | - |
| full | - | - | - | - | - | - |

## 待填：相对基线提升

| 管线 | Recall@5 | 相对提升 |
| --- | --- | --- |
| hybrid | - | - |
| hyde | - | - |
| full | - | - |

> 提醒：相对提升只对同一批用例、同一个 golden set 成立。扩充语料后必须重新标注并复测，
> 否则指标不可比。若实测达不到预期值，以实测数字为准，不改口径凑数。
