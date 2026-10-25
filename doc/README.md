# 语料目录

本目录用于放置待入库的原始文档（`.pdf` / `.docx` / `.xlsx` / `.md`）。

**厂商用户手册受版权保护，且单份几十 MB，因此不纳入版本库。**
仓库只保留本说明文件，`*.pdf` 已在 `.gitignore` 中排除。

## 使用方式

1. 把设备手册 PDF 放进本目录；
2. 通过导入服务上传，或直接跑批处理：

```bash
# 上传（需先启动导入服务与 worker）
curl -F "files=@doc/你的手册.pdf" http://127.0.0.1:8002/upload

# 或者复用已有的解析产物重灌向量（不重跑 MinerU）
python scripts/reembed_existing.py --limit 3
```

## 评测集与语料的关系

`app/eval/golden_set.py` 里的 `relevant_docs` 是**文档标题关键词**，需要与
本目录下的文件名对应。换语料后必须同步更新标注，否则指标不可比。

`scripts/check_golden.py` 会校验每个关键词能否命中真实文件名；如果本目录为空，
该检查会自动跳过并给出提示。
