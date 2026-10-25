"""本地词法向量（无 torch / 无 onnx / 无 API key）。

把中文二元组与英文词做 **signed hashing** 映射到固定维度并 L2 归一化。
这是真实的词法表示（Hashing TF），结果可复现、由内容决定，不是随机数；
但它只做字面匹配，没有语义泛化能力，**仅用于本地演示与链路验证**。

线上请用 dashscope（稠密+稀疏）或 siliconflow（仅稠密）。
"""
import hashlib
import re

import numpy as np

from app.conf.provider_config import EmbeddingProviderConfig
from app.lm.providers.base import EmbeddingProvider, EmbedRecord

_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class LexicalEmbeddingProvider(EmbeddingProvider):
    name = "lexical"
    # Milvus Lite 不支持稀疏向量索引，这里只给稠密，检索自动走 dense-only
    supports_sparse = False

    def __init__(self, cfg: EmbeddingProviderConfig) -> None:
        self._dim = cfg.dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: list[str], text_type: str = "document") -> list[EmbedRecord]:
        return [EmbedRecord(dense=self._encode(text), sparse=None) for text in texts]

    def _encode(self, text: str) -> list[float]:
        text = text or ""
        characters = _CJK_PATTERN.findall(text)
        # 中文用二元组，单字文本退化为单字
        tokens = (
            [characters[i] + characters[i + 1] for i in range(len(characters) - 1)]
            if len(characters) > 1
            else characters
        )
        tokens.extend(word.lower() for word in _WORD_PATTERN.findall(text))

        vector = np.zeros(self._dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dim
            # signed hashing：用一位做符号，抵消哈希碰撞带来的系统性偏置
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        return (vector / norm).tolist() if norm > 0 else vector.tolist()
