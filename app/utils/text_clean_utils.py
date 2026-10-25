"""正文清洗工具。

历史产物里混进了原作者本地 MinIO 的图片链接（http://localhost:9000/...），
既不可访问，也不该出现在新库里。统一替换成保留图片描述的纯文本占位。
"""
import re

_LEGACY_IMAGE_PATTERN = re.compile(
    r"!\[([^\]]*)\]\((?:https?://(?:localhost|127\.0\.0\.1|mock\.example\.com)[^)]*)\)"
)


def sanitize_content(text: str) -> str:
    """把指向本地/占位域名的 Markdown 图片替换成 `[图片：描述]`。"""

    def _replace(match: re.Match) -> str:
        alt = (match.group(1) or "").strip()
        return f"[图片：{alt}]" if alt else "[图片]"

    return _LEGACY_IMAGE_PATTERN.sub(_replace, text or "")
