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


# 联网搜索抓回来的网页正文里常带占位垃圾：孤立的【图片】、尖括号包起来的示例链接、
# Markdown 图片语法。这些进了上下文就会被模型原样抄进答案（实测答案尾部出现过
# 【图片】 + 四行 <https://example.com/image_xxx.jpg>）。
_WEB_IMAGE_MARKERS = re.compile(r"【图片】|\[图片\](?!：)")
_WEB_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_WEB_ANGLE_URL = re.compile(r"<https?://[^>\s]+>")


def clean_web_snippet(text: str) -> str:
    """清掉网页摘要里的图片占位与尖括号链接，只留有信息量的文字。"""
    if not text:
        return ""
    cleaned = _WEB_MARKDOWN_IMAGE.sub(" ", text)
    cleaned = _WEB_ANGLE_URL.sub(" ", cleaned)
    cleaned = _WEB_IMAGE_MARKERS.sub(" ", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
