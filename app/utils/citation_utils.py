"""引用后置校验。

模型有时会凭空写出 [7] 这种不存在的引用编号。这里在答案落库/返回前做一次
强制校验：越界编号直接删除，并把该处降级为无出处表述，保证「答案里出现的
每个编号都能在 references 里找到对应切片」。
"""
import re

# 只认 1~2 位数字，避免把 [2024] 这类年份误判成引用
CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")

# 答案里出现的链接。模型很爱在"帮我配张图"的语境下编造 example.com 之类的 URL，
# 这类东西必须拦：链接错了比没有链接更糟（用户会以为官方文档里有图）。
URL_PATTERN = re.compile(r"https?://[^\s<>）)】\]\"']+", re.I)


def extract_citations(text: str) -> list[int]:
    """抽取文本里出现的引用编号（按出现顺序，含重复）。"""
    return [int(match.group(1)) for match in CITATION_PATTERN.finditer(text or "")]


def validate_and_fix(answer: str, reference_count: int, grounded_text: str = "") -> tuple[str, dict]:
    """校验并修正引用编号，并（在给了 grounded_text 时）拦掉凭空编造的链接。

    返回 (修正后的答案, 校验信息)。越界编号被删除；若某句因此变成"无出处"，
    不做额外改写（避免动到模型的表述），由上层决定是否提示。
    grounded_text：允许出现链接的文本（检索上下文 + 引用条目自带的 url），
    答案里的 URL 不在其中就视为编造，直接删掉。
    """
    answer = answer or ""
    invalid: list[int] = []
    seen: list[int] = []

    def _replace(match: re.Match) -> str:
        number = int(match.group(1))
        if 1 <= number <= reference_count:
            if number not in seen:
                seen.append(number)
            return match.group(0)
        invalid.append(number)
        return ""

    fixed = CITATION_PATTERN.sub(_replace, answer)

    # 删除编号后可能留下孤立的空格或重复标点，做一次轻量规整
    fixed = re.sub(r"[ \t]{2,}", " ", fixed)
    fixed = re.sub(r"\s+([，。；！？])", r"\1", fixed)

    fabricated: list[str] = []
    if grounded_text:
        def _drop_ungrounded(match: re.Match) -> str:
            url = match.group(0)
            if url in grounded_text:
                return url
            fabricated.append(url)
            return ""

        fixed = URL_PATTERN.sub(_drop_ungrounded, fixed)
        if fabricated:
            # 链接被抽走后，残留的【图片】标题行和空尖括号一并清掉，不留空壳
            fixed = re.sub(r"【图片】[ \t]*\n?", "", fixed)
            fixed = re.sub(r"<\s*>", "", fixed)
            fixed = re.sub(r"[ \t]{2,}", " ", fixed)
            fixed = re.sub(r"\n{3,}", "\n\n", fixed)

    info = {
        "reference_count": reference_count,
        "valid_citations": seen,
        "invalid_citations": sorted(set(invalid)),
        "has_invalid": bool(invalid),
        "cited_count": len(seen),
        "all_referenced": reference_count > 0 and len(seen) == reference_count,
        "fabricated_urls": sorted(set(fabricated)),
    }
    return fixed.strip(), info
