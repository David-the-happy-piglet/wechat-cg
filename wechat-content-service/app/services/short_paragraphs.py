"""Keep the short-form article's paragraph limit without another model call."""

import re


def split_short_paragraphs(text: str, limit: int = 50) -> list[str]:
    paragraphs = []
    for line in text.splitlines():
        remaining = line.strip()
        while len(remaining) > limit:
            # Prefer a full sentence boundary, then a clause boundary.
            end = 0
            for pattern in (r'[。！？!?][”」』"]?', r'[，；：,;:]'):
                boundaries = list(re.finditer(pattern, remaining[:limit]))
                if boundaries:
                    end = boundaries[-1].end()
                    break
            end = end or limit
            paragraphs.append(remaining[:end])
            remaining = remaining[end:].strip()
        if remaining:
            paragraphs.append(remaining)
    return paragraphs


def normalize_short_article(article: dict) -> None:
    """Preserve all text and its order, including overflow from opening/ending."""
    opening = split_short_paragraphs(article["opening"])
    conclusion = split_short_paragraphs(article["conclusion"])
    for section in article["sections"]:
        section["heading"] = ""
        section["paragraphs"] = [
            paragraph
            for original in section["paragraphs"]
            for paragraph in split_short_paragraphs(original)
        ]
    article["opening"] = opening[0] if opening else ""
    article["sections"][0]["paragraphs"][:0] = opening[1:]
    article["conclusion"] = conclusion[-1] if conclusion else ""
    article["sections"][-1]["paragraphs"].extend(conclusion[:-1])
