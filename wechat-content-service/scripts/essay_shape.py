"""Measure the shape of an essay so a generated one can be compared with a reference.

    python scripts/essay_shape.py reference.txt dialogue-essays/01-刻舟求剑.txt

Paragraphs are separated by blank lines; a line break inside a paragraph is a soft break.
The numbers recorded in prompts/README.md were taken with this script.
"""

import re
import statistics
import sys
from pathlib import Path

HAN = re.compile(r"[一-鿿]")


def han(text: str) -> int:
    return len(HAN.findall(text))


def spoken_share(text: str) -> float:
    """Share of characters inside quotation marks; speech may run across several paragraphs."""
    depth = inside = 0
    for char in text:
        if char == "“":
            depth += 1
        elif char == "”":
            depth = max(0, depth - 1)
        elif depth and HAN.match(char):
            inside += 1
    return inside / (han(text) or 1)


def measure(text: str) -> dict[str, str]:
    paragraphs = [block for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    sizes = [han(paragraph) for paragraph in paragraphs]
    spoken = [paragraph for paragraph in paragraphs if "“" in paragraph or "”" in paragraph]
    broken = [paragraph for paragraph in paragraphs if "\n" in paragraph.strip()]
    lines = [han(line) for line in text.splitlines() if line.strip()]
    share = len(paragraphs) or 1
    return {
        "汉字": str(han(text)),
        "含标点": str(len(re.sub(r"\s", "", text))),
        "段数": str(len(paragraphs)),
        "每段汉字 均值/中位/最长": f"{statistics.mean(sizes):.0f} / {statistics.median(sizes):.0f} / {max(sizes)}",
        "每行汉字 均值": f"{statistics.mean(lines):.0f}",
        "带引号的段": f"{len(spoken) / share:.0%}",
        "引号内的字数": f"{spoken_share(text):.0%}",
        "段内换行的段": f"{len(broken) / share:.0%}",
        "不超过12字的段": f"{sum(size <= 12 for size in sizes) / share:.0%}",
        "60字以上的段": f"{sum(size >= 60 for size in sizes) / share:.0%}",
        "问号 / 破折号 / 省略号": f"{text.count('？')} / {text.count('——')} / {text.count('……')}",
    }


def main() -> None:
    paths = [Path(argument) for argument in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    columns = [measure(path.read_text(encoding="utf-8")) for path in paths]
    print("\t".join(["", *[path.name for path in paths]]))
    for key in columns[0]:
        print("\t".join([key, *[column[key] for column in columns]]))


if __name__ == "__main__":
    main()
