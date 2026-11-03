import re

RISK_PATTERNS: dict[str, str] = {
    r"根治|彻底治愈|包治": "包含绝对治疗承诺",
    r"不用吃药|立即停药|替代药物": "可能诱导停止或替代正规治疗",
    r"医生不会告诉你|医院不愿告诉你": "包含煽动性医疗表达",
    r"癌细胞最怕|血管全部通了|排出所有毒素": "包含夸大或伪科学表达",
}

DOSAGE_FLAG = "可能包含具体药物或补充剂剂量"
DOSAGE_AMOUNT_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百半两]+)\s*"
    r"(?:克|毫克|微克|片|粒|毫升|单位)"
)
DOSAGE_FREQUENCY_PATTERN = re.compile(r"每天|每日|一日|每早|每晚|每次")
MEDICATION_CONTEXT_PATTERN = re.compile(
    r"用药|药物|服药|服用|口服|剂量|补充剂|保健品|维生素|胶囊|片剂|冲剂"
)


def _contains_specific_dosage(text: str) -> bool:
    # Keep the amount, frequency, and medication context in the same sentence.
    # This avoids joining unrelated phrases across JSON paragraphs, such as
    # “每天吃 200 克水果” and a later “一日三餐”.
    sentences = re.split(r"[。！？!?；;\n]+", text)
    return any(
        DOSAGE_AMOUNT_PATTERN.search(sentence)
        and DOSAGE_FREQUENCY_PATTERN.search(sentence)
        and MEDICATION_CONTEXT_PATTERN.search(sentence)
        for sentence in sentences
    )


def scan_article(text: str) -> list[str]:
    flags: list[str] = []
    for pattern, message in RISK_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            flags.append(message)
    if _contains_specific_dosage(text):
        flags.append(DOSAGE_FLAG)
    return flags
