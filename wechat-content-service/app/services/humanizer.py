import json
from typing import Any

from ..emotional_defaults import (
    CLEAR_SENTENCE_CATEGORY,
    CLEAR_SENTENCE_TITLE_REFERENCE,
    EMOTIONAL_TITLE_REFERENCE,
    PSYCHOLOGY_CATEGORIES,
    emotional_style_rules,
)
from . import dialogue_essay

HUMANIZER_VERSION = "humanizer-zh-v1"


def build_humanizer_prompt(
    article: dict[str, Any],
    *,
    content_type: str = "health_article",
    category: str = "",
    topic: str = "",
    source_instruction: str = "",
    emotional_title_reference: bool = False,
    target_length: int | None = None,
) -> str:
    if content_type == "emotional_article" and dialogue_essay.is_dialogue_essay(category):
        return dialogue_essay.build_polish_prompt(
            article,
            topic=topic,
            source_instruction=source_instruction,
            target_length=target_length or dialogue_essay.DEFAULT_LENGTH,
        )
    if content_type == "emotional_article":
        short_style = category.strip() == CLEAR_SENTENCE_CATEGORY
        title_reference = (
            EMOTIONAL_TITLE_REFERENCE
            + "\n润色时不重新拟题；必须保留正文中兑现候选标题的具体情节、数字、对白和处境，不能删掉或改动它们。"
            if emotional_title_reference else ""
        )
        rewrite_rules = """1. 保留一个具体处境和其中准确的小动作，把没说出口的感受写清楚；每个感受都有场景支撑，不添加另一个主题。
2. 保持初稿的“你”或“我”，不来回换。段落一至三句，允许一句话一段；不用表格、分点、加粗、破折号、英文或课本式词语（真实人名与作品名除外），sections 的 heading 全部为空字符串。
3. 保留有出处、贴合处境的心理解释及其适用范围，不删成纯抒情；不写成脱离场景的理论课、行动清单或星座排行榜，不新增“你应该、建议你、学会……”；不教挽回、不骂对方、不替人物下判决。
4. 保留有内容支撑的对照句，只删除机械连接和空话；结尾停在一句话或画面上，不补总结、升华、口号或“愿你”。
5. 标题和图片提示词原样保留，正文中支撑标题判断的细节不能删除。引用的原句、作者和作品保留，出处自然写在句子里，不新增未经核实的引文、研究、投稿或作者亲历；不附论文链接或参考文献清单。
6. 精修后正文长度保持在原文的 85% 到 110%。在心里按直接性、节奏、信任度、真实性、精炼度自检，目标至少 45/50，不把评分写进结果。"""
        if short_style:
            title_reference = CLEAR_SENTENCE_TITLE_REFERENCE if emotional_title_reference else ""
            rewrite_rules = """1. 保留有依据、有锋芒的短句和逐步说透问题的顺序，删空话、重复和机械对仗，不把文章改成故事。
2. 不描写具体场景，不添加动作、时间地点或环境铺垫；opening、conclusion 和每个 paragraphs 项均不超过50字（含标点），每段一至两句，heading 为空。
3. 保留至少一处来源库中与问题有关的心理学家署名及观点，适用条件不能删；不够短就分段说明。不输出论文链接、DOI或参考文献清单。
4. 标题和图片提示词原样保留。判断要与正文一致，不断言他人动机，不羞辱读者。结尾用一句与主题有关的判断收住。
5. 保留有内容的有力表达，不因去 AI 味删除所有金句；避免同一种句式反复出现，不堆鸡汤，不为了凑篇幅增加长句。"""
        if category.strip() in PSYCHOLOGY_CATEGORIES:
            rewrite_rules += "\n7. 恋爱心理每篇至少保留一处来自来源库的心理学家佐证，作者和观点不能删，保留自然融入正文的年份或作品名；移除论文链接、网址、DOI和独立参考文献清单。初稿缺少时，用下方已提供且已核对的来源补足与场景相关的概述；这不属于新增未经核实的材料。不要把概述变成作者原话。"
        return f"""
你是一位有多年公众号经验的中文情感编辑。请按照 Humanizer-zh 的规则精修下面的情感文章，
让它像一个愿意听完、说话自然且有分寸的人写的。

必须保留：
1. title_candidates、selected_title 和 image_prompts 原样保留。
2. 保留原文的事实边界、关系背景和安全提醒，不新增未经核实的研究或数据，不编造诊断、投稿者或个人经历。
3. 保持完整 JSON 结构，只输出 JSON，不要解释修改过程或给出评分。

本篇原有主题与写作要求：
{source_instruction}

体裁规则：
{emotional_style_rules(category, topic)}
{"本篇按清醒短句规则精修，不加入场景描写。" if short_style else "所有情感分类都按本次共鸣型 Prompt 精修，保留处境、叙述人称和细节，不恢复旧版分体裁写法。"}

{title_reference}

重点改写 summary、opening、sections 和 conclusion：
{rewrite_rules}

待精修文章 JSON：
{json.dumps(article, ensure_ascii=False)}
""".strip()

    return f"""
你是一位有多年公众号经验的中文编辑。请按照 Humanizer-zh 的规则，对下面的健康文章
做一次去除 AI 写作痕迹的精修。

必须保留：
1. title_candidates、selected_title 和 image_prompts 原样保留。
2. 所有健康事实、风险提醒、数字、适用条件和行动建议，不得新增未经原文支持的事实。
3. 保持完整 JSON 结构，只输出 JSON，不要解释修改过程或给出评分。

重点改写 summary、opening、sections 和 conclusion：
1. 删除“此外、值得注意的是、至关重要、深入探讨、总而言之”等机械连接词。
2. 避免每段相同句式、连续三段式列举、过多“第一/第二/第三”和否定式排比。
3. 长短句交替；允许自然停顿和口语化表达，但不要故意写错字。
4. 用具体生活动作和场景替代空泛总结，减少宣传腔、正确但无用的套话。
5. 不使用“专家认为”等模糊权威，不虚构个人经历、医生、患者或统计数字。
6. 不过度使用破折号、粗体、表情符号、金句和整齐得像提纲的段落。
7. 结尾落到一件能做的小事或必要提醒，不写“未来可期”式通用积极结论。
8. 适合爱养生的中老年读者：自然、亲切、有分寸，不居高临下。
9. 精修后正文长度保持在原文的 85% 到 110%。
10. 在心里按直接性、节奏、信任度、真实性、精炼度自检，目标至少 45/50，
    但不要把评分写进结果。

待精修文章 JSON：
{json.dumps(article, ensure_ascii=False)}
""".strip()
