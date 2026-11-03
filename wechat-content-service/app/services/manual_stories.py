"""Human-reviewed handoff to a chat model and back into the publishing flow."""

import json
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..models import Article, Asset, GenerationRun, ManualStoryDraft
from ..story_defaults import (
    DEFAULT_STORY_IMAGE_STYLE,
    FOLKLORE_ELEMENTS,
    STORY_EDITORIAL_RULES,
    folklore_story_guidance,
)
from .images import build_image_provider
from .llm import build_llm_provider
from .renderer import render_story_html

MANUAL_STORY_MODES = {
    "auto": "自由选择：微恐、心理悬疑或怪谈",
    "everyday_uncanny": "日常微恐：熟悉生活里出现一个偏差",
    "psychological_suspense": "心理悬疑：误认、隐瞒与关系中的不安",
    "relationship_mystery": "关系悬疑：亲密关系或家庭里的旧账",
    "transaction_mystery": "交易悬疑：交换、欠账和不对等的承诺",
    "object_mystery": "旧物悬疑：来历不明的物件或遗失物",
    "workplace_mystery": "职场悬疑：同事、交接与反常流程",
    "urban_legend": "都市怪谈：流言、禁忌与无法验证的异常",
    "rule_tale": "规则怪谈：日常场所里一条不合常理的规矩",
    "time_dislocation": "时间错位：日期、记忆或顺序对不上",
    "institutional_tale": "机构怪谈：窗口、登记与无人解释的手续",
    "near_future": "近未来寓言：便利规则逐渐显露代价",
    "folk_horror": "民俗志异：地方传说与旧俗中的异常规则",
}

MANUAL_STORY_DIRECTION_RULES = {
    "psychological_suspense": "允许没有超自然现象；恐惧来自认知差、信息不对称与选择的后果，不以精神疾病标签解释一切。",
    "relationship_mystery": "让亲近者的一句普通话在结尾改义；人物动机要可信，不靠所有人集体撒谎。",
    "transaction_mystery": "用一笔小交易或旧账推动选择；账目、交换条件和代价在前文可核对。",
    "object_mystery": "核心物件必须反复出现并改变含义；不要靠物件突然新增能力解谜。",
    "workplace_mystery": "把异常藏在交接、排班或同事的习惯里；不用读者不懂的行业黑话。",
    "urban_legend": "用一条可传播的传闻或禁忌牵动人物；保留真假余地，不把网络流言冒充真实新闻。",
    "rule_tale": "规则只有一个核心，至少两条限制提前以日常话埋下；人物不是因为犯蠢才触发后果。",
    "time_dislocation": "时间线可回溯；让错位来自具体记录和人物选择，不能靠结尾突然改写时间法则。",
    "institutional_tale": "手续或登记用大白话说明；机构要虚构，恐惧来自流程与人的需要相互卡住。",
    "near_future": "技术只做一个可理解的变化；不堆设定、术语或产品说明，代价要落在日常细节。",
    "folk_horror": "每篇只用一个背景与一项主民俗；人物认知、传闻和真实教义须分清。",
}


def manual_story_mode_label(value: str) -> str:
    return MANUAL_STORY_MODES.get(value, MANUAL_STORY_MODES["auto"])


def suggest_idea(settings: Settings, story_mode: str) -> str:
    """Generate only a premise; the editor must review it before a prompt exists."""
    llm = build_llm_provider(settings)
    label = manual_story_mode_label(story_mode)
    folk_hint = "子方向：民俗志异。" if story_mode == "folk_horror" else ""
    result = llm.generate_json(
        f"[MICRO_HORROR_TOPIC] 为精品微恐短篇提出一个原创、具体、可写的灵感。"
        f"方向：{label}。{folk_hint}"
        "只给一条点子，包含人物处境、一个异常、一个选择，不写正文。"
        '返回 JSON：{"idea":"..."}。',
        operation="manual_story_idea",
    )
    idea = result.get("idea") or result.get("topic")
    if not isinstance(idea, str) or not idea.strip():
        raise ValueError("灵感生成没有返回可用内容，请手动输入灵感")
    return idea.strip()[:1000]


def build_claude_prompt(draft: ManualStoryDraft) -> str:
    direction = manual_story_mode_label(draft.story_mode)
    direction_rule = MANUAL_STORY_DIRECTION_RULES.get(draft.story_mode, "")
    folklore = ""
    if draft.story_mode == "folk_horror":
        folklore = (
            f"\n核心民俗素材：{FOLKLORE_ELEMENTS[draft.folklore_element]}。"
            "只取一种主素材，规则须参与因果。\n"
            + folklore_story_guidance(draft.story_mode)
        )
    return (
        "请写一篇可直接发表的原创中文微恐／心理悬疑／怪谈短篇小说。"
        "这是一篇完整正文，不是大纲或写作分析。\n\n"
        f"审核过的灵感：{draft.idea}\n"
        f"方向：{direction}。{direction_rule}{folklore}\n"
        f"长度：约 {draft.target_length} 个汉字。\n"
        f"读者边界：{draft.reader_boundary}\n\n"
        f"写作要求：\n{draft.style_prompt}\n\n"
        f"统一编辑规范：\n{STORY_EDITORIAL_RULES}\n\n"
        "质量检查（请在内部完成，不要输出检查过程）：异常规则与人物选择要互相牵动；"
        "伏笔能回收；反转可由前文推断；结尾留余味但不故弄玄虚。"
        "避免模仿具体作品的标志性情节或句子。\n\n"
        "输出格式：先写“正文标题：……”和“推送标题：……”各一行；空一行后直接写小说正文。"
        "用自然段分隔，正文不要 Markdown 标题、编号、大纲、点评、配图说明或写作说明。"
    )


def _content_from_body(body_text: str) -> dict:
    normalized = body_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if len(blocks) == 1:
        blocks = [part.strip() for part in normalized.splitlines() if part.strip()]
    if len(blocks) < 2:
        raise ValueError("正文请至少分成两个自然段，方便排版预览")
    return {
        "opening": blocks[0],
        "sections": [{"heading": "", "paragraphs": blocks[1:-1]}],
        "conclusion": blocks[-1],
    }


def _render_article(article: Article) -> None:
    assets = sorted(article.assets, key=lambda item: item.position)
    urls = [asset.source_url or "" for asset in assets]
    credits = [
        {key: getattr(asset, key) for key in (
            "source_kind", "source_page_url", "source_author", "license_name", "license_url"
        )}
        for asset in assets
    ]
    article.content_html = render_story_html(json.loads(article.content_json), urls, credits)


def import_story(
    session: Session,
    draft: ManualStoryDraft,
    *,
    title: str,
    push_title: str,
    summary: str,
    body_text: str,
    audience_id: int,
) -> Article:
    title = title.strip()
    push_title = push_title.strip() or title
    body_text = body_text.strip()
    if not title or len(title) > 300 or len(push_title) > 300:
        raise ValueError("请填写各不超过 300 字的正文标题和推送标题")
    if len(body_text) < 120 or len(body_text) > 30000:
        raise ValueError("正文需要 120–30000 字")
    if draft.status == "approved":
        raise ValueError("成稿已审核通过，不能再修改")
    content = _content_from_body(body_text)
    content["body_title"] = title
    article = session.get(Article, draft.article_id) if draft.article_id else None
    if article is None:
        article = Article(
            content_type="micro_horror",
            story_mode=draft.story_mode,
            audience_profile_id=audience_id,
            topic=draft.idea[:300],
            category="微恐短篇",
            target_length=draft.target_length,
            title_candidates_json=json.dumps([push_title], ensure_ascii=False),
            selected_title=push_title,
            summary="",
            content_json="{}",
            content_html="",
            status="pending_review",
            llm_provider="manual-claude",
            model_name="Claude chat (manual)",
            prompt_version="manual-claude-v1",
        )
        session.add(article)
        session.flush()
        draft.article_id = article.id
    article.selected_title = push_title
    article.title_candidates_json = json.dumps([push_title], ensure_ascii=False)
    article.summary = summary.strip() or content["opening"][:100]
    article.content_json = json.dumps(content, ensure_ascii=False)
    _render_article(article)
    draft.title = title
    draft.push_title = push_title
    draft.summary = summary.strip()
    draft.body_text = body_text
    draft.status = "imported"
    session.add(GenerationRun(
        article_id=article.id,
        step="manual_import",
        provider="manual-claude",
        model="Claude chat (manual)",
        input_text=draft.prompt_text,
        output_text=json.dumps(
            {"body_title": title, "push_title": push_title, "characters": len(body_text)},
            ensure_ascii=False,
        ),
        status="completed",
    ))
    session.commit()
    return article


def add_story_images(
    session: Session,
    draft: ManualStoryDraft,
    settings: Settings,
    *,
    cover_prompt: str,
    inline_prompt: str,
    image_count: int,
) -> None:
    if draft.status == "approved" or not draft.article_id:
        raise ValueError("请先贴回成稿；审核通过后不能再改配图")
    if image_count < 1 or image_count > 3:
        raise ValueError("配图数量只能是 1–3 张")
    article = draft.article
    provider = build_image_provider(settings)
    existing = list(session.scalars(select(Asset).where(Asset.article_id == article.id)))
    for asset in existing:
        session.delete(asset)
    session.flush()
    for index in range(image_count):
        scene = cover_prompt.strip() if index == 0 else inline_prompt.strip()
        if not scene:
            scene = f"《{draft.title}》：{draft.idea[:180]}"
        prompt = f"{scene}。{DEFAULT_STORY_IMAGE_STYLE}。不得出现文字或剧透结尾。"
        generated = provider.generate(prompt, draft.title, index, draft.idea)
        session.add(Asset(
            article_id=article.id,
            asset_type="cover" if index == 0 else "inline",
            position=index,
            prompt=prompt,
            local_path=generated.get("local_path"),
            source_url=generated.get("source_url"),
            source_kind=generated.get("source_kind") or provider.name,
            source_page_url=generated.get("source_page_url"),
            source_author=generated.get("source_author"),
            license_name=generated.get("license_name"),
            license_url=generated.get("license_url"),
        ))
    session.flush()
    session.expire(article, ["assets"])
    _render_article(article)
    session.commit()


def approve_story(session: Session, draft: ManualStoryDraft) -> None:
    if not draft.article_id or draft.status != "imported":
        raise ValueError("请先贴回并预览成稿")
    if not session.scalar(select(Asset.id).where(Asset.article_id == draft.article_id).limit(1)):
        raise ValueError("请先生成至少一张封面图")
    draft.status = "approved"
    draft.article.status = "approved"
    session.commit()
