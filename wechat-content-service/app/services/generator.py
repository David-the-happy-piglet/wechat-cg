import json
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings
from ..content_workflows import (
    ContentWorkflow,
    get_content_workflow,
    register_generation_handler,
)
from ..emotional_defaults import CLEAR_SENTENCE_CATEGORY
from ..models import (
    Article,
    Asset,
    AudienceProfile,
    GenerationRun,
    StoryOutlineDraft,
)
from ..schemas import ArticleGenerateRequest, StoryOutlineCreate
from .humanizer import build_humanizer_prompt
from .images import build_image_provider, normalize_search_query
from .llm import build_llm_provider
from .prompts import (
    build_article_prompt,
    build_emotional_article_prompt,
    build_story_draft_prompt,
    build_story_outline_prompt,
    build_story_review_prompt,
)
from .short_paragraphs import normalize_short_article
from .video.source import fetch_article


def audience_to_context(audience: AudienceProfile) -> dict[str, Any]:
    return {
        "slug": audience.slug,
        "name": audience.name,
        "age_range": audience.age_range,
        "description": audience.description,
        "interests": json.loads(audience.interests_json),
        "writing_style": audience.writing_style,
        "title_preferences": json.loads(audience.title_preferences_json),
        "article_length": audience.article_length,
        "image_style": audience.image_style,
        "prohibited_topics": json.loads(audience.prohibited_topics_json),
    }


def _validate_result(result: dict[str, Any]) -> None:
    required = {
        "title_candidates",
        "selected_title",
        "summary",
        "opening",
        "sections",
        "conclusion",
        "image_prompts",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"LLM response missing keys: {', '.join(sorted(missing))}")
    if not isinstance(result["title_candidates"], list) or not result["title_candidates"]:
        raise ValueError("LLM response must contain title candidates")
    if result["selected_title"] not in result["title_candidates"]:
        raise ValueError("Selected title must be one of the title candidates")
    if not isinstance(result["sections"], list) or not result["sections"]:
        raise ValueError("LLM response must contain sections")
    for section in result["sections"]:
        if not isinstance(section, dict):
            raise TypeError("Each section must be an object")
        if not isinstance(section.get("heading", ""), str):
            raise TypeError("Each section heading must be a string")
        paragraphs = section.get("paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs:
            raise ValueError("Each section must contain paragraphs")
        if not all(isinstance(item, str) and item.strip() for item in paragraphs):
            raise ValueError("Each paragraph must be a non-empty string")
    if not isinstance(result["image_prompts"], list) or not result["image_prompts"]:
        raise ValueError("LLM response image_prompts must be a non-empty list")
    for item in result["image_prompts"]:
        if not isinstance(item, dict):
            raise TypeError("Each image prompt must be an object")
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ValueError("Each image prompt must contain a prompt string")


def _validate_story_outline(outline: dict[str, Any]) -> None:
    required = {
        "premise",
        "protagonist",
        "setting",
        "anomaly",
        "reader_assumption",
        "truth",
        "beats",
        "foreshadows",
        "ending",
    }
    missing = required - set(outline)
    if missing:
        raise ValueError(f"Story outline missing keys: {', '.join(sorted(missing))}")
    for key in ("premise", "reader_assumption", "truth"):
        if not isinstance(outline.get(key), str) or not outline[key].strip():
            raise ValueError(f"Story outline {key} must be a non-empty string")
    protagonist = outline.get("protagonist")
    if not isinstance(protagonist, dict):
        raise TypeError("Story outline protagonist must be an object")
    for key in ("identity", "desire", "critical_choice"):
        if not isinstance(protagonist.get(key), str) or not protagonist[key].strip():
            raise ValueError(f"Story outline protagonist.{key} must be a non-empty string")
    anomaly = outline.get("anomaly")
    if not isinstance(anomaly, dict):
        raise TypeError("Story outline anomaly must be an object")
    for key in ("rule", "limit", "cost"):
        if not isinstance(anomaly.get(key), str) or not anomaly[key].strip():
            raise ValueError(f"Story outline anomaly.{key} must be a non-empty string")
    setting = outline.get("setting")
    if not isinstance(setting, dict):
        raise TypeError("Story outline setting must be an object")
    for key in ("place", "normal_routine"):
        if not isinstance(setting.get(key), str) or not setting[key].strip():
            raise ValueError(f"Story outline setting.{key} must be a non-empty string")
    beats = outline.get("beats")
    if not isinstance(beats, list) or not 5 <= len(beats) <= 7:
        raise ValueError("Story outline must contain 5 to 7 causal beats")
    for index, beat in enumerate(beats):
        if not isinstance(beat, dict):
            raise TypeError(f"Story outline beat {index} must be an object")
        for key in ("phase", "event", "causes_next"):
            if not isinstance(beat.get(key), str) or not beat[key].strip():
                raise ValueError(f"Story outline beat {index}.{key} must be non-empty")
    foreshadows = outline.get("foreshadows")
    if not isinstance(foreshadows, list) or not 2 <= len(foreshadows) <= 4:
        raise ValueError("Story outline must contain 2 to 4 foreshadows")
    for index, clue in enumerate(foreshadows):
        if not isinstance(clue, dict):
            raise TypeError(f"Story outline foreshadow {index} must be an object")
        for key in ("detail", "surface_meaning", "true_meaning", "payoff"):
            if not isinstance(clue.get(key), str) or not clue[key].strip():
                raise ValueError(f"Story outline foreshadow {index}.{key} must be non-empty")
    ending = outline.get("ending")
    if not isinstance(ending, dict):
        raise TypeError("Story outline ending must be an object")
    for key in ("reveal", "final_action_or_image"):
        if not isinstance(ending.get(key), str) or not ending[key].strip():
            raise ValueError(f"Story outline ending.{key} must be a non-empty string")


def _validate_story_result(
    result: dict[str, Any],
    *,
    title_count: int,
    image_count: int,
) -> None:
    _validate_result(result)
    if len(result["title_candidates"]) != title_count:
        raise ValueError(f"Story response must contain exactly {title_count} title candidates")
    if not all(isinstance(title, str) and title.strip() for title in result["title_candidates"]):
        raise ValueError("Story title candidates must be non-empty strings")
    for key in ("summary", "opening", "conclusion"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"Story response {key} must be a non-empty string")
    if len(result["image_prompts"]) != image_count:
        raise ValueError(f"Story response must contain exactly {image_count} image prompts")
    for index, image in enumerate(result["image_prompts"]):
        if image.get("position") != index:
            raise ValueError("Story image prompt positions must be consecutive from zero")
        expected_type = "cover" if index == 0 else "inline"
        if image.get("asset_type") != expected_type:
            raise ValueError(f"Story image prompt {index} must use asset_type {expected_type}")


def _validate_story_review(review: dict[str, Any]) -> dict[str, Any]:
    if review.get("verdict") not in {"pass", "revised"}:
        raise ValueError("Story review verdict must be pass or revised")
    if not isinstance(review.get("scores"), dict):
        raise TypeError("Story review scores must be an object")
    if not isinstance(review.get("checks"), dict):
        raise TypeError("Story review checks must be an object")
    if not isinstance(review.get("issues"), list):
        raise TypeError("Story review issues must be a list")
    revised = review.get("revised_article")
    if not isinstance(revised, dict):
        raise TypeError("Story review must contain revised_article")
    return revised


def _normalize_image_search_queries(
    result: dict[str, Any],
    fallback: str,
) -> None:
    for item in result["image_prompts"]:
        item["search_query"] = normalize_search_query(
            item.get("search_query"),
            fallback,
        )


def _run_data(
    *,
    step: str,
    prompt: str,
    output: dict[str, Any],
    started: float,
    llm,
) -> dict[str, Any]:
    return {
        "step": step,
        "input_text": prompt,
        "output_text": json.dumps(output, ensure_ascii=False),
        "status": "completed",
        "duration_ms": int((time.monotonic() - started) * 1000),
        "prompt_tokens": llm.usage.prompt_tokens,
        "completion_tokens": llm.usage.completion_tokens,
        "total_tokens": llm.usage.total_tokens,
    }


def _recent_story_signatures(
    session: Session,
    content_type: str,
    limit: int = 20,
) -> list[str]:
    outputs = session.scalars(
        select(GenerationRun.output_text)
        .join(Article, GenerationRun.article_id == Article.id)
        .where(
            Article.content_type == content_type,
            GenerationRun.step == "story_outline",
            GenerationRun.status == "completed",
        )
        .order_by(GenerationRun.id.desc())
        .limit(limit)
    ).all()
    signatures: list[str] = []
    for output in outputs:
        try:
            outline = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            continue
        anomaly = outline.get("anomaly") if isinstance(outline, dict) else {}
        ending = outline.get("ending") if isinstance(outline, dict) else {}
        if not isinstance(anomaly, dict) or not isinstance(ending, dict):
            continue
        signatures.append(
            "异常："
            f"{str(anomaly.get('rule') or '')[:100]}｜"
            f"真相：{str(outline.get('truth') or '')[:100]}｜"
            f"结尾：{str(ending.get('reveal') or '')[:100]}"
        )
    return signatures


@register_generation_handler("health_article")
def _generate_health_article(
    *,
    workflow: ContentWorkflow,
    session: Session,
    llm,
    request: ArticleGenerateRequest,
    effective_topic: str,
    audience_context: dict[str, Any],
    source_material: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    prompt = build_article_prompt(
        topic=effective_topic,
        category=request.category,
        audience=audience_context,
        title_count=request.title_count,
        image_count=request.image_count,
        headline_intensity=request.headline_intensity,
        source_material=source_material,
        source_instruction=request.source_prompt,
    )
    context = {
        "topic": effective_topic,
        "category": request.category,
        "audience": audience_context,
        "title_count": request.title_count,
        "image_count": request.image_count,
        "target_length": audience_context["article_length"],
        "headline_intensity": request.headline_intensity,
        "content_type": workflow.content_type,
    }
    started = time.monotonic()
    draft_result = llm.generate_article(prompt, context)
    _validate_result(draft_result)
    _normalize_image_search_queries(draft_result, effective_topic)
    generation_runs = [
        _run_data(
            step="article",
            prompt=prompt,
            output=draft_result,
            started=started,
            llm=llm,
        )
    ]

    result = draft_result
    humanizer_applied = False
    if request.humanize and llm.name != "mock":
        humanizer_prompt = build_humanizer_prompt(draft_result)
        humanizer_started = time.monotonic()
        try:
            humanized = llm.generate_article(humanizer_prompt, context)
            _validate_result(humanized)
            for preserved_key in (
                "title_candidates",
                "selected_title",
                "image_prompts",
            ):
                humanized[preserved_key] = draft_result[preserved_key]
            result = humanized
            generation_runs.append(
                _run_data(
                    step="humanize",
                    prompt=humanizer_prompt,
                    output=humanized,
                    started=humanizer_started,
                    llm=llm,
                )
            )
            humanizer_applied = True
        except Exception as exc:  # noqa: BLE001 - retain a usable first draft
            generation_runs.append(
                {
                    "step": "humanize",
                    "input_text": humanizer_prompt,
                    "output_text": "",
                    "status": "failed",
                    "error_message": str(exc),
                    "duration_ms": int((time.monotonic() - humanizer_started) * 1000),
                }
            )

    result["content_type"] = workflow.content_type
    result["story_mode"] = workflow.default_story_mode
    version = (
        workflow.enhanced_prompt_version
        if humanizer_applied
        else workflow.base_prompt_version
    )
    return result, generation_runs, version


@register_generation_handler("emotional_article")
def _generate_emotional_article(
    *,
    workflow: ContentWorkflow,
    session: Session,
    llm,
    request: ArticleGenerateRequest,
    effective_topic: str,
    audience_context: dict[str, Any],
    source_material: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    del session
    prompt = build_emotional_article_prompt(
        topic=effective_topic,
        category=request.category,
        audience=audience_context,
        title_count=request.title_count,
        image_count=request.image_count,
        headline_intensity=request.headline_intensity,
        emotional_title_reference=request.emotional_title_reference,
        source_material=source_material,
        source_instruction=request.source_prompt,
    )
    context = {
        "topic": effective_topic,
        "category": request.category,
        "audience": audience_context,
        "title_count": request.title_count,
        "image_count": request.image_count,
        "target_length": audience_context["article_length"],
        "headline_intensity": request.headline_intensity,
        "content_type": workflow.content_type,
    }
    started = time.monotonic()
    draft_result = llm.generate_article(prompt, context)
    _validate_result(draft_result)
    _normalize_image_search_queries(draft_result, effective_topic)
    generation_runs = [
        _run_data(
            step="emotional_article",
            prompt=prompt,
            output=draft_result,
            started=started,
            llm=llm,
        )
    ]

    result = draft_result
    humanizer_applied = False
    if request.humanize and llm.name != "mock":
        humanizer_prompt = build_humanizer_prompt(
            draft_result,
            content_type=workflow.content_type,
            category=request.category,
            topic=effective_topic,
            source_instruction=request.source_prompt,
            emotional_title_reference=request.emotional_title_reference,
        )
        humanizer_started = time.monotonic()
        try:
            humanized = llm.generate_article(humanizer_prompt, context)
            _validate_result(humanized)
            for preserved_key in (
                "title_candidates",
                "selected_title",
                "image_prompts",
            ):
                humanized[preserved_key] = draft_result[preserved_key]
            result = humanized
            generation_runs.append(
                _run_data(
                    step="humanize",
                    prompt=humanizer_prompt,
                    output=humanized,
                    started=humanizer_started,
                    llm=llm,
                )
            )
            humanizer_applied = True
        except Exception as exc:  # noqa: BLE001 - retain a usable first draft
            generation_runs.append(
                {
                    "step": "humanize",
                    "input_text": humanizer_prompt,
                    "output_text": "",
                    "status": "failed",
                    "error_message": str(exc),
                    "duration_ms": int(
                        (time.monotonic() - humanizer_started) * 1000
                    ),
                }
            )

    if request.category.strip() == CLEAR_SENTENCE_CATEGORY:
        normalize_short_article(result)
    result["content_type"] = workflow.content_type
    result["story_mode"] = workflow.default_story_mode
    version = (
        workflow.enhanced_prompt_version
        if humanizer_applied
        else workflow.base_prompt_version
    )
    return result, generation_runs, version


def _write_micro_horror_from_outline(
    *,
    workflow: ContentWorkflow,
    llm,
    request: ArticleGenerateRequest,
    effective_topic: str,
    audience_context: dict[str, Any],
    outline: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    context = {
        "topic": effective_topic,
        "category": request.category,
        "title_count": request.title_count,
        "image_count": request.image_count,
        "target_length": audience_context["article_length"],
        "headline_intensity": request.headline_intensity,
        "content_type": workflow.content_type,
        "story_mode": request.story_mode,
    }
    generation_runs: list[dict[str, Any]] = []
    draft_prompt = build_story_draft_prompt(
        topic=effective_topic,
        category=request.category,
        outline=outline,
        title_count=request.title_count,
        image_count=request.image_count,
        target_length=audience_context["article_length"],
        headline_intensity=request.headline_intensity,
        reader_boundary=request.reader_boundary,
        story_mode=request.story_mode,
        source_instruction=request.source_prompt,
    )
    draft_started = time.monotonic()
    draft = llm.generate_json(
        draft_prompt,
        system_prompt=(
            "你是中文微恐短篇作者。用克制、具体的中文写作，"
            "严格遵守给定方案，只输出合法 JSON。"
        ),
        operation="micro_horror_draft",
        context={**context, "story_outline": outline},
    )
    _validate_story_result(
        draft,
        title_count=request.title_count,
        image_count=request.image_count,
    )
    generation_runs.append(
        _run_data(
            step="story_draft",
            prompt=draft_prompt,
            output=draft,
            started=draft_started,
            llm=llm,
        )
    )

    result = draft
    reviewed = False
    if request.humanize:
        review_prompt = build_story_review_prompt(
            draft=draft,
            outline=outline,
            target_length=audience_context["article_length"],
            title_count=request.title_count,
            image_count=request.image_count,
            reader_boundary=request.reader_boundary,
            story_mode=request.story_mode,
            source_instruction=request.source_prompt,
        )
        review_started = time.monotonic()
        try:
            review = llm.generate_json(
                review_prompt,
                system_prompt=(
                    "你是严格的中文悬疑小说编辑。保护因果、视角公平与伏笔，"
                    "去除机械语言，只输出合法 JSON。"
                ),
                operation="micro_horror_review",
                context={
                    **context,
                    "story_outline": outline,
                    "story_draft": draft,
                },
            )
            revised = _validate_story_review(review)
            _validate_story_result(
                revised,
                title_count=request.title_count,
                image_count=request.image_count,
            )
            result = revised
            generation_runs.append(
                _run_data(
                    step="story_review",
                    prompt=review_prompt,
                    output=review,
                    started=review_started,
                    llm=llm,
                )
            )
            reviewed = True
        except Exception as exc:  # noqa: BLE001 - keep the valid story draft
            generation_runs.append(
                {
                    "step": "story_review",
                    "input_text": review_prompt,
                    "output_text": "",
                    "status": "failed",
                    "error_message": str(exc),
                    "duration_ms": int((time.monotonic() - review_started) * 1000),
                }
            )

    result["content_type"] = workflow.content_type
    resolved_mode = result.get("story_mode")
    concrete_modes = {
        mode.value
        for mode in workflow.story_modes
        if mode.value != workflow.default_story_mode
    }
    if request.story_mode in concrete_modes:
        resolved_mode = request.story_mode
    elif resolved_mode not in concrete_modes:
        subgenre = str(outline.get("subgenre") or "")
        if "民俗" in subgenre:
            resolved_mode = "folk_horror"
        elif "近未来" in subgenre:
            resolved_mode = "near_future"
        elif "日常" in subgenre:
            resolved_mode = "everyday_uncanny"
        elif request.story_mode != workflow.default_story_mode:
            resolved_mode = request.story_mode
        else:
            resolved_mode = workflow.fallback_story_mode
    result["story_mode"] = resolved_mode
    _normalize_image_search_queries(result, effective_topic)
    version = (
        workflow.enhanced_prompt_version if reviewed else workflow.base_prompt_version
    )
    return result, generation_runs, version


@register_generation_handler("micro_horror")
def _generate_micro_horror(
    *,
    workflow: ContentWorkflow,
    session: Session,
    llm,
    request: ArticleGenerateRequest,
    effective_topic: str,
    audience_context: dict[str, Any],
    source_material: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    context = {
        "topic": effective_topic,
        "category": request.category,
        "title_count": request.title_count,
        "image_count": request.image_count,
        "target_length": audience_context["article_length"],
        "headline_intensity": request.headline_intensity,
        "content_type": workflow.content_type,
        "story_mode": request.story_mode,
    }
    outline_prompt = build_story_outline_prompt(
        topic=effective_topic,
        category=request.category,
        story_mode=request.story_mode,
        target_length=audience_context["article_length"],
        source_material=source_material,
        source_instruction=request.source_prompt,
        recent_signatures=_recent_story_signatures(session, workflow.content_type),
        reader_boundary=request.reader_boundary,
    )
    outline_started = time.monotonic()
    outline = llm.generate_json(
        outline_prompt,
        system_prompt=(
            "你是中文微恐短篇的故事策划。只输出合法 JSON；"
            "本轮只设计因果、规则与伏笔，不写正文。"
        ),
        operation="micro_horror_outline",
        context=context,
    )
    _validate_story_outline(outline)
    outline_run = _run_data(
        step="story_outline",
        prompt=outline_prompt,
        output=outline,
        started=outline_started,
        llm=llm,
    )
    result, generation_runs, version = _write_micro_horror_from_outline(
        workflow=workflow,
        llm=llm,
        request=request,
        effective_topic=effective_topic,
        audience_context=audience_context,
        outline=outline,
    )
    return result, [outline_run, *generation_runs], version


def generate_article(
    session: Session,
    request: ArticleGenerateRequest,
    audience: AudienceProfile,
    settings: Settings,
    *,
    commit: bool = True,
    generate_images: bool = True,
    story_outline: dict[str, Any] | None = None,
) -> Article:
    source_material = ""
    effective_topic = request.topic.strip()
    if request.source_type == "text":
        source_material = (request.source_text or "").strip()
        effective_topic = effective_topic or " ".join(source_material.split())[:80]
    elif request.source_type == "url":
        extracted = fetch_article(request.source_url or "", settings)
        source_material = extracted.text
        effective_topic = effective_topic or extracted.title

    audience_context = audience_to_context(audience)
    target_length = request.target_length or audience_context["article_length"]
    audience_context["article_length"] = target_length
    llm = build_llm_provider(settings, request.provider)
    workflow = get_content_workflow(request.content_type)
    if story_outline is None:
        result, generation_runs, prompt_version = workflow.generate(
            session=session,
            llm=llm,
            request=request,
            effective_topic=effective_topic,
            audience_context=audience_context,
            source_material=source_material,
        )
    else:
        if workflow.content_type != "micro_horror":
            raise ValueError("A reviewed outline can only generate a micro-horror story")
        _validate_story_outline(story_outline)
        result, generation_runs, prompt_version = _write_micro_horror_from_outline(
            workflow=workflow,
            llm=llm,
            request=request,
            effective_topic=effective_topic,
            audience_context=audience_context,
            outline=story_outline,
        )

    plain_text = json.dumps(result, ensure_ascii=False)
    story_mode = str(result.get("story_mode") or request.story_mode)
    article = Article(
        content_type=workflow.content_type,
        story_mode=story_mode,
        audience_profile_id=audience.id,
        topic=effective_topic,
        category=request.category,
        target_length=target_length,
        title_candidates_json=json.dumps(
            result["title_candidates"],
            ensure_ascii=False,
        ),
        selected_title=result["selected_title"],
        summary=result["summary"],
        content_json=plain_text,
        content_html=workflow.render_article(result, [], []),
        status="image_pending" if not generate_images else "generated",
        llm_provider=llm.name,
        model_name=llm.model,
        prompt_version=prompt_version,
        safety_flags_json="[]",
    )
    session.add(article)
    session.flush()

    for run_data in generation_runs:
        session.add(
            GenerationRun(
                article_id=article.id,
                provider=llm.name,
                model=llm.model,
                **run_data,
            )
        )

    if generate_images:
        generate_article_images(
            session,
            article,
            settings,
            image_count=request.image_count,
        )
    if commit:
        session.commit()
        session.refresh(article)
    else:
        session.flush()
    return article


def generate_article_images(
    session: Session,
    article: Article,
    settings: Settings,
    *,
    image_count: int,
    checkpoint_each: bool = False,
) -> Article:
    """Generate only missing article images and rebuild rendered HTML.

    Automation jobs checkpoint each completed image so an exhausted image quota
    never discards the already generated text or successful earlier images.
    """
    result = json.loads(article.content_json)
    _validate_result(result)
    workflow = get_content_workflow(article.content_type)
    image_specs = result["image_prompts"][:image_count]
    existing_assets = session.scalars(
        select(Asset)
        .where(Asset.article_id == article.id)
        .order_by(Asset.position)
    ).all()
    assets_by_position = {asset.position: asset for asset in existing_assets}
    image_provider = build_image_provider(settings)

    for index, image_spec in enumerate(image_specs):
        if index in assets_by_position:
            continue
        generated = image_provider.generate(
            image_spec["prompt"],
            article.selected_title,
            index,
            image_spec.get("search_query") or article.topic,
        )
        asset = Asset(
            article_id=article.id,
            asset_type=image_spec.get("asset_type", "inline"),
            position=index,
            prompt=image_spec["prompt"],
            local_path=generated.get("local_path"),
            source_url=generated.get("source_url"),
            source_kind=generated.get("source_kind") or image_provider.name,
            source_page_url=generated.get("source_page_url"),
            source_author=generated.get("source_author"),
            license_name=generated.get("license_name"),
            license_url=generated.get("license_url"),
        )
        session.add(asset)
        session.flush()
        assets_by_position[index] = asset
        if checkpoint_each:
            session.commit()

    missing_positions = [
        index for index in range(len(image_specs)) if index not in assets_by_position
    ]
    if missing_positions:
        raise RuntimeError(f"Article images missing positions: {missing_positions}")
    ordered_assets = [assets_by_position[index] for index in range(len(image_specs))]
    asset_urls = [str(asset.source_url or "") for asset in ordered_assets]
    asset_credits = [
        {
            "source_kind": item.source_kind,
            "source_page_url": item.source_page_url,
            "source_author": item.source_author,
            "license_name": item.license_name,
            "license_url": item.license_url,
        }
        for item in ordered_assets
    ]
    article.content_html = workflow.render_article(
        result,
        asset_urls,
        asset_credits,
    )
    article.status = "generated"
    if checkpoint_each:
        session.commit()
        session.refresh(article)
    else:
        session.flush()
    session.expire(article, ["assets"])
    return article


def _workbench_outline_prompt(
    session: Session,
    data: StoryOutlineCreate,
    *,
    previous_outline: dict[str, Any] | None = None,
    review_notes: str = "",
) -> tuple[str, str, dict[str, Any]]:
    effective_topic = data.idea or "自由生成一篇原创中文微恐短篇"
    context = {
        "topic": effective_topic,
        "category": data.category,
        "title_count": data.title_count,
        "image_count": data.image_count,
        "target_length": data.target_length,
        "headline_intensity": data.headline_intensity,
        "content_type": "micro_horror",
        "story_mode": data.story_mode,
        "reader_boundary": data.reader_boundary,
    }
    prompt = build_story_outline_prompt(
        topic=effective_topic,
        category=data.category,
        story_mode=data.story_mode,
        target_length=data.target_length,
        source_material=data.reference_text,
        source_instruction=data.style_prompt,
        recent_signatures=_recent_story_signatures(session, "micro_horror"),
        reader_boundary=data.reader_boundary,
        previous_outline=previous_outline,
        outline_review_notes=review_notes,
    )
    return effective_topic, prompt, context


def create_story_outline_draft(
    session: Session,
    data: StoryOutlineCreate,
    audience: AudienceProfile,
    settings: Settings,
) -> StoryOutlineDraft:
    _, prompt, context = _workbench_outline_prompt(session, data)
    llm = build_llm_provider(settings, data.provider)
    started = time.monotonic()
    outline = llm.generate_json(
        prompt,
        system_prompt=(
            "你是中文微恐短篇的故事策划。只输出合法 JSON；"
            "本轮只设计因果、规则与伏笔，不写正文。"
        ),
        operation="micro_horror_outline",
        context=context,
    )
    _validate_story_outline(outline)
    draft = StoryOutlineDraft(
        audience_profile_id=audience.id,
        status="outline_ready",
        idea=data.idea,
        story_mode=data.story_mode,
        reader_boundary=data.reader_boundary,
        reference_text=data.reference_text,
        style_prompt=data.style_prompt,
        category=data.category,
        provider=str(data.provider or settings.llm_provider),
        target_length=data.target_length,
        headline_intensity=data.headline_intensity,
        title_count=data.title_count,
        image_count=data.image_count,
        polish=data.polish,
        outline_json=json.dumps(outline, ensure_ascii=False),
        review_notes="",
    )
    session.add(draft)
    session.flush()
    session.add(
        GenerationRun(
            story_outline_draft_id=draft.id,
            provider=llm.name,
            model=llm.model,
            **_run_data(
                step="story_outline",
                prompt=prompt,
                output=outline,
                started=started,
                llm=llm,
            ),
        )
    )
    session.commit()
    session.refresh(draft)
    return draft


def revise_story_outline_draft(
    session: Session,
    draft: StoryOutlineDraft,
    review_notes: str,
    settings: Settings,
) -> StoryOutlineDraft:
    if draft.article_id is not None:
        raise ValueError("This outline has already generated a story")
    notes = review_notes.strip()
    if len(notes) < 3:
        raise ValueError("Outline review notes must contain at least 3 characters")
    data = StoryOutlineCreate(
        idea=draft.idea,
        story_mode=draft.story_mode,
        reader_boundary=draft.reader_boundary,
        reference_text=draft.reference_text,
        style_prompt=draft.style_prompt,
        category=draft.category,
        provider=draft.provider,
        target_length=draft.target_length,
        headline_intensity=draft.headline_intensity,
        title_count=draft.title_count,
        image_count=draft.image_count,
        polish=draft.polish,
    )
    previous_outline = json.loads(draft.outline_json)
    _, prompt, context = _workbench_outline_prompt(
        session,
        data,
        previous_outline=previous_outline,
        review_notes=notes,
    )
    llm = build_llm_provider(settings, data.provider)
    started = time.monotonic()
    outline = llm.generate_json(
        prompt,
        system_prompt=(
            "你是中文微恐短篇的故事策划。根据审核意见修订因果、规则与伏笔；"
            "只输出合法 JSON，不写正文。"
        ),
        operation="micro_horror_outline",
        context=context,
    )
    _validate_story_outline(outline)
    draft.outline_json = json.dumps(outline, ensure_ascii=False)
    draft.review_notes = notes
    draft.error_message = None
    draft.status = "outline_ready"
    session.add(
        GenerationRun(
            story_outline_draft_id=draft.id,
            provider=llm.name,
            model=llm.model,
            **_run_data(
                step="story_outline",
                prompt=prompt,
                output=outline,
                started=started,
                llm=llm,
            ),
        )
    )
    session.commit()
    session.refresh(draft)
    return draft


def generate_story_from_outline_draft(
    session: Session,
    draft: StoryOutlineDraft,
    settings: Settings,
    *,
    polish: bool = True,
) -> Article:
    if draft.article_id is not None:
        article = session.get(Article, draft.article_id)
        if article is None:
            raise RuntimeError("The generated story record is missing")
        return article
    if draft.status == "generating":
        raise RuntimeError("This outline is already generating a story")

    draft.status = "generating"
    draft.polish = polish
    draft.error_message = None
    session.commit()
    try:
        outline = json.loads(draft.outline_json)
        topic = (draft.idea or str(outline.get("premise") or "原创微恐短篇"))[:300]
        request = ArticleGenerateRequest(
            content_type="micro_horror",
            story_mode=draft.story_mode,
            topic=topic,
            source_type="topic",
            source_prompt=draft.style_prompt,
            audience_slug=draft.audience.slug,
            category=draft.category,
            provider=draft.provider,
            target_length=draft.target_length,
            headline_intensity=draft.headline_intensity,
            title_count=draft.title_count,
            image_count=draft.image_count,
            humanize=polish,
            reader_boundary=draft.reader_boundary,
        )
        article = generate_article(
            session,
            request,
            draft.audience,
            settings,
            commit=False,
            story_outline=outline,
        )
        for run in draft.generation_runs:
            run.article_id = article.id
        draft.article_id = article.id
        draft.status = "completed"
        session.commit()
        session.refresh(article)
        return article
    except Exception as exc:
        session.rollback()
        current = session.get(StoryOutlineDraft, draft.id)
        if current is not None:
            current.status = "outline_ready"
            current.error_message = str(exc)
            session.commit()
        raise
