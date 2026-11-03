from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from ...config import Settings
from ..llm import TokenUsage, build_llm_provider
from .source import SourceExtractionError, fetch_article


class NewsResearchError(RuntimeError):
    pass


ALLOWED_CLAIM_LEVELS = {
    "CONFIRMED",
    "REPORTED",
    "SINGLE_SOURCE",
    "SOCIAL_CLAIM",
    "RUMOR",
    "ANALYSIS",
    "SPECULATION",
}
WRITABLE_CLAIM_LEVELS = {"CONFIRMED", "REPORTED", "SINGLE_SOURCE"}
PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts" / "news"


@dataclass(frozen=True)
class NewsLLMCall:
    step: str
    provider: str
    model: str
    prompt: str
    output: dict[str, Any]
    duration_ms: int
    usage: TokenUsage


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._href = ""
        self._parts: list[str] = []
        self._in_result = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        values = dict(attrs)
        classes = set(str(values.get("class") or "").split())
        if "result__a" in classes:
            self._in_result = True
            self._href = str(values.get("href") or "")
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._in_result:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._in_result:
            return
        self._in_result = False
        target = _decode_duckduckgo_url(self._href)
        title = " ".join("".join(self._parts).split())
        if target and title:
            self.results.append({"url": target, "title": title})


def _decode_duckduckgo_url(value: str) -> str:
    value = html.unescape(value).strip()
    if value.startswith("//"):
        value = f"https:{value}"
    parsed = urlparse(value)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        encoded = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(encoded)
    return value if value.startswith(("http://", "https://")) else ""


def _prompt(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8").strip()


def _domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    if ".".join(parts[-2:]) in {"com.cn", "net.cn", "org.cn", "gov.cn", "co.uk"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _source_tier(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if any(
        marker in host
        for marker in (
            ".gov.cn",
            ".gov",
            "reuters.com",
            "apnews.com",
            "court.gov",
            "police.",
        )
    ):
        return "A"
    if any(
        marker in host
        for marker in (
            "people.com.cn",
            "xinhuanet.com",
            "news.cn",
            "cctv.com",
            "bbc.",
            "nytimes.com",
            "theguardian.com",
            "caixin.com",
            "thepaper.cn",
        )
    ):
        return "B"
    if any(marker in host for marker in ("weibo.com", "reddit.com", "tiktok.com", "douyin.com", "x.com")):
        return "D"
    return "C"


def _clean_query_list(value: object, fallback: str) -> list[str]:
    raw = value if isinstance(value, list) else []
    queries: list[str] = []
    for item in [*raw, fallback]:
        query = " ".join(str(item or "").split())[:180]
        if query and query not in queries:
            queries.append(query)
    return queries[:4]


def _call_json(
    settings: Settings,
    *,
    step: str,
    provider_name: str,
    instruction_file: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], NewsLLMCall]:
    provider = build_llm_provider(settings, provider_name)
    prompt = f"{_prompt(instruction_file)}\n\n任务输入（JSON）：\n{json.dumps(payload, ensure_ascii=False)}"
    started = time.monotonic()
    if provider.name == "mock":
        result = _mock_stage(step, payload)
    else:
        result = provider.generate_json(
            prompt,
            system_prompt="你是严谨的中文新闻研究台。只输出合法 JSON，不要 Markdown。",
        )
    call = NewsLLMCall(
        step=step,
        provider=provider.name,
        model=provider.model,
        prompt=prompt,
        output=result,
        duration_ms=round((time.monotonic() - started) * 1000),
        usage=provider.usage,
    )
    return result, call


def _mock_stage(step: str, payload: dict[str, Any]) -> dict[str, Any]:
    if step == "news_query_expansion":
        topic = str(payload.get("topic") or "新闻事件")
        return {"queries": [topic, f"{topic} 官方回应", f"{topic} 时间线"]}
    if step == "news_mechanical":
        sources = payload.get("sources") or []
        return {
            "normalized_sources": [
                {"source_id": item.get("source_id"), "tier": item.get("tier")}
                for item in sources
            ],
            "duplicate_groups": [],
            "entities": [],
            "event_labels": ["news"],
        }
    if step == "news_judgment":
        ids = [item.get("source_id") for item in payload.get("sources") or []]
        return {
            "same_event_source_ids": ids,
            "excluded_source_ids": [],
            "story_score": 70,
            "confidence": "medium",
            "recommended_angles": ["straight_news", "timeline"],
            "selection_rationale": "mock research",
        }
    if step == "news_fact_sheet":
        claims = []
        for index, source in enumerate((payload.get("sources") or [])[:4], start=1):
            text = str(source.get("text") or "")
            excerpt = next((part.strip() for part in re.split(r"[。！？\n]", text) if len(part.strip()) >= 8), text[:80])
            if excerpt:
                claims.append(
                    {
                        "claim_id": f"C{index}",
                        "statement": excerpt,
                        "level": "SINGLE_SOURCE",
                        "source_ids": [source.get("source_id")],
                        "evidence": [{"source_id": source.get("source_id"), "excerpt": excerpt}],
                    }
                )
        return {"story": payload.get("topic"), "timeline": [], "claims": claims}
    if step == "news_review":
        return {
            "verdict": "pass",
            "unsupported_claims": [],
            "rumor_flags": [],
            "source_coverage": {"status": "sufficient"},
            "risk_flags": [],
            "notes": "mock review",
            "corrected_script": {},
        }
    return {}


def search_news(query: str, settings: Settings) -> list[dict[str, str]]:
    if settings.news_search_provider == "tavily":
        if not settings.tavily_api_key:
            raise NewsResearchError("NEWS_SEARCH_PROVIDER=tavily requires TAVILY_API_KEY")
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "topic": "news",
                "search_depth": "advanced",
                "max_results": settings.news_search_max_results,
            },
            timeout=settings.news_search_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        return [
            {
                "url": str(item.get("url") or ""),
                "title": str(item.get("title") or ""),
                "snippet": str(item.get("content") or ""),
                "published_at": str(item.get("published_date") or ""),
            }
            for item in body.get("results") or []
            if isinstance(item, dict) and str(item.get("url") or "").startswith("http")
        ]
    if settings.news_search_provider != "duckduckgo":
        raise NewsResearchError(f"Unsupported news search provider: {settings.news_search_provider}")
    response = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": settings.web_image_user_agent},
        timeout=settings.news_search_timeout_seconds,
        follow_redirects=True,
        trust_env=False,
    )
    response.raise_for_status()
    parser = _DuckDuckGoParser()
    parser.feed(response.text)
    return parser.results[: settings.news_search_max_results]


def _source_payload(document: dict[str, Any], *, text_limit: int = 5000) -> dict[str, Any]:
    return {
        "source_id": document["source_id"],
        "url": document["url"],
        "domain": document["domain"],
        "title": document["title"],
        "tier": document["tier"],
        "published_at": document.get("published_at"),
        "text": document["text"][:text_limit],
    }


def _collect_documents(
    *,
    topic: str,
    source_url: str | None,
    source_title: str,
    source_text: str,
    queries: list[str],
    settings: Settings,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    documents: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()

    def add_document(url: str, title: str, text: str, published_at: str = "") -> None:
        normalized_text = " ".join(text.split())
        digest = normalized_text[:5000]
        if not url or url in seen_urls or not normalized_text or digest in seen_hashes:
            return
        seen_urls.add(url)
        seen_hashes.add(digest)
        documents.append(
            {
                "source_id": f"S{len(documents) + 1}",
                "url": url,
                "domain": _domain(url),
                "title": title[:500],
                "tier": _source_tier(url),
                "published_at": published_at or None,
                "text": text[:12000],
            }
        )

    if source_url:
        add_document(source_url, source_title, source_text)

    candidates: list[dict[str, str]] = []
    for query in queries:
        try:
            candidates.extend(search_news(query, settings))
        except (httpx.HTTPError, NewsResearchError) as exc:
            failures.append({"query": query, "error": str(exc)[:500]})
    for candidate in candidates:
        if len(documents) >= settings.news_max_documents:
            break
        url = candidate.get("url") or ""
        if url in seen_urls or (source_url and _domain(url) == _domain(source_url) and len(documents) > 0):
            continue
        try:
            extracted = fetch_article(url, settings)
        except (SourceExtractionError, httpx.HTTPError) as exc:
            failures.append({"url": url, "error": str(exc)[:500]})
            continue
        add_document(
            extracted.url,
            extracted.title or candidate.get("title") or topic,
            extracted.text,
            candidate.get("published_at") or "",
        )
    return documents, failures


def _normalize_fact_sheet(raw: dict[str, Any], documents: list[dict[str, Any]]) -> dict[str, Any]:
    sources = {item["source_id"]: item for item in documents}
    normalized_claims: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, claim in enumerate(raw.get("claims") or [], start=1):
        if not isinstance(claim, dict):
            continue
        statement = " ".join(str(claim.get("statement") or "").split())
        if not statement:
            continue
        claim_id = str(claim.get("claim_id") or f"C{index}").strip()[:30]
        if claim_id in used_ids:
            claim_id = f"C{index}"
        used_ids.add(claim_id)
        evidence: list[dict[str, str]] = []
        for item in claim.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id") or "")
            excerpt = " ".join(str(item.get("excerpt") or "").split())[:300]
            source = sources.get(source_id)
            if source and excerpt and excerpt in " ".join(source["text"].split()):
                evidence.append({"source_id": source_id, "excerpt": excerpt})
        source_ids = list(dict.fromkeys(item["source_id"] for item in evidence))
        level = str(claim.get("level") or "SINGLE_SOURCE").upper()
        if level not in ALLOWED_CLAIM_LEVELS:
            level = "SINGLE_SOURCE"
        independent_domains = {sources[source_id]["domain"] for source_id in source_ids}
        has_primary = any(sources[source_id]["tier"] == "A" for source_id in source_ids)
        if level in {"CONFIRMED", "REPORTED"} and len(independent_domains) < 2 and not has_primary:
            level = "SINGLE_SOURCE"
        if not source_ids:
            level = "SPECULATION"
        normalized_claims.append(
            {
                "claim_id": claim_id,
                "statement": statement,
                "level": level,
                "source_ids": source_ids,
                "evidence": evidence,
            }
        )
    fact_sheet = {**raw, "claims": normalized_claims}
    fact_sheet["writable_claims"] = [
        item
        for item in normalized_claims
        if item["level"] in WRITABLE_CLAIM_LEVELS and item["source_ids"]
    ]
    return fact_sheet


def run_news_research(
    settings: Settings,
    *,
    topic: str,
    source_url: str | None,
    source_title: str,
    source_text: str,
    routes: dict[str, str],
    requested_angle: str,
) -> tuple[dict[str, Any], list[NewsLLMCall]]:
    calls: list[NewsLLMCall] = []
    plan, call = _call_json(
        settings,
        step="news_query_expansion",
        provider_name=routes["research"],
        instruction_file="query_expansion.txt",
        payload={"topic": topic, "source_title": source_title, "source_excerpt": source_text[:2500]},
    )
    calls.append(call)
    queries = _clean_query_list(plan.get("queries"), topic)
    documents, search_failures = _collect_documents(
        topic=topic,
        source_url=source_url,
        source_title=source_title,
        source_text=source_text,
        queries=queries,
        settings=settings,
    )
    independent_domains = {item["domain"] for item in documents if item["tier"] != "D"}
    if len(independent_domains) < settings.news_min_independent_sources:
        raise NewsResearchError(
            f"新闻研究只获得 {len(independent_domains)} 个独立可访问来源；"
            f"至少需要 {settings.news_min_independent_sources} 个，已停止写稿"
        )
    compact_sources = [_source_payload(item, text_limit=2600) for item in documents]
    mechanical, call = _call_json(
        settings,
        step="news_mechanical",
        provider_name=routes["mechanical"],
        instruction_file="mechanical.txt",
        payload={"topic": topic, "sources": compact_sources},
    )
    calls.append(call)
    judgment, call = _call_json(
        settings,
        step="news_judgment",
        provider_name=routes["judgment"],
        instruction_file="judgment.txt",
        payload={
            "topic": topic,
            "requested_angle": requested_angle,
            "sources": [{k: v for k, v in item.items() if k != "text"} for item in compact_sources],
            "mechanical": mechanical,
        },
    )
    calls.append(call)
    available_ids = {item["source_id"] for item in compact_sources}
    selected_ids = {
        str(item)
        for item in (judgment.get("same_event_source_ids") or [])
        if str(item) in available_ids
    }
    if not selected_ids:
        raise NewsResearchError("事件聚类没有找到属于同一事件的可靠来源，已停止写稿")
    fact_sources = [item for item in compact_sources if item["source_id"] in selected_ids]
    selected_domains = {item["domain"] for item in fact_sources if item["tier"] != "D"}
    if len(selected_domains) < settings.news_min_independent_sources:
        raise NewsResearchError(
            "事件聚类后不足两个独立来源，可能混入了相似但不同的事件，已停止写稿"
        )
    raw_fact_sheet, call = _call_json(
        settings,
        step="news_fact_sheet",
        provider_name=routes["research"],
        instruction_file="fact_sheet.txt",
        payload={"topic": topic, "judgment": judgment, "sources": fact_sources},
    )
    calls.append(call)
    fact_sheet = _normalize_fact_sheet(raw_fact_sheet, documents)
    if not fact_sheet["writable_claims"]:
        raise NewsResearchError("Fact Sheet 没有通过逐字证据校验的可写事实，已停止写稿")
    recommended_angles = judgment.get("recommended_angles") or []
    selected_angle = requested_angle
    if selected_angle == "auto":
        selected_angle = str(recommended_angles[0]) if recommended_angles else "straight_news"
    result = {
        "version": "news-research-v1",
        "query_plan": {**plan, "queries": queries},
        "search_provider": settings.news_search_provider,
        "search_failures": search_failures,
        "sources": [{k: v for k, v in item.items() if k != "text"} for item in documents],
        "mechanical": mechanical,
        "judgment": judgment,
        "selected_angle": selected_angle,
        "fact_sheet": fact_sheet,
    }
    return result, calls


def grounded_source_text(research: dict[str, Any]) -> str:
    sources = research.get("sources") or []
    source_notes = {
        item.get("source_id"): {
            "title": item.get("title"),
            "url": item.get("url"),
            "tier": item.get("tier"),
        }
        for item in sources
    }
    return json.dumps(
        {
            "source_notes": source_notes,
            "story_assessment": research.get("judgment") or {},
            "fact_sheet": research.get("fact_sheet") or {},
        },
        ensure_ascii=False,
    )


def audit_claim_ids(
    script: dict[str, Any],
    research: dict[str, Any],
    *,
    require_claim_ids: bool = True,
) -> list[str]:
    allowed = {
        item.get("claim_id")
        for item in (research.get("fact_sheet") or {}).get("writable_claims") or []
    }
    errors: list[str] = []
    for index, scene in enumerate(script.get("scenes") or []):
        claim_ids = scene.get("claim_ids") or []
        if not claim_ids:
            if require_claim_ids:
                errors.append(f"scene {index} 没有 claim_ids")
            continue
        invalid = [claim_id for claim_id in claim_ids if claim_id not in allowed]
        if invalid:
            errors.append(f"scene {index} 引用了无效事实编号：{', '.join(invalid)}")
    return errors


def merge_review_correction(
    script: dict[str, Any],
    review: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    corrected = review.get("corrected_script")
    if not isinstance(corrected, dict):
        return script, ["审核模型没有返回 corrected_script 对象"]
    patch_scenes = corrected.get("scenes")
    if not isinstance(patch_scenes, list):
        return script, ["审核模型没有返回 corrected_script.scenes 补丁"]
    patches: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    original_positions = {
        int(item.get("position", index))
        for index, item in enumerate(script.get("scenes") or [])
        if isinstance(item, dict)
    }
    for item in patch_scenes:
        if not isinstance(item, dict):
            continue
        try:
            position = int(item.get("position"))
        except (TypeError, ValueError):
            errors.append("修订场景缺少有效 position")
            continue
        if position not in original_positions:
            errors.append(f"修订稿包含不存在的 scene {position}")
            continue
        narration = str(item.get("narration") or "").strip()
        claim_ids = item.get("claim_ids")
        if not narration or not isinstance(claim_ids, list) or not claim_ids:
            errors.append(f"scene {position} 的修订补丁缺少 narration 或 claim_ids")
            continue
        patches[position] = {"narration": narration, "claim_ids": claim_ids}
    unsupported_positions: set[int] = set()
    for issue in review.get("unsupported_claims") or []:
        if not isinstance(issue, dict):
            continue
        try:
            unsupported_positions.add(int(issue.get("scene_position")))
        except (TypeError, ValueError):
            continue
    missing = sorted(unsupported_positions - set(patches))
    if missing:
        errors.append(
            "以下审核问题没有对应修订补丁：" + "、".join(str(item) for item in missing)
        )
    merged_scenes: list[dict[str, Any]] = []
    for index, original in enumerate(script.get("scenes") or []):
        if not isinstance(original, dict):
            continue
        position = int(original.get("position", index))
        merged_scenes.append({**original, **patches.get(position, {})})
    return {
        **script,
        "title": str(corrected.get("title") or script.get("title") or ""),
        "summary": str(corrected.get("summary") or script.get("summary") or ""),
        "scenes": merged_scenes,
    }, errors


def review_news_script(
    settings: Settings,
    *,
    script: dict[str, Any],
    research: dict[str, Any],
    provider_name: str,
) -> tuple[dict[str, Any], NewsLLMCall]:
    local_errors = audit_claim_ids(script, research)
    allowed_claim_ids = sorted(
        str(item.get("claim_id"))
        for item in (research.get("fact_sheet") or {}).get("writable_claims") or []
        if item.get("claim_id")
    )
    review, call = _call_json(
        settings,
        step="news_review",
        provider_name=provider_name,
        instruction_file="review.txt",
        payload={
            "fact_sheet": research.get("fact_sheet") or {},
            "sources": research.get("sources") or [],
            "script": script,
            "allowed_claim_ids": allowed_claim_ids,
            "local_claim_id_errors": local_errors,
        },
    )
    verdict = str(review.get("verdict") or "blocked").lower()
    if verdict not in {"pass", "corrected", "blocked"}:
        verdict = "blocked"
    if local_errors and verdict == "pass":
        verdict = "blocked"
    return {**review, "verdict": verdict, "local_claim_id_errors": local_errors}, call
