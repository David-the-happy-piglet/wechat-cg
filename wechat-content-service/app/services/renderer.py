from html import escape
from typing import Any


def _paragraph(text: str) -> str:
    # Dialogue essays break lines inside a paragraph; HTML would otherwise collapse them.
    return (
        '<p style="font-size:18px;line-height:1.9;color:#333;'
        'margin:14px 0;text-align:justify;">'
        f"{escape(text).replace(chr(10), '<br>')}</p>"
    )


def _story_paragraph(text: str, *, opening: bool = False) -> str:
    weight = "font-weight:500;" if opening else ""
    return (
        '<p style="font-size:17px;line-height:2;color:#252525;'
        f'margin:16px 0;text-align:justify;{weight}">'
        f"{escape(text)}</p>"
    )


def _inline_image(url: str, *, radius: int = 8) -> str:
    return (
        '<p style="text-align:center;margin:24px 0;">'
        f'<img src="{escape(url)}" '
        f'style="max-width:100%;height:auto;border-radius:{radius}px;" />'
        "</p>"
    )


def _render_credits(asset_credits: list[dict[str, str | None]]) -> str:
    web_credits = [item for item in asset_credits if item.get("source_kind") == "web"]
    if not web_credits:
        return ""
    html = [
        (
            '<section style="font-size:12px;line-height:1.7;color:#8a8a8a;'
            'margin:18px 0 6px;">'
            "<p>图片来源与许可：</p>"
        )
    ]
    for credit in web_credits:
        author = escape(credit.get("source_author") or "Wikimedia Commons")
        source_url = escape(credit.get("source_page_url") or "")
        license_name = escape(credit.get("license_name") or "开放许可")
        license_url = escape(credit.get("license_url") or "")
        author_html = (
            f'<a href="{source_url}" style="color:#707070;">{author}</a>'
            if source_url
            else author
        )
        license_html = (
            f'<a href="{license_url}" style="color:#707070;">{license_name}</a>'
            if license_url
            else license_name
        )
        html.append(f"<p>{author_html} · {license_html}</p>")
    html.append("</section>")
    return "".join(html)

def render_story_html(
    article: dict[str, Any],
    asset_urls: list[str],
    asset_credits: list[dict[str, str | None]],
) -> str:
    html: list[str] = [
        (
            '<section style="max-width:680px;margin:0 auto;padding:14px 15px;'
            'font-family:Georgia,Songti SC,STSong,serif;background:#fbfaf8;">'
        ),
    ]
    body_title = str(article.get("body_title") or "").strip()
    if body_title:
        html.append(
            '<p style="font-size:23px;font-weight:700;text-align:center;'
            'line-height:1.5;margin:12px 0 28px;color:#252525;">'
            f"{escape(body_title)}</p>"
        )
    html.append(_story_paragraph(article.get("opening", ""), opening=True))
    inline_index = 1
    for section_index, section in enumerate(article.get("sections", [])):
        if section_index > 0 and inline_index < len(asset_urls):
            html.append(_inline_image(asset_urls[inline_index], radius=2))
            inline_index += 1
        heading = str(section.get("heading") or "").strip()
        if heading:
            html.append(
                '<p style="font-size:14px;letter-spacing:3px;color:#777;'
                'text-align:center;margin:30px 0 18px;">'
                f"{escape(heading)}</p>"
            )
        for paragraph in section.get("paragraphs", []):
            html.append(_story_paragraph(paragraph))

    html.append(
        '<div style="width:32px;border-top:1px solid #999;'
        'margin:34px auto 24px;"></div>'
    )
    html.append(_story_paragraph(article.get("conclusion", "")))
    html.append(_render_credits(asset_credits))
    html.append("</section>")
    return "".join(html)


def render_health_html(
    article: dict[str, Any],
    asset_urls: list[str] | None = None,
    asset_credits: list[dict[str, str | None]] | None = None,
) -> str:
    asset_urls = asset_urls or []
    asset_credits = asset_credits or []
    html: list[str] = [
        (
            '<section style="max-width:680px;margin:0 auto;padding:8px 12px;'
            'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;">'
        )
    ]
    html.append(_paragraph(article.get("opening", "")))

    inline_index = 1
    for section_index, section in enumerate(article.get("sections", [])):
        if section_index > 0 and inline_index < len(asset_urls):
            html.append(_inline_image(asset_urls[inline_index]))
            inline_index += 1
        html.append(
            '<h2 style="font-size:21px;line-height:1.5;color:#2f6f4e;'
            'margin:28px 0 12px;border-left:4px solid #70a987;padding-left:10px;">'
            f"{escape(section['heading'])}</h2>"
        )
        for paragraph in section.get("paragraphs", []):
            html.append(_paragraph(paragraph))

    html.append(
        '<section style="background:#f4f8f5;border-radius:8px;padding:12px 16px;margin-top:26px;">'
    )
    html.append(_paragraph(article.get("conclusion", "")))
    html.append("</section>")
    html.append(_render_credits(asset_credits))
    html.append("</section>")
    return "".join(html)


def render_emotional_html(
    article: dict[str, Any],
    asset_urls: list[str] | None = None,
    asset_credits: list[dict[str, str | None]] | None = None,
) -> str:
    asset_urls = asset_urls or []
    asset_credits = asset_credits or []
    html: list[str] = [
        (
            '<section style="max-width:680px;margin:0 auto;padding:12px 15px;'
            'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;'
            'background:#fffaf9;">'
        ),
        (
            '<div style="width:42px;border-top:3px solid #c97b82;'
            'margin:4px 0 22px;"></div>'
        ),
        _paragraph(article.get("opening", "")),
    ]
    sections = article.get("sections", [])
    inline_urls = [url for url in asset_urls[1:] if url]
    paragraph_count = sum(len(section.get("paragraphs", [])) for section in sections)
    image_positions: dict[int, list[str]] = {}
    for index, url in enumerate(inline_urls):
        after_paragraph = max(
            1,
            round(paragraph_count * (index + 1) / (len(inline_urls) + 1)),
        )
        image_positions.setdefault(after_paragraph, []).append(url)

    paragraph_index = 0
    for section in sections:
        heading = str(section.get("heading") or "").strip()
        if heading:
            html.append(
                '<h2 style="font-size:21px;line-height:1.55;color:#8d4c55;'
                'margin:30px 0 12px;padding-bottom:7px;'
                'border-bottom:1px solid #ead6d8;">'
                f"{escape(heading)}</h2>"
            )
        for paragraph in section.get("paragraphs", []):
            html.append(_paragraph(paragraph))
            paragraph_index += 1
            for url in image_positions.get(paragraph_index, []):
                html.append(_inline_image(url, radius=10))

    if paragraph_count == 0:
        for url in inline_urls:
            html.append(_inline_image(url, radius=10))

    html.append(
        '<section style="background:#f9eeee;border-radius:10px;'
        'padding:12px 16px;margin-top:28px;">'
    )
    html.append(_paragraph(article.get("conclusion", "")))
    html.append("</section>")
    html.append(_render_credits(asset_credits))
    html.append("</section>")
    return "".join(html)


def render_wechat_html(
    article: dict[str, Any],
    asset_urls: list[str] | None = None,
    asset_credits: list[dict[str, str | None]] | None = None,
) -> str:
    """Render through the registered workflow while preserving the legacy default."""
    from ..content_workflows import DEFAULT_CONTENT_TYPE, get_content_workflow

    content_type = str(article.get("content_type") or DEFAULT_CONTENT_TYPE)
    return get_content_workflow(content_type).render_article(
        article,
        asset_urls or [],
        asset_credits or [],
    )
