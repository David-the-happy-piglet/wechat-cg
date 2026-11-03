from __future__ import annotations

import json
import math
from typing import Any

from ...config import Settings
from ..llm import LLMProvider, build_llm_provider
from .visual_profiles import DEFAULT_VISUAL_PROFILE, load_visual_profile


class VideoScriptError(RuntimeError):
    pass


def target_scene_count(duration_seconds: int, scene_duration_seconds: float = 3.5) -> int:
    return min(30, max(2, round(duration_seconds / scene_duration_seconds)))


def build_video_script_prompt(
    *,
    topic: str,
    source_text: str,
    target_duration_seconds: int,
    style: str,
    scene_duration_seconds: float = 3.5,
    speech_speed: float = 1.2,
    visual_profile_name: str = DEFAULT_VISUAL_PROFILE,
    script_prompt: str = "",
    use_web_images: bool = False,
    grounded_claim_ids: list[str] | None = None,
) -> str:
    visual_profile = load_visual_profile(visual_profile_name)
    scene_count = target_scene_count(target_duration_seconds, scene_duration_seconds)
    target_chars = max(80, round(target_duration_seconds * 4.65 * speech_speed))
    chars_per_scene = max(12, round(target_chars / scene_count))
    source_excerpt = source_text[:12000]
    image_source_guidance = (
        "优先使用互联网真实图片。search_query 必须包含可检索的具体人物、地点、事件、"
        "机构或年份；有标准英文名称时优先写英文，不要只写抽象概念或画面风格。"
        if use_web_images
        else "使用 AI 生成解释型配图，search_query 仅作为素材检索备用。"
    )
    if visual_profile.allow_photorealism:
        image_prompt_guidance = (
            "image_prompt 必须描述具体、可信的新闻或事件说明画面，包括地点、时间感、"
            "主体和一个明确动作。AI 回退图只能作为情境重建；涉及具名真实人物时，"
            "不得要求复刻其精确面孔，应改用远景、背影、环境或相关物件。"
        )
        rendering_guidance = (
            "允许写实摄影和新闻纪实构图，但禁止伪造真实现场证据、媒体台标、新闻标题、"
            "监控截图或带文字的文件；不要使用电影海报光效、夸张滤镜和戏剧化摆拍。"
        )
    else:
        image_prompt_guidance = (
            "image_prompt 必须描述一个具象、常见、无需猜测的画面，优先使用人物动作和"
            "大众熟悉的物品或图标。最多 3 个主要视觉元素，禁止无意义抽象线条、"
            "漂浮圆点、装饰性几何图形、光环和看不懂的隐喻。"
        )
        rendering_guidance = (
            "避免任何摄影、写实、电影海报、3D渲染式描述，也不要描述复杂人物五官、"
            "手脚或拥挤场面。"
        )
    contract = {
        "title": "视频标题",
        "summary": "一句话摘要",
        "scenes": [
            {
                "position": 0,
                "narration": "像向朋友讲解一样自然、亲切的完整旁白",
                "visual_intent": "这一幕应当让观众看到什么",
                "visual_type": f"从以下值中选择：{'|'.join(visual_profile.allowed_types)}",
                "image_prompt": "只描述本幕应当出现的具体可见内容，不重复风格规范",
                "overlay_labels": [
                    {"text": "后期叠加的短标签", "x": 0.5, "y": 0.4}
                ],
                "search_query": "用于开放图库搜索的简短关键词",
                "claim_ids": ["新闻模式下支撑本幕旁白的事实编号；普通模式为空数组"],
            }
        ],
    }
    grounding_guidance = (
        "这是严格的新闻写稿任务。只能使用 Fact Sheet 的 writable_claims，"
        f"可用事实编号为：{'、'.join(grounded_claim_ids)}。每一幕都必须填写至少一个"
        "有效 claim_ids；同一事实可以支撑多幕。SINGLE_SOURCE 必须在旁白中明确说出"
        "‘据某来源报道/表示’，不得把 REPORTED、传闻或分析改写成已确认事实。"
        if grounded_claim_ids
        else "这是普通讲解任务，每幕 claim_ids 返回空数组。"
    )
    return f"""
你是中文知识解说短视频的编导。请将材料改写为适合配音和静态图片讲解的视频稿。

主题：{topic}
目标时长：约 {target_duration_seconds} 秒
配音语速：常规语速的 {speech_speed:.2f} 倍
建议总旁白字数：约 {target_chars} 个中文字符
建议场景数：{scene_count} 个，每个场景约 {scene_duration_seconds:.1f} 秒
视觉范式：{visual_profile.profile_id}（{visual_profile.name}）
视觉基础规范：{visual_profile.base_prompt}
用户补充风格：{style.strip() or '无'}
用户对口语稿的补充要求：{script_prompt.strip() or '无'}
图片来源策略：{image_source_guidance}

visual_type 选择规则：
{visual_profile.selection_rules()}

原始材料：
{source_excerpt}

要求：
0. 这是一项直接改写和格式化任务，不需要深度推理；请把输出 token 留给最终 JSON。
1. 把它讲给一个聪明但没读过原文的朋友听。语气亲切、直接，可以自然地说“你”，不要端着播新闻，也不要写成课程提纲。
2. 不逐句照搬原文，不虚构数据、引用、人物经历或权威来源。
2.1 {grounding_guidance}
3. 开场直接抛出一个观众可能遇到的具体困惑，不写“你是否曾经想过”。先讲生活里的例子，再带出概念；结尾用一句自然的话收住，不喊口号。
4. 每个场景只表达一个完整自然语义；每段 narration 尽量控制在 {max(12, chars_per_scene - 4)} 到 {chars_per_scene + 4} 个中文字符，以便单张图片停留 3 到 4 秒。
5. narration 必须是能直接朗读的自然中文，不写镜头指令。长短句要有变化，允许“其实”“比如”“换句话说”等少量自然口头连接，但不要每句都用连接词。
6. 避免“核心是、最终答案、首先/其次/最后、值得注意的是、不仅……而且……、底层逻辑、赋能、至关重要”等模板化表达；不要强行凑三点，不写像金句或广告口号的句子。
7. 字幕将由系统直接复制 narration，JSON 中不要另外生成 subtitle。
8. visual_type 只能从 {'、'.join(visual_profile.allowed_types)} 中选择，并严格按照上面的动态规则判断。
9. {image_prompt_guidance} 画面中绝对不要出现文字、字母、数字、水印或 Logo。
10. 只有需要本地文字标签时才填写 overlay_labels，否则返回空数组。每个标签不超过 8 个汉字，x、y 是标签中心在画面中的归一化坐标，范围 0.1 到 0.9。flowchart 的图片只画空白节点、连线、箭头和熟悉的小图标，节点文字全部放进 overlay_labels，不能写入 image_prompt。
11. {rendering_guidance}
12. position 从 0 开始连续递增；scenes 必须有 {max(2, scene_count - 2)} 到 {min(30, scene_count + 2)} 个。
13. 只返回合法 JSON，不要 Markdown。

返回结构：
{json.dumps(contract, ensure_ascii=False)}
""".strip()


def _mock_script(
    topic: str,
    duration_seconds: int,
    style: str,
    *,
    scene_duration_seconds: float,
    speech_speed: float,
    visual_profile_name: str,
    grounded_claim_ids: list[str] | None = None,
) -> dict[str, Any]:
    visual_profile = load_visual_profile(visual_profile_name)
    count = target_scene_count(duration_seconds, scene_duration_seconds)
    templates = [
        f"{topic}，为什么总是很难坚持？",
        "先把目标变具体，明确什么时候做、每次做多久。",
        "把动作缩小到今天就能完成，开始时会轻松很多。",
        "不要一口气改变太多，稳定完成比追求强度更重要。",
        "记录每天的实际感受，才能看见细小但真实的变化。",
        "遇到中断也不用全部推翻，从下一次机会重新开始即可。",
        f"面对{topic}，先从小处开始，再根据反馈逐步调整。",
    ]
    narrations = [templates[index % len(templates)] for index in range(count)]
    narrations[-1] = templates[-1]
    target_chars = round(duration_seconds * 4.65 * speech_speed)
    fillers = [
        "先观察自己的真实情况。",
        "不要急着一次改变太多。",
        "给变化留出足够的时间。",
        "重点是让行动能够重复。",
    ]
    filler_index = 0
    while sum(len(item) for item in narrations) < target_chars * 0.9:
        scene_index = filler_index % len(narrations)
        narrations[scene_index] += fillers[filler_index % len(fillers)]
        filler_index += 1
    return {
        "title": topic[:300],
        "summary": f"用清晰、可执行的方式讲解{topic}",
        "style_bible": visual_profile.style_bible(style),
        "scenes": [
            {
                "position": index,
                "narration": narration,
                "subtitle": narration,
                "visual_intent": f"与第{index + 1}段旁白对应的解释画面",
                "visual_type": (
                    "flowchart"
                    if index == 1 and "flowchart" in visual_profile.types
                    else visual_profile.default_type
                ),
                "image_prompt": (
                    "两个带常见人物图标的空白节点框，由一根粗箭头连接"
                    if index == 1 and "flowchart" in visual_profile.types
                    else f"围绕“{topic}”的第{index + 1}幕，使用熟悉图标，主体清晰"
                ),
                "overlay_labels": (
                    [
                        {"text": "现在", "x": 0.32, "y": 0.42},
                        {"text": "下一步", "x": 0.68, "y": 0.42},
                    ]
                    if index == 1 and "flowchart" in visual_profile.types
                    else []
                ),
                "search_query": topic,
                "claim_ids": [grounded_claim_ids[index % len(grounded_claim_ids)]] if grounded_claim_ids else [],
            }
            for index, narration in enumerate(narrations)
        ],
    }


def _normalize_overlay_labels(value: object) -> list[dict[str, str | float]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str | float]] = []
    for item in value[:3]:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text") or "").split()).strip()
        if not text:
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        normalized.append(
            {
                "text": text[:12],
                "x": min(0.9, max(0.1, x)),
                "y": min(0.82, max(0.12, y)),
            }
        )
    return normalized


def validate_video_script(
    result: dict[str, Any],
    duration_seconds: int,
    *,
    scene_duration_seconds: float = 3.5,
    style: str = "",
    visual_profile_name: str = DEFAULT_VISUAL_PROFILE,
) -> dict[str, Any]:
    visual_profile = load_visual_profile(visual_profile_name)
    if not isinstance(result, dict):
        raise VideoScriptError("Video script response must be an object")
    for key in ("title", "summary"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise VideoScriptError(f"Video script is missing valid {key}")
    scenes = result.get("scenes")
    if not isinstance(scenes, list) or not 1 <= len(scenes) <= 30:
        raise VideoScriptError("Video script scenes must contain 1 to 30 items")
    normalized = []
    for position, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise VideoScriptError("Every video scene must be an object")
        narration = scene.get("narration")
        image_prompt = scene.get("image_prompt")
        if not isinstance(narration, str) or not narration.strip():
            raise VideoScriptError(f"Scene {position} has no narration")
        if not isinstance(image_prompt, str) or not image_prompt.strip():
            raise VideoScriptError(f"Scene {position} has no image_prompt")
        visual_type = visual_profile.normalize_type(scene.get("visual_type"))
        overlay_labels = _normalize_overlay_labels(scene.get("overlay_labels"))
        prompt_content = image_prompt.strip()
        if overlay_labels:
            positions = "、".join(
                f"({item['x']:.2f}, {item['y']:.2f})" for item in overlay_labels
            )
            prompt_content += (
                f"。在归一化坐标 {positions} 放置清楚的空白节点框或留白区域，"
                "供后期叠加标签；不要在图片中渲染标签文字"
            )
        rendered_prompt, visual_type = visual_profile.render_prompt(
            visual_type,
            prompt_content,
            style_notes=style,
        )
        normalized.append(
            {
                "position": position,
                "narration": narration.strip(),
                "subtitle": narration.strip(),
                "visual_intent": str(scene.get("visual_intent") or "").strip(),
                "visual_type": visual_type,
                "image_prompt": rendered_prompt[:1500],
                "overlay_labels": overlay_labels,
                "search_query": str(scene.get("search_query") or result["title"])[:300],
                "claim_ids": list(
                    dict.fromkeys(
                        str(item).strip()[:30]
                        for item in (scene.get("claim_ids") or [])
                        if str(item).strip()
                    )
                )[:6],
            }
        )
    result = {
        **result,
        "visual_profile": visual_profile.profile_id,
        "style_bible": visual_profile.style_bible(style),
        "scenes": normalized,
    }
    expected_scenes = target_scene_count(duration_seconds, scene_duration_seconds)
    minimum_scenes = max(2, expected_scenes - 3)
    if len(normalized) < minimum_scenes:
        raise VideoScriptError(
            f"Video script has {len(normalized)} scenes; at least {minimum_scenes} are required "
            "for the requested visual cadence"
        )
    total_chars = sum(len(item["narration"]) for item in normalized)
    if total_chars < max(30, math.floor(duration_seconds * 1.5)):
        raise VideoScriptError("Video narration is too short for the requested duration")
    return result


def generate_video_script(
    settings: Settings,
    *,
    topic: str,
    source_text: str,
    target_duration_seconds: int,
    style: str,
    provider_name: str | None,
    scene_duration_seconds: float = 3.5,
    speech_speed: float = 1.2,
    visual_profile_name: str = DEFAULT_VISUAL_PROFILE,
    script_prompt: str = "",
    use_web_images: bool = False,
    grounded_claim_ids: list[str] | None = None,
) -> tuple[dict[str, Any], LLMProvider, str]:
    provider = build_llm_provider(settings, provider_name)
    prompt = build_video_script_prompt(
        topic=topic,
        source_text=source_text,
        target_duration_seconds=target_duration_seconds,
        style=style,
        scene_duration_seconds=scene_duration_seconds,
        speech_speed=speech_speed,
        visual_profile_name=visual_profile_name,
        script_prompt=script_prompt,
        use_web_images=use_web_images,
        grounded_claim_ids=grounded_claim_ids,
    )
    if provider.name == "mock":
        result = _mock_script(
            topic,
            target_duration_seconds,
            style,
            scene_duration_seconds=scene_duration_seconds,
            speech_speed=speech_speed,
            visual_profile_name=visual_profile_name,
            grounded_claim_ids=grounded_claim_ids,
        )
    else:
        result = provider.generate_json(
            prompt,
            system_prompt="你是中文知识解说视频编导。只输出合法 JSON，不要 Markdown。",
        )
    return (
        validate_video_script(
            result,
            target_duration_seconds,
            scene_duration_seconds=scene_duration_seconds,
            style=style,
            visual_profile_name=visual_profile_name,
        ),
        provider,
        prompt,
    )
