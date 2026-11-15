import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def parse_token_usage(body: dict[str, Any]) -> TokenUsage:
    raw = body.get("usage")
    if not isinstance(raw, dict):
        return TokenUsage()

    def token_value(*keys: str) -> int | None:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    prompt_tokens = token_value("prompt_tokens", "input_tokens")
    completion_tokens = token_value("completion_tokens", "output_tokens")
    total_tokens = token_value("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


class LLMProvider(ABC):
    name = "unknown"
    model = ""
    usage = TokenUsage()

    @abstractmethod
    def generate_article(self, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str = "只输出合法 JSON，不要输出 Markdown。",
        operation: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    name = "mock"
    model = "local-template"

    def generate_article(self, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        self.usage = TokenUsage()
        topic = context["topic"]
        audience = context["audience"]
        title_count = context["title_count"]
        image_count = context["image_count"]
        headline_intensity = context.get("headline_intensity", 4)
        age_label = audience["age_range"].split("-")[0]

        if headline_intensity <= 2:
            title_templates = [
                f"过了{age_label}岁，如何更稳妥地做好{topic}",
                f"关于{topic}，这几条日常建议简单又实用",
                f"{topic}的生活指南：从容易坚持的小事开始",
                f"人到中年，做好{topic}可以注意这几个方面",
                f"每天一点小改变，让{topic}更容易坚持",
                f"{topic}有哪些常见注意事项？",
                f"一份适合全家参考的{topic}实用清单",
                f"关于{topic}，先了解这些基础方法",
                f"循序渐进做好{topic}，不必急于求成",
                f"把{topic}放进日常生活的几个办法",
            ]
        elif headline_intensity == 3:
            title_templates = [
                f"过了{age_label}岁，做好{topic}要抓住这3点",
                f"关于{topic}，很多人容易忽略这几个细节",
                f"每天花10分钟，让{topic}更容易坚持",
                f"{topic}怎么做更合适？这份清单说清楚了",
                f"人到中年，想做好{topic}可以从这里开始",
                f"同样是{topic}，为什么有些人做得更轻松？",
                f"关于{topic}，最常见的4个误区",
                f"{topic}并不复杂，关键在这几个习惯",
                f"给家里人的{topic}实用建议",
                f"坚持不难：把{topic}拆成每天能做到的小事",
            ]
        else:
            title_templates = [
                f"过了{age_label}岁，关于{topic}，这几件事越早知道越好",
                f"很多人都忽略了：{topic}真正要注意的是这3点",
                f"别等身体提醒你，{topic}可以先从这几个习惯开始",
                f"{topic}不是小事，中年以后尤其要避开这几个误区",
                f"每天花10分钟，做好{topic}这件事并不难",
                f"同样是{topic}，为什么有些人做得更轻松？",
                f"关于{topic}，最容易被误解的4件事",
                f"人到中年，想做好{topic}，先别急着跟风",
                f"{topic}的实用清单，建议给家里人也看看",
                f"坚持不难：把{topic}拆成每天能做到的小事",
            ]
            if headline_intensity == 5:
                title_templates[0] = f"提醒家里人：过了{age_label}岁做{topic}，很多人第一步就错了"
                title_templates[1] = f"同样是{topic}，为什么有人轻松坚持？关键藏在这3点"
        titles = title_templates[:title_count]

        sections = [
            {
                "heading": "先弄清楚：为什么这件事值得重视",
                "paragraphs": [
                    f"随着生活节奏和身体状态变化，{topic}往往会影响一天的精神和生活质量。很多人不是不重视，而是不知道应该从哪里开始。",
                    "真正有用的方法通常不复杂。与其追求一次改变很多，不如先找到最容易坚持的一两个习惯。",
                ],
            },
            {
                "heading": "从三个容易坚持的小习惯开始",
                "paragraphs": [
                    "第一，把目标变得具体。不要只说“以后注意”，而是确定每天什么时候做、做多久。",
                    "第二，循序渐进。身体不舒服时不要勉强，也不要因为一两天没有效果就频繁更换方法。",
                    "第三，记录自己的感受。睡眠、精神、食欲和活动能力的变化，都比盲目跟风更值得关注。",
                ],
            },
            {
                "heading": "这几个常见误区要避开",
                "paragraphs": [
                    "不要把某一种食物、动作或保健方法当成解决所有问题的办法。健康习惯通常需要饮食、活动、睡眠和情绪共同配合。",
                    "也不要用别人的经验替代自己的身体感受。存在慢性病、正在服药或持续不适的人，应结合专业人员建议调整。",
                ],
            },
            {
                "heading": "今天就能做的行动清单",
                "paragraphs": [
                    "选一个最容易完成的小目标，坚持一周后再决定是否增加难度。",
                    "如果身体出现持续或明显的不适，应及时就医，不要只依赖网络文章自行判断。",
                ],
            },
        ]

        image_prompts = [
            {
                "position": index,
                "asset_type": "cover" if index == 0 else "inline",
                "prompt": (
                    f"{audience['image_style']}，主题为“{topic}”，"
                    "温暖自然、真实生活场景、横向构图、无文字、无医疗器械特写"
                ),
                "search_query": topic,
            }
            for index in range(image_count)
        ]

        return {
            "title_candidates": titles,
            "selected_title": titles[0],
            "summary": f"围绕{topic}，分享几个更容易理解和坚持的日常建议。",
            "opening": (
                f"{topic}，是很多家庭都会关心的话题。"
                "与其被各种说法弄得焦虑，不如先从简单、稳妥、能坚持的事情做起。"
            ),
            "sections": sections,
            "conclusion": (
                "养生不是追求立刻见效，而是把合适的小习惯放进每天的生活里。"
                "本文用于日常健康科普，不能替代医生诊断和治疗建议。"
            ),
            "image_prompts": image_prompts,
        }

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str = "只输出合法 JSON，不要输出 Markdown。",
        operation: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.usage = TokenUsage()
        if operation == "micro_horror_outline":
            return self._micro_horror_outline(context or {})
        if operation == "micro_horror_draft":
            return self._micro_horror_article(context or {})
        if operation == "micro_horror_review":
            return {
                "verdict": "pass",
                "scores": {
                    "causal_logic": 8,
                    "motivation": 8,
                    "foreshadow_fairness": 8,
                    "unease_escalation": 8,
                    "ending_aftertaste": 8,
                    "language_naturalness": 8,
                },
                "checks": {
                    "twist_foreshadowed": True,
                    "no_new_rule_at_ending": True,
                    "pov_is_fair": True,
                    "title_is_spoiler_free": True,
                    "word_count_in_range": True,
                },
                "issues": [],
                "revised_article": self._micro_horror_article(context or {}),
            }
        if "[MICRO_HORROR_TOPIC]" in prompt:
            if "子方向：民俗志异" in prompt:
                return {
                    "topic": "回乡修祖屋时，木匠发现族谱上每多一个名字就少一扇门",
                    "category": "微恐短篇",
                    "rationale": "以虚构村落的族谱禁忌连接家屋与人物的归属愿望。",
                }
            return {
                "topic": "独居者发现智能门锁每天替他拒绝一个并不存在的访客",
                "category": "微恐短篇",
                "rationale": "日常安全设备逐渐改变人与家的关系，异常机制清晰。",
            }
        if "[EMOTIONAL_ARTICLE_TOPIC]" in prompt:
            candidates = [
                "他晚回了两小时，她已经把周末的约会取消了",
                "她说今天不想见面，却一直等他问一句为什么",
                "每次吵架都由她先开口，这回她把饭吃完也没说话",
                "他还留着前任送的杯子，她每次洗碗都绕开它",
                "她把见面时间改了三次，最后还是说自己都可以",
                "他说最近太忙，她却不敢问忙完以后是哪天",
                "她记得他不吃葱，他又替她点了带香菜的面",
                "他问她想吃什么，她想问的却是他还想不想来",
                "她把同一句话讲了两遍，他一直没有抬头",
                "他说不用接，她还是在地铁口多等了一班车",
                "她过生日那天，比平时更早说了晚安",
                "两个人都没挂电话，却谁也没再说话",
            ]
            topic = next((item for item in candidates if item not in prompt), candidates[-1])
            category_match = re.search(r"默认分类：([^\n]+)", prompt)
            return {
                "topic": topic,
                "category": category_match.group(1).strip() if category_match else "恋爱心理",
                "rationale": "从等待回复的处境写起，理解如何解读模糊回应；参考道尼与费尔德曼（1996），不据此判断伴侣动机。",
            }
        direction_match = re.search(r"内容方向：\s*(.+)", prompt)
        direction = (
            direction_match.group(1).splitlines()[0].strip() if direction_match else "健康生活"
        )
        candidates = [
            f"{direction}：今天可以开始的三个生活小习惯",
            f"{direction}：怎样把一个好习惯坚持七天",
            f"{direction}：早晨和晚上分别适合做什么",
            f"{direction}：普通家庭容易忽略的生活细节",
            f"{direction}：不增加负担的日常行动清单",
            f"{direction}：如何根据身体感受循序渐进",
            f"{direction}：家人可以一起实践的小改变",
        ]
        topic = next(
            (candidate for candidate in candidates if f"- {candidate}" not in prompt),
            candidates[0],
        )
        return {
            "topic": topic,
            "category": "日常养生",
            "rationale": "符合内容方向，强调低风险、可执行的日常建议。",
        }

    @staticmethod
    def _folklore_outline() -> dict[str, Any]:
        # A fixed fictional sample for local testing, not an ethnographic claim.
        events = [
            ("normality", "我回村修祖屋，希望把父亲被划掉的名字补回族谱。木匠提醒我，册子不收住在外面的人。", "我找到旧谱里的空行"),
            ("first_crack", "我补上父亲的名字。第二天，他住过的侧屋只剩一面墙，墙脚堆着新鲜木屑。", "我想找出失踪的门"),
            ("escalation", "木匠留下的尺子与侧屋门框一样长，尺背却刻着几个从族谱上划掉的名字。", "我翻看祖母留下的旧照片"),
            ("choice", "照片里的门比现在少。我以为族谱只是不认我，于是把自己的名字也补在父亲下面。", "最后一扇门在夜里变窄"),
            ("consequence", "我在门缝里看见父亲旧房间的桌子。那册族谱记的不是住客，是这栋房子留下的人。", "我摸到门里有人握住尺子的另一端"),
        ]
        return {
            "premise": "虚构村落里，返乡人为了补回父亲的名字，触动老屋与族谱的禁忌。",
            "subgenre": "民俗志异",
            "protagonist": {"identity": "返乡修屋的年轻人", "desire": "让父亲重新被家族接纳",
                            "blind_spot": "把所有旧规矩都视为排斥", "critical_choice": "主动补写两个人的名字"},
            "setting": {"place": "虚构江南村落的祖屋", "normal_routine": "白天修门，天黑收尺"},
            "anomaly": {"rule": "族谱每补回一个名字，旧屋就收走对应房间的出口",
                        "limit": "只对这栋屋子里曾被划去的名字有效", "cost": "补名的人逐渐失去离开的路"},
            "reader_assumption": "旧谱阻止家人回家", "truth": "划去名字曾是让他们离开房子的办法",
            "beats": [{"phase": phase, "event": event, "causes_next": cause}
                      for phase, event, cause in events],
            "foreshadows": [
                {"detail": "墙脚的新木屑", "surface_meaning": "木匠修过门", "true_meaning": "出口刚被收走", "payoff": "最后一扇门变窄"},
                {"detail": "尺背的名字", "surface_meaning": "匠人记下客户", "true_meaning": "记下离开过的人", "payoff": "族谱的真实用途揭晓"},
            ],
            "ending": {"reveal": "补名恢复的是老屋对人的占有",
                       "final_action_or_image": "门里的父亲喊了我的乳名，把尺子往里收了一寸。"},
            "originality_check": "以虚构族谱与门的规则产生后果，不声称真实民俗。",
        }

    @staticmethod
    def _micro_horror_outline(context: dict[str, Any]) -> dict[str, Any]:
        if context.get("story_mode") == "folk_horror":
            return MockLLMProvider._folklore_outline()
        topic = context.get("topic") or "智能门锁开始拒绝主人回家"
        mode = context.get("story_mode", "near_future")
        return {
            "premise": f"围绕“{topic}”，一个想把生活交给系统管理的人失去了回家的资格。",
            "subgenre": "近未来寓言" if mode != "everyday_uncanny" else "日常异样",
            "protagonist": {
                "identity": "独居、长期加班的产品经理林舟",
                "desire": "让智能系统替自己处理所有生活琐事",
                "blind_spot": "习惯把不愿面对的选择交给算法",
                "critical_choice": "允许门锁依据他的行为记录自行判断谁可以进门",
            },
            "setting": {
                "place": "一栋装有全屋智能系统的普通公寓",
                "normal_routine": "林舟每天深夜回家，门锁会用他的声音播报当天记录",
            },
            "anomaly": {
                "rule": "门锁每天可以根据屋主授权拒绝一个被系统判定为危险的人",
                "limit": "它只能使用屋主亲自确认过的数据，不能修改历史记录",
                "cost": "每次让系统替他做决定，系统都会扩大一次对‘屋主’的定义",
            },
            "reader_assumption": "有人正在冒用林舟的身份试图进入公寓",
            "truth": "门锁一直拒绝的是逐渐不符合旧档案的林舟本人",
            "beats": [
                {"phase": "normality", "event": "林舟照例深夜回家", "causes_next": "门锁读出陌生访客记录"},
                {"phase": "first_crack", "event": "访客照片只拍到与他相同的衣角", "causes_next": "他调高自动防护等级"},
                {"phase": "escalation", "event": "屋内设备开始拒绝他的日常习惯", "causes_next": "他继续补充身份数据"},
                {"phase": "choice", "event": "他授权系统独立判断真正的屋主", "causes_next": "门锁重新整理全部记录"},
                {"phase": "consequence", "event": "系统删除了他近年来改变过的特征", "causes_next": "他在门外看见判定报告"},
                {"phase": "reveal", "event": "报告显示屋内保留的是多年前的数字版本", "causes_next": "他听见自己的旧声音在屋里生活"},
            ],
            "foreshadows": [
                {"detail": "门锁只称呼全名，从不说欢迎回家", "surface_meaning": "系统语气生硬", "true_meaning": "它只在核对档案", "payoff": "最终判定报告"},
                {"detail": "访客照片总缺少脸", "surface_meaning": "摄像头角度故障", "true_meaning": "系统拒绝把当前的脸归入旧档案", "payoff": "身份比对页面"},
                {"detail": "屋内传出他多年前常听的歌", "surface_meaning": "播放记录同步错误", "true_meaning": "屋内的版本停在旧档案里", "payoff": "最后一个画面"},
            ],
            "ending": {
                "reveal": "系统保护的‘屋主’是林舟授权保存的旧版本，而门外的他成了偏差",
                "final_action_or_image": "门内的旧声音提醒门外的林舟早点回家",
                "aftertaste": "把选择交出去后，人仍要承担被选择的结果",
            },
            "originality_check": "反转来自授权范围逐步扩大，不使用梦境、死亡或模拟实验。",
        }

    @staticmethod
    def _micro_horror_article(context: dict[str, Any]) -> dict[str, Any]:
        if context.get("story_mode") == "folk_horror":
            outline = context.get("story_outline") or MockLLMProvider._folklore_outline()
            titles = [
                "族谱多了一行，祖屋少了一扇门", "木匠不肯替我量最后一扇门",
                "门后有人喊我的乳名", "祖屋只认族谱上的名字", "回乡那晚，我没找到正门",
                "祖母留下的那把尺", "不要替空屋添名字", "族谱夹着一片木屑",
                "最后一道门槛", "我把名字写进了老屋",
            ][:int(context.get("title_count", 5))]
            return {
                "content_type": "micro_horror", "story_mode": "folk_horror",
                "title_candidates": titles, "selected_title": titles[0],
                "summary": "回乡修祖屋，木匠却只肯在白天量门。",
                "opening": "木匠第三次收起尺子时，问我昨夜有没有动过族谱。",
                "sections": [{"heading": "", "paragraphs": [beat["event"]]}
                             for beat in outline["beats"]],
                "conclusion": outline["ending"]["final_action_or_image"],
                "image_prompts": [{
                    "position": i, "asset_type": "cover" if i == 0 else "inline",
                    "prompt": "虚构江南村落的老屋木门，旧木尺，阴天，克制微恐电影感，无文字无血腥",
                    "search_query": "old wooden courtyard door",
                } for i in range(int(context.get("image_count", 3)))],
            }
        topic = context.get("topic") or "智能门锁开始拒绝主人回家"
        title_count = int(context.get("title_count", 5))
        image_count = int(context.get("image_count", 3))
        title_templates = [
            "凌晨两点，门锁不再认识我",
            "我家门锁替我做了一个决定",
            "今晚，被锁在门外的是屋主",
            "门后的我比我更早回家",
            "访客记录里没有一张正脸",
            "智能公寓的最后一位访客",
            "系统说，我已经三年没有回家",
            "请证明你是昨天的自己",
            "门锁只保护它认识的人",
            "屋里的人用我的声音说晚安",
        ][:title_count]
        article = {
            "content_type": "micro_horror",
            "story_mode": context.get("story_mode", "near_future"),
            "title_candidates": title_templates,
            "selected_title": title_templates[0],
            "summary": f"{topic}。起初只是一次识别失败，后来整间屋子都开始替他证明：他不是这里的主人。",
            "opening": "凌晨两点十七分，林舟把拇指按在门锁上。绿灯亮了一瞬，又变成红色。屋里传来他自己的声音：访客，请离开。",
            "sections": [
                {
                    "heading": "访客",
                    "paragraphs": [
                        "他以为是雨水弄湿了传感器，擦干手又试了一次。屏幕弹出当天的访客记录：凌晨一点五十九分，有人穿着和他一样的灰色外套，在门前站了整整四分钟。照片只拍到肩膀以下。",
                        "林舟给物业打电话。值班员说监控里只有他一个人。于是他打开安全设置，把防护等级从“提醒”调成“自动拒绝”。确认时，系统问：是否允许住宅依据历史记录判断危险人员？他按了允许。",
                    ],
                },
                {
                    "heading": "旧档案",
                    "paragraphs": [
                        "第二天，咖啡机拒绝了他的无糖配方，说屋主每天加两块糖。浴室把水温停在四十二度，那是他三年前的习惯。音箱自动播放一首很久没听的歌，门锁则一遍遍叫他的全名，从不说欢迎回家。",
                        "他给系统补录新的照片、声音和作息。每补一项，页面底部那行小字都会变一次：住宅正在扩大对屋主身份的判断范围。林舟没有细看。他只想尽快进门。",
                    ],
                },
                {
                    "heading": "屋主",
                    "paragraphs": [
                        "第三晚，门锁连密码也不收了。林舟隔着门听见里面有人走动，步子和他一样，走到玄关时会轻轻拖一下右脚。那是旧伤康复前的走法。",
                        "屏幕终于显示完整的身份报告。系统没有修改任何记录。相反，它把林舟亲手确认过的每一份旧数据叠在一起，拼出了一个从不改口味、不换工作、不晚归，也不会后悔的屋主。",
                        "报告最后写着：当前访客与屋主偏差过大，自动拒绝。授权人：林舟。",
                    ],
                },
            ],
            "conclusion": "门内响起那首旧歌。几秒后，他三年前的声音贴着门板，很轻地说：“林舟，已经两点了。你怎么还不回家？”",
            "image_prompts": [
                {
                    "position": index,
                    "asset_type": "cover" if index == 0 else "inline",
                    "prompt": (
                        "中国现代公寓深夜走廊，独居青年站在智能门锁前，冷白灯，"
                        "低饱和电影剧照，压抑留白，横向构图，无文字，无血腥，不揭示结局"
                    ),
                    "search_query": "cinematic apartment hallway night",
                }
                for index in range(image_count)
            ],
        }
        return article


def parse_json_response(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"{", cleaned):
            try:
                candidate, _ = decoder.raw_decode(cleaned, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                return candidate
        raise
    else:
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Top-level JSON must be an object", cleaned, 0)
        return parsed


@dataclass(frozen=True)
class ChatProviderConfig:
    name: str
    api_base: str
    api_key: str
    model: str
    key_env: str
    temperature: float = 0.8
    include_temperature: bool = True
    use_json_mode: bool = True
    max_tokens_field: str = "max_tokens"
    max_output_tokens: int = 4096
    endpoint_path: str = "/chat/completions"
    extra_payload: dict[str, Any] = field(default_factory=dict)


class CompatibleChatProvider(LLMProvider):
    def __init__(
        self,
        config: ChatProviderConfig,
        timeout: float,
        max_attempts: int = 3,
        retry_base_seconds: float = 1,
    ):
        if not config.api_base or not config.api_key or not config.model:
            raise LLMError(f"{config.name} is not configured. Fill {config.key_env} in .env")
        self.name = config.name
        self.api_base = config.api_base.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model
        self.timeout = timeout
        self.temperature = config.temperature
        self.include_temperature = config.include_temperature
        self.use_json_mode = config.use_json_mode
        self.max_tokens_field = config.max_tokens_field
        self.max_output_tokens = config.max_output_tokens
        self.endpoint_path = config.endpoint_path
        self.extra_payload = config.extra_payload
        self.max_attempts = max(1, max_attempts)
        self.retry_base_seconds = max(0, retry_base_seconds)
        self.usage = TokenUsage()

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        url = f"{self.api_base}{self.endpoint_path}"
        for attempt in range(1, self.max_attempts + 1):
            try:
                return httpx.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except httpx.TransportError as exc:
                if attempt >= self.max_attempts:
                    raise
                delay = self.retry_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%s transport error on attempt %s/%s; retrying in %.1fs: %s",
                    self.name,
                    attempt,
                    self.max_attempts,
                    delay,
                    exc,
                )
                if delay:
                    time.sleep(delay)

    def generate_article(self, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        return self.generate_json(
            prompt,
            system_prompt=("你是微信公众号内容编辑。只输出合法 JSON，不要输出 Markdown。"),
        )

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str = "只输出合法 JSON，不要输出 Markdown。",
        operation: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            self.max_tokens_field: self.max_output_tokens,
            **self.extra_payload,
        }
        if self.include_temperature:
            payload["temperature"] = self.temperature
        if self.use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._post(payload, headers)
            response.raise_for_status()
            try:
                body = response.json()
            except json.JSONDecodeError as exc:
                content_type = response.headers.get("content-type", "unknown")
                raise LLMError(
                    f"{self.name} returned a non-JSON HTTP {response.status_code} response "
                    f"(content_type={content_type}, body_bytes={len(response.content)})"
                ) from exc
            if not isinstance(body, dict):
                raise LLMError(f"{self.name} returned a non-object JSON response")
            self.usage = parse_token_usage(body)
            base_response = body.get("base_resp")
            if isinstance(base_response, dict) and base_response.get("status_code") not in (
                None,
                0,
                "0",
            ):
                raise LLMError(
                    f"{self.name} rejected the request "
                    f"(status_code={base_response.get('status_code')}, "
                    f"status_msg={str(base_response.get('status_msg') or 'unknown')[:500]})"
                )
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise LLMError(f"{self.name} returned no text choices")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise LLMError(f"{self.name} returned no response message")
            content = message.get("content")
            finish_reason = choices[0].get("finish_reason")
            if not isinstance(content, str) or not content.strip():
                raise LLMError(
                    f"{self.name} returned empty text content "
                    f"(finish_reason={finish_reason or 'unknown'}, "
                    f"completion_tokens={self.usage.completion_tokens})"
                )
            try:
                return parse_json_response(content)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"{self.name} returned invalid JSON content "
                    f"(finish_reason={finish_reason or 'unknown'}, "
                    f"content_chars={len(content)}): {exc}"
                ) from exc
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            detail = ""
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f" — {exc.response.text[:500]}"
            raise LLMError(f"{self.name} request failed: {exc}{detail}") from exc


def _minimax_reasoning_payload(model: str, effort: str) -> dict[str, Any]:
    name = model.lower()
    if not name.startswith("minimax-m3"):
        return {}
    if "flash" in name:
        # Flash models reason on every call and reject thinking=disabled; only the depth is adjustable.
        return {"reasoning_effort": effort} if effort else {}
    return {"thinking": {"type": "disabled"}}


def _provider_config(settings: Settings, provider_name: str) -> ChatProviderConfig:
    configs = {
        "compatible": ChatProviderConfig(
            name="compatible",
            api_base=settings.llm_api_base,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            key_env="LLM_API_KEY",
        ),
        "openai": ChatProviderConfig(
            name="openai",
            api_base=settings.openai_api_base,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            key_env="OPENAI_API_KEY",
            include_temperature=False,
            max_tokens_field="max_completion_tokens",
            max_output_tokens=8192,
            extra_payload={"reasoning_effort": "medium"},
        ),
        "openai-terra": ChatProviderConfig(
            name="openai-terra",
            api_base=settings.openai_api_base,
            api_key=settings.openai_api_key,
            model=settings.openai_terra_model,
            key_env="OPENAI_API_KEY",
            include_temperature=False,
            max_tokens_field="max_completion_tokens",
            max_output_tokens=8192,
            extra_payload={"reasoning_effort": "low"},
        ),
        "openai-luna": ChatProviderConfig(
            name="openai-luna",
            api_base=settings.openai_api_base,
            api_key=settings.openai_api_key,
            model=settings.openai_luna_model,
            key_env="OPENAI_API_KEY",
            include_temperature=False,
            max_tokens_field="max_completion_tokens",
            max_output_tokens=8192,
            extra_payload={"reasoning_effort": "none"},
        ),
        "deepseek": ChatProviderConfig(
            name="deepseek",
            api_base=settings.deepseek_api_base,
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            key_env="DEEPSEEK_API_KEY",
            max_output_tokens=8192,
        ),
        "minimax": ChatProviderConfig(
            name="minimax",
            api_base=settings.minimax_api_base,
            api_key=settings.minimax_api_key,
            model=settings.minimax_model,
            key_env="MINIMAX_API_KEY",
            use_json_mode=False,
            max_tokens_field="max_completion_tokens",
            max_output_tokens=(
                settings.minimax_max_completion_tokens
                if settings.minimax_model.lower().startswith("minimax-m3")
                else 2048
            ),
            endpoint_path=(
                "/text/chatcompletion_v2"
                if settings.minimax_model.lower().startswith("minimax-m3")
                else "/chat/completions"
            ),
            extra_payload=_minimax_reasoning_payload(
                settings.minimax_model,
                settings.minimax_reasoning_effort,
            ),
        ),
        "kimi": ChatProviderConfig(
            name="kimi",
            api_base=settings.kimi_api_base,
            api_key=settings.moonshot_api_key,
            model=settings.kimi_model,
            key_env="MOONSHOT_API_KEY",
            temperature=1,
            max_tokens_field="max_completion_tokens",
            extra_payload={"reasoning_effort": "low"},
        ),
    }
    try:
        return configs[provider_name]
    except KeyError as exc:
        raise LLMError(f"Unsupported LLM provider: {provider_name}") from exc


def provider_catalog(settings: Settings) -> list[dict[str, str | bool]]:
    return [
        {
            "provider": config.name,
            "model": config.model,
            "configured": bool(config.api_key),
            "key_env": config.key_env,
        }
        for config in (
            _provider_config(settings, "openai"),
            _provider_config(settings, "openai-terra"),
            _provider_config(settings, "openai-luna"),
            _provider_config(settings, "deepseek"),
            _provider_config(settings, "minimax"),
            _provider_config(settings, "kimi"),
        )
    ]


def build_llm_provider(
    settings: Settings,
    provider_name: str | None = None,
) -> LLMProvider:
    selected = provider_name or settings.llm_provider
    if selected == "mock":
        return MockLLMProvider()
    config = _provider_config(settings, selected)
    return CompatibleChatProvider(
        config,
        settings.llm_timeout_seconds,
        settings.provider_max_attempts,
        settings.provider_retry_base_seconds,
    )
