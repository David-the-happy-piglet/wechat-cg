import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AudienceProfile
from .schemas import AudienceCreate, AudienceUpdate


def audience_values(data: AudienceCreate | AudienceUpdate) -> dict:
    raw = data.model_dump(exclude_unset=True)
    mapping = {
        "interests": "interests_json",
        "title_preferences": "title_preferences_json",
        "prohibited_topics": "prohibited_topics_json",
    }
    values: dict = {}
    for key, value in raw.items():
        target = mapping.get(key, key)
        values[target] = json.dumps(value, ensure_ascii=False) if key in mapping else value
    return values


def create_audience(session: Session, data: AudienceCreate) -> AudienceProfile:
    audience = AudienceProfile(**audience_values(data))
    session.add(audience)
    session.commit()
    session.refresh(audience)
    return audience


def update_audience(
    session: Session,
    audience: AudienceProfile,
    data: AudienceUpdate,
) -> AudienceProfile:
    for key, value in audience_values(data).items():
        setattr(audience, key, value)
    session.commit()
    session.refresh(audience)
    return audience


def seed_audiences(session: Session) -> None:
    defaults = [
        AudienceCreate(
            slug="senior-wellness",
            name="中老年养生",
            age_range="50-75",
            description="关注日常生活质量、身体舒适度和简单易坚持习惯的人群",
            interests=[
                "睡眠",
                "关节",
                "饮食",
                "心脑血管基础科普",
                "慢性病日常管理",
            ],
            writing_style="亲切、简单、少用术语，短段落，提供明确行动建议",
            title_preferences=[
                "过了50岁……",
                "很多人没注意……",
                "每天坚持这几个习惯……",
            ],
            article_length=1200,
            image_style="温暖明亮、真实自然的中国中老年人物生活场景",
            prohibited_topics=[
                "宣称治愈疾病",
                "建议停药",
                "未经证实的保健品疗效",
            ],
        ),
        AudienceCreate(
            slug="middle-age-health",
            name="中年健康管理",
            age_range="35-55",
            description="工作和家庭压力较大、希望用有限时间改善状态的人群",
            interests=[
                "睡眠不足",
                "职场压力",
                "体重管理",
                "腰颈健康",
                "家庭健康",
            ],
            writing_style="节奏明快，强调时间效率、现实场景和可执行性",
            title_preferences=[
                "人到中年……",
                "每天坐8小时的人……",
                "不是年纪大了，可能是……",
            ],
            article_length=1000,
            image_style="现代都市生活中的中国中年人物，真实自然、干净明亮",
            prohibited_topics=[
                "宣称治愈疾病",
                "制造健康焦虑",
                "建议停药",
            ],
        ),
        AudienceCreate(
            slug="micro-horror-readers",
            name="微恐短篇读者",
            age_range="18+",
            description="喜欢日常微恐、近未来寓言和公平反转的成年读者",
            interests=[
                "日常异样",
                "近未来技术",
                "社会规则",
                "关系错位",
                "伏笔反转",
            ],
            writing_style="克制、具体、短段落，以动作和细节推进，不解释寓意",
            title_preferences=[
                "熟悉场景里出现一个不对劲的细节",
                "标题制造信息缺口但不泄露反转",
                "避免夸张感叹号和恐怖词堆叠",
            ],
            article_length=1800,
            image_style=(
                "冷静克制的电影剧照感，中国当代日常场景，低饱和，"
                "自然光影，留有空白，无文字，无血腥"
            ),
            prohibited_topics=[
                "露骨血腥和虐杀细节",
                "把精神疾病当作反转工具",
                "伤害儿童的细节描写",
                "照搬现有作品的设定或结局",
            ],
        ),
        AudienceCreate(
            slug="relationship-readers",
            name="恋爱与情感成长读者",
            age_range="18-40",
            description="恋爱中的男女，以女性读者为主，也覆盖暧昧、分手、复合与单身成长阶段",
            interests=[
                "恋爱相处",
                "分手修复",
                "两性心理",
                "沟通与边界",
                "追求策略",
                "星座恋爱性格",
            ],
            writing_style="共情但清醒，像有判断力的朋友，结合具体场景、沟通句式和行动步骤",
            title_preferences=[
                "为什么越……的人，越容易……",
                "一段关系出现这些细节……",
                "分手后真正需要做的……",
            ],
            article_length=1500,
            image_style=(
                "当代中国都市生活与亲密关系场景，真实自然，温暖克制，"
                "柔和光线，有情绪但不煽情，无文字"
            ),
            prohibited_topics=[
                "PUA、操控、跟踪、欺骗或羞辱",
                "煽动性别仇恨与绝对化刻板印象",
                "承诺百分百追到或必定复合",
                "把星座当作科学诊断或命运预测",
                "诊断精神疾病或替代专业心理支持",
            ],
        ),
    ]
    existing_slugs = set(session.scalars(select(AudienceProfile.slug)).all())
    for item in defaults:
        if item.slug in existing_slugs:
            continue
        session.add(AudienceProfile(**audience_values(item)))
    session.commit()
