from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database import SessionLocal
from app.main import app
from app.models import Article, ManualStoryDraft


def test_manual_claude_handoff_requires_review_before_upload():
    body = "楼道的灯灭了以后，我听见有人从墙后数门牌。" * 4
    body += "\n\n" + "第二天，邻居说这一层只有三户，可我明明看见第四扇门。" * 4
    body += "\n\n" + "我把钥匙放回口袋，第四扇门从里面轻轻锁上了。" * 4

    with TestClient(app) as client:
        response = client.post(
            "/admin/stories/manual/ideas",
            data={"idea": "公寓里多出一扇门，只有主角能够看见。", "story_mode": "everyday_uncanny"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        detail_path = response.headers["location"]
        draft_id = int(detail_path.rsplit("/", 1)[-1])
        with SessionLocal() as session:
            draft = session.get(ManualStoryDraft, draft_id)
            assert draft.status == "idea_review"
            assert draft.article_id is None
            assert draft.prompt_text == ""

        response = client.post(
            f"{detail_path}/prompt",
            data={
                "idea": "公寓里多出一扇门，只有主角能够看见。",
                "reader_boundary": "成年读者；克制、无血腥。",
                "style_prompt": "第一人称，三次递进，公平反转。",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        page = client.get(detail_path)
        assert "复制完整 Prompt" in page.text
        assert "公寓里多出一扇门" in page.text

        response = client.post(
            f"{detail_path}/import",
            data={
                "title": "第四扇门",
                "push_title": "邻居说这层只有三户，第四把钥匙是谁的？",
                "summary": "公寓里多了一扇门。",
                "body_text": body,
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        article_path = response.headers["location"]
        article_id = int(article_path.rsplit("/", 1)[-1])
        with SessionLocal() as session:
            article = session.get(Article, article_id)
            assert article.llm_provider == "manual-claude"
            assert article.status == "pending_review"
            assert article.assets == []
            assert article.selected_title == "邻居说这层只有三户，第四把钥匙是谁的？"
            assert "第四扇门" in article.content_html
            assert "邻居说这层只有三户" not in article.content_html
        page = client.get(article_path)
        assert "第四扇门" in page.text
        assert "上方是推送标题" in page.text
        assert "一键上传到公众号草稿箱" not in page.text
        blocked = client.post(f"{article_path}/wechat-draft", follow_redirects=False)
        assert "draft_error" in blocked.headers["location"]

        image_response = client.post(
            f"{detail_path}/images",
            data={"image_count": "1", "cover_prompt": "夜晚的旧公寓走廊"},
            follow_redirects=False,
        )
        assert image_response.status_code == 303
        with SessionLocal() as session:
            article = session.get(Article, article_id)
            assert len(article.assets) == 1
        approval = client.post(f"{detail_path}/approve", follow_redirects=False)
        assert approval.headers["location"] == article_path
        with SessionLocal() as session:
            assert session.scalar(select(Article.status).where(Article.id == article_id)) == "approved"
        uploaded = client.post(f"{article_path}/wechat-draft", follow_redirects=False)
        assert "draft_message" in uploaded.headers["location"]


def test_manual_import_escapes_html_and_preserves_single_article():
    with TestClient(app) as client:
        created = client.post(
            "/admin/stories/manual/ideas",
            data={"idea": "同事的工位总在夜里移动。"},
            follow_redirects=False,
        )
        detail_path = created.headers["location"]
        client.post(
            f"{detail_path}/prompt",
            data={"idea": "同事的工位总在夜里移动。", "reader_boundary": "成人微恐", "style_prompt": "克制表达"},
        )
        body = "<script>alert(1)</script>" + "我看见工位移动了。" * 18 + "\n\n" + "第二天它又回来了。" * 18
        imported = client.post(
            f"{detail_path}/import",
            data={"title": "空工位", "body_text": body},
            follow_redirects=False,
        )
        article_path = imported.headers["location"]
        with SessionLocal() as session:
            article_id = session.get(ManualStoryDraft, int(detail_path.rsplit("/", 1)[-1])).article_id
            article = session.get(Article, article_id)
            assert "<script>" not in article.content_html
            assert "&lt;script&gt;" in article.content_html
        edited = client.post(
            f"{detail_path}/import",
            data={"title": "移动的工位", "body_text": body},
            follow_redirects=False,
        )
        assert edited.headers["location"] == article_path
        with SessionLocal() as session:
            assert session.get(Article, article_id).selected_title == "移动的工位"


def test_model_idea_stays_reviewable_and_folklore_prompt_uses_selected_material():
    with TestClient(app) as client:
        created = client.post(
            "/admin/stories/manual/ideas",
            data={"story_mode": "folk_horror", "folklore_element": "ancestral"},
            follow_redirects=False,
        )
        assert created.status_code == 303
        detail_path = created.headers["location"]
        draft_id = int(detail_path.rsplit("/", 1)[-1])
        with SessionLocal() as session:
            draft = session.get(ManualStoryDraft, draft_id)
            assert draft.idea
            assert draft.status == "idea_review"
            assert draft.article_id is None
        reviewed = client.post(
            f"{detail_path}/prompt",
            data={"idea": "族谱上多一个名字，门就少一扇。", "reader_boundary": "成年读者", "style_prompt": "克制叙述"},
            follow_redirects=False,
        )
        assert reviewed.status_code == 303
        with SessionLocal() as session:
            draft = session.get(ManualStoryDraft, draft_id)
            assert "祠堂、族谱与祖先祭祀传说" in draft.prompt_text
            assert "族谱上多一个名字" in draft.prompt_text
            assert draft.article_id is None


def test_psychological_suspense_has_its_own_prompt_rule():
    with TestClient(app) as client:
        page = client.get("/admin/stories/manual")
        assert 'value="psychological_suspense"' in page.text
        assert 'value="urban_legend"' in page.text
        assert 'value="relationship_mystery"' in page.text
        assert 'value="transaction_mystery"' in page.text
        assert 'value="rule_tale"' in page.text
        created = client.post(
            "/admin/stories/manual/ideas",
            data={"idea": "同事突然不认识我。", "story_mode": "psychological_suspense"},
            follow_redirects=False,
        )
        detail_path = created.headers["location"]
        client.post(
            f"{detail_path}/prompt",
            data={"idea": "同事突然不认识我。", "reader_boundary": "成年读者", "style_prompt": "自然语言"},
        )
        with SessionLocal() as session:
            draft = session.get(ManualStoryDraft, int(detail_path.rsplit("/", 1)[-1]))
            assert "允许没有超自然现象" in draft.prompt_text
            assert "正文标题：" in draft.prompt_text
            assert "推送标题：" in draft.prompt_text
            assert "代词指向必须清楚" in draft.prompt_text
            assert "至少提前用闲谈或叮嘱埋下两条限制" in draft.prompt_text
            assert "揭示之后最多六行" in draft.prompt_text


def test_existing_unapproved_prompt_can_be_refreshed_to_latest_rules():
    with TestClient(app) as client:
        created = client.post(
            "/admin/stories/manual/ideas",
            data={"idea": "旧账本里多出一笔交易。", "story_mode": "transaction_mystery"},
            follow_redirects=False,
        )
        detail_path = created.headers["location"]
        draft_id = int(detail_path.rsplit("/", 1)[-1])
        client.post(
            f"{detail_path}/prompt",
            data={"idea": "旧账本里多出一笔交易。", "reader_boundary": "成年读者", "style_prompt": "短句"},
        )
        with SessionLocal() as session:
            draft = session.get(ManualStoryDraft, draft_id)
            assert "账目、交换条件和代价" in draft.prompt_text
            draft.prompt_text = "旧版 Prompt"
            session.commit()
        refreshed = client.post(f"{detail_path}/refresh-prompt", follow_redirects=False)
        assert refreshed.status_code == 303
        with SessionLocal() as session:
            draft = session.get(ManualStoryDraft, draft_id)
            assert "推送标题：" in draft.prompt_text
            assert "代词指向必须清楚" in draft.prompt_text
