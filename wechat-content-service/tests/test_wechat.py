import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import BytesIO

import httpx
import pytest
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import Settings, wechat_settings_for_content
from app.database import Base
from app.models import Article, Asset, PublishJob
from app.services.publishing import upload_article_to_draft
from app.services.wechat import WechatClient, WechatError


@pytest.fixture
def account_settings():
    return Settings(
        wechat_enabled=True, wechat_app_id='health-id', wechat_app_secret='health-secret',
        wechat_author='养生作者', wechat_emotional_enabled=True,
        wechat_emotional_app_id='emotion-id', wechat_emotional_app_secret='emotion-secret',
        wechat_emotional_author='情感作者',
    )


@pytest.fixture
def session():
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value
    engine.dispose()


def make_article(session, content_type='emotional_article', **kwargs):
    article = Article(
        content_type=content_type, audience_profile_id=1, topic='关系',
        selected_title='好好相处', content_html='<p>本地预览</p>',
        content_json=json.dumps({'opening': '开头', 'sections': [
            {'heading': '一', 'paragraphs': ['第一段']},
            {'heading': '二', 'paragraphs': ['第二段']},
        ], 'conclusion': '结尾'}, ensure_ascii=False), **kwargs,
    )
    session.add(article)
    session.commit()
    return article


def response(payload):
    return httpx.Response(200, json=payload)


def test_account_selection_never_falls_back_to_health(account_settings):
    emotion = wechat_settings_for_content(account_settings, 'emotional_article')
    assert (emotion.wechat_app_id, emotion.wechat_app_secret, emotion.wechat_author) == (
        'emotion-id', 'emotion-secret', '情感作者',
    )
    assert wechat_settings_for_content(account_settings, 'health_article') is account_settings
    disabled = wechat_settings_for_content(replace(
        account_settings, wechat_emotional_enabled=False, wechat_emotional_app_id='',
        wechat_emotional_app_secret='',
    ), 'emotional_article')
    assert not disabled.wechat_enabled
    assert disabled.wechat_app_id == disabled.wechat_app_secret == ''


def test_tokens_cached_independently_and_expired_token_retried(monkeypatch, account_settings):
    calls = []
    failed_once = False

    def post(url, **kwargs):
        nonlocal failed_once
        calls.append((url, kwargs))
        if url.endswith('/stable_token'):
            assert kwargs['json']['force_refresh'] is False
            return response({'access_token': kwargs['json']['appid'] + '-token', 'expires_in': 7200})
        if not failed_once:
            failed_once = True
            return response({'errcode': 42001, 'errmsg': 'expired'})
        return response({'news_item': [{'title': '好好相处'}]})

    monkeypatch.setattr('app.services.wechat.httpx.post', post)
    health = WechatClient(account_settings)
    emotion = WechatClient(wechat_settings_for_content(account_settings, 'emotional_article'))
    assert health.access_token() == 'health-id-token'
    assert emotion.access_token() == 'emotion-id-token'
    assert health.access_token() == 'health-id-token'
    emotion.verify_draft('draft-id', '好好相处')
    token_calls = [args['json']['appid'] for url, args in calls if url.endswith('/stable_token')]
    assert token_calls == ['health-id', 'emotion-id', 'emotion-id']
    draft_calls = [args for url, args in calls if url.endswith('/draft/get')]
    assert len(draft_calls) == 2
    assert all(args['params']['access_token'] == 'emotion-id-token' for args in draft_calls)


def test_upload_converts_images_preserves_credits_and_emotional_layout(
    monkeypatch, account_settings, session, tmp_path,
):
    article = make_article(session, summary='摘要' * 100)
    path = tmp_path / 'source.webp'
    Image.new('RGBA', (2200, 1200), (200, 100, 150, 150)).save(path)
    for position in range(2):
        article.assets.append(Asset(
            position=position, prompt='图', local_path=str(path), source_url='http://localhost/image',
            source_kind='web', source_author='配图作者', license_name='CC BY',
            source_page_url='https://example.com/credit',
        ))
    session.commit()
    payloads = []

    def post(url, **kwargs):
        if url.endswith('/stable_token'):
            return response({'access_token': 'token', 'expires_in': 7200})
        if 'files' in kwargs:
            name, content, mime = kwargs['files']['media']
            assert name.endswith('.jpg') and mime == 'image/jpeg'
            assert len(content) < 1024 * 1024
            assert Image.open(BytesIO(content)).format == 'JPEG'
        if url.endswith('/add_material'):
            return response({'media_id': 'cover-id'})
        if url.endswith('/uploadimg'):
            return response({'url': 'https://mmbiz.qpic.cn/inline.jpg'})
        payloads.append(kwargs['json']['articles'][0])
        return response({'media_id': 'real-draft'})

    monkeypatch.setattr('app.services.wechat.httpx.post', post)
    client = WechatClient(wechat_settings_for_content(account_settings, article.content_type))
    assert client.create_draft(article) == 'real-draft'
    payload = payloads[0]
    assert payload['author'] == '情感作者'
    assert len(payload['digest']) == 120
    assert payload['thumb_media_id'] == 'cover-id'
    assert 'localhost' not in payload['content']
    assert 'https://mmbiz.qpic.cn/inline.jpg' in payload['content']
    assert '配图作者' in payload['content'] and 'CC BY' in payload['content']
    assert '#fffaf9' in payload['content']
    assert path.exists()


def test_update_draft_rerenders_single_section_with_inline_images(
    monkeypatch, account_settings, session,
):
    article = make_article(session, wechat_media_id="existing-draft")
    article.content_json = json.dumps({
        "opening": "开头",
        "sections": [{"heading": "", "paragraphs": [f"第{i}段" for i in range(8)]}],
        "conclusion": "结尾",
    }, ensure_ascii=False)
    article.assets.extend([
        Asset(position=0, prompt="封面", source_kind="ai", wechat_media_id="cover-id"),
        Asset(position=1, prompt="插图1", source_kind="ai", wechat_url="https://mmbiz/1.jpg"),
        Asset(position=2, prompt="插图2", source_kind="ai", wechat_url="https://mmbiz/2.jpg"),
    ])
    session.commit()
    requests = []
    client = WechatClient(wechat_settings_for_content(account_settings, article.content_type))

    def request(path, **kwargs):
        requests.append((path, kwargs["json"]))
        return {"errcode": 0}

    monkeypatch.setattr(client, "_request", request)
    assert client.update_draft(article) == "existing-draft"
    path, payload = requests[0]
    assert path == "/cgi-bin/draft/update"
    assert payload["media_id"] == "existing-draft"
    assert payload["articles"]["content"].count("<img") == 2


def test_publishing_routes_both_accounts_and_upgrades_mock(monkeypatch, session, account_settings):
    created = []
    verified = []

    def create(self, article):
        created.append((self.settings.wechat_app_id, article.id))
        return f'{self.settings.wechat_app_id}-draft-{article.id}'

    monkeypatch.setattr(WechatClient, 'create_draft', create)
    monkeypatch.setattr(WechatClient, 'verify_draft', lambda self, media, title: verified.append(media))
    health = make_article(session, 'health_article')
    emotion = make_article(session, wechat_media_id='mock-draft-9')
    upload_article_to_draft(session, health, account_settings)
    media, _ = upload_article_to_draft(session, emotion, account_settings)
    upload_article_to_draft(session, emotion, account_settings)
    assert created == [('health-id', health.id), ('emotion-id', emotion.id)]
    assert media.startswith('emotion-id-draft-')
    assert len(verified) == 3


def test_verification_timeout_keeps_id_and_recheck_recovers(monkeypatch, session, account_settings):
    article = make_article(session)
    created = []

    def create(self, article):
        created.append(article.id)
        return 'created-id'

    def fail(self, media, title):
        raise WechatError('微信接口连接中断或超时')

    monkeypatch.setattr(WechatClient, 'create_draft', create)
    monkeypatch.setattr(WechatClient, 'verify_draft', fail)
    media, message = upload_article_to_draft(session, article, account_settings)
    assert media == article.wechat_media_id == 'created-id'
    assert article.status == 'wechat_draft_unverified'
    assert '无需重复上传' in message
    job = session.scalar(select(PublishJob))
    assert job.status == 'completed_unverified'
    monkeypatch.setattr(WechatClient, 'verify_draft', lambda *args: None)
    upload_article_to_draft(session, article, account_settings)
    assert article.status == 'wechat_draft'
    assert job.status == 'completed' and job.error_message is None
    assert created == [article.id]


def test_transport_error_is_not_retried_or_leaked(monkeypatch, account_settings, caplog):
    # Alembic's logging configuration can disable pre-existing loggers in the full suite.
    monkeypatch.setattr(logging.getLogger('httpx'), 'disabled', False)
    client = WechatClient(account_settings)
    client._token = 'private-token'
    client._token_expires_at = datetime.now(UTC) + timedelta(minutes=10)
    calls = []

    def fail(url, **kwargs):
        calls.append(url)
        logging.getLogger('httpx').info('POST %s?access_token=%s', url, client._token)
        raise httpx.ReadTimeout('https://example.com?access_token=private-token')

    monkeypatch.setattr('app.services.wechat.httpx.post', fail)
    with caplog.at_level(logging.INFO, logger='httpx'), pytest.raises(WechatError) as error:
        client._request('/cgi-bin/draft/add', json={})
    assert len(calls) == 1
    assert 'private-token' not in str(error.value) + caplog.text
    assert '[REDACTED]' in caplog.text


@pytest.mark.parametrize('payload', [[], {'errcode': 48001, 'errmsg': 'api unauthorized'}])
def test_invalid_response_and_permission_errors(account_settings, payload):
    with pytest.raises(WechatError):
        WechatClient(account_settings)._check_response(response(payload))


def test_api_and_admin_uploads_use_separate_accounts(monkeypatch, account_settings):
    from fastapi.testclient import TestClient

    from app import admin, api
    from app.main import app

    health_client = WechatClient(account_settings)
    emotion_client = WechatClient(wechat_settings_for_content(account_settings, 'emotional_article'))
    monkeypatch.setattr(api, 'wechat_client', health_client)
    monkeypatch.setattr(api, 'emotional_wechat_client', emotion_client)
    monkeypatch.setattr(admin, 'settings', account_settings)
    created = []
    published = []

    def create(self, article):
        created.append((article.content_type, self.settings.wechat_app_id))
        return f'{self.settings.wechat_app_id}-{article.id}'

    def publish(self, media):
        published.append((self.settings.wechat_app_id, media))
        return 'publish-id'

    monkeypatch.setattr(WechatClient, 'create_draft', create)
    monkeypatch.setattr(WechatClient, 'verify_draft', lambda *args: None)
    monkeypatch.setattr(WechatClient, 'publish', publish)
    monkeypatch.setattr(WechatClient, 'check_connection', lambda self: {
        'app_id': self.settings.wechat_app_id, 'connected': True,
    })
    with TestClient(app) as client:
        for kind, audience, expected in (
            ('health_article', 'middle-age-health', 'health-id'),
            ('emotional_article', 'relationship-readers', 'emotion-id'),
        ):
            result = client.post('/api/articles/generate', json={
                'content_type': kind, 'audience_slug': audience, 'topic': '好好生活',
                'provider': 'mock', 'image_count': 1,
            })
            assert result.status_code == 201
            article_id = result.json()['id']
            draft = client.post(f'/api/articles/{article_id}/wechat-draft')
            assert draft.status_code == 200
            assert draft.json()['media_id'] == f'{expected}-{article_id}'
            result = client.post(f'/api/articles/{article_id}/publish')
            assert result.status_code == 200
            assert published[-1] == (expected, f'{expected}-{article_id}')
            assert client.get(f'/api/wechat/status?content_type={kind}').json()['app_id'] == expected
        assert created == [('health_article', 'health-id'), ('emotional_article', 'emotion-id')]
        result = client.post('/admin/emotions/generate', data={
            'idea': '好好相处', 'provider': 'mock', 'auto_upload_draft': 'true', 'review_topic': 'false',
        }, follow_redirects=False)
        assert result.status_code == 303
        assert 'draft_message=' in result.headers['location']
        assert created[-1] == ('emotional_article', 'emotion-id')
        assert 'emotion-id' in client.get('/admin/emotions').text
