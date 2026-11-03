import json
import logging
import re
import tempfile
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import Settings
from ..models import Article, Asset
from .images import is_stock_preview_url
from .renderer import render_wechat_html


class _CredentialFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = re.sub(
            r"([?&](?:access_token|secret)=)[^&\s\"']+",
            r"\1[REDACTED]",
            record.getMessage(),
        )
        record.args = ()
        return True


# httpx logs request URLs at INFO, including WeChat's query-string token.
logging.getLogger("httpx").addFilter(_CredentialFilter())


class WechatError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None):
        super().__init__(message)
        self.code = code


class WechatClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_base = settings.wechat_api_base.rstrip("/")
        self._token: str | None = None
        self._token_expires_at: datetime | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.wechat_enabled

    def _redact(self, message: str) -> str:
        for secret in (self.settings.wechat_app_secret, self._token):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message

    def _check_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise WechatError(f"微信接口 HTTP {response.status_code}，请稍后核验草稿箱")
        try:
            data = response.json()
        except ValueError:
            raise WechatError("微信接口返回了非 JSON 响应，请稍后重试") from None
        if not isinstance(data, dict):
            raise WechatError("微信接口返回格式异常")
        code = data.get("errcode", 0)
        if code != 0:
            hints = {
                40164: "请将错误中的出口 IP 加入公众号后台的 IP 白名单。",
                40013: "请核对公众号 App ID。",
                40125: "请核对公众号 App Secret。",
                48001: "当前公众号没有此接口权限，请在公众号后台检查接口权限。",
                40007: "素材或草稿不存在，请检查是否已在微信后台删除或发布。",
                45009: "接口调用次数已达上限，请稍后重试。",
                40001: "访问令牌无效，请重新连接公众号。",
                40014: "访问令牌无效，请重新连接公众号。",
                42001: "访问令牌已过期，请重新连接公众号。",
            }
            detail = self._redact(str(data.get("errmsg", "")))[:500]
            raise WechatError(
                f"微信接口错误 {code}：{detail} {hints.get(code, '')}", code=code
            )
        return data

    @staticmethod
    def _required(data: dict[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise WechatError(f"微信接口响应缺少 {key}")
        return value

    def access_token(self) -> str:
        if self._token and self._token_expires_at and datetime.now(UTC) < self._token_expires_at:
            return self._token
        if not self.settings.wechat_app_id or not self.settings.wechat_app_secret:
            raise WechatError("请配置 WECHAT_APP_ID 和 WECHAT_APP_SECRET")
        try:
            response = httpx.post(
                f"{self.api_base}/cgi-bin/stable_token",
                json={
                    "grant_type": "client_credential",
                    "appid": self.settings.wechat_app_id,
                    "secret": self.settings.wechat_app_secret,
                    "force_refresh": False,
                },
                timeout=30,
            )
        except httpx.RequestError:
            raise WechatError("无法连接微信令牌接口，请检查网络") from None
        data = self._check_response(response)
        token = self._required(data, "access_token")
        expires_in = int(data.get("expires_in", 7200))
        self._token = token
        self._token_expires_at = datetime.now(UTC) + timedelta(
            seconds=max(expires_in - min(300, expires_in // 2), 0)
        )
        return token

    def _request(
        self, path: str, *, method: str = "POST", params: dict | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        # Only explicit invalid-token responses may be replayed. A network timeout
        # after draft/add may already have created a draft, so never retry it here.
        for attempt in range(2):
            token = self.access_token()
            try:
                request = httpx.get if method == "GET" else httpx.post
                response = request(
                    f"{self.api_base}{path}",
                    params={**(params or {}), "access_token": token},
                    timeout=60,
                    **kwargs,
                )
            except httpx.RequestError:
                raise WechatError(
                    "微信接口连接中断或超时；若正在上传，请先检查公众号草稿箱再重试"
                ) from None
            try:
                return self._check_response(response)
            except WechatError as exc:
                if attempt == 0 and exc.code in {40001, 40014, 42001}:
                    self._token = None
                    self._token_expires_at = None
                    continue
                raise
        raise AssertionError("unreachable")

    def check_connection(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "connected": False, "message": "当前为模拟模式，草稿不会上传到微信"}
        data = self._request("/cgi-bin/draft/count", method="GET")
        if not isinstance(data.get("total_count"), int):
            raise WechatError("微信草稿箱查询响应缺少 total_count")
        return {
            "enabled": True, "connected": True,
            "app_id": self.settings.wechat_app_id,
            "draft_count": data["total_count"],
            "message": "公众号连接正常，草稿箱查询成功",
        }

    def _material_file(self, asset: Asset) -> tuple[Path, bool]:
        if asset.local_path and Path(asset.local_path).is_file():
            return Path(asset.local_path), False
        if not asset.source_url:
            raise WechatError(f"图片 {asset.id} 没有可用文件")
        try:
            response = httpx.get(asset.source_url, timeout=60, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError:
            raise WechatError(f"无法下载图片 {asset.id}，请检查图片是否可访问") from None
        with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as temporary:
            path = Path(temporary.name)
            try:
                temporary.write(response.content)
            except Exception:
                path.unlink(missing_ok=True)
                raise
        return path, True

    def _image_bytes(self, asset: Asset) -> bytes:
        path, remove_after = self._material_file(asset)
        try:
            with Image.open(path) as original:
                img = ImageOps.exif_transpose(original).convert("RGBA")
                background = Image.new("RGB", img.size, "white")
                background.paste(img, mask=img.getchannel("A"))
                background.thumbnail((1920, 1920))
                for size in (1920, 1440, 1080, 800):
                    background.thumbnail((size, size))
                    for quality in (90, 80, 65):
                        output = BytesIO()
                        background.save(output, format="JPEG", quality=quality, optimize=True)
                        if output.tell() < 1024 * 1024:
                            return output.getvalue()
                raise WechatError(f"图片 {asset.id} 压缩后仍过大")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise WechatError(f"图片 {asset.id} 无法读取，请重新生成配图") from None
        finally:
            if remove_after:
                path.unlink(missing_ok=True)

    def upload_inline_image(self, asset: Asset) -> str:
        data = self._request(
            "/cgi-bin/media/uploadimg",
            files={"media": ("inline.jpg", self._image_bytes(asset), "image/jpeg")},
        )
        return self._required(data, "url")

    def upload_cover(self, asset: Asset) -> str:
        data = self._request(
            "/cgi-bin/material/add_material", params={"type": "image"},
            files={"media": ("cover.jpg", self._image_bytes(asset), "image/jpeg")},
        )
        return self._required(data, "media_id")

    def create_draft(self, article: Article) -> str:
        payload = self._draft_article_payload(article)
        data = self._request(
            "/cgi-bin/draft/add",
            json={"articles": [payload]},
        )
        return self._required(data, "media_id")

    def update_draft(self, article: Article) -> str:
        if not article.wechat_media_id:
            raise WechatError("文章还没有微信草稿 ID，无法更新草稿")
        payload = self._draft_article_payload(article)
        self._request(
            "/cgi-bin/draft/update",
            json={
                "media_id": article.wechat_media_id,
                "index": 0,
                "articles": payload,
            },
        )
        return article.wechat_media_id

    def _draft_article_payload(self, article: Article) -> dict[str, Any]:
        assets = sorted(article.assets, key=lambda item: item.position)
        if not assets:
            raise WechatError("文章至少需要一张封面图，请先生成配图")
        if not article.selected_title.strip():
            raise WechatError("请先选择文章标题")
        cover = assets[0]
        if cover.source_kind == "web" and is_stock_preview_url(cover.source_page_url or ""):
            raise WechatError("封面是带水印的图库预览图，请先更换封面再上传草稿")
        cover.wechat_media_id = cover.wechat_media_id or self.upload_cover(cover)
        # Position zero is reserved for the cover; body renderers start at one.
        urls = [""]
        for asset in assets[1:]:
            asset.wechat_url = asset.wechat_url or self.upload_inline_image(asset)
            urls.append(asset.wechat_url)
        credits = [
            {key: getattr(asset, key) for key in (
                "source_kind", "source_page_url", "source_author", "license_name", "license_url"
            )}
            for asset in assets
        ]
        content = render_wechat_html(article_content(article), urls, credits)
        article.content_html = content
        return {
            "article_type": "news",
            "title": article.selected_title,
            "author": self.settings.wechat_author,
            "digest": (article.summary or "")[:120],
            "content": content,
            "content_source_url": "",
            "thumb_media_id": cover.wechat_media_id,
            "need_open_comment": 0,
            "only_fans_can_comment": 0,
        }

    def get_draft(self, media_id: str) -> dict[str, Any]:
        return self._request("/cgi-bin/draft/get", json={"media_id": media_id})

    def verify_draft(self, media_id: str, expected_title: str) -> None:
        draft = self.get_draft(media_id)
        items = draft.get("news_item")
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            raise WechatError(f"WeChat draft verification returned no articles for {media_id}")
        title = items[0].get("title")
        if title != expected_title:
            raise WechatError(
                "WeChat draft verification title mismatch: "
                f"expected {expected_title!r}, got {title!r}"
            )

    def publish(self, media_id: str) -> str:
        data = self._request("/cgi-bin/freepublish/submit", json={"media_id": media_id})
        return self._required(data, "publish_id")


def article_content(article: Article) -> dict[str, Any]:
    content = json.loads(article.content_json)
    content["content_type"] = article.content_type
    return content
