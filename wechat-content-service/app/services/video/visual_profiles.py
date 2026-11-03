from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
DEFAULT_VISUAL_PROFILE = "minimal_explainer"


class VisualProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class VisualType:
    name: str
    use_when: str
    prompt: str


@dataclass(frozen=True)
class VisualProfile:
    profile_id: str
    name: str
    base_prompt: str
    default_type: str
    types: dict[str, VisualType]
    allow_photorealism: bool = False

    @property
    def allowed_types(self) -> tuple[str, ...]:
        return tuple(self.types)

    def normalize_type(self, value: object) -> str:
        candidate = str(value or "").strip().lower()
        return candidate if candidate in self.types else self.default_type

    def selection_rules(self) -> str:
        return "\n".join(
            f"- {item.name}: {item.use_when}" for item in self.types.values()
        )

    def style_bible(self, style_notes: str = "") -> str:
        parts = [self.base_prompt]
        if style_notes.strip():
            parts.append(f"本项目补充要求：{style_notes.strip()}")
        return "\n".join(parts)

    def render_prompt(
        self,
        visual_type: object,
        scene_prompt: str,
        *,
        style_notes: str = "",
    ) -> tuple[str, str]:
        normalized_type = self.normalize_type(visual_type)
        parts = [
            self.base_prompt,
            f"本幕视觉类型：{normalized_type}。{self.types[normalized_type].prompt}",
        ]
        if style_notes.strip():
            parts.append(f"本项目补充风格：{style_notes.strip()}")
        parts.append(f"本幕具体内容：{scene_prompt.strip()}")
        return "\n".join(parts), normalized_type


def visual_profile_root() -> Path:
    return Path(__file__).resolve().parents[3] / "prompts" / "video" / "styles"


def list_visual_profiles(*, root: Path | None = None) -> list[VisualProfile]:
    profile_root = root or visual_profile_root()
    profiles: list[VisualProfile] = []
    for path in sorted(profile_root.glob("*.json")):
        if PROFILE_ID_PATTERN.fullmatch(path.stem):
            profiles.append(load_visual_profile(path.stem, root=profile_root))
    return profiles


def _required_text(data: dict[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VisualProfileError(f"Visual profile {path} has no valid {key}")
    return value.strip()


def load_visual_profile(
    profile_id: str = DEFAULT_VISUAL_PROFILE,
    *,
    root: Path | None = None,
) -> VisualProfile:
    if not PROFILE_ID_PATTERN.fullmatch(profile_id):
        raise VisualProfileError(f"Invalid visual profile id: {profile_id}")
    path = (root or visual_profile_root()) / f"{profile_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VisualProfileError(f"Visual profile does not exist: {profile_id}") from exc
    except json.JSONDecodeError as exc:
        raise VisualProfileError(f"Visual profile is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise VisualProfileError(f"Visual profile must be a JSON object: {path}")
    configured_id = _required_text(data, "id", path)
    if configured_id != profile_id:
        raise VisualProfileError(
            f"Visual profile id mismatch: requested {profile_id}, found {configured_id}"
        )
    types_data = data.get("types")
    if not isinstance(types_data, dict) or not types_data:
        raise VisualProfileError(f"Visual profile {path} must define at least one type")
    types: dict[str, VisualType] = {}
    for name, item in types_data.items():
        if not isinstance(name, str) or not PROFILE_ID_PATTERN.fullmatch(name):
            raise VisualProfileError(f"Visual profile {path} has invalid type id: {name}")
        if not isinstance(item, dict):
            raise VisualProfileError(f"Visual type {name} in {path} must be an object")
        types[name] = VisualType(
            name=name,
            use_when=_required_text(item, "use_when", path),
            prompt=_required_text(item, "prompt", path),
        )
    default_type = _required_text(data, "default_type", path)
    if default_type not in types:
        raise VisualProfileError(
            f"Visual profile {path} default_type is not present in types"
        )
    return VisualProfile(
        profile_id=profile_id,
        name=_required_text(data, "name", path),
        base_prompt=_required_text(data, "base_prompt", path),
        default_type=default_type,
        types=types,
        allow_photorealism=bool(data.get("allow_photorealism", False)),
    )
