from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from src.models import (
    CopyConfig,
    DetectionConfig,
    DiscordProfileConfig,
    EditingConfig,
    InstagramPublishConfig,
    ProfileConfig,
    PublishingConfig,
    TikTokPublishConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_environment(env_path: Optional[Path] = None) -> None:
    target = env_path or (PROJECT_ROOT / ".env")
    if target.exists():
        load_dotenv(dotenv_path=target, override=False)
    else:
        load_dotenv(override=False)


def load_profiles(config_path: Optional[Path] = None, only_enabled: bool = True) -> List[ProfileConfig]:
    path = config_path or (PROJECT_ROOT / "config" / "profiles.yaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    profiles: List[ProfileConfig] = []
    for item in raw.get("profiles", []):
        det_raw = item.get("detection", {}) or {}
        edt_raw = item.get("editing", {}) or {}
        cpy_raw = item.get("copy", {}) or {}
        dsc_raw = item.get("discord", {}) or {}
        pub_raw = item.get("publishing", {}) or {}
        ig_raw = pub_raw.get("instagram", {}) or {}
        tt_raw = pub_raw.get("tiktok", {}) or {}

        prof = ProfileConfig(
            id=str(item["id"]),
            twitch_login=str(item["twitch_login"]).strip().lower(),
            display_name=str(item.get("display_name") or item["twitch_login"]),
            enabled=bool(item.get("enabled", True)),
            language=str(item.get("language", "pt")),
            youtube_channels=[str(u).strip() for u in (item.get("youtube_channels") or []) if str(u).strip()],
            detection=DetectionConfig(**det_raw),
            editing=EditingConfig(**edt_raw),
            copy=CopyConfig(**cpy_raw),
            discord=DiscordProfileConfig(**dsc_raw),
            publishing=PublishingConfig(
                drip_interval_minutes=int(pub_raw.get("drip_interval_minutes", 75)),
                quiet_hours_brt=pub_raw.get("quiet_hours_brt", [3, 9]),
                timezone=str(pub_raw.get("timezone", "America/Sao_Paulo")),
                instagram=InstagramPublishConfig(**ig_raw),
                tiktok=TikTokPublishConfig(**tt_raw),
            ),
        )
        if not only_enabled or prof.enabled:
            profiles.append(prof)
    return profiles


def load_subtitle_theme(theme_path: Optional[Path] = None) -> Dict[str, Any]:
    path = theme_path or (PROJECT_ROOT / "config" / "subtitle_theme.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_env_secret(env_key: str, required: bool = False, default: str = "") -> str:
    val = os.getenv(env_key, default).strip()
    if required and not val:
        raise RuntimeError(f"Variável de ambiente obrigatória ausente: {env_key}")
    return val
