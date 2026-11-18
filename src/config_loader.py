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


def _parse_profiles_data(raw: Any, only_enabled: bool = True) -> List[ProfileConfig]:
    """Converte estrutura dict/list (YAML ou JSON do Discord Bot) em objetos ProfileConfig."""
    items = raw if isinstance(raw, list) else (raw or {}).get("profiles", [])
    profiles: List[ProfileConfig] = []
    for item in items:
        det_raw = item.get("detection", {}) or {}
        edt_raw = item.get("editing", {}) or {}
        cpy_raw = item.get("copy", {}) or {}
        dsc_raw = item.get("discord", {}) or {}
        pub_raw = item.get("publishing", {}) or {}
        ig_raw = pub_raw.get("instagram", {}) or {}
        tt_raw = pub_raw.get("tiktok", {}) or {}

        twitch_login = str(item.get("twitch_login") or "").strip().lower()
        kick_list = [str(u).strip() for u in (item.get("kick_channels") or []) if str(u).strip()]
        yt_list = [str(u).strip() for u in (item.get("youtube_channels") or []) if str(u).strip()]
        fallback_name = twitch_login or (kick_list[0] if kick_list else str(item.get("id", "AutoClipper")))

        prof = ProfileConfig(
            id=str(item.get("id") or f"perfil_{len(profiles) + 1:02d}"),
            twitch_login=twitch_login,
            display_name=str(item.get("display_name") or fallback_name),
            enabled=bool(item.get("enabled", True)),
            language=str(item.get("language", "pt")),
            youtube_channels=yt_list,
            kick_channels=kick_list,
            detection=DetectionConfig(**det_raw),
            editing=EditingConfig(**edt_raw),
            copy=CopyConfig(**cpy_raw),
            discord=DiscordProfileConfig(**dsc_raw),
            publishing=PublishingConfig(
                drip_interval_minutes=int(pub_raw.get("drip_interval_minutes", 75)),
                quiet_hours_brt=pub_raw.get("quiet_hours_brt", [3, 9]),
                timezone=str(pub_raw.get("timezone", "America/Sao_Paulo")),
                drive_retention_days=int(pub_raw.get("drive_retention_days", 10)),
                instagram=InstagramPublishConfig(**ig_raw),
                tiktok=TikTokPublishConfig(**tt_raw),
            ),
        )
        if not only_enabled or prof.enabled:
            profiles.append(prof)
    return profiles


def load_profiles(config_path: Optional[Path] = None, only_enabled: bool = True) -> List[ProfileConfig]:
    """
    Carrega perfis na ordem de prioridade:
      1. `AUTOCLIPPER_PROFILES_JSON` (injetado pelo Bot Discord multi-usuario) se definido e sem `config_path` explícito.
      2. `config_path` explícito ou `config/profiles.yaml`.
      3. Fallback para `config/profiles.example.yaml` (Open Template / Modelo C).
    """
    env_profiles_json = os.getenv("AUTOCLIPPER_PROFILES_JSON", "").strip()
    if config_path is None and env_profiles_json:
        raw_json = json.loads(env_profiles_json)
        return _parse_profiles_data(raw_json, only_enabled=only_enabled)

    if config_path is not None:
        path = config_path
    else:
        primary = PROJECT_ROOT / "config" / "profiles.yaml"
        example = PROJECT_ROOT / "config" / "profiles.example.yaml"
        path = primary if primary.exists() else example

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _parse_profiles_data(raw, only_enabled=only_enabled)


def load_subtitle_theme(theme_path: Optional[Path] = None) -> Dict[str, Any]:
    path = theme_path or (PROJECT_ROOT / "config" / "subtitle_theme.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_env_secret(env_key: str, required: bool = False, default: str = "") -> str:
    val = os.getenv(env_key, default).strip()
    if required and not val:
        raise RuntimeError(f"Variável de ambiente obrigatória ausente: {env_key}")
    return val
