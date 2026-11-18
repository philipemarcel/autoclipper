from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import TwitchClip

logger = logging.getLogger(__name__)

TWITCH_OAUTH_URL = "https://id.twitch.tv/oauth2/token"
TWITCH_HELIX_BASE = "https://api.twitch.tv/helix"


def parse_twitch_duration_seconds(duration_str: str) -> int:
    """Converte durações do VOD da Twitch como '6h24m12s', '45m10s' ou '58s' em segundos."""
    if not duration_str:
        return 0
    hours = re.search(r"(\d+)h", duration_str)
    minutes = re.search(r"(\d+)m", duration_str)
    seconds = re.search(r"(\d+)s", duration_str)
    total = 0
    if hours:
        total += int(hours.group(1)) * 3600
    if minutes:
        total += int(minutes.group(1)) * 60
    if seconds:
        total += int(seconds.group(1))
    return total


class TwitchHelixClient:
    """Cliente oficial Twitch Helix API usando apenas conta comum de espectador (Client Credentials)."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        self.client_id = (client_id or os.getenv("TWITCH_CLIENT_ID", "")).strip()
        self.client_secret = (client_secret or os.getenv("TWITCH_CLIENT_SECRET", "")).strip()
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "Credenciais TWITCH_CLIENT_ID e TWITCH_CLIENT_SECRET são obrigatórias."
            )
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._session = requests.Session()

    def _ensure_token(self) -> str:
        now = time.time()
        if self._access_token and now < (self._token_expires_at - 120):
            return self._access_token

        resp = self._session.post(
            TWITCH_OAUTH_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._access_token = payload["access_token"]
        self._token_expires_at = now + int(payload.get("expires_in", 3600))
        return self._access_token

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        token = self._ensure_token()
        headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {token}",
        }
        resp = self._session.get(f"{TWITCH_HELIX_BASE}/{endpoint.lstrip('/')}", headers=headers, params=params, timeout=25)
        if resp.status_code == 401:
            self._access_token = None
            token = self._ensure_token()
            headers["Authorization"] = f"Bearer {token}"
            resp = self._session.get(f"{TWITCH_HELIX_BASE}/{endpoint.lstrip('/')}", headers=headers, params=params, timeout=25)
        resp.raise_for_status()
        return resp.json()

    def get_user_by_login(self, login: str) -> Optional[Dict[str, Any]]:
        data = self._get("users", {"login": login.lower().strip()})
        items = data.get("data", [])
        return items[0] if items else None

    def get_active_stream(self, broadcaster_id: str) -> Optional[Dict[str, Any]]:
        data = self._get("streams", {"user_id": broadcaster_id, "first": 1})
        items = data.get("data", [])
        if items and items[0].get("type") == "live":
            return items[0]
        return None

    def get_recent_vods(self, broadcaster_id: str, first: int = 5) -> List[Dict[str, Any]]:
        data = self._get(
            "videos",
            {
                "user_id": broadcaster_id,
                "type": "archive",
                "first": min(max(first, 1), 20),
            },
        )
        return data.get("data", [])

    def get_clips_in_window(
        self,
        broadcaster_id: str,
        started_at_iso: str,
        ended_at_iso: Optional[str] = None,
        max_pages: int = 5,
    ) -> List[TwitchClip]:
        params: Dict[str, Any] = {
            "broadcaster_id": broadcaster_id,
            "started_at": started_at_iso,
            "first": 100,
        }
        if ended_at_iso:
            params["ended_at"] = ended_at_iso

        clips: List[TwitchClip] = []
        cursor: Optional[str] = None

        for _ in range(max_pages):
            page_params = dict(params)
            if cursor:
                page_params["after"] = cursor

            data = self._get("clips", page_params)
            for raw in data.get("data", []):
                vod_offset_raw = raw.get("vod_offset")
                clips.append(
                    TwitchClip(
                        id=str(raw["id"]),
                        url=str(raw["url"]),
                        title=str(raw.get("title", "")).strip(),
                        view_count=int(raw.get("view_count", 0)),
                        created_at=str(raw["created_at"]),
                        duration=float(raw.get("duration", 30.0)),
                        creator_name=str(raw.get("creator_name", "viewer")),
                        broadcaster_id=str(raw["broadcaster_id"]),
                        broadcaster_name=str(raw.get("broadcaster_name", "")),
                        game_id=str(raw.get("game_id", "")),
                        video_id=str(raw.get("video_id", "") or ""),
                        vod_offset=int(vod_offset_raw) if vod_offset_raw is not None else None,
                    )
                )

            cursor = data.get("pagination", {}).get("cursor")
            if not cursor:
                break

        return clips

    def get_game_name(self, game_id: str) -> str:
        if not game_id:
            return ""
        data = self._get("games", {"id": game_id})
        items = data.get("data", [])
        return str(items[0].get("name", "")) if items else ""

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
