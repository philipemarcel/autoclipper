from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config_loader import get_env_secret
from src.models import ProfileConfig

logger = logging.getLogger(__name__)

YOUTUBE_UPLOAD_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class YouTubeShortsPublisher:
    """
    Publicador oficial de YouTube Shorts via YouTube Data API v3 (`videos.insert` resumable).
    O YouTube classifica automaticamente como Short qualquer vídeo vertical (`9:16`) com até 60s
    (nossos cortes já saem em `1080x1920` e `<= 58s`, e ainda anexamos `#Shorts`).
    """

    def __init__(
        self,
        profile: ProfileConfig,
        token_json_str: Optional[str] = None,
    ) -> None:
        yt_cfg = profile.publishing.youtube_shorts
        self.enabled = bool(yt_cfg.enabled)
        self.privacy_status = (yt_cfg.privacy_status or "public").strip().lower()
        if self.privacy_status not in {"public", "unlisted", "private"}:
            self.privacy_status = "public"
        self.category_id = str(yt_cfg.category_id or "20").strip()
        self.token_json_str = (
            token_json_str
            if token_json_str is not None
            else get_env_secret(yt_cfg.token_json_env, required=False)
        ).strip()

    @property
    def is_ready(self) -> bool:
        if not self.enabled or not self.token_json_str or not self.token_json_str.startswith("{"):
            return False
        try:
            data = json.loads(self.token_json_str)
            return bool(data.get("refresh_token") or data.get("token"))
        except Exception:
            return False

    def _build_youtube_service(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        info = json.loads(self.token_json_str)
        scopes = info.get("scopes") or YOUTUBE_UPLOAD_SCOPES
        creds = Credentials.from_authorized_user_info(info, scopes=scopes)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("youtube", "v3", credentials=creds, cache_discovery=False)

    @staticmethod
    def build_video_metadata(
        caption: str,
        title_hint: str = "",
        category_id: str = "20",
        privacy_status: str = "public",
    ) -> Dict[str, Any]:
        """Monta título (<=100 chars), descrição com `#Shorts` e tags limpas para o YouTube."""
        raw_caption = str(caption or "").strip()
        lines = [ln.strip() for ln in raw_caption.splitlines() if ln.strip()]
        non_tag_lines = [ln for ln in lines if not all(tok.startswith("#") for tok in ln.split())]
        base_title = (non_tag_lines[0] if non_tag_lines else str(title_hint or "Corte da live")).strip()
        base_title = re.sub(r"\s+", " ", base_title)

        # Mantém título <= 58 chars para não cortar no feed mobile do YouTube Shorts; #Shorts vai sempre na descrição
        if "#shorts" not in base_title.lower() and len(base_title) <= 48:
            final_title = f"{base_title} #Shorts"
        else:
            if len(base_title) > 58:
                cut_at = base_title.rfind(" ", 0, 56)
                final_title = base_title[: (cut_at if cut_at > 25 else 56)].rstrip(" ,;:") + "..."
            else:
                final_title = base_title

        description = raw_caption or base_title
        if "#shorts" not in description.lower():
            description = f"{description}\n\n#Shorts".strip()

        found_tags: List[str] = []
        for match in re.findall(r"#([A-Za-zÀ-ÿ0-9_]{2,30})", raw_caption):
            if match.lower() not in {t.lower() for t in found_tags}:
                found_tags.append(match)
        if "shorts" not in {t.lower() for t in found_tags}:
            found_tags.append("Shorts")
        found_tags = found_tags[:15]

        return {
            "snippet": {
                "title": final_title[:100],
                "description": description[:5000],
                "tags": found_tags,
                "categoryId": str(category_id or "20"),
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }

    def publish_short(self, video_path: Path, caption: str, title_hint: str = "") -> str:
        if not self.is_ready:
            raise RuntimeError("YouTube Shorts não configurado (YOUTUBE_TOKEN_JSON ausente).")

        from googleapiclient.http import MediaFileUpload

        service = self._build_youtube_service()
        body = self.build_video_metadata(
            caption=caption,
            title_hint=title_hint,
            category_id=self.category_id,
            privacy_status=self.privacy_status,
        )
        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=8 * 1024 * 1024,
        )
        request = service.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            _, response = request.next_chunk()

        video_id = str((response or {}).get("id") or "")
        short_url = f"https://youtube.com/shorts/{video_id}" if video_id else ""
        logger.info("YouTube Short enviado com sucesso! url=%s", short_url)
        return short_url or video_id
