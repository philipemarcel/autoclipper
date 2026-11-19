from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import requests

from src.config_loader import get_env_secret
from src.models import ProfileConfig

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_FB_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
GRAPH_VIDEO_BASE = f"https://graph-video.facebook.com/{GRAPH_API_VERSION}"


class FacebookReelsPublisher:
    """
    Publicador oficial de Reels em Páginas do Facebook via Meta Graph API (`/{page_id}/video_reels`),
    com fallback automático para `/{page_id}/videos`.
    Se `FB_PAGE_ID` / `FB_ACCESS_TOKEN` não estiverem definidos explicitamente, descobre
    automaticamente a Página vinculada ao `IG_ACCESS_TOKEN` (token `EAA...`) via `/me/accounts`.
    """

    def __init__(
        self,
        profile: ProfileConfig,
        page_id: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        fb_cfg = getattr(profile.publishing, "facebook", None)
        self.enabled = bool(fb_cfg.enabled) if fb_cfg is not None else True
        page_id_env = fb_cfg.page_id_env if fb_cfg is not None else "FB_PAGE_ID"
        token_env = fb_cfg.access_token_env if fb_cfg is not None else "FB_ACCESS_TOKEN"

        self.page_id = (page_id or get_env_secret(page_id_env, required=False)).strip()
        explicit_tok = (access_token or get_env_secret(token_env, required=False)).strip()
        fallback_ig_tok = get_env_secret(profile.publishing.instagram.access_token_env, required=False).strip()
        self.user_or_page_token = explicit_tok or (
            fallback_ig_tok if not fallback_ig_tok.upper().startswith("IG") else ""
        )
        self._has_explicit_fb_token = bool(explicit_tok)
        self._resolved_page_token: Optional[str] = explicit_tok or None

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and (self.page_id or self.user_or_page_token) and self.user_or_page_token)

    def has_publish_permission(self) -> bool:
        """Verifica se o token atual possui `pages_manage_posts` para publicar na Página do Facebook."""
        if not self.is_ready:
            return False
        if self._has_explicit_fb_token:
            return True
        try:
            resp = requests.get(
                f"{GRAPH_FB_BASE}/me/permissions",
                params={"access_token": self.user_or_page_token},
                timeout=15,
            )
            if not resp.ok:
                return False
            perms = {
                str(p.get("permission")): str(p.get("status"))
                for p in (resp.json().get("data") or [])
            }
            return perms.get("pages_manage_posts") == "granted"
        except Exception:
            return False

    def _resolve_page_credentials(self) -> tuple[str, str]:
        if self.page_id and self._resolved_page_token:
            return self.page_id, self._resolved_page_token

        # Se temos um token EAA... (User ou Page token), busca Página e Page Access Token via /me/accounts
        resp = requests.get(
            f"{GRAPH_FB_BASE}/me/accounts",
            params={"fields": "id,name,access_token", "access_token": self.user_or_page_token},
            timeout=20,
        )
        if resp.ok:
            pages = resp.json().get("data") or []
            if self.page_id:
                for p in pages:
                    if str(p.get("id")) == self.page_id and p.get("access_token"):
                        self._resolved_page_token = str(p["access_token"])
                        return self.page_id, self._resolved_page_token
            elif pages:
                first = pages[0]
                self.page_id = str(first["id"])
                self._resolved_page_token = str(first.get("access_token") or self.user_or_page_token)
                return self.page_id, self._resolved_page_token

        if not self.page_id:
            raise RuntimeError(
                "Facebook Page não configurada (defina FB_PAGE_ID + FB_ACCESS_TOKEN com permissão pages_manage_posts)."
            )
        return self.page_id, self.user_or_page_token

    def publish_reel(self, video_path: Path, caption: str) -> str:
        if not self.is_ready:
            raise RuntimeError("Facebook Reels não configurado (FB_PAGE_ID / FB_ACCESS_TOKEN ausentes).")

        page_id, page_token = self._resolve_page_credentials()
        video_bytes = video_path.read_bytes()
        file_size = len(video_bytes)

        # Fluxo oficial Facebook Page Reels (3 etapas: start -> rupload -> finish)
        start_resp = requests.post(
            f"{GRAPH_FB_BASE}/{page_id}/video_reels",
            data={"upload_phase": "start", "access_token": page_token},
            timeout=30,
        )
        if not start_resp.ok:
            err_txt = start_resp.text[:400]
            if "pages_manage_posts" in err_txt or start_resp.status_code in (400, 403):
                raise RuntimeError(
                    f"Facebook Reels HTTP {start_resp.status_code}: {err_txt}. "
                    "Para postar direto na Página do Facebook, gere um token com a permissão `pages_manage_posts` e salve em FB_ACCESS_TOKEN."
                )
            start_resp.raise_for_status()

        start_data = start_resp.json()
        video_id = str(start_data["video_id"])
        upload_url = str(start_data.get("upload_url") or f"https://rupload.facebook.com/video-upload/{GRAPH_API_VERSION}/{video_id}")

        up_resp = requests.post(
            upload_url,
            headers={
                "Authorization": f"OAuth {page_token}",
                "offset": "0",
                "file_size": str(file_size),
                "Content-Length": str(file_size),
                "Content-Type": "application/octet-stream",
            },
            data=video_bytes,
            timeout=300,
        )
        if not up_resp.ok:
            raise RuntimeError(f"Facebook rupload erro {up_resp.status_code}: {up_resp.text[:300]}")

        finish_resp = requests.post(
            f"{GRAPH_FB_BASE}/{page_id}/video_reels",
            data={
                "upload_phase": "finish",
                "video_id": video_id,
                "video_state": "PUBLISHED",
                "description": caption[:2000],
                "access_token": page_token,
            },
            timeout=45,
        )
        if not finish_resp.ok:
            # Fallback para upload de vídeo comum na Página se /video_reels finish falhar
            with open(video_path, "rb") as fh:
                vid_resp = requests.post(
                    f"{GRAPH_VIDEO_BASE}/{page_id}/videos",
                    data={"description": caption[:2000], "access_token": page_token},
                    files={"source": (video_path.name, fh, "video/mp4")},
                    timeout=300,
                )
            if not vid_resp.ok:
                raise RuntimeError(f"Facebook publish erro {finish_resp.status_code}: {finish_resp.text[:300]}")
            fb_id = str(vid_resp.json().get("id") or video_id)
            logger.info("Vídeo publicado na Página do Facebook com sucesso! id=%s", fb_id)
            return fb_id

        logger.info("Facebook Reel publicado com sucesso! video_id=%s", video_id)
        return video_id
