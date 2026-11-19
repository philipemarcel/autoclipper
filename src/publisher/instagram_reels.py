from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import requests

from src.config_loader import get_env_secret
from src.models import ProfileConfig

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_FB_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
GRAPH_IG_BASE = f"https://graph.instagram.com/{GRAPH_API_VERSION}"
RUPLOAD_BASE = f"https://rupload.facebook.com/ig-api-upload/{GRAPH_API_VERSION}"

_PLACEHOLDER_VALUES = {
    "",
    "id_conta_instagram_creator_ou_business",
    "token_longa_duracao_meta_graph_api",
    "seu_ig_account_id",
}


class InstagramReelsPublisher:
    """
    Publicador oficial Instagram Reels via Meta API:
      - Suporta **Instagram API com Login do Instagram (`IG...` -> `graph.instagram.com`)**,
        que não exige Página do Facebook e publica usando URL HTTPS pública (`download_url` do Drive).
      - Suporta **Instagram Graph API com Facebook Login (`EAA...` -> `graph.facebook.com` + `rupload`)**,
        que faz upload binário resumable direto do arquivo local `master_1080p60.mp4`.
    """

    def __init__(
        self,
        profile: ProfileConfig,
        account_id: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        ig_cfg = profile.publishing.instagram
        self.enabled = ig_cfg.enabled
        self.account_id = (account_id or get_env_secret(ig_cfg.account_id_env, required=False)).strip()
        self.access_token = (access_token or get_env_secret(ig_cfg.access_token_env, required=False)).strip()

    @property
    def is_instagram_login_token(self) -> bool:
        return self.access_token.upper().startswith("IG")

    @property
    def is_ready(self) -> bool:
        return bool(
            self.enabled
            and self.account_id
            and self.access_token
            and self.account_id.lower() not in _PLACEHOLDER_VALUES
            and self.access_token.lower() not in _PLACEHOLDER_VALUES
        )

    def publish_reel(
        self,
        video_path: Path,
        caption: str,
        max_wait_sec: int = 240,
        video_url: str = "",
    ) -> str:
        if not self.is_ready:
            raise RuntimeError("Instagram Reels não configurado (IG_ACCOUNT_ID / IG_ACCESS_TOKEN ausentes).")

        if self.is_instagram_login_token:
            return self._publish_via_instagram_login(caption=caption, video_url=video_url, max_wait_sec=max_wait_sec)
        try:
            return self._publish_via_facebook_resumable(video_path=video_path, caption=caption, max_wait_sec=max_wait_sec)
        except Exception as exc:
            clean_url = str(video_url or "").strip()
            if clean_url.startswith("http"):
                logger.warning("Instagram rupload falhou (%s); tentando fallback via video_url pública.", exc)
                return self._publish_via_facebook_url(caption=caption, video_url=clean_url, max_wait_sec=max_wait_sec)
            raise

    def _publish_via_facebook_url(self, caption: str, video_url: str, max_wait_sec: int) -> str:
        init_resp = requests.post(
            f"{GRAPH_FB_BASE}/{self.account_id}/media",
            data={
                "media_type": "REELS",
                "video_url": video_url,
                "caption": caption,
                "share_to_feed": "true",
                "access_token": self.access_token,
            },
            timeout=30,
        )
        if not init_resp.ok:
            raise RuntimeError(f"Instagram FB video_url init erro {init_resp.status_code}: {init_resp.text[:350]}")
        container_id = str(init_resp.json()["id"])
        return self._wait_and_publish(GRAPH_FB_BASE, container_id, max_wait_sec=max_wait_sec)

    def _publish_via_instagram_login(self, caption: str, video_url: str, max_wait_sec: int) -> str:
        clean_url = str(video_url or "").strip()
        if not clean_url.startswith("http"):
            raise RuntimeError(
                "Token `IG...` (Instagram Login) requer URL HTTPS pública do vídeo (`download_url` do Drive)."
            )

        init_resp = requests.post(
            f"{GRAPH_IG_BASE}/{self.account_id}/media",
            data={
                "media_type": "REELS",
                "video_url": clean_url,
                "caption": caption,
                "share_to_feed": "true",
                "access_token": self.access_token,
            },
            timeout=30,
        )
        init_resp.raise_for_status()
        container_id = str(init_resp.json()["id"])
        return self._wait_and_publish(GRAPH_IG_BASE, container_id, max_wait_sec=max_wait_sec)

    def _publish_via_facebook_resumable(self, video_path: Path, caption: str, max_wait_sec: int) -> str:
        raw_bytes = video_path.read_bytes()
        file_size = len(raw_bytes)

        # Passo 1: Inicializa Container Resumable REELS
        init_resp = requests.post(
            f"{GRAPH_FB_BASE}/{self.account_id}/media",
            data={
                "media_type": "REELS",
                "upload_type": "resumable",
                "caption": caption,
                "share_to_feed": "true",
                "access_token": self.access_token,
            },
            timeout=30,
        )
        if not init_resp.ok:
            raise RuntimeError(f"Instagram container init erro {init_resp.status_code}: {init_resp.text[:350]}")
        init_data = init_resp.json()
        container_id = str(init_data["id"])
        upload_uri = str(init_data.get("uri") or f"{RUPLOAD_BASE}/{container_id}")

        # Passo 2: Envia o vídeo binário com Content-Length explícito (evita 400 chunked no rupload)
        headers = {
            "Authorization": f"OAuth {self.access_token}",
            "offset": "0",
            "file_size": str(file_size),
            "Content-Length": str(file_size),
            "Content-Type": "application/octet-stream",
        }
        up_resp = requests.post(upload_uri, headers=headers, data=raw_bytes, timeout=300)
        if not up_resp.ok:
            raise RuntimeError(f"Instagram rupload erro {up_resp.status_code}: {up_resp.text[:350]}")

        return self._wait_and_publish(GRAPH_FB_BASE, container_id, max_wait_sec=max_wait_sec)

    def _wait_and_publish(self, graph_base: str, container_id: str, max_wait_sec: int = 240) -> str:
        # Passo 3: Aguarda o processamento do container (`FINISHED`)
        deadline = time.time() + max_wait_sec
        status_code = "IN_PROGRESS"
        while time.time() < deadline:
            st_resp = requests.get(
                f"{graph_base}/{container_id}",
                params={"fields": "status_code,status", "access_token": self.access_token},
                timeout=20,
            )
            st_resp.raise_for_status()
            st_json = st_resp.json()
            status_code = str(st_json.get("status_code", ""))
            if status_code == "FINISHED":
                break
            if status_code in {"ERROR", "EXPIRED"}:
                raise RuntimeError(f"Erro no processamento do Reel na Meta: {st_json}")
            time.sleep(6)

        if status_code != "FINISHED":
            raise TimeoutError(f"Timeout aguardando processamento do Reel ({container_id}): {status_code}")

        # Passo 4: Publica o Reel
        pub_resp = requests.post(
            f"{graph_base}/{self.account_id}/media_publish",
            data={"creation_id": container_id, "access_token": self.access_token},
            timeout=30,
        )
        pub_resp.raise_for_status()
        media_id = str(pub_resp.json().get("id", ""))
        logger.info("Instagram Reel publicado com sucesso! media_id=%s", media_id)
        return media_id
