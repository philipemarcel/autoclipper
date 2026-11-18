from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.config_loader import get_env_secret
from src.models import ProfileConfig

logger = logging.getLogger(__name__)

TIKTOK_API_BASE = "https://open.tiktokapis.com/v2"
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB por chunk


class TikTokPublisher:
    """
    Publicador oficial TikTok Content Posting API v2:
      - `mode == "inbox"` (Padrão imediato sem auditoria comercial): envia o vídeo direto para a
        Caixa de Entrada/Rascunhos do seu app TikTok pronto para 1 toque final.
      - `mode == "direct"` (Pós-auditoria do app no TikTok Developers): publica 100% direto no feed público.
    """

    def __init__(
        self,
        profile: ProfileConfig,
        access_token: Optional[str] = None,
    ) -> None:
        tt_cfg = profile.publishing.tiktok
        self.enabled = tt_cfg.enabled
        self.mode = tt_cfg.mode.lower().strip()
        self.access_token = (access_token or get_env_secret(tt_cfg.access_token_env, required=False)).strip()
        self.refresh_token = os.getenv("TIKTOK_REFRESH_TOKEN", "").strip()
        self.client_key = os.getenv("TIKTOK_CLIENT_KEY", "").strip()
        self.client_secret = os.getenv("TIKTOK_CLIENT_SECRET", "").strip()

    @property
    def is_ready(self) -> bool:
        return bool(
            self.enabled
            and (self.access_token or (self.refresh_token and self.client_key and self.client_secret))
        )

    def _ensure_valid_access_token(self) -> str:
        if self.refresh_token and self.client_key and self.client_secret:
            try:
                resp = requests.post(
                    f"{TIKTOK_API_BASE}/oauth/token/",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={
                        "client_key": self.client_key,
                        "client_secret": self.client_secret,
                        "grant_type": "refresh_token",
                        "refresh_token": self.refresh_token,
                    },
                    timeout=20,
                )
                if resp.ok and "access_token" in resp.json():
                    self.access_token = str(resp.json()["access_token"])
            except Exception as exc:
                logger.warning("Aviso ao renovar TIKTOK_REFRESH_TOKEN (%s); usando access_token atual.", exc)
        return self.access_token

    def publish_video(self, video_path: Path, caption: str) -> str:
        if not self.is_ready:
            raise RuntimeError("TikTok não configurado (TIKTOK_ACCESS_TOKEN ausente).")

        video_size = video_path.stat().st_size
        if video_size <= CHUNK_SIZE:
            chunk_size = video_size
            total_chunks = 1
        else:
            chunk_size = CHUNK_SIZE
            total_chunks = math.floor(video_size / chunk_size)
            # No TikTok v2, o último chunk absorve os bytes residuais se `floor` for usado; ou `ceil` se exato
            if total_chunks == 0:
                total_chunks = 1

        token = self._ensure_valid_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

        if self.mode == "direct":
            endpoint = f"{TIKTOK_API_BASE}/post/publish/video/init/"
            init_body: Dict[str, Any] = {
                "post_info": {
                    "title": caption[:2200],
                    "privacy_level": "PUBLIC_TO_EVERYONE",
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": video_size,
                    "chunk_size": video_size if total_chunks == 1 else chunk_size,
                    "total_chunk_count": total_chunks,
                },
            }
        else:
            endpoint = f"{TIKTOK_API_BASE}/post/publish/inbox/video/init/"
            init_body = {
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": video_size,
                    "chunk_size": video_size if total_chunks == 1 else chunk_size,
                    "total_chunk_count": total_chunks,
                }
            }

        resp = requests.post(endpoint, headers=headers, json=init_body, timeout=30)
        resp.raise_for_status()
        payload = resp.json().get("data", {})
        publish_id = str(payload["publish_id"])
        upload_url = str(payload["upload_url"])

        # Envio binário em chunks via PUT
        with open(video_path, "rb") as fh:
            for idx in range(total_chunks):
                start_byte = idx * chunk_size
                if idx == total_chunks - 1:
                    data_bytes = fh.read()
                else:
                    data_bytes = fh.read(chunk_size)
                end_byte = start_byte + len(data_bytes) - 1

                put_headers = {
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(data_bytes)),
                    "Content-Range": f"bytes {start_byte}-{end_byte}/{video_size}",
                }
                up_resp = requests.put(upload_url, headers=put_headers, data=data_bytes, timeout=240)
                up_resp.raise_for_status()

        logger.info("TikTok (%s) enviado com sucesso! publish_id=%s", self.mode, publish_id)
        return publish_id
