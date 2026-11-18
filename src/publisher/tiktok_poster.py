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
      - `mode == "direct"` (Video Init `/v2/post/publish/video/init/`): envia vídeo COM descrição/hashtags em `post_info.title`.
        Antes da aprovação em Production, só funciona com `privacy_level="SELF_ONLY"` e conta TikTok configurada como Privada.
      - `mode == "inbox"` (Inbox Init `/v2/post/publish/inbox/video/init/`): envia para Caixa de Entrada/Rascunhos sem auditoria,
        mas a API de inbox do TikTok não aceita `post_info` (sem descrição automática).
    """

    def __init__(
        self,
        profile: ProfileConfig,
        access_token: Optional[str] = None,
    ) -> None:
        tt_cfg = profile.publishing.tiktok
        self.enabled = tt_cfg.enabled
        self.mode = tt_cfg.mode.lower().strip()
        self.privacy_level = (tt_cfg.privacy_level or "PUBLIC_TO_EVERYONE").upper().strip()
        self.fallback_to_inbox = bool(tt_cfg.fallback_to_inbox)
        self.last_publish_mode = ""
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

        source_info = {
            "source": "FILE_UPLOAD",
            "video_size": video_size,
            "chunk_size": video_size if total_chunks == 1 else chunk_size,
            "total_chunk_count": total_chunks,
        }

        payload, used_mode = self._init_publish_session(headers, caption, source_info)
        self.last_publish_mode = used_mode
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

        logger.info("TikTok (%s / %s) enviado com sucesso! publish_id=%s", self.mode, used_mode, publish_id)
        return publish_id

    @staticmethod
    def _extract_error_code(resp: requests.Response) -> str:
        try:
            err = resp.json().get("error", {})
            return str(err.get("code") or "").strip().lower()
        except Exception:
            return ""

    def _post_init_request(self, endpoint: str, headers: Dict[str, str], body: Dict[str, Any]) -> requests.Response:
        return requests.post(endpoint, headers=headers, json=body, timeout=30)

    def _init_publish_session(
        self,
        headers: Dict[str, str],
        caption: str,
        source_info: Dict[str, Any],
    ) -> tuple[Dict[str, Any], str]:
        direct_endpoint = f"{TIKTOK_API_BASE}/post/publish/video/init/"
        inbox_endpoint = f"{TIKTOK_API_BASE}/post/publish/inbox/video/init/"

        if self.mode == "direct":
            tried_privacy = self.privacy_level or "PUBLIC_TO_EVERYONE"
            for privacy in ([tried_privacy, "SELF_ONLY"] if tried_privacy != "SELF_ONLY" else ["SELF_ONLY"]):
                init_body: Dict[str, Any] = {
                    "post_info": {
                        "title": caption[:2200],
                        "privacy_level": privacy,
                        "disable_duet": False,
                        "disable_comment": False,
                        "disable_stitch": False,
                    },
                    "source_info": source_info,
                }
                resp = self._post_init_request(direct_endpoint, headers, init_body)
                if resp.ok:
                    return resp.json().get("data", {}), f"direct:{privacy}"

                err_code = self._extract_error_code(resp)
                if err_code == "unaudited_client_can_only_post_to_private_accounts":
                    logger.warning(
                        "TikTok Direct Post com `%s` bloqueado pré-aprovação (%s).",
                        privacy,
                        err_code,
                    )
                    continue
                resp.raise_for_status()

            if self.fallback_to_inbox:
                logger.warning(
                    "Conta TikTok não está Privada ou app aguarda auditoria; fazendo fallback para TikTok Inbox."
                )
                inbox_resp = self._post_init_request(inbox_endpoint, headers, {"source_info": source_info})
                inbox_resp.raise_for_status()
                return inbox_resp.json().get("data", {}), "inbox_fallback"

            raise RuntimeError(
                "TikTok Direct Post pré-aprovação exige `privacy_level=SELF_ONLY` E conta TikTok configurada como Privada."
            )

        inbox_resp = self._post_init_request(inbox_endpoint, headers, {"source_info": source_info})
        inbox_resp.raise_for_status()
        return inbox_resp.json().get("data", {}), "inbox"
