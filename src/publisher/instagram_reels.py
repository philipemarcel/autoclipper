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
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
RUPLOAD_BASE = f"https://rupload.facebook.com/ig-api-upload/{GRAPH_API_VERSION}"


class InstagramReelsPublisher:
    """
    Publicador oficial Instagram Reels via Meta Graph API (Fluxo Resumable Binary Upload):
      - Não exige expor arquivos publicamente no Google Drive (mantém seu Drive 100% privado).
      - Sobe o binário `master_1080p60.mp4` direto em `rupload.facebook.com`, aguarda processamento e publica.
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
    def is_ready(self) -> bool:
        return bool(self.enabled and self.account_id and self.access_token)

    def publish_reel(self, video_path: Path, caption: str, max_wait_sec: int = 240) -> str:
        if not self.is_ready:
            raise RuntimeError("Instagram Reels não configurado (IG_ACCOUNT_ID / IG_ACCESS_TOKEN ausentes).")

        file_size = video_path.stat().st_size

        # Passo 1: Inicializa Container Resumable REELS
        init_resp = requests.post(
            f"{GRAPH_BASE}/{self.account_id}/media",
            data={
                "media_type": "REELS",
                "upload_type": "resumable",
                "caption": caption,
                "share_to_feed": "true",
                "access_token": self.access_token,
            },
            timeout=30,
        )
        init_resp.raise_for_status()
        init_data = init_resp.json()
        container_id = str(init_data["id"])
        upload_uri = str(init_data.get("uri") or f"{RUPLOAD_BASE}/{container_id}")

        # Passo 2: Envia o vídeo binário direto para `rupload.facebook.com`
        headers = {
            "Authorization": f"OAuth {self.access_token}",
            "offset": "0",
            "file_size": str(file_size),
            "Content-Type": "application/octet-stream",
        }
        with open(video_path, "rb") as fh:
            up_resp = requests.post(upload_uri, headers=headers, data=fh, timeout=300)
        up_resp.raise_for_status()

        # Passo 3: Aguarda o processamento do container (`FINISHED`)
        deadline = time.time() + max_wait_sec
        status_code = "IN_PROGRESS"
        while time.time() < deadline:
            st_resp = requests.get(
                f"{GRAPH_BASE}/{container_id}",
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
            f"{GRAPH_BASE}/{self.account_id}/media_publish",
            data={"creation_id": container_id, "access_token": self.access_token},
            timeout=30,
        )
        pub_resp.raise_for_status()
        media_id = str(pub_resp.json().get("id", ""))
        logger.info("Instagram Reel publicado com sucesso! media_id=%s", media_id)
        return media_id
