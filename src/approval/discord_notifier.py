from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.config_loader import get_env_secret
from src.models import CutRecord, ProfileConfig

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"


class DiscordNotifier:
    """
    Envia a prévia compacta (<9 MB) direto no player do Discord junto com:
      - Link do arquivo mestre `1080p60` nos 20 TB do Google Drive
      - Copy minimalista pronta
      - Botões interativos de 1 clique (Aprovar Fila, Postar Agora, Só Reels, Só TikTok, Limpar Música IA, Descartar)
    """

    def __init__(
        self,
        profile: ProfileConfig,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> None:
        self.profile = profile
        self.bot_token = (bot_token or get_env_secret(profile.discord.bot_token_env, required=False)).strip()
        self.channel_id = (channel_id or get_env_secret(profile.discord.channel_id_env, required=False)).strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.channel_id)

    def send_cut_for_approval(self, cut: CutRecord, preview_video_path: Path) -> str:
        if not self.is_configured:
            logger.warning("Discord não configurado (DISCORD_BOT_TOKEN/DISCORD_CHANNEL_ID ausentes). Pulando envio.")
            return ""

        url = f"{DISCORD_API_BASE}/channels/{self.channel_id}/messages"
        headers = {"Authorization": f"Bot {self.bot_token}"}

        if cut.mode == "youtube_live":
            mode_badge = "🔴 YOUTUBE LIVE"
        elif cut.mode.startswith("youtube"):
            mode_badge = "▶️ YOUTUBE"
        elif cut.mode == "kick_live":
            mode_badge = "🟢 KICK AO VIVO"
        elif cut.mode == "kick_clip":
            mode_badge = "🟢 KICK CLIPE"
        elif cut.mode.startswith("kick"):
            mode_badge = "🟢 KICK VOD"
        elif cut.mode == "live_hot":
            mode_badge = "🔥 AO VIVO"
        else:
            mode_badge = "🏁 REPESCAGEM PÓS-LIVE"
        audio_badge = "🔇 Voz Isolada (Sem Música)" if cut.audio_cleaned else "🎵 Áudio Original"
        drip_min = self.profile.publishing.drip_interval_minutes

        embed = {
            "title": f"{mode_badge} • @{cut.twitch_login} (Score {cut.score:.0f})",
            "color": 0x9146FF if not cut.audio_cleaned else 0x00D26A,
            "fields": [
                {
                    "name": "📝 Copy pronta (Reels / TikTok)",
                    "value": f"```\n{cut.caption}\n```",
                    "inline": False,
                },
                {
                    "name": "⏱️ Duração",
                    "value": f"`{cut.total_duration:.1f}s` (<=58s)",
                    "inline": True,
                },
                {
                    "name": "📐 Auto-Layout",
                    "value": f"`{cut.layout_used}`",
                    "inline": True,
                },
                {
                    "name": "🔊 Trilha",
                    "value": audio_badge,
                    "inline": True,
                },
                {
                    "name": f"📥 Download Direto & Drive (Auto-exclusão em {self.profile.publishing.drive_retention_days}d)",
                    "value": (
                        f"[⬇️ Baixar Vídeo Direto (Sem acesso ao Drive)]({cut.download_url or cut.master_file_url})\n"
                        f"[👁️ Abrir no Navegador]({cut.master_file_url}) • [📂 Pasta do Proprietário]({cut.drive_folder_url})"
                    ),
                    "inline": False,
                },
            ],
            "footer": {"text": f"ID: {cut.cut_id} • Fonte: {cut.source_type}"},
        }

        components = self._build_action_rows(
            cut.cut_id,
            drip_min,
            audio_cleaned=cut.audio_cleaned,
            ig_enabled=self.profile.publishing.instagram.enabled,
            tt_enabled=self.profile.publishing.tiktok.enabled,
        )
        payload_json = {
            "content": f"**Novo corte pronto de `{self.profile.display_name}`!** Assista à prévia abaixo e aprove com 1 clique:",
            "embeds": [embed],
            "components": components,
        }

        with open(preview_video_path, "rb") as vf:
            files = {
                "files[0]": (f"{cut.cut_id}_preview.mp4", vf, "video/mp4"),
            }
            data = {"payload_json": json.dumps(payload_json, ensure_ascii=False)}
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=90)

        resp.raise_for_status()
        msg_id = str(resp.json().get("id", ""))
        logger.info("Prévia enviada ao Discord com sucesso (message_id=%s).", msg_id)
        return msg_id

    def post_status_reply(self, reply_to_message_id: str, text: str) -> None:
        if not self.is_configured:
            return
        url = f"{DISCORD_API_BASE}/channels/{self.channel_id}/messages"
        headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {"content": text}
        if reply_to_message_id:
            body["message_reference"] = {"message_id": reply_to_message_id, "fail_if_not_exists": False}
        resp = requests.post(url, headers=headers, json=body, timeout=25)
        if not resp.ok:
            logger.warning("Falha ao enviar status no Discord: %s", resp.text)

    @staticmethod
    def _build_action_rows(
        cut_id: str,
        drip_min: int,
        audio_cleaned: bool = False,
        ig_enabled: bool = True,
        tt_enabled: bool = True,
    ) -> List[Dict[str, Any]]:
        if ig_enabled and tt_enabled:
            row1_buttons = [
                {
                    "type": 2,
                    "style": 3,
                    "label": f"Aprovar (Fila {drip_min}m)",
                    "emoji": {"name": "✅"},
                    "custom_id": f"approve_both:{cut_id}",
                },
                {
                    "type": 2,
                    "style": 1,
                    "label": "Postar Agora",
                    "emoji": {"name": "⚡"},
                    "custom_id": f"now_both:{cut_id}",
                },
                {
                    "type": 2,
                    "style": 2,
                    "label": "Só Reels",
                    "emoji": {"name": "📸"},
                    "custom_id": f"approve_reels:{cut_id}",
                },
                {
                    "type": 2,
                    "style": 2,
                    "label": "Só TikTok",
                    "emoji": {"name": "🎵"},
                    "custom_id": f"approve_tiktok:{cut_id}",
                },
            ]
        elif tt_enabled:
            row1_buttons = [
                {
                    "type": 2,
                    "style": 3,
                    "label": f"Aprovar TikTok (Fila {drip_min}m)",
                    "emoji": {"name": "🎵"},
                    "custom_id": f"approve_tiktok:{cut_id}",
                },
                {
                    "type": 2,
                    "style": 1,
                    "label": "Enviar TikTok Agora",
                    "emoji": {"name": "⚡"},
                    "custom_id": f"now_tiktok:{cut_id}",
                },
            ]
        else:
            row1_buttons = [
                {
                    "type": 2,
                    "style": 3,
                    "label": f"Aprovar Reels (Fila {drip_min}m)",
                    "emoji": {"name": "📸"},
                    "custom_id": f"approve_reels:{cut_id}",
                }
            ]

        row2_buttons: List[Dict[str, Any]] = [
            {
                "type": 2,
                "style": 2,
                "label": "Copiar Descrição",
                "emoji": {"name": "📋"},
                "custom_id": f"copy_caption:{cut_id}",
            }
        ]
        if not audio_cleaned:
            row2_buttons.append(
                {
                    "type": 2,
                    "style": 2,
                    "label": "Limpar Música IA",
                    "emoji": {"name": "🔇"},
                    "custom_id": f"clean_audio:{cut_id}",
                }
            )
        row2_buttons.append(
            {
                "type": 2,
                "style": 4,  # Danger (Vermelho)
                "label": "Descartar",
                "emoji": {"name": "❌"},
                "custom_id": f"reject:{cut_id}",
            }
        )

        return [
            {"type": 1, "components": row1_buttons},
            {"type": 1, "components": row2_buttons},
        ]
