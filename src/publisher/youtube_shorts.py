from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.config_loader import get_env_secret
from src.ffmpeg_utils import get_ffmpeg_bin
from src.models import ProfileConfig

logger = logging.getLogger(__name__)

YOUTUBE_UPLOAD_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

DEFAULT_NICHE_KEYWORDS = [
    "cortes de live",
    "cortes de games",
    "gamer brasil",
    "melhores momentos",
    "humor gamer",
    "clips twitch",
]


def compute_youtube_publish_at(
    now_utc: Optional[datetime] = None,
    min_wait_minutes: int = 25,
    avoid_afternoon_crowd: bool = True,
    tz_name: str = "America/Sao_Paulo",
) -> str:
    """
    Agenda a liberação pública (`publishAt`) no YouTube:
      - Sobe sempre em `private` e espera no mínimo 25 min (`min_wait_minutes`) para o YouTube
        processar a qualidade HD completa antes de ficar público.
      - Se cair na janela congestionada da tarde (12h-17h59 BRT), adia para a noite (19h-22h BRT)
        com pequeno escalonamento de minutos para maximizar entrega.
    """
    now = now_utc or datetime.now(timezone.utc)
    wait = max(20, int(min_wait_minutes))
    target_utc = now + timedelta(minutes=wait)

    if avoid_afternoon_crowd:
        tz = ZoneInfo(tz_name)
        local_dt = target_utc.astimezone(tz)
        if 12 <= local_dt.hour < 18:
            # Move para janela noturna recomendada (19h / 21h / 22h BRT), preservando espaçamento
            slot_hour = 19 if local_dt.hour < 15 else (21 if local_dt.hour < 17 else 22)
            local_dt = local_dt.replace(hour=slot_hour, minute=min(55, local_dt.minute), second=0, microsecond=0)
            target_utc = local_dt.astimezone(timezone.utc)

    return target_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_striking_thumbnail(video_path: Path, out_jpg: Path) -> Optional[Path]:
    """Extrai o frame visualmente mais nítido e vibrante para thumbnail do Short."""
    if not video_path.exists():
        return None
    out_jpg.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            best_score = -1.0
            best_frame = None
            for ratio in (0.15, 0.30, 0.48, 0.65, 0.82):
                idx = max(0, min(frame_count - 1, int(frame_count * ratio)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                saturation = float(hsv[:, :, 1].mean())
                brightness = float(gray.mean())
                if brightness < 25:
                    continue
                score = sharpness * 0.7 + saturation * 2.5
                if score > best_score:
                    best_score = score
                    best_frame = frame
            cap.release()
            if best_frame is not None:
                cv2.imwrite(str(out_jpg), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if out_jpg.exists() and out_jpg.stat().st_size > 0:
                    return out_jpg
        else:
            cap.release()
    except Exception:
        pass

    # Fallback via FFmpeg em 35% do vídeo
    ffmpeg = get_ffmpeg_bin()
    proc = subprocess.run(
        [ffmpeg, "-y", "-ss", "2.0", "-i", str(video_path), "-frames:v", "1", "-q:v", "3", str(out_jpg)],
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0 and out_jpg.exists() and out_jpg.stat().st_size > 0:
        return out_jpg
    return None


class YouTubeShortsPublisher:
    """
    Publicador oficial de YouTube Shorts via YouTube Data API v3 (`videos.insert` + `thumbnails.set`).
    Aplica boas práticas de SEO e processamento:
      - Primeiras palavras do título em MAIÚSCULAS + apenas `#Shorts` (sem spam de hashtags)
      - Descrição SEO com frase-resumo + 5–6 palavras-chave separadas por vírgula
      - Tags abundantes (até 15 tags relevantes)
      - Upload em `private` com `publishAt` >= 25 min no futuro (e fora do pico congestionado da tarde)
      - Upload de thumbnail chamativa extraída do vídeo
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
        self.timezone = profile.publishing.timezone or "America/Sao_Paulo"
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
    def _emphasize_opening_words(text: str, count: int = 4) -> str:
        """Coloca as primeiras 3-4 palavras fortes em MAIÚSCULAS para destacar no feed Shorts."""
        words = re.sub(r"\s+", " ", text).strip().split(" ")
        styled: List[str] = []
        alpha_seen = 0
        for w in words:
            if w.startswith("#"):
                continue
            if alpha_seen < count:
                styled.append(w.upper())
                if any(ch.isalpha() for ch in w):
                    alpha_seen += 1
            else:
                styled.append(w)
        return " ".join(styled).strip()

    @staticmethod
    def build_video_metadata(
        caption: str,
        title_hint: str = "",
        category_id: str = "20",
        privacy_status: str = "public",
        creator_name: str = "",
        publish_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Monta título chamativo (#Shorts único), descrição SEO (resumo + keywords separadas por vírgula) e até 15 tags."""
        raw_caption = str(caption or "").strip()
        lines = [ln.strip() for ln in raw_caption.splitlines() if ln.strip()]
        non_tag_lines = [ln for ln in lines if not all(tok.startswith("#") for tok in ln.split())]
        hook_line = (non_tag_lines[0] if non_tag_lines else str(title_hint or "Momento insano na live")).strip()
        hook_clean = re.sub(r"#\S+", "", hook_line).strip() or "Momento insano na live"

        # 1) Título: 3-4 primeiras palavras em MAIÚSCULAS + apenas #Shorts (evita spam de hashtags)
        emphasized = YouTubeShortsPublisher._emphasize_opening_words(hook_clean, count=4)
        if len(emphasized) > 48:
            cut_at = emphasized.rfind(" ", 0, 46)
            emphasized = emphasized[: (cut_at if cut_at > 20 else 46)].rstrip(" ,;:") + "..."
        final_title = f"{emphasized} #Shorts"

        # 2) Extração de palavras-chave (SEO) e tags abundantes
        found_tags: List[str] = []
        if creator_name.strip():
            found_tags.append(creator_name.strip())
        for match in re.findall(r"#([A-Za-zÀ-ÿ0-9_]{2,30})", raw_caption):
            if match.lower() not in {t.lower() for t in found_tags}:
                found_tags.append(match)
        for kw in DEFAULT_NICHE_KEYWORDS:
            if kw.lower() not in {t.lower() for t in found_tags}:
                found_tags.append(kw)
        for word in re.findall(r"[A-Za-zÀ-ÿ0-9]{4,}", hook_clean.lower()):
            if word not in {t.lower() for t in found_tags}:
                found_tags.append(word)
        if "shorts" not in {t.lower() for t in found_tags}:
            found_tags.append("Shorts")
        found_tags = found_tags[:15]

        # 5 a 6 palavras-chave relevantes separadas por vírgula na descrição (SEO do YouTube)
        seo_keywords = [t for t in found_tags if t.lower() != "shorts"][:6]
        keywords_line = ", ".join(seo_keywords)
        summary_sentence = hook_clean[0].upper() + hook_clean[1:] if len(hook_clean) > 1 else hook_clean
        description = f"{summary_sentence}\n\nPalavras-chave: {keywords_line}\n\n#Shorts".strip()

        status_block: Dict[str, Any] = {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        }
        # Se publish_at fornecido (ex: +25 min e fora da tarde), exige privacyStatus="private" na API v3
        if publish_at and privacy_status == "public":
            status_block["privacyStatus"] = "private"
            status_block["publishAt"] = publish_at

        return {
            "snippet": {
                "title": final_title[:100],
                "description": description[:5000],
                "tags": found_tags,
                "categoryId": str(category_id or "20"),
            },
            "status": status_block,
        }

    def publish_short(
        self,
        video_path: Path,
        caption: str,
        title_hint: str = "",
        creator_name: str = "",
        delay_publish: bool = True,
    ) -> str:
        if not self.is_ready:
            raise RuntimeError("YouTube Shorts não configurado (YOUTUBE_TOKEN_JSON ausente).")

        from googleapiclient.http import MediaFileUpload

        service = self._build_youtube_service()
        publish_at = (
            compute_youtube_publish_at(tz_name=self.timezone)
            if (delay_publish and self.privacy_status == "public")
            else None
        )
        body = self.build_video_metadata(
            caption=caption,
            title_hint=title_hint,
            category_id=self.category_id,
            privacy_status=self.privacy_status,
            creator_name=creator_name,
            publish_at=publish_at,
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
        if video_id:
            # Tenta enviar thumbnail chamativa extraída do vídeo (silencioso se canal ainda não verificado)
            try:
                thumb_path = video_path.with_suffix(".thumb.jpg")
                extracted = extract_striking_thumbnail(video_path, thumb_path)
                if extracted is not None:
                    service.thumbnails().set(
                        videoId=video_id,
                        media_body=MediaFileUpload(str(extracted), mimetype="image/jpeg"),
                    ).execute()
                    extracted.unlink(missing_ok=True)
            except Exception as exc:
                logger.info("Thumbnail customizada não aplicada ao Short %s (%s).", video_id, exc)

        short_url = f"https://youtube.com/shorts/{video_id}" if video_id else ""
        logger.info("YouTube Short enviado com sucesso (publishAt=%s)! url=%s", publish_at or "imediato", short_url)
        return short_url or video_id
