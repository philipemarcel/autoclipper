"""Instagram Reels publisher tuned for Reels ranking signals (watch time, sends per reach, SEO, cover frame)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

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

DEFAULT_REELS_NICHE_TAGS: List[str] = [
    "#cortes",
    "#reelsbrasil",
    "#gamerbrasil",
    "#cortesdegames",
    "#clips",
]

# Tags genéricas de baixo sinal que diluem a categorização de nicho no Reels
_GENERIC_LOW_SIGNAL_TAGS = {
    "#fyp",
    "#fy",
    "#foryou",
    "#foryoupage",
    "#viral",
    "#explore",
    "#trending",
    "#instagood",
    "#reelsvideo",
    "#shorts",
}

_SENDS_SAVES_MICRO_CTAS = (
    "manda pra quem precisava ver isso",
    "envia pro amigo que faria igual",
    "salva pra rever essa reação",
    "manda no grupo que ia rir disso",
)


def _slugify_tag(text: str) -> str:
    raw = str(text or "").strip().lower()
    if raw.startswith("#"):
        raw = raw[1:]
    norm = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9]", "", norm)
    if len(cleaned) < 2:
        return ""
    return f"#{cleaned[:30]}"


def compute_best_thumb_offset_ms(video_path: Path) -> Optional[int]:
    """
    Seleciona o frame mais nítido e vibrante para a capa do Reel (`thumb_offset` em ms),
    evitando frame 0 escuro/borrado e respeitando estritamente `0 < offset < duration_ms`.
    """
    if not video_path.exists() or video_path.stat().st_size < 1024:
        return None
    try:
        import cv2
    except Exception:
        return None

    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 2 or fps <= 0.0:
            return None
        total_ms = int((frame_count / fps) * 1000)
        if total_ms < 1500:
            return None

        best_score = -1.0
        best_idx = int(frame_count * 0.25)
        for ratio in (0.08, 0.16, 0.25, 0.38, 0.52, 0.68, 0.82):
            idx = max(1, min(frame_count - 2, int(frame_count * ratio)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            if brightness < 25.0:
                continue
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = float(hsv[:, :, 1].mean())
            score = sharpness * 0.7 + saturation * 2.5
            if score > best_score:
                best_score = score
                best_idx = idx

        offset_ms = int((best_idx / fps) * 1000)
        return max(200, min(total_ms - 500, offset_ms))
    except Exception:
        return None
    finally:
        cap.release()


class InstagramReelsPublisher:
    """
    Publicador oficial Instagram Reels via Meta API alinhado às melhores práticas do algoritmo:
      - Gancho forte logo na 1ª linha da legenda (< 100 chars antes do "... mais") + contexto SEO
      - Micro-CTA variado focado em **Sends Per Reach** (compartilhamentos no Direct) e **Saves**
      - Exatamente **3 a 5 hashtags de nicho** (remove `#fyp`/`#viral`/`#explore` genéricas)
      - Seleção automática do melhor frame para capa (`thumb_offset`), nome de áudio original (`audio_name`),
        `share_to_feed` e suporte opcional a **Trial Reels** (`trial_params`).
      - Dual auth: Instagram Login (`IG...` -> `graph.instagram.com`) ou Facebook Login (`EAA...` -> `rupload`).
    """

    def __init__(
        self,
        profile: ProfileConfig,
        account_id: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        ig_cfg = profile.publishing.instagram
        self.enabled = ig_cfg.enabled
        self.share_to_feed = bool(getattr(ig_cfg, "share_to_feed", True))
        self.trial_mode = str(getattr(ig_cfg, "trial_mode", "") or "").strip().upper()
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

    @staticmethod
    def build_audio_name(creator_name: str = "") -> str:
        clean = re.sub(r"\s+", " ", str(creator_name or "").strip())
        if not clean:
            return ""
        return f"Som original - {clean[:30]}"

    @staticmethod
    def build_reels_metadata(
        caption: str,
        title_hint: str = "",
        creator_name: str = "",
        game_name: str = "",
        max_hashtags: int = 5,
    ) -> Dict[str, Any]:
        """
        Formata legenda de Reels otimizada para Busca (SEO), retenção acima da dobra e Sends/Saves:
          - Linha 1: gancho humano conciso (<= 100 chars visíveis antes de "... mais")
          - Linha 2: micro-CTA de envio/salvamento + palavra-chave de criador/jogo se ausente no gancho
          - Linha 3: 3 a 5 hashtags específicas de nicho
        """
        raw_caption = str(caption or "").strip()
        lines = [ln.strip() for ln in raw_caption.splitlines() if ln.strip()]
        non_tag_lines = [ln for ln in lines if not all(tok.startswith("#") for tok in ln.split())]
        raw_hook = (non_tag_lines[0] if non_tag_lines else str(title_hint or "momento absurdo na live")).strip()
        hook_clean = re.sub(r"#\S+", "", raw_hook)
        hook_clean = re.sub(r"\s+", " ", hook_clean).strip()
        hook_clean = re.sub(
            r"\s*(?:o que (?:vocês|vcs) acharam\??|comenta aí!?)\s*$",
            "",
            hook_clean,
            flags=re.IGNORECASE,
        ).strip()
        if not hook_clean:
            hook_clean = (re.sub(r"#\S+", "", str(title_hint or "")).strip() or "momento absurdo na live").lower()

        if len(hook_clean) > 98:
            cut_at = hook_clean.rfind(" ", 0, 95)
            hook_clean = hook_clean[: (cut_at if cut_at > 40 else 95)].rstrip(" ,;:") + "..."

        # Extrai contexto SEO ausente para a 2ª linha sem sujar o gancho inicial
        clean_creator = re.sub(r"\s+", " ", str(creator_name or "").strip())
        clean_game = re.sub(r"\s+", " ", str(game_name or "").strip())
        if clean_game.lower() in {"live", "games", "youtube", "kick", "twitch", "just chatting", "irl"}:
            clean_game = ""

        hook_low = hook_clean.lower()
        seo_bits: List[str] = []
        if clean_creator and clean_creator.lower() not in hook_low:
            seo_bits.append(clean_creator)
        if clean_game and clean_game.lower() not in hook_low:
            seo_bits.append(clean_game)

        # Micro-CTA focado em DM Shares (Sends per Reach) & Saves — só quando gancho é frase real
        already_has_cta = any(k in raw_caption.lower() for k in ("manda ", "envia ", "salva ", "compartilha "))
        cta_line = ""
        if len(hook_clean) >= 12 and not already_has_cta:
            digest = hashlib.sha1(hook_clean.encode("utf-8")).digest()
            cta_line = _SENDS_SAVES_MICRO_CTAS[digest[0] % len(_SENDS_SAVES_MICRO_CTAS)]

        second_parts: List[str] = []
        if cta_line:
            second_parts.append(cta_line)
        if seo_bits:
            second_parts.append(" · ".join(seo_bits))
        engagement_line = " — ".join(second_parts) if (cta_line and seo_bits) else "".join(second_parts)

        # Hashtags: 3 a 5 específicas de nicho, filtrando tags genéricas de spam
        target_max = max(3, min(5, int(max_hashtags)))
        tags: List[str] = []
        seen_tags: set[str] = set()

        def _add_tag(candidate: str) -> None:
            tag = _slugify_tag(candidate)
            if not tag or tag in _GENERIC_LOW_SIGNAL_TAGS or tag in seen_tags:
                return
            seen_tags.add(tag)
            tags.append(tag)

        if clean_creator:
            _add_tag(clean_creator)
        if clean_game:
            _add_tag(clean_game)
        for match in re.findall(r"#([A-Za-zÀ-ÿ0-9_]{2,30})", raw_caption):
            _add_tag(match)
        for fallback_tag in DEFAULT_REELS_NICHE_TAGS:
            if len(tags) >= target_max:
                break
            _add_tag(fallback_tag)
        tags = tags[:target_max]

        blocks = [hook_clean]
        if engagement_line:
            blocks.append(engagement_line)
        if tags:
            blocks.append(" ".join(tags))
        full_caption = "\n\n".join(blocks)[:2190]

        return {
            "caption": full_caption,
            "hook_line": hook_clean,
            "engagement_line": engagement_line,
            "hashtags": tags,
        }

    @classmethod
    def build_reels_caption(
        cls,
        caption: str,
        title_hint: str = "",
        creator_name: str = "",
        game_name: str = "",
    ) -> str:
        return str(
            cls.build_reels_metadata(
                caption=caption,
                title_hint=title_hint,
                creator_name=creator_name,
                game_name=game_name,
            )["caption"]
        )

    def publish_reel(
        self,
        video_path: Path,
        caption: str,
        max_wait_sec: int = 240,
        video_url: str = "",
        title_hint: str = "",
        creator_name: str = "",
        game_name: str = "",
        thumb_offset_ms: Optional[int] = None,
        audio_name: str = "",
        share_to_feed: Optional[bool] = None,
        trial_mode: Optional[str] = None,
    ) -> str:
        if not self.is_ready:
            raise RuntimeError("Instagram Reels não configurado (IG_ACCOUNT_ID / IG_ACCESS_TOKEN ausentes).")

        formatted_caption = self.build_reels_caption(
            caption=caption,
            title_hint=title_hint,
            creator_name=creator_name,
            game_name=game_name,
        )
        resolved_thumb = thumb_offset_ms if thumb_offset_ms is not None else compute_best_thumb_offset_ms(video_path)
        resolved_audio = audio_name or self.build_audio_name(creator_name)
        resolved_share = self.share_to_feed if share_to_feed is None else bool(share_to_feed)
        resolved_trial = (trial_mode if trial_mode is not None else self.trial_mode).strip().upper()

        optional_data: Dict[str, Any] = {}
        if resolved_thumb is not None and resolved_thumb >= 0:
            optional_data["thumb_offset"] = str(int(resolved_thumb))
        if resolved_audio:
            optional_data["audio_name"] = resolved_audio[:50]
        if resolved_trial in {"MANUAL", "SS_PERFORMANCE"}:
            optional_data["trial_params"] = json.dumps({"graduation_strategy": resolved_trial})

        if self.is_instagram_login_token:
            return self._publish_via_url(
                graph_base=GRAPH_IG_BASE,
                caption=formatted_caption,
                video_url=video_url,
                share_to_feed=resolved_share,
                optional_data=optional_data,
                max_wait_sec=max_wait_sec,
                err_prefix="Instagram Login",
            )
        try:
            return self._publish_via_facebook_resumable(
                video_path=video_path,
                caption=formatted_caption,
                share_to_feed=resolved_share,
                optional_data=optional_data,
                max_wait_sec=max_wait_sec,
            )
        except Exception as exc:
            clean_url = str(video_url or "").strip()
            if clean_url.startswith("http"):
                logger.warning("Instagram rupload falhou (%s); tentando fallback via video_url pública.", exc)
                return self._publish_via_url(
                    graph_base=GRAPH_FB_BASE,
                    caption=formatted_caption,
                    video_url=clean_url,
                    share_to_feed=resolved_share,
                    optional_data=optional_data,
                    max_wait_sec=max_wait_sec,
                    err_prefix="Instagram FB video_url",
                )
            raise

    def _create_reel_container(
        self,
        graph_base: str,
        base_data: Dict[str, Any],
        optional_data: Dict[str, Any],
        err_prefix: str,
    ) -> Dict[str, Any]:
        payload = {**base_data, **optional_data}
        init_resp = requests.post(
            f"{graph_base}/{self.account_id}/media",
            data=payload,
            timeout=30,
        )
        if not init_resp.ok and optional_data and init_resp.status_code == 400:
            # Tenta primeiro remover apenas `trial_params` / `audio_name` mantendo `thumb_offset`
            reduced = {k: v for k, v in optional_data.items() if k not in {"trial_params", "audio_name"}}
            if reduced and reduced != optional_data:
                init_resp = requests.post(
                    f"{graph_base}/{self.account_id}/media",
                    data={**base_data, **reduced},
                    timeout=30,
                )
            if not init_resp.ok and init_resp.status_code == 400:
                logger.warning(
                    "%s container init retornou 400 (%s); reenviando sem parâmetros opcionais (%s).",
                    err_prefix,
                    init_resp.text[:200],
                    list(optional_data.keys()),
                )
                init_resp = requests.post(
                    f"{graph_base}/{self.account_id}/media",
                    data=base_data,
                    timeout=30,
                )
        if not init_resp.ok:
            raise RuntimeError(f"{err_prefix} container init erro {init_resp.status_code}: {init_resp.text[:350]}")
        return dict(init_resp.json())

    def _publish_via_url(
        self,
        graph_base: str,
        caption: str,
        video_url: str,
        share_to_feed: bool,
        optional_data: Dict[str, Any],
        max_wait_sec: int,
        err_prefix: str,
    ) -> str:
        clean_url = str(video_url or "").strip()
        if not clean_url.startswith("http"):
            raise RuntimeError(
                "Token `IG...` (Instagram Login) requer URL HTTPS pública do vídeo (`download_url` do Drive)."
            )

        base_data = {
            "media_type": "REELS",
            "video_url": clean_url,
            "caption": caption,
            "share_to_feed": "true" if share_to_feed else "false",
            "access_token": self.access_token,
        }
        init_data = self._create_reel_container(
            graph_base=graph_base,
            base_data=base_data,
            optional_data=optional_data,
            err_prefix=err_prefix,
        )
        container_id = str(init_data["id"])
        return self._wait_and_publish(graph_base, container_id, max_wait_sec=max_wait_sec)

    def _publish_via_facebook_resumable(
        self,
        video_path: Path,
        caption: str,
        share_to_feed: bool,
        optional_data: Dict[str, Any],
        max_wait_sec: int,
    ) -> str:
        raw_bytes = video_path.read_bytes()
        file_size = len(raw_bytes)

        base_data = {
            "media_type": "REELS",
            "upload_type": "resumable",
            "caption": caption,
            "share_to_feed": "true" if share_to_feed else "false",
            "access_token": self.access_token,
        }
        init_data = self._create_reel_container(
            graph_base=GRAPH_FB_BASE,
            base_data=base_data,
            optional_data=optional_data,
            err_prefix="Instagram FB resumable",
        )
        container_id = str(init_data["id"])
        upload_uri = str(init_data.get("uri") or f"{RUPLOAD_BASE}/{container_id}")

        # Envia o vídeo binário com Content-Length explícito (evita 400 chunked no rupload)
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
        # Aguarda o processamento do container (`FINISHED`)
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

        # Publica o Reel
        pub_resp = requests.post(
            f"{graph_base}/{self.account_id}/media_publish",
            data={"creation_id": container_id, "access_token": self.access_token},
            timeout=30,
        )
        pub_resp.raise_for_status()
        media_id = str(pub_resp.json().get("id", ""))
        logger.info("Instagram Reel publicado com sucesso! media_id=%s", media_id)
        return media_id
