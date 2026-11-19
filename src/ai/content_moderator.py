from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import imageio_ffmpeg
import requests

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = "qwen/qwen3.8-27b"

EXPLICIT_KEYWORDS = (
    "pelado",
    "pelada",
    "nudez",
    "nudes",
    "porno",
    "pornô",
    "sexo explicito",
    "genitalia",
    "mamilo",
    "topless",
    "hentai",
    "nsfw",
    "xvideos",
    "onlyfans vazado",
)

MILD_BODY_KEYWORDS = (
    "biquini",
    "biquíni",
    "sunga",
    "lingerie",
    "calcinha",
    "sutiã",
    "sutia",
    "strip club",
    "boate de strip",
    "sem camisa",
    "seminua",
    "seminu",
    "decote",
)


@dataclass(slots=True)
class PlatformModerationDecision:
    """Platform-specific content decency decision for a rendered video."""

    exposure_level: str  # "clean" | "mild_body_exposure" | "explicit_indecent"
    instagram_safe: bool
    youtube_shorts_safe: bool
    tiktok_safe: bool
    allowed_platforms: List[str]
    blocked_platforms: List[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_ffmpeg_exe() -> str:
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def _probe_duration(video_path: Path) -> float:
    ffmpeg = _get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg, "-i", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 30.0


def extract_sample_frames_b64(video_path: Path, max_frames: int = 4) -> List[str]:
    """Extract evenly spaced low-res JPEG frames as base64 strings for vision moderation."""
    if not video_path.exists():
        return []

    ffmpeg = _get_ffmpeg_exe()
    duration = max(1.0, _probe_duration(video_path))
    ratios = [0.15, 0.40, 0.65, 0.85][: max(1, max_frames)]
    frames_b64: List[str] = []

    with tempfile.TemporaryDirectory(prefix="mod_frames_") as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, ratio in enumerate(ratios):
            ts = max(0.0, min(duration - 0.1, duration * ratio))
            out_jpg = tmp_path / f"frame_{idx}.jpg"
            cmd = [
                ffmpeg,
                "-y",
                "-ss",
                f"{ts:.2f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=360:-2",
                "-q:v",
                "6",
                str(out_jpg),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=False)
            if out_jpg.exists() and out_jpg.stat().st_size > 0:
                b64 = base64.b64encode(out_jpg.read_bytes()).decode("ascii")
                frames_b64.append(b64)

    return frames_b64


def evaluate_text_heuristics(title: str, transcript: str) -> tuple[str, str]:
    """Check title and transcript for explicit nudity or mild body exposure cues."""
    combined = f"{title} {transcript}".lower()
    for kw in EXPLICIT_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", combined):
            return "explicit_indecent", f"Termo explícito detectado no áudio/título ('{kw}')."
    for kw in MILD_BODY_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", combined):
            return "mild_body_exposure", f"Exposição corporal moderada detectada ('{kw}') — permitido no Instagram, restrito no Shorts/TikTok."
    return "clean", "Conteúdo limpo para todas as plataformas."


def build_decision_from_level(
    exposure_level: str,
    requested_platforms: Sequence[str],
    reason: str,
    ig_override: Optional[bool] = None,
    yt_override: Optional[bool] = None,
    tt_override: Optional[bool] = None,
) -> PlatformModerationDecision:
    """Apply platform-specific decency rules to an exposure level."""
    norm = (exposure_level or "clean").strip().lower()
    if norm == "explicit_indecent":
        ig_safe = False
        yt_safe = False
        tt_safe = False
    elif norm == "mild_body_exposure":
        # Instagram allows showing more body parts (swimwear, shirtless, mild cleavage/skin) without explicit nudity.
        # YouTube Shorts and TikTok are stricter on body exposure / suggestive visuals.
        ig_safe = True if ig_override is None else bool(ig_override)
        yt_safe = False if yt_override is None else bool(yt_override)
        tt_safe = False if tt_override is None else bool(tt_override)
    else:
        norm = "clean"
        ig_safe = True if ig_override is None else bool(ig_override)
        yt_safe = True if yt_override is None else bool(yt_override)
        tt_safe = True if tt_override is None else bool(tt_override)

    safety_map = {
        "instagram": ig_safe,
        "facebook": ig_safe,
        "youtube_shorts": yt_safe,
        "youtube": yt_safe,
        "tiktok": tt_safe,
    }
    allowed = [p for p in requested_platforms if safety_map.get(p, True)]
    blocked = [p for p in requested_platforms if not safety_map.get(p, True)]

    return PlatformModerationDecision(
        exposure_level=norm,
        instagram_safe=ig_safe,
        youtube_shorts_safe=yt_safe,
        tiktok_safe=tt_safe,
        allowed_platforms=allowed,
        blocked_platforms=blocked,
        reason=reason.strip() or "Verificação concluída.",
    )


def _call_groq_vision_moderation(
    frames_b64: Sequence[str],
    title: str,
    transcript_text: str,
    api_key: str,
    timeout_sec: int = 25,
) -> Optional[dict[str, Any]]:
    prompt = (
        "Você é um moderador de conteúdo visual e de áudio para publicação automática em redes sociais.\n"
        "Analise os frames do vídeo e o contexto abaixo para classificar o nível de exposição corporal/nudez de acordo com cada plataforma:\n"
        "1. 'clean': Gameplay normal, reação, humor, conversa sem exposição corporal sensualizada e sem nudez. (Seguro em TODAS: Instagram, YouTube Shorts e TikTok).\n"
        "2. 'mild_body_exposure': Mostra mais partes do corpo (ex: biquíni, sunga, praia, piscina, sem camisa com foco corporal, decote acentuado, lingerie, strip club de GTA/jogos, dança sensual) MAS SEM nudez explícita. "
        "No Instagram ('instagram_safe': true) isso é permitido sem problemas; já no YouTube Shorts ('youtube_shorts_safe': false) e TikTok ('tiktok_safe': false) deve ser bloqueado por diretrizes mais rígidas.\n"
        "3. 'explicit_indecent': Nudez explícita (mamilos femininos/topless, genitália, sexo explícito, pornografia) ou gore extremo. Bloqueado em TODAS as plataformas.\n\n"
        f"Título: {title[:150]}\n"
        f"Transcrição: {transcript_text[:600]}\n\n"
        "Responda APENAS um JSON válido neste formato exato:\n"
        '{"exposure_level": "clean", "instagram_safe": true, "youtube_shorts_safe": true, "tiktok_safe": true, "reason": "motivo curto em portugues"}'
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64[:4]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    resp = requests.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 220,
        },
        timeout=timeout_sec,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq Vision moderation HTTP {resp.status_code}: {resp.text[:250]}")

    raw_text = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(0))


def moderate_video_for_platforms(
    video_path: Path,
    requested_platforms: Sequence[str],
    title: str = "",
    transcript_text: str = "",
    groq_api_key: Optional[str] = None,
) -> PlatformModerationDecision:
    """Inspect video frames and transcript to decide which platforms can automatically receive the video."""
    text_level, text_reason = evaluate_text_heuristics(title, transcript_text)
    if text_level == "explicit_indecent":
        return build_decision_from_level("explicit_indecent", requested_platforms, text_reason)

    api_key = (groq_api_key if groq_api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
    frames_b64 = extract_sample_frames_b64(video_path, max_frames=4)

    if api_key and frames_b64:
        try:
            vision_res = _call_groq_vision_moderation(
                frames_b64=frames_b64,
                title=title,
                transcript_text=transcript_text,
                api_key=api_key,
            )
            if vision_res:
                v_level = str(vision_res.get("exposure_level") or "clean").strip().lower()
                # If text heuristic already spotted mild_body_exposure, keep at least mild_body_exposure
                if text_level == "mild_body_exposure" and v_level == "clean":
                    v_level = "mild_body_exposure"
                reason = str(vision_res.get("reason") or text_reason)
                if v_level == "explicit_indecent":
                    return build_decision_from_level("explicit_indecent", requested_platforms, reason)
                if v_level == "mild_body_exposure":
                    return build_decision_from_level("mild_body_exposure", requested_platforms, reason)
                return build_decision_from_level(
                    v_level,
                    requested_platforms,
                    reason,
                    ig_override=vision_res.get("instagram_safe"),
                    yt_override=vision_res.get("youtube_shorts_safe"),
                    tt_override=vision_res.get("tiktok_safe"),
                )
        except Exception as exc:
            logger.warning("Moderação visual via Groq falhou, usando heurística textual (%s)", exc)

    return build_decision_from_level(text_level, requested_platforms, text_reason)
