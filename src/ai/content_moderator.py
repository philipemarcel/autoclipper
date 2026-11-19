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
GROQ_VISION_MAX_IMAGES = 3  # Limite estrito do modelo qwen3.8-27b na Groq (HTTP 400 se > 3)

EXPLICIT_KEYWORDS = (
    "pelado",
    "pelada",
    "nudez",
    "nudismo",
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
    "onlyfans",
    "strip club",
    "boate de strip",
    "lingerie",
    "calcinha",
    "sutia",
    "sutiã",
    "seminua",
    "seminu",
    "naked",
    "nudity",
)

MILD_BODY_KEYWORDS = (
    "biquini",
    "biquíni",
    "sunga",
    "sem camisa",
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
    vision_checked: bool = False

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


def _stitch_frames_grid_b64(jpg_paths: Sequence[Path]) -> Optional[str]:
    """Assemble up to 6 frames into one 2x3 contact-sheet JPEG so 1 Groq Vision call stays well under 7k ITPM."""
    if len(jpg_paths) < 2:
        return None
    try:
        import cv2
        import numpy as np

        imgs = []
        for p in jpg_paths[:6]:
            im = cv2.imread(str(p))
            if im is not None:
                im = cv2.resize(im, (270, 480), interpolation=cv2.INTER_AREA)
                imgs.append(im)
        if len(imgs) < 2:
            return None
        while len(imgs) % 3 != 0:
            imgs.append(np.zeros_like(imgs[0]))
        rows = [np.hstack(imgs[i : i + 3]) for i in range(0, len(imgs), 3)]
        grid = np.vstack(rows)
        ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def extract_sample_frames_b64(video_path: Path, max_frames: int = 6) -> List[str]:
    """Extract evenly spaced low-res JPEG frames (stitched into 1 contact sheet when possible) for vision moderation."""
    if not video_path.exists():
        return []

    ffmpeg = _get_ffmpeg_exe()
    duration = max(1.0, _probe_duration(video_path))
    ratios = [0.10, 0.25, 0.42, 0.58, 0.74, 0.90][: max(1, max_frames)]
    frames_b64: List[str] = []

    with tempfile.TemporaryDirectory(prefix="mod_frames_") as tmpdir:
        tmp_path = Path(tmpdir)
        jpg_paths: List[Path] = []
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
                "scale=270:-2",
                "-q:v",
                "6",
                str(out_jpg),
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=False)
            if out_jpg.exists() and out_jpg.stat().st_size > 0:
                jpg_paths.append(out_jpg)
                frames_b64.append(base64.b64encode(out_jpg.read_bytes()).decode("ascii"))

        stitched = _stitch_frames_grid_b64(jpg_paths)
        if stitched:
            return [stitched]

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
    vision_checked: bool = False,
) -> PlatformModerationDecision:
    """Apply platform-specific decency rules to an exposure level."""
    norm = (exposure_level or "clean").strip().lower()
    if norm in {"explicit_indecent", "mild_body_exposure"}:
        # Política rigorosa: qualquer nudez, seminudez, lingerie, biquíni sensual ou strip club é bloqueado em todas as redes.
        ig_safe = False
        yt_safe = False
        tt_safe = False
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
        vision_checked=vision_checked,
    )


def _call_groq_vision_moderation(
    frames_b64: Sequence[str],
    title: str,
    transcript_text: str,
    api_key: str,
    timeout_sec: int = 25,
) -> Optional[dict[str, Any]]:
    prompt = (
        "Você é um moderador rigoroso de conteúdo visual e de áudio para publicação em redes sociais.\n"
        "Analise TODAS as imagens (inclusive personagens de jogos como GTA/RP, IRL, transmissões ou memes) e classifique o nível de nudez/exposição corporal:\n"
        "1. 'clean': Gameplay normal, reação, rosto em webcam, conversa ou humor SEM nudez, SEM seminudez, SEM foco sensual/erótico.\n"
        "2. 'mild_body_exposure': Qualquer seminudez, roupa íntima (calcinha/sutiã/cueca), biquíni/sunga sensualizado, decote explícito, pessoa sem camisa focando o corpo ou dança sensual. Deve bloquear em TODAS as plataformas.\n"
        "3. 'explicit_indecent': Nudez explícita (seios/mamilos expostos, nádegas/genitália à mostra, topless, strip club de jogos/IRL, sexo simulado ou explícito, pornografia, hentai) ou gore extremo. Deve bloquear em TODAS as plataformas.\n\n"
        f"Título: {title[:150]}\n"
        f"Transcrição: {transcript_text[:600]}\n\n"
        "Responda APENAS um JSON válido neste formato exato:\n"
        '{"exposure_level": "clean", "instagram_safe": true, "youtube_shorts_safe": true, "tiktok_safe": true, "reason": "motivo curto em portugues"}'
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64[:GROQ_VISION_MAX_IMAGES]:
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
    """Inspect video frames (in <=3-image batches) and transcript to block nudity/seminudity."""
    text_level, text_reason = evaluate_text_heuristics(title, transcript_text)
    if text_level == "explicit_indecent":
        return build_decision_from_level("explicit_indecent", requested_platforms, text_reason, vision_checked=False)

    api_key = (groq_api_key if groq_api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
    frames_b64 = extract_sample_frames_b64(video_path, max_frames=6)

    severity_rank = {"clean": 0, "mild_body_exposure": 1, "explicit_indecent": 2}
    worst_level = text_level
    worst_reason = text_reason
    worst_vision: Optional[dict[str, Any]] = None
    vision_ok = False

    if api_key and frames_b64:
        batches = [frames_b64[i : i + GROQ_VISION_MAX_IMAGES] for i in range(0, len(frames_b64), GROQ_VISION_MAX_IMAGES)]
        for batch in batches:
            try:
                vision_res = _call_groq_vision_moderation(
                    frames_b64=batch,
                    title=title,
                    transcript_text=transcript_text,
                    api_key=api_key,
                )
                if not vision_res:
                    continue
                vision_ok = True
                v_level = str(vision_res.get("exposure_level") or "clean").strip().lower()
                if v_level not in severity_rank:
                    v_level = "explicit_indecent"
                if severity_rank[v_level] >= severity_rank.get(worst_level, 0):
                    worst_level = v_level
                    worst_reason = str(vision_res.get("reason") or worst_reason)
                    worst_vision = vision_res
                if worst_level == "explicit_indecent":
                    return build_decision_from_level(
                        "explicit_indecent",
                        requested_platforms,
                        worst_reason,
                        vision_checked=True,
                    )
            except Exception as exc:
                logger.warning("Moderação visual via Groq falhou em um lote (%s)", exc)

    if worst_level in {"explicit_indecent", "mild_body_exposure"}:
        return build_decision_from_level(worst_level, requested_platforms, worst_reason, vision_checked=vision_ok)

    if worst_vision is not None:
        return build_decision_from_level(
            worst_level,
            requested_platforms,
            worst_reason,
            ig_override=worst_vision.get("instagram_safe"),
            yt_override=worst_vision.get("youtube_shorts_safe"),
            tt_override=worst_vision.get("tiktok_safe"),
            vision_checked=vision_ok,
        )

    return build_decision_from_level(worst_level, requested_platforms, worst_reason, vision_checked=vision_ok)
