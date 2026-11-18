from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import requests

from src.ffmpeg_utils import get_ffmpeg_bin
from src.models import WordTimestamp

logger = logging.getLogger(__name__)

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def extract_mono_audio(video_path: Path, out_audio_path: Path) -> Path:
    """Extrai áudio leve 16kHz mono MP3 (< 1.5 MB para 2 min) ideal para Whisper."""
    out_audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_bin(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(out_audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_audio_path


def _has_cuda_gpu() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class WordTranscriber:
    """
    Transcritor word-level híbrido:
      - No Google Colab Ultra (GPU CUDA disponível + faster-whisper instalado): roda `large-v3` em GPU!
      - No GitHub Actions (nuvem CPU): usa `Groq Whisper large-v3-turbo` gratuito (transcreve 2 min em ~2s!).
      - Fallback local: `faster-whisper` em CPU (`int8`).
    """

    def __init__(
        self,
        language: str = "pt",
        groq_api_key: Optional[str] = None,
        prefer_engine: str = "auto",  # auto | groq | faster_whisper
        context_hint: str = "",
    ) -> None:
        self.language = language
        self.groq_api_key = (groq_api_key if groq_api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
        self.prefer_engine = prefer_engine
        self.context_hint = context_hint or (
            "Live na Twitch Brasil, streamer conversando com o chat durante gameplay. "
            "Vocabulário comum: chat, mano, velho, caraca, nossa, FPS, bugou, tancou, intancável."
        )
        self._fw_model = None

    def transcribe_video(self, video_path: Path, work_dir: Path) -> List[WordTimestamp]:
        audio_path = extract_mono_audio(video_path, work_dir / f"{video_path.stem}_16k.mp3")

        if self.prefer_engine == "faster_whisper" or (
            self.prefer_engine == "auto" and _has_cuda_gpu() and self._faster_whisper_installed()
        ):
            return self._transcribe_faster_whisper(audio_path)

        if self.groq_api_key:
            try:
                return self._transcribe_groq(audio_path)
            except Exception as exc:
                logger.warning("Groq Whisper falhou (%s). Tentando fallback local faster-whisper...", exc)

        if self._faster_whisper_installed():
            return self._transcribe_faster_whisper(audio_path)

        raise RuntimeError(
            "Nenhum motor de transcrição disponível: defina GROQ_API_KEY ou instale `faster-whisper`."
        )

    @staticmethod
    def _faster_whisper_installed() -> bool:
        try:
            import faster_whisper  # noqa: F401 # type: ignore

            return True
        except Exception:
            return False

    def _transcribe_groq(self, audio_path: Path) -> List[WordTimestamp]:
        headers = {"Authorization": f"Bearer {self.groq_api_key}"}
        with open(audio_path, "rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/mpeg")}
            data = [
                ("model", "whisper-large-v3-turbo"),
                ("language", self.language),
                ("prompt", self.context_hint[:220]),
                ("temperature", "0.0"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ]
            resp = requests.post(GROQ_TRANSCRIPTION_URL, headers=headers, files=files, data=data, timeout=90)
        resp.raise_for_status()
        payload = resp.json()

        words: List[WordTimestamp] = []
        raw_words = payload.get("words") or []
        if raw_words:
            for w in raw_words:
                txt = str(w.get("word", "")).strip()
                if not txt:
                    continue
                start = float(w.get("start", 0.0))
                end = max(start + 0.06, float(w.get("end", start + 0.15)))
                words.append(WordTimestamp(word=txt, start=start, end=end))
            return words

        # Caso raro em que a API devolve apenas segments
        for seg in payload.get("segments", []):
            seg_words = str(seg.get("text", "")).strip().split()
            if not seg_words:
                continue
            s_start = float(seg.get("start", 0.0))
            s_end = max(s_start + 0.2, float(seg.get("end", s_start + 1.0)))
            step = (s_end - s_start) / len(seg_words)
            for idx, sw in enumerate(seg_words):
                words.append(
                    WordTimestamp(
                        word=sw,
                        start=round(s_start + idx * step, 3),
                        end=round(s_start + (idx + 1) * step, 3),
                    )
                )
        return words

    def _transcribe_faster_whisper(self, audio_path: Path) -> List[WordTimestamp]:
        from faster_whisper import WhisperModel  # type: ignore

        if self._fw_model is None:
            use_cuda = _has_cuda_gpu()
            model_size = "large-v3" if use_cuda else "small"
            device = "cuda" if use_cuda else "cpu"
            compute_type = "float16" if use_cuda else "int8"
            logger.info("Carregando faster-whisper (%s em %s/%s)...", model_size, device, compute_type)
            self._fw_model = WhisperModel(model_size, device=device, compute_type=compute_type)

        segments, _ = self._fw_model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=True,
            vad_filter=True,
        )
        words: List[WordTimestamp] = []
        for seg in segments:
            for w in seg.words or []:
                txt = str(w.word).strip()
                if txt:
                    words.append(
                        WordTimestamp(
                            word=txt,
                            start=float(w.start),
                            end=max(float(w.start) + 0.06, float(w.end)),
                        )
                    )
        return words
