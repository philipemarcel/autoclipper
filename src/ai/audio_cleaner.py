from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from src.ffmpeg_utils import get_ffmpeg_bin

logger = logging.getLogger(__name__)


class AudioCleaner:
    """
    Remove músicas com direitos autorais (DMCA) isolando apenas a voz do criador via IA (`Demucs htdemucs`),
    e substitui a trilha de áudio sem re-encodar os pixels do vídeo (`-c:v copy`).
    """

    def clean_video_music(
        self,
        input_video: Path,
        output_video: Path,
        work_dir: Path,
        device: str = "auto",
    ) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        wav_in = work_dir / f"{input_video.stem}_full.wav"

        # 1. Extrai WAV estéreo 44.1kHz para o Demucs
        subprocess.run(
            [
                get_ffmpeg_bin(),
                "-y",
                "-i",
                str(input_video),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                str(wav_in),
            ],
            check=True,
            capture_output=True,
        )

        # 2. Executa Demucs (`htdemucs --two-stems=vocals`)
        demucs_out = work_dir / "demucs_stems"
        demucs_cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "--two-stems",
            "vocals",
            "-o",
            str(demucs_out),
        ]
        if device in {"cuda", "cpu"}:
            demucs_cmd.extend(["-d", device])
        demucs_cmd.append(str(wav_in))

        logger.info("Executando isolamento vocal Demucs em %s...", input_video.name)
        subprocess.run(demucs_cmd, check=True)

        vocals_wav = demucs_out / "htdemucs" / wav_in.stem / "vocals.wav"
        if not vocals_wav.exists():
            raise RuntimeError(f"Arquivo de voz isolada não encontrado em {vocals_wav}")

        # 3. Remux instantâneo (`-c:v copy`) mantendo 100% da qualidade visual + normalização de voz
        tmp_out = work_dir / f"{output_video.stem}_vocalclean.mp4"
        subprocess.run(
            [
                get_ffmpeg_bin(),
                "-y",
                "-i",
                str(input_video),
                "-i",
                str(vocals_wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-af",
                "highpass=f=75,loudnorm=I=-14:TP=-1.5:LRA=11",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(tmp_out),
            ],
            check=True,
            capture_output=True,
        )

        output_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_out), str(output_video))
        return output_video
