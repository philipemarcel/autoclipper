from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from src.ffmpeg_utils import get_ffmpeg_bin
from src.models import DetectionConfig, MomentCluster

logger = logging.getLogger(__name__)


@dataclass
class DownloadedRawMedia:
    video_path: Path
    source_type: str  # "vod" | "clip_fallback"
    slice_start_in_vod: Optional[int]
    peak_Hint_sec_in_file: float
    duration_sec: float


def probe_video_duration(video_path: Path) -> float:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(proc.stdout.strip())
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        return max(1.0, frames / max(1.0, fps))


class TwitchDownloader:
    """
    Extrator inteligente focado em Fidelidade de Contexto:
      1. Prioriza o MP4 nativo do melhor clipe do cluster quando ele já tem >= 18s (98% dos clipes na Twitch têm 26s–60s):
         - O clipe nativo já foi enquadrado por um humano exatamente na piada/jogada/reação.
         - Evita o desalinhamento de PTS (`start: 64.04s`) do DVR HLS (`index-dvr.m3u8`) da Twitch durante lives ao vivo.
      2. Se o clipe mais visto for curto (< 22s) mas houver outro clipe no mesmo cluster com mais contexto (>= 28s),
         usa o clipe mais completo do cluster.
      3. Só recorre ao recorte bruto do VOD quando todos os clipes do cluster forem muito curtos (< 18s).
    """

    def __init__(self, detection_cfg: DetectionConfig) -> None:
        self.cfg = detection_cfg

    def download_moment(self, cluster: MomentCluster, work_dir: Path) -> DownloadedRawMedia:
        work_dir.mkdir(parents=True, exist_ok=True)
        chosen_clip = self._select_best_context_clip(cluster)

        if chosen_clip.duration >= 18.0 or not cluster.video_id or cluster.vod_start_offset is None:
            return self._download_clip_direct(cluster, chosen_clip, work_dir)

        try:
            return self._download_from_vod(cluster, work_dir)
        except Exception as exc:
            logger.warning(
                "Falha ao extrair janela do VOD %s (%s). Usando clipe nativo...",
                cluster.video_id,
                exc,
            )
            return self._download_clip_direct(cluster, chosen_clip, work_dir)

    @staticmethod
    def _select_best_context_clip(cluster: MomentCluster):
        """
        Usa os clipes da página da Twitch como referência, priorizando dentro do cluster
        o clipe popular que tenha MAIOR tempo de contexto (ex: 40s-60s em vez de um corte seco de 25s).
        """
        best = cluster.best_clip
        if best.duration >= 42.0:
            return best
        # Procura no mesmo momento se algum espectador fez um clipe mais longo (com mais contexto antes/depois)
        longer_candidates = [
            c
            for c in cluster.clips
            if c.duration > best.duration + 6.0
            and (c.view_count >= max(15, int(best.view_count * 0.25)) or c.view_count >= 150)
        ]
        if longer_candidates:
            return max(longer_candidates, key=lambda c: (c.duration * 1.5) + (c.view_count * 0.5))
        return best

    def _download_from_vod(self, cluster: MomentCluster, work_dir: Path) -> DownloadedRawMedia:
        assert cluster.vod_start_offset is not None
        vod_end = cluster.vod_end_offset or (cluster.vod_start_offset + 30)

        slice_start = max(0, cluster.vod_start_offset - self.cfg.vod_padding_before_seconds)
        slice_end = vod_end + self.cfg.vod_padding_after_seconds
        # Trava janela bruta máxima em 150s para manter download/transcrição rápidos
        if slice_end - slice_start > 150:
            center = cluster.peak_offset or cluster.vod_start_offset
            slice_start = max(0, center - 65)
            slice_end = slice_start + 135

        out_path = work_dir / f"{cluster.cluster_id}_raw_vod.mp4"
        vod_url = f"https://www.twitch.tv/videos/{cluster.video_id}"

        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ffmpeg-location",
            get_ffmpeg_bin(),
            "--no-part",
            "--no-playlist",
            "--concurrent-fragments",
            "4",
            "-f",
            "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "--merge-output-format",
            "mp4",
            "--download-sections",
            f"*{slice_start}-{slice_end}",
            "-o",
            str(out_path),
            vod_url,
        ]
        logger.info("Baixando trecho do VOD %s [%ss -> %ss]...", cluster.video_id, slice_start, slice_end)
        subprocess.run(cmd, check=True, timeout=300)

        if not out_path.exists() or out_path.stat().st_size < 50_000:
            raise RuntimeError("Arquivo de trecho do VOD veio vazio ou não foi gerado.")

        duration = probe_video_duration(out_path)
        peak_offset = cluster.peak_offset if cluster.peak_offset is not None else cluster.vod_start_offset
        peak_in_file = max(0.0, min(duration, float(peak_offset - slice_start)))

        return DownloadedRawMedia(
            video_path=out_path,
            source_type="vod",
            slice_start_in_vod=slice_start,
            peak_Hint_sec_in_file=peak_in_file,
            duration_sec=duration,
        )

    def _download_clip_direct(self, cluster: MomentCluster, clip, work_dir: Path) -> DownloadedRawMedia:
        out_path = work_dir / f"{cluster.cluster_id}_raw_clip.mp4"
        out_path.unlink(missing_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ffmpeg-location",
            get_ffmpeg_bin(),
            "--force-overwrites",
            "--no-continue",
            "--no-part",
            "--no-playlist",
            "-f",
            "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            str(out_path),
            clip.url,
        ]
        logger.info(
            "Baixando clipe nativo de alta precisão: %s ('%s' - %d views, %.1fs)...",
            clip.url,
            clip.title,
            clip.view_count,
            clip.duration,
        )
        subprocess.run(cmd, check=True, timeout=180)

        if not out_path.exists() or out_path.stat().st_size < 20_000:
            raise RuntimeError(f"Falha ao baixar clipe: {clip.url}")

        duration = probe_video_duration(out_path)
        return DownloadedRawMedia(
            video_path=out_path,
            source_type="clip_native",
            slice_start_in_vod=clip.vod_offset,
            peak_Hint_sec_in_file=duration * 0.65,
            duration_sec=duration,
        )
