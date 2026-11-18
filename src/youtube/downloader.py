from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from src.ffmpeg_utils import get_ffmpeg_bin
from src.models import MomentCluster
from src.twitch.downloader import DownloadedRawMedia, probe_video_duration
from src.youtube.ytdlp_opts import get_ytdlp_youtube_args

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """
    Baixa apenas o segmento cirúrgico (`start_sec -> end_sec`) de um vídeo ou live do YouTube em `1080p60`
    usando `yt-dlp --download-sections`, sem precisar baixar o vídeo inteiro de várias horas.
    """

    def download_moment(self, cluster: MomentCluster, work_dir: Path) -> DownloadedRawMedia:
        work_dir.mkdir(parents=True, exist_ok=True)
        url = cluster.best_clip.url
        out_path = work_dir / f"{cluster.cluster_id}_raw_yt.mp4"
        out_path.unlink(missing_ok=True)

        dl_start = cluster.vod_start_offset

        # Caso 1: Trecho com timestamp definido (`vod_start_offset` e `vod_end_offset`)
        if cluster.vod_start_offset is not None and cluster.vod_end_offset is not None:
            orig_s0 = max(0, int(cluster.vod_start_offset))
            orig_s1 = max(orig_s0 + 18, int(cluster.vod_end_offset))
            # Baixa buffer generoso antes e depois para o Diretor achar o início/fim exatos da história sem amputar contexto
            buf_before = 25 if cluster.mode != "youtube_manual" else 8
            buf_after = 25 if cluster.mode != "youtube_manual" else 8
            s0 = max(0, orig_s0 - buf_before)
            s1 = orig_s1 + buf_after
            dl_start = s0

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
                "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
                "--merge-output-format",
                "mp4",
                "--download-sections",
                f"*{s0}-{s1}",
                *get_ytdlp_youtube_args(work_dir),
                "-o",
                str(out_path),
                url,
            ]
            logger.info("Baixando trecho + buffer YouTube [%ds -> %ds] (alvo %ds-%ds): %s", s0, s1, orig_s0, orig_s1, url)
            subprocess.run(cmd, check=True, timeout=360)
        else:
            # Caso 2: Live do YouTube em andamento — grava 58s da ponta ao vivo para preservar contexto completo da conversa
            logger.info("Capturando 58s ao vivo da Live do YouTube: %s", url)
            cmd = [
                sys.executable,
                "-m",
                "yt_dlp",
                "--ffmpeg-location",
                get_ffmpeg_bin(),
                "--force-overwrites",
                "--no-part",
                "--no-playlist",
                "-f",
                "best[height<=1080]/best",
                "--downloader",
                "ffmpeg",
                "--downloader-args",
                "ffmpeg_i:-rw_timeout 8000000 ffmpeg:-t 58",
                *get_ytdlp_youtube_args(work_dir, is_live=True),
                "-o",
                str(out_path),
                url,
            ]
            try:
                subprocess.run(cmd, check=True, timeout=135)
            except subprocess.TimeoutExpired:
                # Se o ffmpeg já gravou um arquivo válido antes do timeout, reaproveita!
                if not out_path.exists() or out_path.stat().st_size < 200_000:
                    raise

        if not out_path.exists() or out_path.stat().st_size < 25_000:
            raise RuntimeError(f"Falha ao baixar trecho do YouTube: {url}")

        duration = probe_video_duration(out_path)
        if dl_start is not None and cluster.peak_offset is not None:
            peak_in_file = max(5.0, min(max(6.0, duration - 5.0), float(cluster.peak_offset - dl_start)))
        else:
            peak_in_file = duration * 0.50

        return DownloadedRawMedia(
            video_path=out_path,
            source_type="youtube",
            slice_start_in_vod=dl_start,
            peak_Hint_sec_in_file=peak_in_file,
            duration_sec=duration,
        )
