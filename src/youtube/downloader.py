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

        # Caso 1: Trecho com timestamp definido (`vod_start_offset` e `vod_end_offset`)
        if cluster.vod_start_offset is not None and cluster.vod_end_offset is not None:
            s0 = max(0, int(cluster.vod_start_offset))
            s1 = max(s0 + 18, int(cluster.vod_end_offset))
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
            logger.info("Baixando trecho do YouTube [%ds -> %ds]: %s", s0, s1, url)
            subprocess.run(cmd, check=True, timeout=300)
        else:
            # Caso 2: Live do YouTube em andamento — grava 45 segundos ao vivo do stream atual
            logger.info("Capturando 45s ao vivo da Live do YouTube: %s", url)
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
                "ffmpeg:-t 45",
                *get_ytdlp_youtube_args(work_dir),
                "-o",
                str(out_path),
                url,
            ]
            subprocess.run(cmd, check=True, timeout=180)

        if not out_path.exists() or out_path.stat().st_size < 25_000:
            raise RuntimeError(f"Falha ao baixar trecho do YouTube: {url}")

        duration = probe_video_duration(out_path)
        return DownloadedRawMedia(
            video_path=out_path,
            source_type="youtube",
            slice_start_in_vod=cluster.vod_start_offset,
            peak_Hint_sec_in_file=duration * 0.55,
            duration_sec=duration,
        )
