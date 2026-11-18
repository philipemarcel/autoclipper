from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from src.ffmpeg_utils import get_ffmpeg_bin
from src.models import MomentCluster
from src.twitch.downloader import DownloadedRawMedia, probe_video_duration

logger = logging.getLogger(__name__)


class KickDownloader:
    """
    Baixa clipes, trechos de VOD (`.m3u8` com buffer de contexto ±25s) ou 58s de live da Kick
    via FFmpeg HLS nativo com fallback automático para `yt-dlp`.
    """

    def download_moment(self, cluster: MomentCluster, work_dir: Path) -> DownloadedRawMedia:
        work_dir.mkdir(parents=True, exist_ok=True)
        out_path = work_dir / f"{cluster.cluster_id}_raw_kick.mp4"
        out_path.unlink(missing_ok=True)

        stream_url = str(cluster.best_clip.url or "").strip()
        page_url = str(cluster.best_clip.video_id or stream_url).strip()
        ffmpeg_bin = get_ffmpeg_bin()
        dl_start = cluster.vod_start_offset

        if cluster.mode == "kick_clip":
            logger.info("Baixando clipe da Kick: %s", stream_url)
            self._download_hls_or_ytdlp(ffmpeg_bin, stream_url, page_url, out_path)
            dur = probe_video_duration(out_path)
            # Para clipes longos (>65s), não envia dica falsa do meio (dur*0.55) para não induzir o Diretor a cortar no meio morto
            peak_hint = (dur * 0.55) if dur <= 65.0 else -1.0
            return DownloadedRawMedia(
                video_path=out_path,
                source_type="kick_clip" if dur <= 65.0 else "kick_vod",
                slice_start_in_vod=0,
                peak_Hint_sec_in_file=peak_hint,
                duration_sec=dur,
            )

        if cluster.mode == "kick_vod" and cluster.vod_start_offset is not None and cluster.vod_end_offset is not None:
            orig_s0 = max(0, int(cluster.vod_start_offset))
            orig_s1 = max(orig_s0 + 18, int(cluster.vod_end_offset))
            s0 = max(0, orig_s0 - 25)
            s1 = orig_s1 + 25
            span = s1 - s0
            dl_start = s0

            logger.info("Baixando trecho + buffer VOD Kick [%ds -> %ds]: %s", s0, s1, stream_url)
            cmd = [
                ffmpeg_bin,
                "-y",
                "-fflags",
                "+genpts",
                "-ss",
                str(s0),
                "-i",
                stream_url,
                "-t",
                str(span),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 25_000:
                self._download_hls_or_ytdlp(ffmpeg_bin, stream_url, page_url, out_path, section=(s0, s1))
        else:
            # Live da Kick ao vivo: captura 58s contínuos do stream HLS com PTS zerado
            logger.info("Capturando 58s ao vivo da Kick: %s", stream_url)
            cmd = [
                ffmpeg_bin,
                "-y",
                "-fflags",
                "+genpts",
                "-rw_timeout",
                "8000000",
                "-i",
                stream_url,
                "-t",
                "58",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=135)
                if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 100_000:
                    raise RuntimeError(proc.stderr[-300:])
            except Exception:
                self._download_hls_or_ytdlp(ffmpeg_bin, stream_url, page_url, out_path, live_seconds=58)

        if not out_path.exists() or out_path.stat().st_size < 25_000:
            raise RuntimeError(f"Falha ao baixar mídia da Kick: {stream_url}")

        duration = probe_video_duration(out_path)
        if dl_start is not None and cluster.peak_offset is not None:
            peak_in_file = max(5.0, min(max(6.0, duration - 5.0), float(cluster.peak_offset - dl_start)))
        else:
            peak_in_file = duration * 0.50

        return DownloadedRawMedia(
            video_path=out_path,
            source_type="kick_vod",
            slice_start_in_vod=dl_start,
            peak_Hint_sec_in_file=peak_in_file,
            duration_sec=duration,
        )

    @staticmethod
    def _download_hls_or_ytdlp(
        ffmpeg_bin: str,
        stream_url: str,
        page_url: str,
        out_path: Path,
        section: tuple[int, int] | None = None,
        live_seconds: int | None = None,
    ) -> None:
        if stream_url.endswith(".m3u8") or ".m3u8" in stream_url:
            cmd = [ffmpeg_bin, "-y", "-fflags", "+genpts"]
            if section:
                cmd.extend(["-ss", str(section[0])])
            cmd.extend(["-i", stream_url])
            if section:
                cmd.extend(["-t", str(section[1] - section[0])])
            elif live_seconds:
                cmd.extend(["-t", str(live_seconds)])
            cmd.extend(["-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out_path)])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size >= 25_000:
                return

        target_url = page_url if "kick.com" in page_url else stream_url
        yt_cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ffmpeg-location",
            ffmpeg_bin,
            "--force-overwrites",
            "--no-part",
            "-f",
            "best[height<=1080]/best",
            "--merge-output-format",
            "mp4",
        ]
        if section:
            yt_cmd.extend(["--download-sections", f"*{section[0]}-{section[1]}"])
        yt_cmd.extend(["-o", str(out_path), target_url])
        subprocess.run(yt_cmd, check=True, timeout=300)
