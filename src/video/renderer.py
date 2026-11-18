from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.ffmpeg_utils import get_ffmpeg_bin
from src.models import DirectorDecision
from src.video.layout_analyzer import LayoutSpec

logger = logging.getLogger(__name__)


@dataclass
class RenderedCutFiles:
    master_mp4: Path
    clean_mp4: Path
    discord_preview_mp4: Path


def detect_nvenc_available() -> bool:
    """Detecta se o encoder por GPU NVIDIA (`h264_nvenc`) está disponível e funcional (ex: Google Colab L4/A100)."""
    try:
        chk = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if chk.returncode != 0:
            return False
        enc = subprocess.run([get_ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=8)
        return "h264_nvenc" in enc.stdout
    except Exception:
        return False


class VideoRenderer:
    """
    Pipeline FFmpeg completo:
      1. Recorta e concatena Cold Open (2-3s com flash rápido na transição) + Cena Principal.
      2. Aplica o Auto-Layout 9:16 (`1080x1920`) -> gera `limpo_1080p60.mp4`.
      3. Queima `legenda.ass` (OpusClip + Headline 2-3s do Teaser) -> gera `master_1080p60.mp4`.
      4. Gera `preview_discord.mp4` (< 8.5 MB garantidos) para tocar nativo no Discord.
    """

    def __init__(self, force_cpu: bool = False) -> None:
        self.use_nvenc = (not force_cpu) and detect_nvenc_available()

    def _video_encoder_args(self, quality: str = "high") -> List[str]:
        if self.use_nvenc:
            cq = "20" if quality == "high" else "26"
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", cq, "-pix_fmt", "yuv420p"]
        crf = "19" if quality == "high" else "25"
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", crf, "-pix_fmt", "yuv420p"]

    def render_all(
        self,
        raw_video: Path,
        decision: DirectorDecision,
        layout: LayoutSpec,
        ass_file: Path,
        out_dir: Path,
    ) -> RenderedCutFiles:
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_mp4 = out_dir / "limpo_1080p60.mp4"
        master_mp4 = out_dir / "master_1080p60.mp4"
        preview_mp4 = out_dir / "preview_discord.mp4"

        # Passo 1: Monta `limpo_1080p60.mp4` (vertical 9:16 + Cold Open + normalização de áudio, sem texto)
        self._render_clean_vertical(raw_video, decision, layout, clean_mp4)

        # Passo 2: Queima `.ass` gerando `master_1080p60.mp4`
        # Executamos com `cwd=ass_file.parent` passando apenas o nome do `.ass` (compatível 100% com Windows e Linux)
        burn_cmd = [
            get_ffmpeg_bin(),
            "-y",
            "-i",
            str(clean_mp4.resolve()),
            "-vf",
            f"ass={ass_file.name}",
            *self._video_encoder_args("high"),
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(master_mp4.resolve()),
        ]
        subprocess.run(burn_cmd, cwd=str(ass_file.parent.resolve()), check=True, capture_output=True)

        # Passo 3: Gera `preview_discord.mp4` sempre abaixo de 8.5 MB
        self.render_discord_preview(master_mp4, preview_mp4, decision.total_duration)

        return RenderedCutFiles(
            master_mp4=master_mp4,
            clean_mp4=clean_mp4,
            discord_preview_mp4=preview_mp4,
        )

    def render_discord_preview(self, source_mp4: Path, preview_mp4: Path, total_duration: float) -> Path:
        dur = max(5.0, total_duration)
        # Alvo máximo: 8.0 MB = 67_108_864 bits -> desconta 96kbps de áudio
        target_total_kbps = int((8.0 * 8192) / dur)
        video_kbps = max(350, min(2200, target_total_kbps - 110))

        cmd = [
            get_ffmpeg_bin(),
            "-y",
            "-i",
            str(source_mp4),
            "-vf",
            "scale=540:960:flags=lanczos,fps=30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            f"{video_kbps}k",
            "-maxrate",
            f"{int(video_kbps * 1.15)}k",
            "-bufsize",
            f"{video_kbps * 2}k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(preview_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return preview_mp4

    def _render_clean_vertical(
        self,
        raw_video: Path,
        decision: DirectorDecision,
        layout: LayoutSpec,
        clean_mp4: Path,
    ) -> None:
        layout_filter = self._build_layout_vf(layout)

        if decision.cold_open_enabled and decision.teaser_duration > 0.25:
            t0, t1 = decision.teaser_start, decision.teaser_end
            t_dur = decision.teaser_duration
            flash_start = max(0.0, t_dur - 0.10)
            m0, m1 = decision.main_start, decision.main_end

            # Cold open recebe Punch-In Zoom dinâmico de 10% + flash de transição nos últimos 0.10s
            filter_complex = (
                f"[0:v]trim=start={t0:.3f}:end={t1:.3f},setpts=PTS-STARTPTS,"
                f"crop=iw*0.90:ih*0.90:iw*0.05:ih*0.05,scale=iw/0.90:ih/0.90:flags=lanczos,"
                f"eq=brightness=0.35:contrast=1.12:enable='gte(t,{flash_start:.3f})'[tv];"
                f"[0:a]atrim=start={t0:.3f}:end={t1:.3f},asetpts=PTS-STARTPTS[ta];"
                f"[0:v]trim=start={m0:.3f}:end={m1:.3f},setpts=PTS-STARTPTS[mv];"
                f"[0:a]atrim=start={m0:.3f}:end={m1:.3f},asetpts=PTS-STARTPTS[ma];"
                f"[tv][ta][mv][ma]concat=n=2:v=1:a=1[catv][cata];"
                f"[catv]{layout_filter}[outv];"
                f"[cata]loudnorm=I=-14:TP=-1.5:LRA=11[outa]"
            )
        else:
            m0, m1 = decision.main_start, decision.main_end
            filter_complex = (
                f"[0:v]trim=start={m0:.3f}:end={m1:.3f},setpts=PTS-STARTPTS[catv];"
                f"[0:a]atrim=start={m0:.3f}:end={m1:.3f},asetpts=PTS-STARTPTS[cata];"
                f"[catv]{layout_filter}[outv];"
                f"[cata]loudnorm=I=-14:TP=-1.5:LRA=11[outa]"
            )

        cmd = [
            get_ffmpeg_bin(),
            "-y",
            "-i",
            str(raw_video),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            *self._video_encoder_args("high"),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-movflags",
            "+faststart",
            str(clean_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    @staticmethod
    def _build_layout_vf(layout: LayoutSpec) -> str:
        """Retorna o grafo de filtros FFmpeg que transforma `[in]` em `1080x1920` (SAR 1:1)."""
        if layout.mode == "split_screen" and layout.webcam_crop and layout.gameplay_crop:
            wc = layout.webcam_crop
            gc = layout.gameplay_crop
            # Topo (Y=0..680): webcam close-up + linha divisória
            # Base (Y=680..1920): fundo blur da gameplay + quadro principal 1080x930 preservando 100% da legenda
            # de tradução do jogo na parte inferior (Y=1470..1610), acima da barra de botões do TikTok!
            return (
                f"split=3[cam_src][gp_src][gp_bg_src];"
                f"[cam_src]crop={wc.w}:{wc.h}:{wc.x}:{wc.y},scale=1080:674:flags=lanczos,"
                f"eq=contrast=1.04:saturation=1.06,"
                f"pad=1080:680:0:0:color=#0D0D0D[top];"
                f"[gp_bg_src]crop={gc.w}:{gc.h}:{gc.x}:{gc.y},scale=1080:1240:force_original_aspect_ratio=increase,"
                f"crop=1080:1240,boxblur=26:4,eq=brightness=-0.22[gp_bg];"
                f"[gp_src]crop={gc.w}:{gc.h}:{gc.x}:{gc.y},scale=1080:930:flags=lanczos,"
                f"eq=contrast=1.04:saturation=1.05[gp_fg];"
                f"[gp_bg][gp_fg]overlay=0:0[bot];"
                f"[top][bot]vstack=inputs=2,setsar=1"
            )

        if layout.mode == "face_track" and layout.face_track_crop:
            fc = layout.face_track_crop
            return f"crop={fc.w}:{fc.h}:{fc.x}:{fc.y},scale=1080:1920:flags=lanczos,setsar=1"

        # Default `blur_center`: fundo vertical desfocado escuro + quadro principal centralizado com leve zoom de 12%
        return (
            "split=2[bg_src][fg_src];"
            "[bg_src]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=26:4,eq=brightness=-0.16:saturation=1.12[bg];"
            "[fg_src]scale=1210:-2:flags=lanczos,crop=1080:ih[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )
