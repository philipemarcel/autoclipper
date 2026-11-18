from __future__ import annotations

import shutil


def get_ffmpeg_bin() -> str:
    """Retorna `ffmpeg` do sistema ou usa automaticamente o binário embutido do `imageio-ffmpeg` (ótimo no Windows)."""
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg  # type: ignore

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return "ffmpeg"
