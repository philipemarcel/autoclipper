from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional


def get_ytdlp_youtube_args(work_dir: Optional[Path] = None) -> List[str]:
    """
    Constrói flags resilientes do `yt-dlp` para YouTube em 2026:
      - Ativa runtime JS (`deno` ou `node`, obrigatório nas versões recentes do yt-dlp)
      - Usa clientes resilientes (`android_vr,tv,web`)
      - Injeta `--cookies` automaticamente caso `YOUTUBE_COOKIES_TXT` esteja configurado no `.env` / GitHub Secrets.
    """
    args: List[str] = []
    if shutil.which("deno"):
        args.extend(["--js-runtimes", "deno"])
    elif shutil.which("node"):
        args.extend(["--js-runtimes", "node"])

    args.extend(["--extractor-args", "youtube:player_client=android_vr,tv,web"])

    cookies_txt = os.getenv("YOUTUBE_COOKIES_TXT", "").strip()
    if cookies_txt and work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        cookie_file = work_dir / "yt_cookies.txt"
        cookie_file.write_text(cookies_txt, encoding="utf-8")
        args.extend(["--cookies", str(cookie_file)])

    return args
