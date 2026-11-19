"""
Autorizador OAuth2 de 1 passo para enviar cortes como YouTube Shorts (YouTube Data API v3).
Reaproveita automaticamente o `client_id` e `client_secret` já existentes em `gdrive_token.json`
ou no `.env` (`GDRIVE_TOKEN_JSON`), sem precisar baixar `client_secret.json` de novo!

Pré-requisito no Google Cloud Console (1 clique):
  - Ativar a "YouTube Data API v3" no mesmo projeto Google Cloud usado no Google Drive.

Uso:
  .\.venv\Scripts\python.exe scripts\auth_youtube.py --save
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_google_client(args: argparse.Namespace) -> Tuple[str, str]:
    if args.client_id and args.client_secret:
        return args.client_id.strip(), args.client_secret.strip()

    for candidate in (PROJECT_ROOT / "gdrive_token.json",):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if data.get("client_id") and data.get("client_secret"):
                    return str(data["client_id"]).strip(), str(data["client_secret"]).strip()
            except Exception:
                pass

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    env_gdrive = os.getenv("GDRIVE_TOKEN_JSON", "").strip()
    if env_gdrive.startswith("{"):
        try:
            data = json.loads(env_gdrive)
            if data.get("client_id") and data.get("client_secret"):
                return str(data["client_id"]).strip(), str(data["client_secret"]).strip()
        except Exception:
            pass

    return "", ""


def save_youtube_to_env_and_profile(compact_json: str, enable_profile: bool = True) -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        escaped_val = f"'{compact_json}'"
        if re.search(r"^YOUTUBE_TOKEN_JSON=.*$", content, flags=re.MULTILINE):
            content = re.sub(
                r"^YOUTUBE_TOKEN_JSON=.*$",
                lambda _: f"YOUTUBE_TOKEN_JSON={escaped_val}",
                content,
                flags=re.MULTILINE,
            )
        else:
            content = content.rstrip("\n") + f"\nYOUTUBE_TOKEN_JSON={escaped_val}\n"
        env_path.write_text(content, encoding="utf-8")
        print(f"✅ YOUTUBE_TOKEN_JSON gravado em {env_path}")

    if enable_profile:
        prof_path = PROJECT_ROOT / "config" / "profiles.yaml"
        if prof_path.exists():
            yaml_txt = prof_path.read_text(encoding="utf-8")
            yaml_txt = re.sub(
                r"(youtube_shorts:\s*(?:\n\s*#[^\n]*)*\n\s*enabled:\s*)(false|true)",
                r"\1true",
                yaml_txt,
                count=1,
            )
            prof_path.write_text(yaml_txt, encoding="utf-8")
            print(f"✅ `youtube_shorts.enabled: true` ativado em {prof_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoriza seu canal YouTube para publicar Shorts via API v3.")
    parser.add_argument("--client-secrets", type=Path, default=Path("client_secret.json"))
    parser.add_argument("--client-id", type=str, default="")
    parser.add_argument("--client-secret", type=str, default="")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "youtube_token.json")
    parser.add_argument("--save", action="store_true", help="Salva no .env e ativa youtube_shorts no profiles.yaml")
    args = parser.parse_args()

    cid, csec = _resolve_google_client(args)
    if cid and csec:
        client_config = {
            "installed": {
                "client_id": cid,
                "client_secret": csec,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    elif args.client_secrets.exists():
        flow = InstalledAppFlow.from_client_secrets_file(str(args.client_secrets), SCOPES)
    else:
        raise SystemExit(
            "Não encontrei client_id/client_secret em gdrive_token.json/.env nem client_secret.json."
        )

    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    compact_json = json.dumps(json.loads(creds.to_json()), separators=(",", ":"))
    args.out.write_text(compact_json, encoding="utf-8")

    try:
        yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
        ch = yt.channels().list(part="snippet", mine=True).execute()
        items = ch.get("items") or []
        if items:
            sn = items[0].get("snippet") or {}
            print(f"✅ Canal YouTube conectado: {sn.get('title')} ({sn.get('customUrl', items[0].get('id'))})")
    except Exception as exc:
        print(f"⚠️ Token gerado, mas aviso ao consultar canal (confira se a YouTube Data API v3 está ativada): {exc}")

    if args.save:
        save_youtube_to_env_and_profile(compact_json, enable_profile=True)
    else:
        print("\nToken salvo em:", args.out)
        print("Rode novamente com `--save` para gravar no `.env` e ativar `youtube_shorts` no `config/profiles.yaml`.")


if __name__ == "__main__":
    main()
