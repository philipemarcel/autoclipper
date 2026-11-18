"""
Utilitário de 30 segundos para autorizar sua conta Google AI Ultra (20 TB) via OAuth2
e gerar o JSON pronto para colar no segredo `GDRIVE_TOKEN_JSON` do GitHub Actions.

Uso:
  1. Baixe o arquivo `client_secret.json` (OAuth Client ID tipo "Desktop App" no Google Cloud Console).
  2. Rode: python scripts/auth_gdrive.py --client-secrets client_secret.json
  3. Faça login na sua conta Google AI Ultra no navegador que abrirá.
  4. Copie a linha JSON impressa no terminal e salve em `GDRIVE_TOKEN_JSON` no GitHub Secrets!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera GDRIVE_TOKEN_JSON para Google Drive 20 TB.")
    parser.add_argument(
        "--client-secrets",
        type=Path,
        default=Path("client_secret.json"),
        help="Caminho para o client_secret.json baixado do Google Cloud Console.",
    )
    parser.add_argument("--client-id", type=str, default="", help="OAuth Client ID (Desktop App)")
    parser.add_argument("--client-secret", type=str, default="", help="OAuth Client Secret (Desktop App)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gdrive_token.json"),
        help="Arquivo local onde o token também será salvo (ignorado pelo git).",
    )
    args = parser.parse_args()

    if args.client_id and args.client_secret:
        client_config = {
            "installed": {
                "client_id": args.client_id.strip(),
                "client_secret": args.client_secret.strip(),
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
            f"Arquivo {args.client_secrets} não encontrado e --client-id/--client-secret não informados."
        )
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

    compact_json = json.dumps(json.loads(creds.to_json()), separators=(",", ":"))
    args.out.write_text(compact_json, encoding="utf-8")

    print("\n" + "=" * 78)
    print("SUCESSO! Copie a linha abaixo para o Secret `GDRIVE_TOKEN_JSON` no GitHub:")
    print("=" * 78)
    print(compact_json)
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
