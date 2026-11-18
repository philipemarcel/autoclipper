"""Sincroniza os segredos configurados no `.env` local diretamente para GitHub Actions Secrets."""

from __future__ import annotations

import argparse
import base64
import os
import requests
from nacl import encoding, public

from src.config_loader import load_environment

SECRET_KEYS = [
    "TWITCH_CLIENT_ID",
    "TWITCH_CLIENT_SECRET",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "GDRIVE_TOKEN_JSON",
    "DISCORD_BOT_TOKEN",
    "DISCORD_CHANNEL_ID",
    "IG_ACCOUNT_ID",
    "IG_ACCESS_TOKEN",
    "TIKTOK_CLIENT_KEY",
    "TIKTOK_CLIENT_SECRET",
    "TIKTOK_ACCESS_TOKEN",
    "TIKTOK_REFRESH_TOKEN",
    "YOUTUBE_COOKIES_TXT",
]

PLACEHOLDERS = {
    "",
    "seu_twitch_client_id",
    "seu_twitch_client_secret",
    "sua_chave_gemini_ai_studio",
    "sua_chave_gratuita_groq_cloud",
    "seu_token_do_bot_discord",
    "id_do_canal_de_aprovacao",
    "id_conta_instagram_creator_ou_business",
    "token_longa_duracao_meta_graph_api",
    "token_acesso_tiktok_api",
}


def encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    pk = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder())
    sealed = public.SealedBox(pk).encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(sealed).decode("utf-8")


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="philipemarcel")
    parser.add_argument("--repo", default="autoclipper")
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    headers = {
        "Authorization": f"Bearer {args.token.strip()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base = f"https://api.github.com/repos/{args.owner}/{args.repo}/actions/secrets"
    r_pk = requests.get(f"{base}/public-key", headers=headers, timeout=20)
    r_pk.raise_for_status()
    pk_data = r_pk.json()

    synced = []
    for key in SECRET_KEYS:
        val = os.getenv(key, "").strip()
        if val in PLACEHOLDERS:
            continue
        enc_val = encrypt_secret(pk_data["key"], val)
        r_put = requests.put(
            f"{base}/{key}",
            headers=headers,
            json={"encrypted_value": enc_val, "key_id": pk_data["key_id"]},
            timeout=20,
        )
        r_put.raise_for_status()
        synced.append(key)

    print("SECRETS_SYNCED:", len(synced), synced)


if __name__ == "__main__":
    main()
