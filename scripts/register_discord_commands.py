"""
Registra o Slash Command `/cortar` no bot do Discord (tanto globalmente quanto instantaneamente no servidor).
"""

from __future__ import annotations

import argparse
import os
import requests

from src.config_loader import load_environment

DISCORD_API_BASE = "https://discord.com/api/v10"

COMMANDS_PAYLOAD = [
    {
        "name": "cortar",
        "description": "Gera cortes verticais 9:16 automáticos de um vídeo, live ou clipe (YouTube ou Kick)",
        "type": 1,
        "options": [
            {
                "name": "url",
                "description": "Link do YouTube ou Kick (ex: https://youtu.be/... ou https://kick.com/canal)",
                "type": 3,  # STRING
                "required": True,
            },
            {
                "name": "quantidade",
                "description": "Quantos cortes gerar dos melhores momentos (1 a 10, padrão: 3)",
                "type": 4,  # INTEGER
                "required": False,
                "min_value": 1,
                "max_value": 10,
            },
            {
                "name": "trecho",
                "description": "Opcional: intervalo específico (ex: 12:30-15:00 ou minuto 25:00)",
                "type": 3,  # STRING
                "required": False,
            },
        ],
    }
]


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description="Registra o comando /cortar no Discord.")
    parser.add_argument("--guild-id", default="1572595139040383136", help="ID do servidor Discord para ativação instantânea.")
    args = parser.parse_args()

    bot_token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not bot_token:
        raise SystemExit("DISCORD_BOT_TOKEN ausente no .env")

    headers = {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }

    # Descobre o Application ID do próprio bot
    r_me = requests.get(f"{DISCORD_API_BASE}/oauth2/applications/@me", headers=headers, timeout=15)
    r_me.raise_for_status()
    app_id = r_me.json()["id"]

    # 1. Registra no servidor específico (aparece em 0 segundos no Discord!)
    if args.guild_id:
        r_guild = requests.put(
            f"{DISCORD_API_BASE}/applications/{app_id}/guilds/{args.guild_id}/commands",
            headers=headers,
            json=COMMANDS_PAYLOAD,
            timeout=15,
        )
        r_guild.raise_for_status()
        print(f"✅ Comando `/cortar` ativado INSTANTANEAMENTE no servidor {args.guild_id}!")

    # 2. Registra também globalmente
    r_glob = requests.put(
        f"{DISCORD_API_BASE}/applications/{app_id}/commands",
        headers=headers,
        json=COMMANDS_PAYLOAD,
        timeout=15,
    )
    r_glob.raise_for_status()
    print(f"✅ Comando `/cortar` registrado globalmente no App {app_id}!")


if __name__ == "__main__":
    main()
