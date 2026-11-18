"""
Descobre automaticamente o `IG_ACCOUNT_ID` (Instagram Creator/Business) e opcionalmente
converte o token curto do Graph API Explorer em um Token de Longa Duração (60 dias).
"""

from __future__ import annotations

import argparse
import requests

GRAPH_BASE = "https://graph.facebook.com/v21.0"


def discover_instagram_account(access_token: str, app_id: str = "", app_secret: str = "") -> dict:
    final_token = access_token.strip()

    # Se App ID e App Secret forem informados, troca automaticamente por Long-Lived Token (60 dias)
    if app_id.strip() and app_secret.strip():
        r_ll = requests.get(
            f"{GRAPH_BASE}/oauth/access_token",
            params={
                "grant_type": "fb_exchange_token",
                "client_id": app_id.strip(),
                "client_secret": app_secret.strip(),
                "fb_exchange_token": final_token,
            },
            timeout=20,
        )
        if r_ll.ok and "access_token" in r_ll.json():
            final_token = r_ll.json()["access_token"]
            print("✅ Token convertido para Longa Duração (60 dias)!")

    # Busca todas as páginas conectadas que possuem conta Instagram Business/Creator vinculada
    r_pages = requests.get(
        f"{GRAPH_BASE}/me/accounts",
        params={
            "fields": "id,name,instagram_business_account{id,username,name}",
            "access_token": final_token,
        },
        timeout=20,
    )
    r_pages.raise_for_status()
    pages = r_pages.json().get("data", [])

    found = []
    for p in pages:
        ig = p.get("instagram_business_account")
        if ig and ig.get("id"):
            found.append(
                {
                    "page_name": p.get("name"),
                    "ig_account_id": str(ig["id"]),
                    "ig_username": ig.get("username", ""),
                    "access_token": final_token,
                }
            )
    return {"accounts": found, "access_token": final_token}


def main() -> None:
    parser = argparse.ArgumentParser(description="Descobre IG_ACCOUNT_ID a partir do token da Meta.")
    parser.add_argument("--token", required=True, help="User Access Token gerado no Graph API Explorer.")
    parser.add_argument("--app-id", default="", help="App ID da Meta (opcional, para gerar token de 60 dias).")
    parser.add_argument("--app-secret", default="", help="App Secret da Meta (opcional, para gerar token de 60 dias).")
    args = parser.parse_args()

    res = discover_instagram_account(args.token, args.app_id, args.app_secret)
    if not res["accounts"]:
        print("⚠️ Nenhuma conta Instagram Creator/Business vinculada a uma Página do Facebook foi encontrada nesse token.")
        print("Verifique se marcou as permissões: instagram_basic, instagram_content_publish, pages_show_list, pages_read_engagement.")
        return

    for acc in res["accounts"]:
        print(f"✅ Conta encontrada: @{acc['ig_username']} (Página: {acc['page_name']})")
        print(f"   IG_ACCOUNT_ID={acc['ig_account_id']}")
        print(f"   IG_ACCESS_TOKEN={acc['access_token'][:25]}...")


if __name__ == "__main__":
    main()
