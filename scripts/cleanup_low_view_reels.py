from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_FB_BASE = "https://graph.facebook.com/v21.0"


def fetch_reel_views(media_id: str, token: str) -> Optional[int]:
    """Fetch total views/plays for a single Instagram Reel using instagram_manage_insights."""
    for metric in ("views", "plays"):
        resp = requests.get(
            f"{GRAPH_FB_BASE}/{media_id}/insights",
            params={"metric": metric, "access_token": token},
            timeout=15,
        )
        if resp.ok:
            data = resp.json().get("data") or []
            if data and data[0].get("values"):
                val = data[0]["values"][0].get("value")
                if val is not None:
                    return int(val)
    return None


def list_all_reels(ig_account_id: str, token: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    url: Optional[str] = f"{GRAPH_FB_BASE}/{ig_account_id}/media"
    params: Optional[dict[str, Any]] = {
        "fields": "id,caption,media_type,media_product_type,timestamp,permalink",
        "limit": 50,
        "access_token": token,
    }
    while url:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        items.extend(body.get("data") or [])
        url = (body.get("paging") or {}).get("next")
        params = None
    return items


def check_permissions(token: str) -> set[str]:
    resp = requests.get(f"{GRAPH_FB_BASE}/me/permissions", params={"access_token": token}, timeout=15)
    if not resp.ok:
        return set()
    return {
        str(p.get("permission"))
        for p in (resp.json().get("data") or [])
        if p.get("status") == "granted"
    }


def delete_reel(media_id: str, token: str) -> tuple[bool, str]:
    resp = requests.delete(f"{GRAPH_FB_BASE}/{media_id}", params={"access_token": token}, timeout=20)
    if resp.ok and resp.json().get("success", True):
        return True, "deleted"
    return False, resp.text[:250]


def main() -> int:
    parser = argparse.ArgumentParser(description="Lista e exclui Reels do Instagram com poucas visualizacoes.")
    parser.add_argument("--min-views", type=int, default=10, help="Limite minimo de views (default: 10)")
    parser.add_argument("--delete", action="store_true", help="Tenta excluir via API os Reels abaixo de --min-views")
    parser.add_argument("--token", default="", help="Override opcional de IG_ACCESS_TOKEN")
    args = parser.parse_args()

    env = dotenv_values(PROJECT_ROOT / ".env")
    ig_id = (os.getenv("IG_ACCOUNT_ID") or env.get("IG_ACCOUNT_ID") or "").strip()
    token = (args.token or os.getenv("IG_ACCESS_TOKEN") or env.get("IG_ACCESS_TOKEN") or "").strip()

    if not ig_id or not token:
        print("Erro: IG_ACCOUNT_ID ou IG_ACCESS_TOKEN nao configurados.")
        return 1

    perms = check_permissions(token)
    print(f"Permissoes ativas: {sorted(perms)}")
    if "instagram_manage_insights" not in perms:
        print("Aviso: permissao 'instagram_manage_insights' ausente neste token — nao sera possivel ler as views.")
        return 2

    media = list_all_reels(ig_id, token)
    print(f"Total de midias encontradas: {len(media)}")
    low_reels: list[dict[str, Any]] = []

    for item in media:
        mid = str(item["id"])
        views = fetch_reel_views(mid, token)
        cap = (item.get("caption") or "").splitlines()[0][:55] if item.get("caption") else "(sem legenda)"
        link = item.get("permalink") or ""
        print(f"  {mid} | views={views} | {item.get('timestamp')} | {link} | {cap}")
        if views is not None and views < args.min_views:
            low_reels.append({"id": mid, "views": views, "link": link, "caption": cap})

    print(f"\nReels com menos de {args.min_views} views: {len(low_reels)}")
    if args.delete and low_reels:
        for r in low_reels:
            ok, msg = delete_reel(r["id"], token)
            status = "EXCLUIDO" if ok else f"BLOQUEADO PELA API ({msg})"
            print(f"  [{status}] {r['views']} views | {r['link']} | {r['caption']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
