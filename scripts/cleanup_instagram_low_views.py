from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_FB_BASE = "https://graph.facebook.com/v21.0"


def fetch_media_views(media_id: str, access_token: str) -> Optional[int]:
    """Fetch Reel view count from Meta Graph API (requires instagram_manage_insights)."""
    for metric in ("views", "plays", "ig_reels_aggregated_all_plays_count"):
        resp = requests.get(
            f"{GRAPH_FB_BASE}/{media_id}/insights",
            params={"metric": metric, "access_token": access_token},
            timeout=15,
        )
        if resp.ok:
            data = resp.json().get("data") or []
            if data and data[0].get("values"):
                val = data[0]["values"][0].get("value")
                if val is not None:
                    return int(val)
        else:
            err_msg = resp.text
            if "(#10)" in err_msg or "OAuthException" in err_msg and "permission" in err_msg.lower():
                raise PermissionError(
                    "Token sem permissão `instagram_manage_insights` para ler views dos Reels."
                )
    return None


def list_all_ig_media(ig_account_id: str, access_token: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    url: Optional[str] = f"{GRAPH_FB_BASE}/{ig_account_id}/media"
    params: Optional[Dict[str, Any]] = {
        "fields": "id,caption,media_type,media_product_type,timestamp,permalink",
        "limit": 50,
        "access_token": access_token,
    }
    while url:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        items.extend(payload.get("data") or [])
        url = (payload.get("paging") or {}).get("next")
        params = None
    return items


def delete_ig_media(media_id: str, access_token: str) -> tuple[bool, str]:
    resp = requests.delete(
        f"{GRAPH_FB_BASE}/{media_id}",
        params={"access_token": access_token},
        timeout=20,
    )
    if resp.ok:
        return True, "ok"
    return False, resp.text[:250]


def find_low_view_media(
    ig_account_id: str,
    access_token: str,
    min_views: int = 10,
) -> List[Dict[str, Any]]:
    media_list = list_all_ig_media(ig_account_id, access_token)
    matches: List[Dict[str, Any]] = []
    for item in media_list:
        mid = str(item["id"])
        views = fetch_media_views(mid, access_token)
        if views is not None and views < min_views:
            cap_first = (str(item.get("caption") or "").splitlines() or [""])[0][:60]
            matches.append(
                {
                    "id": mid,
                    "views": views,
                    "timestamp": str(item.get("timestamp") or ""),
                    "permalink": str(item.get("permalink") or ""),
                    "caption": cap_first,
                }
            )
    matches.sort(key=lambda x: (x["views"], x["timestamp"]))
    return matches


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Lista e opcionalmente exclui Reels do Instagram com poucas views.")
    parser.add_argument("--min-views", type=int, default=10, help="Limite mínimo de views (abaixo disso entra na lista)")
    parser.add_argument("--delete", action="store_true", help="Tenta excluir via Graph API os Reels listados")
    parser.add_argument("--token", default="", help="Token Meta com instagram_manage_insights (sobrescreve IG_ACCESS_TOKEN)")
    args = parser.parse_args()

    ig_id = os.getenv("IG_ACCOUNT_ID", "").strip()
    token = (args.token or os.getenv("IG_ACCESS_TOKEN", "")).strip()
    if not ig_id or not token:
        print("Erro: IG_ACCOUNT_ID e IG_ACCESS_TOKEN são obrigatórios.", file=sys.stderr)
        return 1

    try:
        low = find_low_view_media(ig_id, token, min_views=args.min_views)
    except PermissionError as exc:
        print(f"Erro de permissão: {exc}", file=sys.stderr)
        return 2

    print(f"Encontrados {len(low)} vídeo(s) com menos de {args.min_views} views:")
    for it in low:
        print(f"  - views={it['views']} | {it['timestamp']} | {it['permalink']} | {it['caption']}")

    if args.delete and low:
        deleted = 0
        for it in low:
            ok, msg = delete_ig_media(it["id"], token)
            if ok:
                deleted += 1
                print(f"  [EXCLUIDO] {it['id']} ({it['permalink']})")
            else:
                print(f"  [BLOQUEADO PELA API] {it['id']} ({it['permalink']}) -> {msg}")
        print(f"Resumo: {deleted}/{len(low)} vídeo(s) excluído(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
