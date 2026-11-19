"""
Configurador 1-passo para conectar Instagram Reels no AutoClipper:
  1. Suporta **Instagram API com Login do Instagram (`IG...` — recomendado, não precisa Página do Facebook!)**
     via OAuth (`--oauth` com App ID/Secret) OU colando o token gerado no painel da Meta (`--token IG...`).
  2. Suporta **Facebook Login Graph API (`EAA...`)** descobrindo `IG_ACCOUNT_ID` via `/me/accounts`.
  3. Opcionalmente troca o token curto por um **Token de Longa Duração (60 dias)**.
  4. Com `--save`, grava direto em `.env` e habilita `publishing.instagram.enabled: true` em `config/profiles.yaml`.
"""

from __future__ import annotations

import argparse
import http.server
import re
import urllib.parse
from pathlib import Path

import requests

GRAPH_FB_BASE = "https://graph.facebook.com/v21.0"
GRAPH_IG_BASE = "https://graph.instagram.com/v21.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _exchange_ig_long_lived(short_token: str, app_secret: str) -> str:
    resp = requests.get(
        "https://graph.instagram.com/access_token",
        params={
            "grant_type": "ig_exchange_token",
            "client_secret": app_secret.strip(),
            "access_token": short_token.strip(),
        },
        timeout=20,
    )
    if resp.ok and "access_token" in resp.json():
        print("✅ Token Instagram convertido para Longa Duração (60 dias)!")
        return str(resp.json()["access_token"])
    return short_token.strip()


def discover_instagram_account(access_token: str, app_id: str = "", app_secret: str = "") -> dict:
    final_token = access_token.strip()

    # Caminho moderno: Instagram API com Login do Instagram (`IG...`, sem Página do Facebook)
    if final_token.upper().startswith("IG"):
        if app_secret.strip():
            final_token = _exchange_ig_long_lived(final_token, app_secret)

        r_me = requests.get(
            f"{GRAPH_IG_BASE}/me",
            params={"fields": "id,user_id,username,account_type", "access_token": final_token},
            timeout=20,
        )
        r_me.raise_for_status()
        me = r_me.json()
        ig_uid = str(me.get("user_id") or me.get("id") or "")
        return {
            "accounts": [
                {
                    "page_name": f"Instagram Login ({me.get('account_type', 'PROFESSIONAL')})",
                    "ig_account_id": ig_uid,
                    "ig_username": str(me.get("username") or ""),
                    "account_type": str(me.get("account_type") or ""),
                    "access_token": final_token,
                }
            ]
            if ig_uid
            else [],
            "access_token": final_token,
        }

    # Caminho clássico: Facebook Login (`EAA...`) ligado a Página do Facebook
    if app_id.strip() and app_secret.strip():
        r_ll = requests.get(
            f"{GRAPH_FB_BASE}/oauth/access_token",
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
            print("✅ Token Facebook convertido para Longa Duração (60 dias)!")

    r_pages = requests.get(
        f"{GRAPH_FB_BASE}/me/accounts",
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


def run_instagram_oauth_local(app_id: str, app_secret: str, port: int = 8767) -> dict:
    """Abre servidor HTTP local para concluir OAuth `Instagram API with Instagram Login`."""
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    auth_url = "https://www.instagram.com/oauth/authorize?" + urllib.parse.urlencode(
        {
            "client_id": app_id.strip(),
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "instagram_business_basic,instagram_business_content_publish",
        }
    )
    print(f"AUTH_URL={auth_url}", flush=True)
    captured: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in qs:
                captured["code"] = qs["code"][0].replace("#_", "").strip()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write("<h2>✅ Instagram autorizado! Pode voltar ao terminal.</h2>".encode("utf-8"))
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Erro OAuth Instagram: {qs}".encode("utf-8"))

        def log_message(self, format, *args):
            return

    with http.server.HTTPServer(("127.0.0.1", port), Handler) as httpd:
        httpd.handle_request()

    if "code" not in captured:
        raise RuntimeError("Nenhum código OAuth recebido.")

    tok_resp = requests.post(
        "https://api.instagram.com/oauth/access_token",
        data={
            "client_id": app_id.strip(),
            "client_secret": app_secret.strip(),
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code": captured["code"],
        },
        timeout=25,
    )
    tok_resp.raise_for_status()
    short_tok = str(tok_resp.json()["access_token"])
    return discover_instagram_account(short_tok, app_id=app_id, app_secret=app_secret)


def save_to_env_and_enable_profile(ig_account_id: str, ig_access_token: str, enable_profile: bool = True) -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        for key, val in [("IG_ACCOUNT_ID", ig_account_id), ("IG_ACCESS_TOKEN", ig_access_token)]:
            if re.search(rf"^{key}=.*$", content, flags=re.MULTILINE):
                content = re.sub(rf"^{key}=.*$", f"{key}={val}", content, flags=re.MULTILINE)
            else:
                content = content.rstrip("\n") + f"\n{key}={val}\n"
        env_path.write_text(content, encoding="utf-8")
        print(f"✅ Credenciais gravadas em {env_path}")

    if enable_profile:
        prof_path = PROJECT_ROOT / "config" / "profiles.yaml"
        if prof_path.exists():
            yaml_txt = prof_path.read_text(encoding="utf-8")
            yaml_txt = re.sub(
                r"(instagram:\s*\n\s*enabled:\s*)(false|true)",
                r"\1true",
                yaml_txt,
                count=1,
            )
            prof_path.write_text(yaml_txt, encoding="utf-8")
            print(f"✅ `instagram.enabled: true` ativado em {prof_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descobre IG_ACCOUNT_ID e configura Instagram Reels.")
    parser.add_argument("--token", default="", help="Token gerado na Meta (IG... ou EAA...).")
    parser.add_argument("--oauth", action="store_true", help="Abre login OAuth local do Instagram.")
    parser.add_argument("--app-id", default="", help="Instagram App ID ou Meta App ID.")
    parser.add_argument("--app-secret", default="", help="Instagram App Secret ou Meta App Secret.")
    parser.add_argument("--save", action="store_true", help="Grava no .env e ativa instagram no profiles.yaml.")
    args = parser.parse_args()

    if args.oauth:
        if not args.app_id or not args.app_secret:
            raise SystemExit("Para --oauth informe --app-id e --app-secret.")
        res = run_instagram_oauth_local(args.app_id, args.app_secret)
    elif args.token:
        res = discover_instagram_account(args.token, args.app_id, args.app_secret)
    else:
        raise SystemExit("Informe --token <TOKEN> ou --oauth --app-id ... --app-secret ...")

    if not res["accounts"]:
        print("⚠️ Nenhuma conta Instagram Profissional (Business/Creator) encontrada nesse token.")
        return

    acc = res["accounts"][0]
    for item in res["accounts"]:
        print(f"✅ Conta encontrada: @{item['ig_username']} ({item['page_name']})")
        print(f"   IG_ACCOUNT_ID={item['ig_account_id']}")
        print(f"   IG_ACCESS_TOKEN={item['access_token'][:25]}...")

    if args.save:
        save_to_env_and_enable_profile(acc["ig_account_id"], acc["access_token"], enable_profile=True)


if __name__ == "__main__":
    main()
