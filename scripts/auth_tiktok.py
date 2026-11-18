"""
Autorizador OAuth2 para o TikTok Content Posting API v2 (com PKCE S256).
Suporta tanto callback local (`127.0.0.1`) quanto Redirect URI HTTPS (ex: `https://github.com/philipemarcel/autoclipper`),
onde basta colar a URL final redirecionada pelo navegador após autorizar.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import secrets
import urllib.parse
import requests

DEFAULT_REDIRECT_URI = "https://github.com/philipemarcel/autoclipper"
SCOPES = "user.info.basic,video.upload,video.publish"


def _generate_pkce() -> tuple[str, str]:
    # TikTok Desktop exige code_challenge em HEX do SHA-256 (e não base64url padrão RFC 7636)
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    verifier = "".join(secrets.choice(chars) for _ in range(64))
    challenge = hashlib.sha256(verifier.encode("utf-8")).hexdigest()
    return verifier, challenge


def exchange_code(
    client_key: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict:
    resp = requests.post(
        "https://open.tiktokapis.com/v2/oauth/token/",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_key": client_key.strip(),
            "client_secret": client_secret.strip(),
            "code": code.strip(),
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri.strip(),
            "code_verifier": code_verifier,
        },
        timeout=25,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoriza sua conta TikTok via OAuth2.")
    parser.add_argument("--client-key", required=True, help="Client Key do seu app no TikTok Developers")
    parser.add_argument("--client-secret", required=True, help="Client Secret do seu app no TikTok Developers")
    parser.add_argument("--redirect-uri", default=DEFAULT_REDIRECT_URI, help="Redirect URI cadastrada no TikTok")
    parser.add_argument("--verifier", default="", help="PKCE verifier salvo (para troca direta via --code-url)")
    parser.add_argument("--code-url", default="", help="URL completa copiada do navegador após autorizar")
    parser.add_argument("--listen", action="store_true", help="Abre servidor local em 127.0.0.1:8766/callback/")
    args = parser.parse_args()

    if args.code_url:
        parsed = urllib.parse.urlparse(args.code_url.strip())
        qs = urllib.parse.parse_qs(parsed.query)
        code = (qs.get("code") or [""])[0]
        if not code:
            raise SystemExit(f"Nenhum parâmetro ?code= encontrado na URL: {args.code_url}")
        token_data = exchange_code(
            args.client_key,
            args.client_secret,
            code,
            args.redirect_uri,
            args.verifier,
        )
        print("TIKTOK_TOKEN_JSON=" + json.dumps(token_data, ensure_ascii=False), flush=True)
        return

    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(12)

    params = {
        "client_key": args.client_key.strip(),
        "scope": SCOPES,
        "response_type": "code",
        "redirect_uri": args.redirect_uri.strip(),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"https://www.tiktok.com/v2/auth/authorize/?{urllib.parse.urlencode(params)}"
    print(f"PKCE_VERIFIER={verifier}", flush=True)
    print(f"AUTH_URL={auth_url}", flush=True)

    if args.listen:
        parsed_r = urllib.parse.urlparse(args.redirect_uri.strip())
        host = parsed_r.hostname or "127.0.0.1"
        port = parsed_r.port or 8766
        captured: dict[str, str] = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if "code" in qs:
                    captured["code"] = qs["code"][0]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write("<h2>✅ TikTok autorizado! Pode voltar ao chat.</h2>".encode("utf-8"))
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(f"Erro OAuth TikTok: {qs}".encode("utf-8"))

            def log_message(self, format, *args):
                return

        with http.server.HTTPServer((host, port), Handler) as httpd:
            httpd.handle_request()

        if "code" in captured:
            token_data = exchange_code(
                args.client_key,
                args.client_secret,
                captured["code"],
                args.redirect_uri,
                verifier,
            )
            print("TIKTOK_TOKEN_JSON=" + json.dumps(token_data, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
