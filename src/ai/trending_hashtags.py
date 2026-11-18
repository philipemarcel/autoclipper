from __future__ import annotations

import logging
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

GOOGLE_TRENDS_BR_RSS = "https://trends.google.com/trending/rss?geo=BR"
_TRENDS_CACHE: Dict[str, Tuple[float, List[str]]] = {}
_CACHE_TTL_SEC = 900.0  # 15 minutos

GAME_HASHTAG_ALIASES: Dict[str, List[str]] = {
    "grand theft auto v": ["#gtarp", "#gtav"],
    "grand theft auto vi": ["#gta6", "#gta"],
    "counter-strike": ["#cs2"],
    "valorant": ["#valorant"],
    "league of legends": ["#lolbr"],
    "just chatting": ["#podcast", "#react"],
    "irl": ["#irl"],
    "fortnite": ["#fortnitebr"],
    "minecraft": ["#minecraftbr"],
    "arc raiders": ["#arcraiders"],
    "dead by daylight": ["#dbd"],
}


def slugify_hashtag(text: str) -> str:
    """Converte termo livre em hashtag ASCII/Unicode limpa para TikTok/Reels."""
    raw = str(text or "").strip().lower()
    if raw.startswith("#"):
        raw = raw[1:]
    norm = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9]", "", norm)
    if len(cleaned) < 2:
        return ""
    return f"#{cleaned[:32]}"


def fetch_brazil_realtime_trends(limit: int = 18) -> List[str]:
    """
    Busca tópicos reais em alta no Brasil neste exato momento via Google Trends RSS (sem necessidade de chave paga).
    Faz cache em memória por 15 minutos e retorna lista vazia com segurança se a rede falhar.
    """
    now = time.time()
    cached = _TRENDS_CACHE.get("BR")
    if cached and (now - cached[0]) < _CACHE_TTL_SEC:
        return cached[1][:limit]

    try:
        resp = requests.get(
            GOOGLE_TRENDS_BR_RSS,
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt;q=0.9"},
            timeout=8,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        titles: List[str] = []
        for item in root.findall(".//item"):
            title = str(item.findtext("title") or "").strip()
            if title and title.lower() not in {t.lower() for t in titles}:
                titles.append(title)
        _TRENDS_CACHE["BR"] = (now, titles)
        return titles[:limit]
    except Exception as exc:
        logger.debug("Aviso ao buscar Google Trends BR em tempo real: %s", exc)
        return cached[1][:limit] if cached else []


def suggest_dynamic_hashtags(
    creator_handle: str,
    platform_mode: str = "",
    game_name: str = "",
    title_text: str = "",
    transcript_snippet: str = "",
    max_tags: int = 5,
    base_discovery_tags: Optional[List[str]] = None,
) -> List[str]:
    """
    Monta lista dinâmica de hashtags voltadas para o que pode entrar em alta naquele momento:
      1. Tag do criador (`#alanzoka`, `#davyjones`, `#bistecone`, `#gameplayrj`, `#paulinholoko`).
      2. Tags específicas do jogo/evento em alta (`#gta6`, `#gtarp`, `#cs2`, etc.).
      3. Tópico real do Google Trends Brasil se aparecer no título/fala do corte!
      4. Tags dinâmicas de descoberta viral (`#fy`, `#fyp`, `#viral`, `#clips`).
    """
    tags: List[str] = []

    def _add(tag: str) -> None:
        t = slugify_hashtag(tag)
        if t and t.lower() not in {x.lower() for x in tags} and len(tags) < max_tags:
            tags.append(t)

    _add(creator_handle)

    haystack = f"{game_name} {title_text} {transcript_snippet}".lower()
    game_key = str(game_name or "").strip().lower()
    for alias_k, alias_tags in GAME_HASHTAG_ALIASES.items():
        if alias_k in game_key or alias_k in haystack:
            _add(alias_tags[0])
            break

    # Termos reais em alta agora no Brasil que casem com o contexto do momento
    for trend in fetch_brazil_realtime_trends(limit=20):
        t_clean = trend.strip().lower()
        if len(t_clean) >= 4 and t_clean in haystack:
            _add(trend)

    # Descoberta de algoritmo ForYou / Reels
    for disc in (base_discovery_tags or ["#fy", "#fyp", "#viral", "#cortes"]):
        _add(disc)

    if platform_mode.startswith("youtube"):
        _add("#shorts")
    elif platform_mode.startswith("kick"):
        _add("#kickbrasil")

    return tags[:max_tags]
