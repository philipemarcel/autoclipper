from __future__ import annotations

import logging
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

GOOGLE_TRENDS_BR_RSS = "https://trends.google.com/trending/rss?geo=BR"
YT_SEARCH_URL = "https://www.youtube.com/results"
YT_SUGGEST_URL = "https://suggestqueries.google.com/complete/search"

_TRENDS_CACHE: Dict[str, Tuple[float, List[str]]] = {}
_BENCHMARK_CACHE: Dict[str, Tuple[float, Dict[str, List[str]]]] = {}
_CACHE_TTL_SEC = 900.0  # 15 minutos
_BENCHMARK_TTL_SEC = 1800.0  # 30 minutos

GAME_HASHTAG_ALIASES: Dict[str, List[str]] = {
    "grand theft auto v": ["#gtarp", "#gtav"],
    "grand theft auto vi": ["#gta6", "#gta"],
    "gta 6": ["#gta6", "#gta"],
    "counter-strike": ["#cs2"],
    "valorant": ["#valorantbr"],
    "league of legends": ["#lolbr"],
    "just chatting": ["#podcast", "#react"],
    "irl": ["#irl"],
    "fortnite": ["#fortnitebr"],
    "minecraft": ["#minecraftbr"],
    "arc raiders": ["#arcraiders"],
    "dead by daylight": ["#dbd"],
}

# Hashtags 100% de nicho gamer BR (sem tags genéricas vazias como #fyp/#fy/#viral que desperdiçam slots)
GAMING_CLIP_BENCHMARK_TAGS: List[str] = [
    "#cortesdegames",
    "#gamerbrasil",
    "#humorgamer",
    "#clips",
    "#gamesbrasil",
    "#livestream",
]


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


def fetch_web_clip_benchmarks(
    creator_name: str,
    game_name: str = "",
    limit: int = 8,
) -> Dict[str, List[str]]:
    """
    Consulta referências reais de cortes/clips de jogos na web (títulos virais de Shorts/Reels/TikTok,
    hashtags mais usadas nesses clipes e termos reais buscados agora no Autocomplete Web/YouTube).
    """
    clean_creator = re.sub(r"\s+", " ", str(creator_name or "").strip())
    clean_game = re.sub(r"\s+", " ", str(game_name or "").strip())
    if clean_game.lower() in {"live", "youtube", "kick", "twitch", "games"}:
        clean_game = ""
    cache_key = f"{clean_creator.lower()}|{clean_game.lower()}"

    now = time.time()
    cached = _BENCHMARK_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _BENCHMARK_TTL_SEC:
        return {
            "titles": cached[1]["titles"][:limit],
            "hashtags": cached[1]["hashtags"][:12],
            "search_terms": cached[1]["search_terms"][:8],
        }

    titles: List[str] = []
    tag_counter: Counter[str] = Counter()
    search_terms: List[str] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    }

    query_str = " ".join(x for x in ["cortes", clean_creator, clean_game] if x).strip() or "cortes games"
    try:
        r = requests.get(
            YT_SEARCH_URL,
            params={"search_query": query_str},
            headers=headers,
            timeout=7,
        )
        if r.ok:
            shorts_hits = re.findall(r'"overlayMetadata":\{"primaryText":\{"content":"([^"]{8,120})"\}', r.text)
            video_hits = re.findall(
                r'"videoRenderer":\{"videoId":"[^"]+","thumbnail":.*?"title":\{"runs":\[\{"text":"([^"]{8,120})"\}',
                r.text,
            )
            for raw_t in shorts_hits + video_hits:
                clean_t = raw_t.encode("utf-8", "ignore").decode("utf-8").replace('\\"', '"').strip()
                if not clean_t or any(bad in clean_t.lower() for bad in ("inscreva-se", "filtros de pesquisa", "atalhos do teclado")):
                    continue
                for ht in re.findall(r"#[A-Za-zÀ-ÿ0-9_]{2,30}", clean_t):
                    norm_ht = slugify_hashtag(ht)
                    if norm_ht and norm_ht not in {"#shorts", "#youtubeshorts", "#short"}:
                        tag_counter[norm_ht] += 1
                # Guarda versão limpa do título sem excesso de # no fim para servir de referência de gancho
                no_tags_t = re.sub(r"\s*#[A-Za-zÀ-ÿ0-9_]+", "", clean_t).strip()
                if len(no_tags_t) >= 12 and no_tags_t not in titles:
                    titles.append(no_tags_t[:95])
    except Exception as exc:
        logger.debug("Aviso ao buscar referências web de cortes (%s): %s", query_str, exc)

    for ds, q_val in [("yt", f"{clean_creator} {clean_game}".strip()), ("", f"{clean_creator} tiktok".strip())]:
        if not q_val:
            continue
        try:
            params = {"client": "firefox", "q": q_val}
            if ds:
                params["ds"] = ds
            sr = requests.get(YT_SUGGEST_URL, params=params, timeout=5)
            if sr.ok:
                data = sr.json()
                if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
                    for term in data[1][:6]:
                        t_s = str(term).strip()
                        if t_s and t_s not in search_terms:
                            search_terms.append(t_s)
        except Exception:
            pass

    top_hashtags = [t for t, _ in tag_counter.most_common(12)]
    result = {
        "titles": titles[:10],
        "hashtags": top_hashtags,
        "search_terms": search_terms[:10],
    }
    _BENCHMARK_CACHE[cache_key] = (now, result)
    return {
        "titles": result["titles"][:limit],
        "hashtags": result["hashtags"][:12],
        "search_terms": result["search_terms"][:8],
    }


def suggest_dynamic_hashtags(
    creator_handle: str,
    platform_mode: str = "",
    game_name: str = "",
    title_text: str = "",
    transcript_snippet: str = "",
    max_tags: int = 5,
    base_discovery_tags: Optional[List[str]] = None,
    web_benchmark_tags: Optional[List[str]] = None,
) -> List[str]:
    """
    Monta lista dinâmica de hashtags comparando tendências reais + benchmarks de cortes de jogos na web:
      1. Tag do criador (`#alanzoka`, `#davyjones`, `#bistecone`, `#gameplayrj`, `#paulinholoko`).
      2. Tag específica do jogo/assunto (`#gta6`, `#gtarp`, `#cs2`, etc.).
      3. Tópico real do Google Trends Brasil se aparecer no título/fala do corte.
      4. Hashtag viral extraída de benchmarks reais na web (ex.: `#cortesdegames`, `#gamerbrasil`, `#humorgamer`).
      5. Tags dinâmicas de descoberta no TikTok/Reels (`#fyp`, `#fy`, `#viral`).
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

    # Hashtags de benchmarks reais na web de clipes parecidos
    for wtag in (web_benchmark_tags or []):
        _add(wtag)

    # Nicho gamer BR + ForYou / Reels
    for disc in (base_discovery_tags or GAMING_CLIP_BENCHMARK_TAGS):
        _add(disc)

    for fallback_tag in GAMING_CLIP_BENCHMARK_TAGS:
        _add(fallback_tag)

    if platform_mode.startswith("kick"):
        _add("#kickbrasil")

    return tags[:max_tags]
