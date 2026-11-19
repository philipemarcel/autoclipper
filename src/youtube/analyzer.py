from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.ai.deduplicator import find_duplicate_candidate
from src.ffmpeg_utils import get_ffmpeg_bin
from src.models import MomentCluster, ProfileConfig, TwitchClip
from src.youtube.ytdlp_opts import get_ytdlp_youtube_args

logger = logging.getLogger(__name__)


def parse_time_point(val: str) -> Optional[int]:
    """Converte '12:30', '01:15:20' ou '750' em segundos."""
    s = val.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    parts = [p.strip() for p in s.split(":")]
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def parse_time_range(range_str: str) -> Optional[Tuple[int, int]]:
    """
    Interpreta um intervalo opcional enviado no Discord `/cortar`:
      - '12:30-15:00' -> (750, 900)
      - '10:00' -> (575, 650) (janela centrada em 10:00)
    """
    clean = range_str.strip()
    if not clean:
        return None
    if "-" in clean:
        left, right = clean.split("-", 1)
        s0 = parse_time_point(left)
        s1 = parse_time_point(right)
        if s0 is not None and s1 is not None and s1 > s0:
            return (max(0, s0), s1)
    single = parse_time_point(clean)
    if single is not None:
        return (max(0, single - 15), single + 50)
    return None


class YouTubeMomentAnalyzer:
    """
    Identifica os momentos mais engajados/virais de qualquer vídeo ou live do YouTube:
      1. Se o usuário informou um `time_range` específico no Discord (ex: `12:10-13:00` ou `25:00`),
         foca cirurgicamente nessa janela.
      2. Se o vídeo possui dados de **Most Replayed (`heatmap`)** do YouTube, encontra os picos locais
         onde o público mais voltou para assistir.
      3. Se for um vídeo novo ou live sem heatmap, extrai as legendas automáticas do YouTube (`json3`)
         e usa o LLM (Groq 120B / Gemini) para escolher os `max_cuts` trechos mais engraçados/virais.
      4. Se for uma **Live do YouTube em andamento (`is_live`)**, captura os últimos momentos ao vivo.
    """

    def __init__(self, profile: ProfileConfig) -> None:
        self.profile = profile
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_DIRECTOR_MODEL", "openai/gpt-oss-120b").strip()

    def fetch_metadata(self, url: str, work_dir: Path) -> Dict[str, Any]:
        work_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ffmpeg-location",
            get_ffmpeg_bin(),
            "--no-playlist",
            "--dump-single-json",
            *get_ytdlp_youtube_args(work_dir),
            url.strip(),
        ]
        logger.info("Extraindo metadados do YouTube: %s", url)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode != 0:
            err_msg = (proc.stderr or proc.stdout or "").strip()[-400:]
            raise RuntimeError(f"yt-dlp metadata falhou ({proc.returncode}): {err_msg}")
        return json.loads(proc.stdout)

    def discover_channel_recent_urls(self, channel_url: str, max_items: int = 2, work_dir: Optional[Path] = None) -> List[str]:
        """
        Dado um canal do YouTube (ex: `https://www.youtube.com/@Gameplayrj`), verifica automaticamente:
          1. A aba `/streams` (Lives ao vivo + Repescagem das últimas lives encerradas!)
          2. A aba `/videos` (Vídeos novos publicados no canal!)
        Usa extração HTTP rápida + feed RSS oficial (evitando timeouts do yt-dlp em canais grandes).
        """
        base = channel_url.strip().rstrip("/")
        if base.endswith(("/videos", "/streams", "/shorts")):
            tabs = [base]
        else:
            tabs = [f"{base}/streams", f"{base}/videos"]

        urls: List[str] = []
        channel_id: Optional[str] = None
        req_headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8"}

        for tab_url in tabs:
            tab_found = 0
            try:
                resp = requests.get(tab_url, headers=req_headers, timeout=15)
                if resp.ok:
                    if not channel_id:
                        cid_m = re.search(r'"externalId":"(UC[A-Za-z0-9_-]{20,})"', resp.text)
                        if cid_m:
                            channel_id = cid_m.group(1)
                    for vid in re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', resp.text):
                        watch_url = f"https://www.youtube.com/watch?v={vid}"
                        if watch_url not in urls:
                            urls.append(watch_url)
                            tab_found += 1
                            if tab_found >= max(1, max_items):
                                break
            except Exception as exc:
                logger.warning("Aviso ao consultar aba YouTube %s via HTTP: %s", tab_url, exc)

            if tab_found > 0:
                continue

            # Fallback via yt-dlp apenas se HTTP direto não retornou vídeos nesta aba
            yt_args = get_ytdlp_youtube_args(work_dir)
            cmd = [
                sys.executable,
                "-m",
                "yt_dlp",
                "--flat-playlist",
                "--playlist-end",
                str(max(1, max_items)),
                "--dump-single-json",
                *yt_args,
                tab_url,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
                data = json.loads(proc.stdout)
                for entry in data.get("entries") or []:
                    vid = entry.get("id")
                    e_url = entry.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
                    if e_url and "youtube.com" not in e_url and vid:
                        e_url = f"https://www.youtube.com/watch?v={vid}"
                    if e_url and e_url not in urls:
                        urls.append(e_url)
            except Exception as exc:
                logger.warning("Aviso ao listar aba YouTube %s: %s", tab_url, exc)

        if not urls and channel_id:
            try:
                rss = requests.get(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}", timeout=12)
                if rss.ok:
                    for vid in re.findall(r"<yt:videoId>([A-Za-z0-9_-]{11})</yt:videoId>", rss.text)[: max(1, max_items)]:
                        watch_url = f"https://www.youtube.com/watch?v={vid}"
                        if watch_url not in urls:
                            urls.append(watch_url)
            except Exception:
                pass
        return urls

    def discover_moments(
        self,
        url: str,
        max_cuts: int,
        work_dir: Path,
        time_range_str: str = "",
        processed_state: Optional[Dict[str, Any]] = None,
        max_age_hours: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], List[MomentCluster]]:
        info = self.fetch_metadata(url, work_dir)
        video_id = str(info.get("id") or hashlib.sha1(url.encode("utf-8")).hexdigest()[:10])
        title = str(info.get("title") or "YouTube Video")
        handle_match = re.search(r"/@([A-Za-z0-9._-]+)", url)
        url_handle = handle_match.group(1) if handle_match else ""
        uploader = str(info.get("uploader") or info.get("channel") or info.get("uploader_id") or url_handle or "YouTube")
        is_live = bool(info.get("is_live"))
        duration = float(info.get("duration") or 0.0)

        user_range = parse_time_range(time_range_str)

        if not is_live and not user_range and max_age_hours is not None:
            ts = info.get("release_timestamp") or info.get("timestamp")
            if ts:
                import time as _time

                age_h = (_time.time() - float(ts)) / 3600.0
                if age_h > float(max_age_hours):
                    logger.info("Pulando repescagem YouTube antiga (%.1fh > %dh): %s", age_h, max_age_hours, title)
                    return info, []

        # ------------------------------------------------------------------
        # CASO 1: LIVE DO YOUTUBE EM ANDAMENTO (`is_live=True`)
        # ------------------------------------------------------------------
        if is_live and not user_range:
            logger.info("YouTube Live em andamento detectada (%s)! Preparando captura ao vivo...", title)
            clusters: List[MomentCluster] = []
            for idx in range(min(max_cuts, 2)):
                cid = self._make_cluster_id(video_id, f"live_{idx}_{int(duration)}")
                clusters.append(
                    self._build_cluster(
                        cluster_id=cid,
                        video_id=video_id,
                        url=url,
                        title=title,
                        uploader=uploader,
                        start_sec=None,
                        end_sec=None,
                        score=95.0 - idx * 5.0,
                        mode="youtube_live",
                        moment_label=f"Ao Vivo: {title[:45]}",
                    )
                )
            return info, clusters

        # ------------------------------------------------------------------
        # CASO 2: TRECHO ESPECÍFICO PEDIDO PELO USUÁRIO NO DISCORD (`/cortar ... trecho:10:00-14:00`)
        # ------------------------------------------------------------------
        if user_range:
            r_start, r_end = user_range
            if duration > 0:
                r_start = min(int(duration) - 10, max(0, r_start))
                r_end = min(int(duration), max(r_start + 18, r_end))
            span = r_end - r_start

            # Se o usuário passou uma janela curta (<= 95s), gera 1 corte direto dessa janela!
            if span <= 95 or max_cuts == 1:
                cid = self._make_cluster_id(video_id, f"range_{r_start}_{r_end}")
                return info, [
                    self._build_cluster(
                        cluster_id=cid,
                        video_id=video_id,
                        url=url,
                        title=title,
                        uploader=uploader,
                        start_sec=r_start,
                        end_sec=min(r_start + 75, r_end),
                        score=99.0,
                        mode="youtube_manual",
                        moment_label=title[:50],
                    )
                ]

            # Se passou uma janela maior (ex: 10:00-25:00 com 3 cortes), busca os melhores picos dentro dela
            windows = self._find_moments_in_bounds(info, url, work_dir, r_start, r_end, max_cuts)
        else:
            # ------------------------------------------------------------------
            # CASO 3: VÍDEO COMPLETO / VOD — BUSCA AUTOMÁTICA DOS TOP N MOMENTOS
            # ------------------------------------------------------------------
            total_dur = int(duration) if duration > 0 else 1800
            windows = self._find_moments_in_bounds(info, url, work_dir, 0, total_dur, max_cuts)

        existing_anchors = [
            int(c["vod_anchor_sec"])
            for c in (processed_state or {}).get("cuts", [])
            if c.get("video_id") == f"yt_{video_id}" and c.get("vod_anchor_sec") is not None
        ]

        clusters = []
        accepted_records: List[Dict[str, Any]] = []
        base_cuts = list((processed_state or {}).get("cuts", []))
        for w_start, w_end, w_score, w_label in windows:
            center = (w_start + w_end) // 2
            if not user_range and any(abs(center - ex) < 40 for ex in existing_anchors):
                continue
            cid = self._make_cluster_id(video_id, f"sec_{w_start}")
            cand_rec = {
                "cut_id": cid,
                "profile_id": self.profile.id,
                "twitch_login": uploader,
                "creator_name": uploader,
                "stream_key": f"yt_{video_id}",
                "video_id": f"yt_{video_id}",
                "vod_anchor_sec": w_start,
                "total_duration": float(max(20, w_end - w_start)),
                "mode": "youtube_auto",
                "dominant_title": w_label or title[:50],
            }
            if not user_range:
                dup = find_duplicate_candidate(
                    cand_rec,
                    {"cuts": base_cuts + accepted_records},
                    anchor_proximity_sec=45.0,
                )
                if dup.is_duplicate:
                    continue
            accepted_records.append(cand_rec)
            clusters.append(
                self._build_cluster(
                    cluster_id=cid,
                    video_id=video_id,
                    url=url,
                    title=title,
                    uploader=uploader,
                    start_sec=w_start,
                    end_sec=w_end,
                    score=w_score,
                    mode="youtube_auto",
                    moment_label=w_label or title[:50],
                )
            )
            if len(clusters) >= max_cuts:
                break

        return info, clusters

    def _find_moments_in_bounds(
        self,
        info: Dict[str, Any],
        url: str,
        work_dir: Path,
        min_sec: int,
        max_sec: int,
        max_cuts: int,
    ) -> List[Tuple[int, int, float, str]]:
        """Retorna lista de `(start_sec, end_sec, score, label)`."""
        # 1. Tenta o Heatmap Oficial ("Most Replayed" do YouTube)
        heatmap = info.get("heatmap") or []
        if heatmap:
            hm_windows = self._extract_heatmap_peaks(heatmap, min_sec, max_sec, max_cuts, info.get("chapters") or [])
            if hm_windows:
                logger.info("YouTube Heatmap ('Most Replayed') encontrado! %d pico(s) extraído(s).", len(hm_windows))
                return hm_windows

        # 2. Tenta Legendas Automáticas do YouTube + Diretor LLM (Groq 120B) para achar momentos mais engraçados
        ai_windows = self._extract_ai_transcript_peaks(info, url, work_dir, min_sec, max_sec, max_cuts)
        if ai_windows:
            logger.info("IA encontrou %d momento(s) virais pela transcrição do YouTube.", len(ai_windows))
            return ai_windows

        # 3. Fallback por Chapters ou divisão uniforme inteligente
        chapters = info.get("chapters") or []
        if chapters:
            ch_windows: List[Tuple[int, int, float, str]] = []
            for idx, ch in enumerate(chapters):
                cs = int(float(ch.get("start_time", 0)))
                ce = int(float(ch.get("end_time", cs + 60)))
                if ce <= min_sec or cs >= max_sec:
                    continue
                w0 = max(min_sec, cs)
                w1 = min(max_sec, w0 + 62)
                ch_windows.append((w0, w1, round(88.0 - idx * 2.0, 1), str(ch.get("title") or "")))
                if len(ch_windows) >= max_cuts:
                    break
            if ch_windows:
                return ch_windows

        # 4. Divisão uniforme segura da janela
        span = max(30, max_sec - min_sec)
        if span <= 70:
            return [(min_sec, max_sec, 85.0, str(info.get("title") or ""))]

        step = span / (max_cuts + 1)
        fallback_wins: List[Tuple[int, int, float, str]] = []
        for i in range(1, max_cuts + 1):
            center = int(min_sec + step * i)
            w0 = max(min_sec, center - 28)
            w1 = min(max_sec, w0 + 58)
            fallback_wins.append((w0, w1, round(85.0 - i * 3.0, 1), str(info.get("title") or "")))
        return fallback_wins

    @staticmethod
    def _extract_heatmap_peaks(
        heatmap: List[Dict[str, Any]],
        min_sec: int,
        max_sec: int,
        max_cuts: int,
        chapters: List[Dict[str, Any]],
    ) -> List[Tuple[int, int, float, str]]:
        points = []
        for pt in heatmap:
            st = float(pt.get("start_time", 0.0))
            en = float(pt.get("end_time", st + 5.0))
            val = float(pt.get("value", 0.0))
            mid = (st + en) / 2.0
            if min_sec <= mid <= max_sec and val > 0.05:
                points.append((mid, val))

        if not points:
            return []

        # Ordena pelos maiores picos de replay
        points.sort(key=lambda x: x[1], reverse=True)
        selected: List[Tuple[int, int, float, str]] = []
        used_centers: List[float] = []

        for mid, val in points:
            # Garante distância mínima de 65s entre dois cortes diferentes
            if any(abs(mid - uc) < 65.0 for uc in used_centers):
                continue

            # Enquadra ~24s de contexto antes do pico e ~36s durante/após o pico (janela completa de 60s!)
            w0 = max(min_sec, int(round(mid - 24)))
            w1 = min(max_sec, w0 + 60)
            if w1 - w0 < 20:
                continue

            # Descobre nome do capítulo se houver
            ch_label = ""
            for ch in chapters:
                if float(ch.get("start_time", 0)) <= mid <= float(ch.get("end_time", 0)):
                    ch_label = str(ch.get("title") or "")
                    break

            score = round(100.0 + val * 300.0, 1)
            selected.append((w0, w1, score, ch_label))
            used_centers.append(mid)
            if len(selected) >= max_cuts * 2:
                break

        return selected

    def _extract_ai_transcript_peaks(
        self,
        info: Dict[str, Any],
        url: str,
        work_dir: Path,
        min_sec: int,
        max_sec: int,
        max_cuts: int,
    ) -> List[Tuple[int, int, float, str]]:
        if not self.groq_api_key:
            return []

        sub_lines = self._fetch_youtube_auto_captions(info, min_sec, max_sec)
        if not sub_lines:
            return []

        transcript_sample = self._sample_transcript_for_ai(sub_lines, max_chars=24000)

        prompt = f"""
Você é um editor viral de páginas de cortes no TikTok.
Analise a transcrição abaixo do vídeo do YouTube "{info.get('title', '')}" e escolha os {max_cuts} momentos MAIS ENGRAÇADOS, AUTO-CONTIDOS e DE MAIOR RETENÇÃO entre {min_sec}s e {max_sec}s.

Regras OBRIGATÓRIAS de contexto:
- Escolha histórias/acontecimentos COMPLETOS (introdução do assunto + desenvolvimento + clímax/reação).
- PROIBIDO escolher frases soltas de 2–4 segundos (ex.: elogio rápido a um NPC, saudação, leitura de donate/corações do chat ou silêncio caminhando depois).
- Comece `start_sec` de **12 a 25 segundos ANTES** da fala principal/clímax (na abertura real do assunto!) e termine `end_sec` só depois da reação final, com duração entre **42s e 60s** (`end_sec - start_sec`).
- Os momentos NÃO podem se sobrepor (distância mínima de 65s entre eles).
- `score`: nota inteira ou decimal de `0` a `100` (ex.: `92`).
- `label`: frase curta de 3 a 6 palavras resumindo o assunto completo do corte.

TRANSCRIÇÃO:
{transcript_sample}

Responda ESTRITAMENTE em JSON válido:
{{
  "moments": [
    {{"start_sec": int, "end_sec": int, "score": float, "label": string}}
  ]
}}
""".strip()

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.groq_model,
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "Retorne apenas JSON válido."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=35,
            )
            resp.raise_for_status()
            data = json.loads(resp.json()["choices"][0]["message"]["content"])
            out: List[Tuple[int, int, float, str]] = []
            for m in data.get("moments", []):
                s0 = max(min_sec, int(m.get("start_sec", min_sec)))
                s1 = min(max_sec, int(m.get("end_sec", s0 + 45)))
                if s1 - s0 < 18:
                    s1 = min(max_sec, s0 + 42)
                if s1 - s0 > 62:
                    s1 = s0 + 56
                raw_sc = float(m.get("score", 90.0))
                if raw_sc <= 10.0:
                    raw_sc *= 10.0
                out.append((s0, s1, round(raw_sc, 1), str(m.get("label", ""))))
            return out[:max_cuts]
        except Exception as exc:
            logger.warning("Aviso ao buscar picos por transcrição YouTube via IA: %s", exc)
            return []

    @staticmethod
    def _sample_transcript_for_ai(sub_lines: str, max_chars: int = 24000) -> str:
        """Mantém 100% da transcrição se couber no orçamento; caso seja VOD longo, amostra 4 janelas ao longo de todo o vídeo."""
        if len(sub_lines) <= max_chars:
            return sub_lines
        lines = sub_lines.splitlines()
        if len(lines) <= 40:
            return sub_lines[:max_chars]

        bucket_budget = max_chars // 4
        n = len(lines)
        anchors = [0, n // 4, n // 2, (3 * n) // 4]
        parts: List[str] = []
        for idx, start_i in enumerate(anchors):
            chunk_lines: List[str] = []
            acc = 0
            limit_i = anchors[idx + 1] if idx + 1 < len(anchors) else n
            for ln in lines[start_i:limit_i]:
                if acc + len(ln) + 1 > bucket_budget:
                    break
                chunk_lines.append(ln)
                acc += len(ln) + 1
            if chunk_lines:
                parts.append("\n".join(chunk_lines))
        return "\n...\n".join(parts)

    @staticmethod
    def _fetch_youtube_auto_captions(info: Dict[str, Any], min_sec: int, max_sec: int) -> str:
        """Extrai as legendas `json3` diretamente da URL retornada no metadado do `yt-dlp` (em <1s, sem baixar vídeo!)."""
        subs = info.get("subtitles") or {}
        auto_subs = info.get("automatic_captions") or {}

        chosen_tracks = None
        for lang_key in ("pt", "pt-BR", "pt-PT", "en", "es"):
            if lang_key in subs:
                chosen_tracks = subs[lang_key]
                break
            if lang_key in auto_subs:
                chosen_tracks = auto_subs[lang_key]
                break

        if not chosen_tracks and auto_subs:
            first_key = next(iter(auto_subs.keys()))
            chosen_tracks = auto_subs[first_key]

        if not chosen_tracks:
            return ""

        json3_url = next((t.get("url") for t in chosen_tracks if t.get("ext") == "json3" and t.get("url")), None)
        if not json3_url:
            return ""

        try:
            r = requests.get(json3_url, timeout=15)
            r.raise_for_status()
            payload = r.json()
            lines: List[str] = []
            for ev in payload.get("events", []):
                t_sec = float(ev.get("tStartMs", 0)) / 1000.0
                if t_sec < min_sec or t_sec > max_sec:
                    continue
                segs = ev.get("segs") or []
                txt = "".join(str(s.get("utf8", "")) for s in segs).replace("\n", " ").strip()
                txt = re.sub(r"\s+", " ", txt)
                if txt:
                    lines.append(f"[{int(t_sec)}s] {txt}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _make_cluster_id(self, video_id: str, tag: str) -> str:
        digest = hashlib.sha1(f"yt:{self.profile.id}:{video_id}:{tag}".encode("utf-8")).hexdigest()[:10]
        return f"yt_{digest}"

    def _build_cluster(
        self,
        cluster_id: str,
        video_id: str,
        url: str,
        title: str,
        uploader: str,
        start_sec: Optional[int],
        end_sec: Optional[int],
        score: float,
        mode: str,
        moment_label: str,
    ) -> MomentCluster:
        dur = float((end_sec - start_sec) if (start_sec is not None and end_sec is not None) else 45.0)
        peak = ((start_sec + end_sec) // 2) if (start_sec is not None and end_sec is not None) else None

        pseudo_clip = TwitchClip(
            id=f"yt_{video_id}_{start_sec or 'live'}",
            url=url,
            title=moment_label or title,
            view_count=int(score * 10),
            created_at="",
            duration=dur,
            creator_name=uploader,
            broadcaster_id="youtube",
            broadcaster_name=uploader,
            video_id=f"yt_{video_id}",
            vod_offset=start_sec,
        )

        clean_uploader = re.sub(r"[^\w]", "", uploader).lower() or "youtube"
        return MomentCluster(
            cluster_id=cluster_id,
            profile_id=self.profile.id,
            broadcaster_login=clean_uploader,
            stream_or_vod_key=f"yt_{video_id}",
            mode=mode,
            clips=[pseudo_clip],
            video_id=f"yt_{video_id}",
            vod_start_offset=start_sec,
            vod_end_offset=end_sec,
            peak_offset=peak,
            unique_clippers=1,
            total_views=int(score * 10),
            max_views=int(score * 10),
            score=score,
            dominant_title=moment_label or title,
            game_name="YouTube",
            stream_title=title,
        )
