from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests

from src.models import MomentCluster, ProfileConfig, TwitchClip
from src.youtube.analyzer import parse_time_range

logger = logging.getLogger(__name__)

KICK_API_V2 = "https://kick.com/api/v2"
KICK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}


def extract_kick_slug(url_or_slug: str) -> str:
    """Extrai o nome do canal (`slug`) a partir de uma URL ou login da Kick."""
    raw = url_or_slug.strip()
    if not raw:
        return ""
    if "://" not in raw and "/" not in raw:
        return raw.lstrip("@").lower()
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    parts = [p for p in parsed.path.split("/") if p]
    return parts[0].lstrip("@").lower() if parts else ""


def parse_kick_url(url: str) -> Dict[str, str]:
    """Classifica links da Kick em `clip`, `vod` ou `channel`."""
    parsed = urlparse(url.strip())
    parts = [p for p in parsed.path.split("/") if p]
    qs = parse_qs(parsed.query)
    slug = parts[0].lstrip("@").lower() if parts else ""

    if qs.get("clip"):
        return {"kind": "clip", "slug": slug, "clip_id": qs["clip"][0]}
    if len(parts) >= 3 and parts[1].lower() in ("clips", "clip"):
        return {"kind": "clip", "slug": slug, "clip_id": parts[2]}
    if len(parts) >= 3 and parts[1].lower() in ("videos", "video", "vod"):
        return {"kind": "vod", "slug": slug, "vod_id": parts[2]}
    return {"kind": "channel", "slug": slug}


def _ms_or_sec_to_seconds(val: Any) -> float:
    try:
        num = float(val or 0)
    except (TypeError, ValueError):
        return 0.0
    # Kick `/videos` retorna duration em ms (ex: 29723000), `/clips` em segundos (ex: 30)
    return num / 1000.0 if num > 86400 else num


class KickMomentAnalyzer:
    """
    Analisador e descobridor de momentos na Kick.com:
      1. Modo Clipes em Alta (`kick_clip`): busca os clipes criados pela comunidade em `/clips?sort=view`.
      2. Modo Ao Vivo (`kick_live`): captura 58s do `playback_url` HLS quando a live está no ar e sem clipes suficientes.
      3. Modo Repescagem/VOD (`kick_vod`): analisa VODs encerrados (`/videos`) com `source` `.m3u8` para extrair momentos.
    """

    def __init__(self, profile: ProfileConfig) -> None:
        self.profile = profile
        self.cfg = profile.detection

    def _get_json(self, path: str) -> Any:
        url = f"{KICK_API_V2}/{path.lstrip('/')}"
        resp = requests.get(url, headers=KICK_HEADERS, timeout=18)
        resp.raise_for_status()
        return resp.json()

    def fetch_channel_info(self, slug: str) -> Dict[str, Any]:
        data = self._get_json(f"channels/{slug}")
        return data if isinstance(data, dict) else {}

    def fetch_channel_clips(self, slug: str, time_window: str = "day", sort: str = "view") -> List[Dict[str, Any]]:
        data = self._get_json(f"channels/{slug}/clips?sort={sort}&time={time_window}")
        if isinstance(data, dict):
            return list(data.get("clips") or [])
        return []

    def fetch_channel_videos(self, slug: str) -> List[Dict[str, Any]]:
        data = self._get_json(f"channels/{slug}/videos")
        return list(data) if isinstance(data, list) else []

    def fetch_single_clip(self, clip_id: str) -> Dict[str, Any]:
        data = self._get_json(f"clips/{clip_id}")
        if isinstance(data, dict):
            return dict(data.get("clip") or data)
        return {}

    def discover_channel_moments(
        self,
        channel_ref: str,
        max_cuts: int = 1,
        processed_state: Optional[Dict[str, Any]] = None,
    ) -> List[MomentCluster]:
        slug = extract_kick_slug(channel_ref)
        if not slug:
            return []

        info = self.fetch_channel_info(slug)
        display_name = str((info.get("user") or {}).get("username") or info.get("slug") or slug)
        livestream = info.get("livestream")
        is_live = bool(livestream)
        playback_url = str(info.get("playback_url") or "").strip()

        seen_clip_ids = self._collect_seen_clip_ids(processed_state)
        clusters: List[MomentCluster] = []

        # 1. Prioriza clipes reais feitos pela comunidade na Kick (`day` + repescagem `week`), ranqueados por qualidade real
        min_views = self.cfg.min_views_single_clip_live if is_live else self.cfg.min_views_post_live
        day_clips = self.fetch_channel_clips(slug, time_window="day", sort="view")
        week_clips = self.fetch_channel_clips(slug, time_window="week", sort="view")
        day_ids = {str(c.get("id") or "").strip() for c in day_clips if c.get("id")}

        merged_clips: List[Dict[str, Any]] = []
        seen_raw: set[str] = set()
        for clip in list(day_clips) + list(week_clips):
            cid = str(clip.get("id") or "").strip()
            if cid and cid not in seen_raw:
                seen_raw.add(cid)
                merged_clips.append(clip)

        recent_cut_times = self._collect_recent_clip_times(slug, processed_state)
        eligible_clips: List[tuple[float, Dict[str, Any]]] = []
        fallback_offline_clips: List[tuple[float, Dict[str, Any]]] = []
        min_dur = max(10.0, self.profile.editing.min_duration_seconds - 4.0)

        for clip in merged_clips:
            cid = str(clip.get("id") or "").strip()
            if not cid or cid in seen_clip_ids:
                continue
            dur = float(clip.get("duration") or 0.0)
            views = int(clip.get("views") or 0)
            if dur < min_dur:
                continue
            c_ts = self._parse_iso(str(clip.get("created_at") or ""))
            if c_ts is not None and any(abs((c_ts - pt).total_seconds()) <= 150.0 for pt in recent_cut_times):
                continue

            # Prefere clipes bem enquadrados (24-65s) a recortes soltos gigantes (110-180s cheios de tempo morto)
            frame_bonus = 60.0 if 24.0 <= dur <= 65.0 else (-45.0 if dur >= 110.0 else 10.0)
            recency_bonus = 30.0 if cid in day_ids else 0.0
            rank_score = float(views * 2.0 + frame_bonus + recency_bonus)

            if views >= min_views:
                eligible_clips.append((rank_score, clip))
            elif not is_live and views >= 40:
                fallback_offline_clips.append((rank_score, clip))

        eligible_clips.sort(key=lambda pair: pair[0], reverse=True)
        fallback_offline_clips.sort(key=lambda pair: pair[0], reverse=True)
        chosen_clips = eligible_clips or (fallback_offline_clips if not is_live else [])
        for rank_score, clip in chosen_clips:
            cid = str(clip.get("id") or "").strip()
            dur = float(clip.get("duration") or 0.0)
            views = int(clip.get("views") or 0)
            stream_url = str(clip.get("clip_url") or clip.get("video_url") or "").strip()
            page_url = f"https://kick.com/{slug}/clips/{cid}"
            if not stream_url:
                stream_url = page_url

            vod_obj = clip.get("vod") or {}
            vod_uuid = str(vod_obj.get("id") or f"kick_{slug}")
            title = str(clip.get("title") or f"Clipe de {display_name}")
            clusters.append(
                self._build_clip_cluster(
                    slug=slug,
                    display_name=display_name,
                    clip_id=cid,
                    title=title,
                    views=views,
                    duration=dur,
                    stream_url=stream_url,
                    page_url=page_url,
                    stream_key=f"kick_{slug}_{vod_uuid}",
                    score=rank_score,
                    created_at=str(clip.get("created_at") or ""),
                )
            )
            if len(clusters) >= max_cuts:
                return clusters

        if clusters:
            return clusters

        # 2. Se está AO VIVO e ainda não teve clipe forte, captura janela ao vivo do HLS
        if is_live and playback_url:
            ls_id = str(livestream.get("id") or slug)
            ls_title = str(livestream.get("session_title") or f"Live na Kick: {display_name}")
            existing_live = sum(
                1 for c in (processed_state or {}).get("cuts", []) if c.get("stream_key") == f"kick_live_{ls_id}"
            )
            if existing_live < 2:
                tag = f"live_{ls_id}_{existing_live}"
                cid = self._make_cluster_id(slug, tag)
                clusters.append(
                    self._build_vod_or_live_cluster(
                        cluster_id=cid,
                        slug=slug,
                        display_name=display_name,
                        stream_key=f"kick_live_{ls_id}",
                        stream_url=playback_url,
                        fallback_url=f"https://kick.com/{slug}",
                        title=ls_title,
                        start_sec=None,
                        end_sec=None,
                        score=95.0,
                        mode="kick_live",
                    )
                )
                return clusters

        # 3. Repescagem de VOD recente caso esteja offline e nenhum clipe bateu o piso de views
        videos = self.fetch_channel_videos(slug)
        for v in videos[:2]:
            vod_id = str((v.get("video") or {}).get("uuid") or v.get("uuid") or v.get("id") or "")
            source_m3u8 = str(v.get("source") or "").strip()
            dur_sec = _ms_or_sec_to_seconds(v.get("duration"))
            if not vod_id or not source_m3u8 or dur_sec < 60.0:
                continue
            s_key = f"kick_vod_{vod_id}"
            already_vod = [c for c in (processed_state or {}).get("cuts", []) if c.get("stream_key") == s_key]
            if len(already_vod) >= 2:
                continue

            slice_idx = len(already_vod) + 1
            anchor = int(dur_sec * (0.35 if slice_idx == 1 else 0.70))
            w0 = max(0, anchor - 28)
            w1 = min(int(dur_sec), w0 + 60)
            cid = self._make_cluster_id(slug, f"vod_{vod_id}_{w0}")
            title = str(v.get("session_title") or f"Live Kick {display_name}")
            clusters.append(
                self._build_vod_or_live_cluster(
                    cluster_id=cid,
                    slug=slug,
                    display_name=display_name,
                    stream_key=s_key,
                    stream_url=source_m3u8,
                    fallback_url=f"https://kick.com/{slug}/videos/{vod_id}",
                    title=title,
                    start_sec=w0,
                    end_sec=w1,
                    score=86.0 - slice_idx * 3.0,
                    mode="kick_vod",
                )
            )
            if len(clusters) >= max_cuts:
                break

        return clusters

    def discover_manual_url(
        self,
        url: str,
        max_cuts: int = 3,
        time_range_str: str = "",
        processed_state: Optional[Dict[str, Any]] = None,
    ) -> List[MomentCluster]:
        parsed = parse_kick_url(url)
        slug = parsed.get("slug") or extract_kick_slug(url)
        kind = parsed.get("kind", "channel")
        user_range = parse_time_range(time_range_str)

        if kind == "clip":
            clip_id = parsed["clip_id"]
            clip = self.fetch_single_clip(clip_id)
            if not clip:
                return []
            c_slug = str((clip.get("channel") or {}).get("slug") or slug or "kick")
            c_user = str((clip.get("creator") or {}).get("username") or (clip.get("channel") or {}).get("username") or c_slug)
            stream_url = str(clip.get("clip_url") or clip.get("video_url") or url)
            return [
                self._build_clip_cluster(
                    slug=c_slug.lower(),
                    display_name=c_user,
                    clip_id=clip_id,
                    title=str(clip.get("title") or f"Clipe Kick {c_slug}"),
                    views=int(clip.get("views") or 100),
                    duration=float(clip.get("duration") or 45.0),
                    stream_url=stream_url,
                    page_url=url,
                    stream_key=f"kick_clip_{clip_id}",
                    score=99.0,
                )
            ]

        if kind == "vod":
            vod_id = parsed["vod_id"]
            videos = self.fetch_channel_videos(slug) if slug else []
            target = next(
                (
                    v
                    for v in videos
                    if str((v.get("video") or {}).get("uuid") or "") == vod_id
                    or str(v.get("uuid") or "") == vod_id
                    or str(v.get("id") or "") == vod_id
                ),
                None,
            )
            if not target or not target.get("source"):
                return []
            dur_sec = max(60.0, _ms_or_sec_to_seconds(target.get("duration")))
            source_m3u8 = str(target["source"])
            title = str(target.get("session_title") or f"VOD Kick {slug}")
            s_key = f"kick_vod_{vod_id}"

            if user_range:
                r0, r1 = user_range
                r0 = max(0, min(int(dur_sec) - 15, r0))
                r1 = min(int(dur_sec), max(r0 + 20, min(r0 + 70, r1)))
                cid = self._make_cluster_id(slug, f"man_{vod_id}_{r0}")
                return [
                    self._build_vod_or_live_cluster(
                        cluster_id=cid,
                        slug=slug,
                        display_name=slug,
                        stream_key=s_key,
                        stream_url=source_m3u8,
                        fallback_url=url,
                        title=title,
                        start_sec=r0,
                        end_sec=r1,
                        score=99.0,
                        mode="kick_vod",
                    )
                ]

            n_cuts = max(1, min(6, int(max_cuts)))
            step = dur_sec / (n_cuts + 1)
            out: List[MomentCluster] = []
            for i in range(1, n_cuts + 1):
                mid = int(step * i)
                w0 = max(0, mid - 28)
                w1 = min(int(dur_sec), w0 + 60)
                cid = self._make_cluster_id(slug, f"man_{vod_id}_{w0}")
                out.append(
                    self._build_vod_or_live_cluster(
                        cluster_id=cid,
                        slug=slug,
                        display_name=slug,
                        stream_key=s_key,
                        stream_url=source_m3u8,
                        fallback_url=url,
                        title=title,
                        start_sec=w0,
                        end_sec=w1,
                        score=90.0 - i * 2.0,
                        mode="kick_vod",
                    )
                )
            return out

        return self.discover_channel_moments(slug, max_cuts=max_cuts, processed_state=processed_state)

    @staticmethod
    def _parse_iso(ts: str) -> Optional[datetime]:
        raw = ts.strip()
        if not raw:
            return None
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def _collect_recent_clip_times(
        self,
        slug: str,
        processed_state: Optional[Dict[str, Any]],
    ) -> List[datetime]:
        times: List[datetime] = []
        for c in (processed_state or {}).get("cuts", []):
            if str(c.get("twitch_login") or "").lower() != slug:
                continue
            dt = self._parse_iso(str(c.get("clip_created_at") or ""))
            if dt is not None:
                times.append(dt)
        return times

    @staticmethod
    def _collect_seen_clip_ids(processed_state: Optional[Dict[str, Any]]) -> set[str]:
        seen: set[str] = set()
        for c in (processed_state or {}).get("cuts", []):
            for cid in c.get("clip_ids") or []:
                seen.add(str(cid))
            if c.get("cut_id"):
                seen.add(str(c["cut_id"]))
        return seen

    def _make_cluster_id(self, slug: str, tag: str) -> str:
        digest = hashlib.sha1(f"kick:{self.profile.id}:{slug}:{tag}".encode("utf-8")).hexdigest()[:10]
        return f"kick_{digest}"

    def _build_clip_cluster(
        self,
        slug: str,
        display_name: str,
        clip_id: str,
        title: str,
        views: int,
        duration: float,
        stream_url: str,
        page_url: str,
        stream_key: str,
        score: float,
        created_at: str = "",
    ) -> MomentCluster:
        cid = self._make_cluster_id(slug, clip_id)
        tc = TwitchClip(
            id=clip_id,
            url=stream_url,
            title=title,
            view_count=views,
            created_at=created_at,
            duration=duration,
            creator_name=display_name,
            broadcaster_id="kick",
            broadcaster_name=display_name,
            video_id=page_url,
            vod_offset=0,
        )
        clean_slug = re.sub(r"[^\w]", "", slug).lower() or self.profile.twitch_login
        return MomentCluster(
            cluster_id=cid,
            profile_id=self.profile.id,
            broadcaster_login=clean_slug,
            stream_or_vod_key=stream_key,
            mode="kick_clip",
            clips=[tc],
            video_id=clip_id,
            vod_start_offset=0,
            vod_end_offset=int(round(duration)),
            peak_offset=int(round(duration * 0.5)) if duration <= 65.0 else None,
            unique_clippers=1,
            total_views=views,
            max_views=views,
            score=score,
            dominant_title=title,
            game_name="Kick",
            stream_title=title,
        )

    def _build_vod_or_live_cluster(
        self,
        cluster_id: str,
        slug: str,
        display_name: str,
        stream_key: str,
        stream_url: str,
        fallback_url: str,
        title: str,
        start_sec: Optional[int],
        end_sec: Optional[int],
        score: float,
        mode: str,
    ) -> MomentCluster:
        dur = float((end_sec - start_sec) if (start_sec is not None and end_sec is not None) else 58.0)
        peak = ((start_sec + end_sec) // 2) if (start_sec is not None and end_sec is not None) else None
        tc = TwitchClip(
            id=cluster_id,
            url=stream_url,
            title=title,
            view_count=int(score * 10),
            created_at="",
            duration=dur,
            creator_name=display_name,
            broadcaster_id="kick",
            broadcaster_name=display_name,
            video_id=fallback_url,
            vod_offset=start_sec,
        )
        clean_slug = re.sub(r"[^\w]", "", slug).lower() or self.profile.twitch_login
        return MomentCluster(
            cluster_id=cluster_id,
            profile_id=self.profile.id,
            broadcaster_login=clean_slug,
            stream_or_vod_key=stream_key,
            mode=mode,
            clips=[tc],
            video_id=stream_key,
            vod_start_offset=start_sec,
            vod_end_offset=end_sec,
            peak_offset=peak,
            unique_clippers=1,
            total_views=int(score * 10),
            max_views=int(score * 10),
            score=score,
            dominant_title=title,
            game_name="Kick",
            stream_title=title,
        )
