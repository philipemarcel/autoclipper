from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.models import MomentCluster, ProfileConfig, TwitchClip
from src.twitch.client import TwitchHelixClient, parse_twitch_duration_seconds

logger = logging.getLogger(__name__)


def _parse_iso_utc(ts: str) -> datetime:
    cleaned = ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _format_iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MomentDetector:
    """
    Detector em Modo Duplo:
      1. `live_hot`: enquanto a live está online, detecta clusters explosivos de clipes no mesmo minuto
         ou picos de velocidade de views em tempo real.
      2. `post_live`: nas lives encerradas nas últimas 36h, faz a repescagem dos momentos mais vistos
         sem duplicar trechos já cortados durante a live.
    """

    def __init__(self, profile: ProfileConfig) -> None:
        self.profile = profile
        self.cfg = profile.detection

    def discover_candidates(
        self,
        helix: TwitchHelixClient,
        processed_state: Dict[str, Any],
        now_utc: Optional[datetime] = None,
    ) -> List[MomentCluster]:
        now = now_utc or datetime.now(timezone.utc)
        user = helix.get_user_by_login(self.profile.twitch_login)
        if not user:
            logger.warning("Canal Twitch '%s' não encontrado.", self.profile.twitch_login)
            return []

        broadcaster_id = str(user["id"])
        active_stream = helix.get_active_stream(broadcaster_id)
        recent_vods = helix.get_recent_vods(broadcaster_id, first=4)

        candidates: List[MomentCluster] = []

        # ------------------------------------------------------------------
        # MODO 1: QUENTE (LIVE AO VIVO)
        # ------------------------------------------------------------------
        if active_stream:
            stream_id = str(active_stream["id"])
            started_at = str(active_stream["started_at"])
            game_name = str(active_stream.get("game_name", ""))
            stream_title = str(active_stream.get("title", ""))

            # Se houver VOD publicado da live atual (geralmente o 1º VOD cujo stream_id bate)
            matching_vod_id = ""
            for v in recent_vods:
                if str(v.get("stream_id", "")) == stream_id:
                    matching_vod_id = str(v["id"])
                    break

            stream_key = matching_vod_id or f"stream_{stream_id}"
            already_cut_count = self._count_cuts_for_stream(processed_state, stream_key, stream_id)

            if already_cut_count < self.cfg.max_cuts_per_stream:
                clips = helix.get_clips_in_window(
                    broadcaster_id=broadcaster_id,
                    started_at_iso=started_at,
                    ended_at_iso=_format_iso_utc(now),
                )
                clusters = self.cluster_clips(
                    clips=clips,
                    stream_key=stream_key,
                    mode="live_hot",
                    now_utc=now,
                    game_name=game_name,
                    stream_title=stream_title,
                )
                filtered = self._filter_and_deduplicate(
                    clusters=clusters,
                    processed_state=processed_state,
                    stream_key=stream_key,
                    alt_stream_key=f"stream_{stream_id}",
                    remaining_slots=self.cfg.max_cuts_per_stream - already_cut_count,
                )
                candidates.extend(filtered)

        # ------------------------------------------------------------------
        # MODO 2: REPESCAGEM PÓS-LIVE (VODs recentes das últimas 36h)
        # ------------------------------------------------------------------
        active_stream_id = str(active_stream["id"]) if active_stream else ""
        cutoff = now - timedelta(hours=36)

        for vod in recent_vods:
            vod_stream_id = str(vod.get("stream_id", ""))
            if active_stream_id and vod_stream_id == active_stream_id:
                # Essa é a live que ainda está ao vivo; já foi tratada no Modo Quente
                continue

            vod_created = _parse_iso_utc(str(vod["created_at"]))
            if vod_created < cutoff:
                continue

            vod_id = str(vod["id"])
            vod_duration_sec = parse_twitch_duration_seconds(str(vod.get("duration", "")))
            vod_end = vod_created + timedelta(seconds=max(vod_duration_sec, 600) + 1800)
            stream_title = str(vod.get("title", ""))

            already_cut_count = self._count_cuts_for_stream(
                processed_state, vod_id, f"stream_{vod_stream_id}" if vod_stream_id else ""
            )
            if already_cut_count >= self.cfg.max_cuts_per_stream:
                continue

            clips = helix.get_clips_in_window(
                broadcaster_id=broadcaster_id,
                started_at_iso=_format_iso_utc(vod_created - timedelta(minutes=5)),
                ended_at_iso=_format_iso_utc(min(now, vod_end + timedelta(hours=12))),
            )
            # Filtra clipes pertencentes a este VOD (ou criados durante a janela exata da live caso VOD esteja oculto)
            vod_clips = [
                c
                for c in clips
                if c.video_id == vod_id
                or (not c.video_id and vod_created <= _parse_iso_utc(c.created_at) <= vod_end)
            ]
            if not vod_clips:
                continue

            game_name = helix.get_game_name(vod_clips[0].game_id) if vod_clips[0].game_id else ""
            clusters = self.cluster_clips(
                clips=vod_clips,
                stream_key=vod_id,
                mode="post_live",
                now_utc=now,
                game_name=game_name,
                stream_title=stream_title,
            )
            filtered = self._filter_and_deduplicate(
                clusters=clusters,
                processed_state=processed_state,
                stream_key=vod_id,
                alt_stream_key=f"stream_{vod_stream_id}" if vod_stream_id else "",
                remaining_slots=self.cfg.max_cuts_per_stream - already_cut_count,
            )
            candidates.extend(filtered)

        # ------------------------------------------------------------------
        # MODO 3: PÁGINA DE CLIPES DO CANAL (Top Clipes 7 dias / 24h como referência direta)
        # Pega clipes em destaque na página da Twitch mesmo quando vod_offset é null ou fora dos VODs acima
        # ------------------------------------------------------------------
        try:
            top_page_clips = helix.get_clips_in_window(
                broadcaster_id=broadcaster_id,
                started_at_iso=_format_iso_utc(now - timedelta(days=7)),
                ended_at_iso=_format_iso_utc(now),
                max_pages=3,
            )
            if top_page_clips:
                page_clusters = self.cluster_clips(
                    clips=top_page_clips,
                    stream_key="twitch_top_7d",
                    mode="post_live",
                    now_utc=now,
                    game_name="",
                    stream_title="Top Clipes da Página Twitch",
                )
                already_grabbed_ids = {c.id for cl in candidates for c in cl.clips}
                page_clusters = [
                    pc for pc in page_clusters if not ({c.id for c in pc.clips} & already_grabbed_ids)
                ]
                page_filtered = self._filter_and_deduplicate(
                    clusters=page_clusters,
                    processed_state=processed_state,
                    stream_key="twitch_top_7d",
                    alt_stream_key="",
                    remaining_slots=self.cfg.max_cuts_per_run,
                )
                candidates.extend(page_filtered)
        except Exception as exc:
            logger.warning("Aviso ao consultar página de Top Clipes 7d da Twitch: %s", exc)

        # Prioriza momentos da live AO VIVO (`live_hot`) sobre repescagem de lives passadas (`post_live`)
        candidates.sort(key=lambda c: (1 if c.mode == "live_hot" else 0, c.score), reverse=True)
        return candidates[: self.cfg.max_cuts_per_run]

    def cluster_clips(
        self,
        clips: List[TwitchClip],
        stream_key: str,
        mode: str,
        now_utc: datetime,
        game_name: str = "",
        stream_title: str = "",
    ) -> List[MomentCluster]:
        if not clips:
            return []

        # Deduplica clipes por ID (a paginação ao vivo da Twitch pode repetir um clipe entre páginas)
        dedup_by_id: Dict[str, TwitchClip] = {}
        for c in clips:
            if c.id not in dedup_by_id or c.view_count > dedup_by_id[c.id].view_count:
                dedup_by_id[c.id] = c
        unique_clips = list(dedup_by_id.values())

        with_offset = [c for c in unique_clips if c.vod_offset is not None]
        without_offset = [c for c in unique_clips if c.vod_offset is None]

        raw_groups: List[List[TwitchClip]] = []

        # 1. Clusterização precisa por posição no VOD (`vod_offset`)
        if with_offset:
            with_offset.sort(key=lambda c: int(c.vod_offset or 0))
            current_group = [with_offset[0]]
            group_anchor = int(with_offset[0].vod_offset or 0)

            for clip in with_offset[1:]:
                offset = int(clip.vod_offset or 0)
                if offset - group_anchor <= self.cfg.cluster_window_seconds:
                    current_group.append(clip)
                else:
                    raw_groups.append(current_group)
                    current_group = [clip]
                    group_anchor = offset
            raw_groups.append(current_group)

        # 2. Clusterização por `created_at` para clipes fresquíssimos (vod_offset ainda null ou VOD fechado)
        if without_offset:
            without_offset.sort(key=lambda c: _parse_iso_utc(c.created_at))
            current_group = [without_offset[0]]
            group_time = _parse_iso_utc(without_offset[0].created_at)

            for clip in without_offset[1:]:
                ctime = _parse_iso_utc(clip.created_at)
                if (ctime - group_time).total_seconds() <= self.cfg.cluster_window_seconds:
                    current_group.append(clip)
                else:
                    raw_groups.append(current_group)
                    current_group = [clip]
                    group_time = ctime
            raw_groups.append(current_group)

        clusters: List[MomentCluster] = []
        for grp in raw_groups:
            unique_clippers = len({c.creator_name.lower() for c in grp})
            total_views = sum(c.view_count for c in grp)
            max_views = max(c.view_count for c in grp)
            best = max(grp, key=lambda c: (c.view_count, c.duration))

            # Critério de disparo conforme o modo
            if mode == "live_hot":
                qualifies = (
                    unique_clippers >= self.cfg.min_clips_live_cluster
                    or max_views >= self.cfg.min_views_single_clip_live
                )
            else:
                qualifies = (
                    max_views >= self.cfg.min_views_post_live
                    or unique_clippers >= self.cfg.min_clips_live_cluster
                )

            if not qualifies:
                continue

            age_minutes = max(
                2.0,
                min((now_utc - _parse_iso_utc(c.created_at)).total_seconds() / 60.0 for c in grp),
            )
            velocity = total_views / age_minutes

            # Score real sem teto artificial em 99.9 (garante que 600 views ranqueie muito acima de 50 views)
            raw_score = (
                (unique_clippers * 14.0)
                + (math.log1p(total_views) * 22.0)
                + (math.sqrt(max_views) * 4.5)
                + min(60.0, velocity * 2.0)
            )
            score = round(raw_score, 1)

            offsets = [int(c.vod_offset) for c in grp if c.vod_offset is not None]
            vod_start = min(offsets) if offsets else None
            vod_end = (
                max(int(c.vod_offset or 0) + int(math.ceil(c.duration)) for c in grp if c.vod_offset is not None)
                if offsets
                else None
            )
            peak_offset = best.vod_offset if best.vod_offset is not None else vod_start
            video_id = next((c.video_id for c in grp if c.video_id), "")

            anchor_tag = f"vod{peak_offset}" if peak_offset is not None else f"clip_{best.id}"
            digest = hashlib.sha1(f"{self.profile.id}:{stream_key}:{anchor_tag}".encode("utf-8")).hexdigest()[:10]
            cluster_id = f"{self.profile.id}_{digest}"

            clusters.append(
                MomentCluster(
                    cluster_id=cluster_id,
                    profile_id=self.profile.id,
                    broadcaster_login=self.profile.twitch_login,
                    stream_or_vod_key=stream_key,
                    mode=mode,
                    clips=grp,
                    video_id=video_id,
                    vod_start_offset=vod_start,
                    vod_end_offset=vod_end,
                    peak_offset=peak_offset,
                    unique_clippers=unique_clippers,
                    total_views=total_views,
                    max_views=max_views,
                    score=score,
                    dominant_title=best.title,
                    game_name=game_name,
                    stream_title=stream_title,
                )
            )

        clusters.sort(key=lambda c: c.score, reverse=True)
        return clusters

    def _filter_and_deduplicate(
        self,
        clusters: List[MomentCluster],
        processed_state: Dict[str, Any],
        stream_key: str,
        alt_stream_key: str,
        remaining_slots: int,
    ) -> List[MomentCluster]:
        if remaining_slots <= 0:
            return []

        existing_records: List[Dict[str, Any]] = processed_state.get("cuts", [])
        relevant_existing = [
            r
            for r in existing_records
            if r.get("profile_id") == self.profile.id
            and r.get("stream_key") in {stream_key, alt_stream_key}
        ]
        used_clip_ids = {
            cid
            for r in existing_records
            if r.get("profile_id") == self.profile.id
            for cid in r.get("clip_ids", [])
        }

        accepted: List[MomentCluster] = []
        for cluster in clusters:
            if len(accepted) >= remaining_slots:
                break

            # 1. Evita repetir os mesmos clipes já usados
            cluster_clip_ids = {c.id for c in cluster.clips}
            if cluster_clip_ids & used_clip_ids:
                continue

            # 2. Evita sobreposição temporal no VOD (`dedup_overlap_seconds`)
            if cluster.peak_offset is not None:
                overlaps_existing = any(
                    r.get("vod_anchor_sec") is not None
                    and abs(int(r["vod_anchor_sec"]) - cluster.peak_offset) < self.cfg.dedup_overlap_seconds
                    for r in relevant_existing
                )
                overlaps_accepted = any(
                    a.peak_offset is not None
                    and abs(a.peak_offset - cluster.peak_offset) < self.cfg.dedup_overlap_seconds
                    for a in accepted
                )
                if overlaps_existing or overlaps_accepted:
                    continue

            accepted.append(cluster)

        return accepted

    def _count_cuts_for_stream(
        self, processed_state: Dict[str, Any], stream_key: str, alt_stream_key: str = ""
    ) -> int:
        keys = {k for k in (stream_key, alt_stream_key) if k}
        return sum(
            1
            for r in processed_state.get("cuts", [])
            if r.get("profile_id") == self.profile.id and r.get("stream_key") in keys
        )
