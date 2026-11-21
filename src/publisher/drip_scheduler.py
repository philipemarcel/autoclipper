from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.ai.content_moderator import moderate_video_for_platforms
from src.ai.deduplicator import find_duplicate_cut
from src.models import ProfileConfig
from src.publisher.facebook_reels import FacebookReelsPublisher
from src.publisher.instagram_reels import InstagramReelsPublisher
from src.publisher.tiktok_poster import TikTokPublisher
from src.publisher.youtube_shorts import YouTubeShortsPublisher
from src.storage.gdrive import DriveStore

logger = logging.getLogger(__name__)

DRIP_STATE_FILE = "drip_queue.json"

SUPPORTED_PLATFORMS: Tuple[str, ...] = ("instagram", "facebook", "youtube", "tiktok")

# Defasagem (min) entre as filas das redes: TikTok primeiro, +30 min Instagram/Facebook, +30 min YouTube.
DEFAULT_NETWORK_OFFSETS: Dict[str, int] = {"tiktok": 0, "instagram": 30, "facebook": 30, "youtube": 60}

PLATFORM_LABELS: Dict[str, str] = {
    "instagram": "Instagram Reels",
    "facebook": "Facebook Reels",
    "youtube": "YouTube Shorts",
    "tiktok": "TikTok",
}

_PLATFORM_RESULT_KEYS: Dict[str, Tuple[str, ...]] = {
    "instagram": ("instagram_reel_id", "instagram"),
    "facebook": ("facebook_reel_id", "facebook"),
    "youtube": ("youtube_short_url", "youtube_shorts", "youtube"),
    "tiktok": ("tiktok_publish_id", "tiktok_mode", "tiktok"),
}

_PLATFORM_DONE_KEY: Dict[str, str] = {
    "instagram": "instagram_reel_id",
    "facebook": "facebook_reel_id",
    "youtube": "youtube_short_url",
    "tiktok": "tiktok_publish_id",
}


def _parse_iso(ts: str) -> datetime:
    cleaned = ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


_DEDUP_ITEM_KEYS = (
    "caption",
    "teaser_headline",
    "twitch_login",
    "creator_name",
    "game_name",
    "stream_key",
    "video_id",
    "vod_anchor_sec",
    "clip_ids",
    "clip_created_at",
    "dominant_title",
    "transcript_snippet",
    "director",
    "mode",
)


def _dedup_fields_for_item(cut_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: cut_dict[k] for k in _DEDUP_ITEM_KEYS if k in cut_dict and cut_dict[k] not in (None, "", [], {})}


def _to_iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DripScheduler:
    """
    Fila Espaçada Inteligente por Rede Social (Separate Per-Network Smart Drip Queues):
      - Cada rede social (`instagram`, `facebook`, `youtube`, `tiktok`) possui sua própria fila independente,
        com horário agendado, último envio e controle de retries isolados.
      - Se não houve post naquela rede nos últimos `drip_interval_minutes` (ex: 75 min) E estamos fora da janela de
        madrugada (`quiet_hours_brt`), o corte sobe IMEDIATAMENTE naquela rede!
      - Cortes aprovados em sequência entram espaçados automaticamente a cada `drip_interval_minutes` por rede,
        pulando horários mortos da madrugada para proteger o alcance no Reels, Shorts e TikTok.
      - Falhas ou rate limits em uma rede (ex: TikTok Inbox cheia) não atrasam nem bloqueiam as outras redes.
    """

    def __init__(self, store: DriveStore) -> None:
        self.store = store

    @staticmethod
    def _canonical_platform(name: str) -> str:
        low = str(name or "").strip().lower()
        if low in ("youtube", "youtube_shorts", "yt"):
            return "youtube"
        if low in ("instagram", "ig"):
            return "instagram"
        if low in ("facebook", "fb"):
            return "facebook"
        if low in ("tiktok", "tt"):
            return "tiktok"
        return low

    @classmethod
    def _queue_item_id(cls, cut_id: str, platform: str) -> str:
        canon = cls._canonical_platform(platform)
        return f"{cut_id}::{canon}" if cut_id and canon else cut_id

    @classmethod
    def _item_platform(cls, item: Dict[str, Any]) -> str:
        if item.get("platform"):
            return cls._canonical_platform(str(item["platform"]))
        plats = item.get("platforms") or []
        if plats:
            return cls._canonical_platform(str(plats[0]))
        return ""

    @classmethod
    def normalize_queue_state(cls, state: Dict[str, Any]) -> bool:
        """
        Normaliza `drip_queue.json` garantindo que cada rede social tenha um item próprio e separado.
        Converte automaticamente itens legados multi-plataforma (`platforms: ["instagram", "tiktok"]`)
        em itens individuais por rede preservando resultados e erros já registrados.
        """
        changed = False
        if not isinstance(state.get("last_published_by_profile"), dict):
            state["last_published_by_profile"] = {}
            changed = True
        if not isinstance(state.get("last_published_by_platform"), dict):
            state["last_published_by_platform"] = {}
            changed = True
        if not isinstance(state.get("items"), list):
            state["items"] = []
            changed = True

        # Migração única da linha de base legada `last_published_by_profile` -> `last_published_by_platform`
        by_plat = state["last_published_by_platform"]
        for pid, ts in list(state["last_published_by_profile"].items()):
            if ts and pid not in by_plat:
                by_plat[pid] = {p: ts for p in SUPPORTED_PLATFORMS}
                changed = True

        new_items: List[Dict[str, Any]] = []
        seen_ids: Dict[str, int] = {}

        for raw_item in state["items"]:
            if not isinstance(raw_item, dict):
                changed = True
                continue

            raw_plats = list(raw_item.get("platforms") or [])
            if not raw_plats and raw_item.get("platform"):
                raw_plats = [raw_item["platform"]]

            canon_plats: List[str] = []
            for p in raw_plats:
                c = cls._canonical_platform(str(p))
                if c and c not in canon_plats:
                    canon_plats.append(c)

            if not canon_plats:
                new_items.append(raw_item)
                continue

            cid = str(raw_item.get("cut_id") or "").strip()

            if len(canon_plats) == 1:
                plat = canon_plats[0]
                item = dict(raw_item)
                if item.get("platform") != plat:
                    item["platform"] = plat
                    changed = True
                if item.get("platforms") != [plat]:
                    item["platforms"] = [plat]
                    changed = True
                expected_id = cls._queue_item_id(cid, plat) if cid else str(item.get("id") or "")
                if expected_id and not item.get("id"):
                    item["id"] = expected_id
                    changed = True

                item_key = str(item.get("id") or "")
                if item_key and item_key in seen_ids:
                    # Prefere item já publicado ou com mais informações
                    prev_idx = seen_ids[item_key]
                    if new_items[prev_idx].get("status") != "published" and item.get("status") == "published":
                        new_items[prev_idx] = item
                    changed = True
                else:
                    if item_key:
                        seen_ids[item_key] = len(new_items)
                    new_items.append(item)
                continue

            # Item legado com múltiplas redes -> separa 1 item independente por rede social
            changed = True
            raw_results = dict(raw_item.get("results") or {})
            raw_errors = dict(raw_item.get("platform_errors") or {})
            base_id = cid or str(raw_item.get("id") or "").strip()

            for plat in canon_plats:
                sub = dict(raw_item)
                sub["platform"] = plat
                sub["platforms"] = [plat]
                if base_id:
                    sub["id"] = cls._queue_item_id(base_id, plat)

                allowed_res_keys = _PLATFORM_RESULT_KEYS.get(plat, (plat,))
                sub_res = {k: v for k, v in raw_results.items() if k in allowed_res_keys}
                if sub_res:
                    sub["results"] = sub_res
                else:
                    sub.pop("results", None)

                plat_err = raw_errors.get(plat) or (raw_errors.get("youtube_shorts") if plat == "youtube" else None)
                if plat_err:
                    sub["platform_errors"] = {plat: plat_err}
                    sub["last_error"] = f"{plat}: {plat_err}"
                else:
                    sub.pop("platform_errors", None)
                    if raw_errors:
                        sub.pop("last_error", None)

                done_key = _PLATFORM_DONE_KEY.get(plat, "")
                if done_key and sub_res.get(done_key):
                    sub["status"] = "published"
                    if not sub.get("published_at"):
                        sub["published_at"] = raw_item.get("published_at") or raw_item.get("scheduled_for") or raw_item.get("created_at") or ""

                sub_key = str(sub.get("id") or "")
                if sub_key and sub_key in seen_ids:
                    prev_idx = seen_ids[sub_key]
                    if new_items[prev_idx].get("status") != "published" and sub.get("status") == "published":
                        new_items[prev_idx] = sub
                else:
                    if sub_key:
                        seen_ids[sub_key] = len(new_items)
                    new_items.append(sub)

        if changed:
            state["items"] = new_items
        return changed

    def _load_queue_state(self) -> Dict[str, Any]:
        state = self.store.load_state_json(
            DRIP_STATE_FILE,
            default={"last_published_by_profile": {}, "last_published_by_platform": {}, "items": []},
        )
        self.normalize_queue_state(state)
        return state

    def _save_queue_state(self, state: Dict[str, Any]) -> None:
        self.store.save_state_json(DRIP_STATE_FILE, state)

    def get_effective_interval_minutes(
        self,
        profile: ProfileConfig,
        state: Optional[Dict[str, Any]] = None,
        platform: Optional[str] = None,
    ) -> int:
        q_state = state if state is not None else self._load_queue_state()
        canon = self._canonical_platform(platform) if platform else ""

        if canon:
            plat_overrides = q_state.get("drip_interval_override_by_platform") or {}
            if canon in plat_overrides and plat_overrides[canon] is not None:
                try:
                    return max(5, min(1440, int(plat_overrides[canon])))
                except (TypeError, ValueError):
                    pass

            plat_cfg_map = {
                "instagram": getattr(profile.publishing, "instagram", None),
                "facebook": getattr(profile.publishing, "facebook", None),
                "youtube": getattr(profile.publishing, "youtube_shorts", None),
                "tiktok": getattr(profile.publishing, "tiktok", None),
            }
            cfg = plat_cfg_map.get(canon)
            plat_min = getattr(cfg, "drip_interval_minutes", None) if cfg is not None else None
            if plat_min is not None:
                try:
                    return max(5, min(1440, int(plat_min)))
                except (TypeError, ValueError):
                    pass

        override = q_state.get("drip_interval_override_minutes")
        if override is not None:
            try:
                return max(5, min(1440, int(override)))
            except (TypeError, ValueError):
                pass

        return max(5, int(profile.publishing.drip_interval_minutes))

    @staticmethod
    def get_effective_quiet_hours(
        profile: ProfileConfig,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[int]]:
        if state and "quiet_hours_override" in state:
            ov = state.get("quiet_hours_override")
            if not ov or len(ov) != 2:
                return None
            return [int(ov[0]), int(ov[1])]
        return profile.publishing.quiet_hours_brt

    @classmethod
    def _get_last_published_iso(
        cls,
        state: Dict[str, Any],
        profile_id: str,
        platform: Optional[str] = None,
    ) -> Optional[str]:
        by_plat = state.get("last_published_by_platform") or {}
        if platform:
            canon = cls._canonical_platform(platform)
            if profile_id in by_plat and isinstance(by_plat[profile_id], dict):
                return by_plat[profile_id].get(canon)
            return (state.get("last_published_by_profile") or {}).get(profile_id)

        candidates: List[str] = []
        legacy = (state.get("last_published_by_profile") or {}).get(profile_id)
        if legacy:
            candidates.append(str(legacy))
        prof_plats = by_plat.get(profile_id)
        if isinstance(prof_plats, dict):
            for ts in prof_plats.values():
                if ts:
                    candidates.append(str(ts))
        if not candidates:
            return None
        return max(candidates, key=_parse_iso)

    @classmethod
    def _record_last_published(
        cls,
        state: Dict[str, Any],
        profile_id: str,
        platform: str,
        dt_utc: datetime,
    ) -> None:
        iso = _to_iso_utc(dt_utc)
        canon = cls._canonical_platform(platform)
        by_plat = state.setdefault("last_published_by_platform", {})
        if not isinstance(by_plat.get(profile_id), dict):
            by_plat[profile_id] = {}
        by_plat[profile_id][canon] = iso
        state.setdefault("last_published_by_profile", {})[profile_id] = iso

    def update_timer_override(
        self,
        profiles_by_id: Dict[str, ProfileConfig],
        interval_minutes: int,
        quiet_hours_str: str = "",
        now_utc: Optional[datetime] = None,
        platform: str = "",
    ) -> Dict[str, Any]:
        """
        Atualiza o intervalo da Fila Espaçada (em minutos) diretamente em `drip_queue.json`
        e recalcula os horários dos itens ainda `pending` na fila (de forma independente por rede social).
        """
        now = now_utc or datetime.now(timezone.utc)
        state = self._load_queue_state()
        new_interval = max(5, min(1440, int(interval_minutes)))
        canon_filter = self._canonical_platform(platform) if platform.strip() else ""

        if canon_filter:
            state.setdefault("drip_interval_override_by_platform", {})[canon_filter] = new_interval
        else:
            state["drip_interval_override_minutes"] = new_interval
            state.pop("drip_interval_override_by_platform", None)

        clean_qh = str(quiet_hours_str or "").strip().lower()
        if clean_qh:
            if clean_qh in {"off", "desativar", "desativado", "none", "nao", "não", "0"}:
                state["quiet_hours_override"] = []
            elif clean_qh in {"default", "padrao", "padrão", "reset"}:
                state.pop("quiet_hours_override", None)
            elif "-" in clean_qh:
                parts = clean_qh.replace("h", "").split("-", 1)
                try:
                    h0 = max(0, min(23, int(parts[0].strip())))
                    h1 = max(0, min(23, int(parts[1].strip())))
                    state["quiet_hours_override"] = [h0, h1]
                except ValueError:
                    pass

        pending_items = [it for it in state.get("items", []) if it.get("status") == "pending"]
        pending_items.sort(key=lambda it: str(it.get("created_at") or it.get("scheduled_for") or ""))

        cursor_by_queue: Dict[Tuple[str, str], datetime] = {}
        rescheduled = 0
        earliest_slot: Optional[datetime] = None
        earliest_tz: Optional[ZoneInfo] = None

        for item in pending_items:
            pid = str(item.get("profile_id") or "")
            plat = self._item_platform(item)
            if canon_filter and plat != canon_filter:
                continue
            prof = profiles_by_id.get(pid)
            if not prof:
                continue

            eff_min = self.get_effective_interval_minutes(prof, state, platform=plat or None)
            delta = timedelta(minutes=eff_min)
            tz = ZoneInfo(prof.publishing.timezone)
            quiet_hours = self.get_effective_quiet_hours(prof, state)
            q_key = (pid, plat)

            if q_key not in cursor_by_queue:
                last_pub = self._get_last_published_iso(state, pid, platform=plat or None)
                base_dt = max(now, _parse_iso(last_pub) + delta) if last_pub else now
            else:
                base_dt = max(now, cursor_by_queue[q_key] + delta)

            slot = self._push_past_quiet_hours(base_dt, quiet_hours, tz)
            cursor_by_queue[q_key] = slot
            item["scheduled_for"] = _to_iso_utc(slot)
            rescheduled += 1
            if earliest_slot is None or slot < earliest_slot:
                earliest_slot = slot
                earliest_tz = tz

        next_local = (
            earliest_slot.astimezone(earliest_tz).strftime("%d/%m às %H:%M")
            if earliest_slot and earliest_tz
            else ""
        )

        self._save_queue_state(state)
        return {
            "interval_minutes": new_interval,
            "platform": canon_filter or "all",
            "quiet_hours": state.get("quiet_hours_override", "default"),
            "rescheduled_count": rescheduled,
            "next_scheduled_local": next_local,
        }

    def stagger_network_queues(
        self,
        profiles_by_id: Dict[str, ProfileConfig],
        offsets_minutes: Optional[Dict[str, int]] = None,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Escalona as filas por rede social: cada rede passa a começar `offsets_minutes[rede]` minutos
        depois da rede de referência (offset 0), mantendo o espaçamento `drip_interval` dentro de cada fila.
        Ex.: {"tiktok": 0, "instagram": 30, "facebook": 30, "youtube": 60}.
        """
        now = now_utc or datetime.now(timezone.utc)
        offsets = {self._canonical_platform(k): max(0, int(v)) for k, v in (offsets_minutes or DEFAULT_NETWORK_OFFSETS).items()}
        state = self._load_queue_state()
        state["network_stagger_minutes"] = offsets

        reflowed = 0
        first_slots: Dict[str, str] = {}
        for pid, prof in profiles_by_id.items():
            tz = ZoneInfo(prof.publishing.timezone)
            quiet_hours = self.get_effective_quiet_hours(prof, state)

            # Referência: a rede com offset 0 que publicou/agendou primeiro
            ref_plat = min(offsets, key=lambda p: offsets[p])
            ref_eff = timedelta(minutes=self.get_effective_interval_minutes(prof, state, platform=ref_plat))
            ref_last = self._get_last_published_iso(state, pid, platform=ref_plat)
            ref_base = max(now, _parse_iso(ref_last) + ref_eff) if ref_last else now

            for plat, off in offsets.items():
                queue = [
                    it for it in state.get("items", [])
                    if it.get("profile_id") == pid and it.get("status") == "pending" and self._item_platform(it) == plat
                ]
                if not queue:
                    continue
                queue.sort(key=lambda it: (str(it.get("scheduled_for") or ""), str(it.get("created_at") or "")))
                delta = timedelta(minutes=self.get_effective_interval_minutes(prof, state, platform=plat))
                cursor = self._push_past_quiet_hours(ref_base + timedelta(minutes=off), quiet_hours, tz)
                for idx, it in enumerate(queue):
                    if idx > 0:
                        cursor = self._push_past_quiet_hours(cursor + delta, quiet_hours, tz)
                    new_iso = _to_iso_utc(cursor)
                    if it.get("scheduled_for") != new_iso:
                        it["scheduled_for"] = new_iso
                        reflowed += 1
                first_slots[plat] = _to_iso_utc(self._push_past_quiet_hours(ref_base + timedelta(minutes=off), quiet_hours, tz))

        self._save_queue_state(state)
        return {"offsets": offsets, "rescheduled_count": reflowed, "first_slots": first_slots}

    def compute_next_slot(
        self,
        profile: ProfileConfig,
        state: Dict[str, Any],
        now_utc: Optional[datetime] = None,
        platform: Optional[str] = None,
    ) -> Tuple[datetime, bool]:
        """
        Calcula `(scheduled_utc, is_immediate)` para a fila da rede social `platform`
        (ou para o perfil inteiro se `platform` não for informado).
        """
        now = now_utc or datetime.now(timezone.utc)
        canon = self._canonical_platform(platform) if platform else ""
        eff_minutes = self.get_effective_interval_minutes(profile, state, platform=canon or None)
        interval = timedelta(minutes=max(5, eff_minutes))
        quiet_hours = self.get_effective_quiet_hours(profile, state)
        tz = ZoneInfo(profile.publishing.timezone)

        last_pub_iso = self._get_last_published_iso(state, profile.id, platform=canon or None)
        pending_for_queue: List[datetime] = []
        for it in state.get("items", []):
            if it.get("profile_id") != profile.id or it.get("status") != "pending":
                continue
            if canon and self._item_platform(it) != canon:
                continue
            if it.get("scheduled_for"):
                pending_for_queue.append(_parse_iso(it["scheduled_for"]))

        anchors: List[datetime] = []
        if last_pub_iso:
            anchors.append(_parse_iso(last_pub_iso) + interval)
        if pending_for_queue:
            anchors.append(max(pending_for_queue) + interval)

        candidate = max([now] + anchors)
        candidate = self._push_past_quiet_hours(candidate, quiet_hours, tz)
        is_immediate = (candidate - now).total_seconds() <= 30
        return candidate, is_immediate

    @staticmethod
    def _push_past_quiet_hours(
        dt_utc: datetime,
        quiet_hours: Optional[List[int]],
        tz: ZoneInfo,
    ) -> datetime:
        if not quiet_hours or len(quiet_hours) != 2:
            return dt_utc
        q_start, q_end = int(quiet_hours[0]), int(quiet_hours[1])
        local_dt = dt_utc.astimezone(tz)

        if q_start < q_end:
            in_quiet = q_start <= local_dt.hour < q_end
            if in_quiet:
                local_dt = local_dt.replace(hour=q_end, minute=5, second=0, microsecond=0)
        elif q_start > q_end:
            # Ex: [23, 8] atravessando meia-noite
            if local_dt.hour >= q_start:
                local_dt = (local_dt + timedelta(days=1)).replace(hour=q_end, minute=5, second=0, microsecond=0)
            elif local_dt.hour < q_end:
                local_dt = local_dt.replace(hour=q_end, minute=5, second=0, microsecond=0)

        return local_dt.astimezone(timezone.utc)

    def _already_done_platforms_for_cut(self, state: Dict[str, Any], cut_id: str) -> set[str]:
        done: set[str] = set()
        for it in state.get("items", []):
            if it.get("cut_id") != cut_id:
                continue
            res = it.get("results") or {}
            if res.get("instagram_reel_id"):
                done.add("instagram")
            if res.get("facebook_reel_id"):
                done.add("facebook")
            if res.get("tiktok_publish_id"):
                done.add("tiktok")
            if res.get("youtube_short_url"):
                done.add("youtube")
            if it.get("status") == "published" and not res:
                plat = self._item_platform(it)
                if plat:
                    done.add(plat)
                else:
                    done.update(self._canonical_platform(p) for p in (it.get("platforms") or []))
        return done

    def _reflow_queue_after_advance(
        self,
        state: Dict[str, Any],
        profile: ProfileConfig,
        platform: str,
        anchor_utc: datetime,
    ) -> int:
        """
        Recalcula os horários dos itens `pending` na fila `(profile.id, platform)` a partir de
        `anchor_utc + interval`, puxando a fila para frente quando um vídeo é adiantado.
        """
        canon = self._canonical_platform(platform)
        if not canon:
            return 0
        eff_min = self.get_effective_interval_minutes(profile, state, platform=canon)
        delta = timedelta(minutes=max(5, eff_min))
        quiet_hours = self.get_effective_quiet_hours(profile, state)
        tz = ZoneInfo(profile.publishing.timezone)

        queue_pending = [
            it
            for it in state.get("items", [])
            if it.get("profile_id") == profile.id
            and it.get("status") == "pending"
            and self._item_platform(it) == canon
        ]
        queue_pending.sort(key=lambda it: (str(it.get("scheduled_for") or ""), str(it.get("created_at") or "")))

        cursor = anchor_utc
        updated = 0
        for it in queue_pending:
            base_dt = cursor + delta
            slot = self._push_past_quiet_hours(base_dt, quiet_hours, tz)
            if int(it.get("retry_count") or 0) > 0 and it.get("scheduled_for"):
                existing_dt = _parse_iso(str(it["scheduled_for"]))
                if existing_dt > slot:
                    slot = existing_dt
            new_iso = _to_iso_utc(slot)
            if it.get("scheduled_for") != new_iso:
                it["scheduled_for"] = new_iso
                updated += 1
            cursor = slot
        return updated

    def enqueue_or_publish_cut(
        self,
        profile: ProfileConfig,
        cut_dict: Dict[str, Any],
        platforms: List[str],  # ["instagram", "facebook", "tiktok", "youtube_shorts"]
        work_dir: Path,
        force_immediate: bool = False,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = now_utc or datetime.now(timezone.utc)
        state = self._load_queue_state()
        cut_id = str(cut_dict["cut_id"])

        already_done_plats = self._already_done_platforms_for_cut(state, cut_id)
        clean_platforms: List[str] = []
        seen_canon: set[str] = set()
        for p in platforms:
            canon = self._canonical_platform(p)
            if not canon or canon in already_done_plats or canon in seen_canon:
                continue
            seen_canon.add(canon)
            clean_platforms.append(canon)

        if not clean_platforms:
            return {"mode": "already_published", "platforms": []}

        # Bloqueia cut_id novo cujo conteúdo/descrição/horário extraído duplique item já na fila ou publicado
        same_cut_exists = any(it.get("cut_id") == cut_id for it in state.get("items", []))
        if not same_cut_exists:
            dup = find_duplicate_cut(cut_dict, processed_state=None, queue_state=state)
            if dup.is_duplicate:
                logger.warning(
                    "Bloqueando enfileiramento/publicação de %s (duplicata de %s): %s",
                    cut_id,
                    dup.matched_cut_id or "item na fila",
                    dup.reason,
                )
                return {
                    "mode": "skipped_duplicate",
                    "matched_cut_id": dup.matched_cut_id,
                    "reason": dup.reason,
                    "score": round(dup.score, 3),
                }

        dedup_meta = _dedup_fields_for_item(cut_dict)

        # Verifica itens já pendentes por rede social (sem bloquear uma rede porque outra já está na fila!)
        pending_by_plat: Dict[str, Dict[str, Any]] = {}
        for it in state.get("items", []):
            if it.get("cut_id") == cut_id and it.get("status") == "pending":
                p = self._item_platform(it)
                if p:
                    pending_by_plat[p] = it

        if not force_immediate:
            to_enqueue = [p for p in clean_platforms if p not in pending_by_plat]
            if not to_enqueue:
                earliest = min(str(pending_by_plat[p].get("scheduled_for") or "") for p in clean_platforms)
                return {
                    "mode": "already_queued",
                    "scheduled_for": earliest,
                    "by_platform": {
                        p: {"mode": "already_queued", "scheduled_for": pending_by_plat[p]["scheduled_for"]}
                        for p in clean_platforms
                    },
                }
        else:
            to_enqueue = clean_platforms
            # Remove apenas entradas pendentes das redes sendo forçadas agora
            forced_set = set(to_enqueue)
            state["items"] = [
                it
                for it in state.get("items", [])
                if not (
                    it.get("cut_id") == cut_id
                    and it.get("status") == "pending"
                    and self._item_platform(it) in forced_set
                )
            ]

        tz = ZoneInfo(profile.publishing.timezone)
        by_platform: Dict[str, Dict[str, Any]] = {}
        combined_results: Dict[str, str] = {}
        failed_platforms: List[str] = []
        queued_slots: List[datetime] = []
        immediate_count = 0
        queued_count = 0

        for plat in to_enqueue:
            if force_immediate:
                is_immediate = True
                slot_utc = now
            else:
                slot_utc, is_immediate = self.compute_next_slot(profile, state, now_utc=now, platform=plat)

            item_id = self._queue_item_id(cut_id, plat)
            base_item: Dict[str, Any] = {
                "id": item_id,
                "cut_id": cut_id,
                "profile_id": profile.id,
                "platform": plat,
                "platforms": [plat],
                "discord_message_id": cut_dict.get("discord_message_id", ""),
                "created_at": _to_iso_utc(now),
                **dedup_meta,
            }

            if is_immediate:
                immediate_count += 1
                result, failed_plats, plat_errors = self._execute_publish_resilient(
                    profile, cut_dict, [plat], work_dir
                )
                combined_results.update(result)
                if "moderation" in plat_errors or "moderation_platform_blocked" in plat_errors:
                    err_msg = plat_errors.get("moderation") or plat_errors.get("moderation_platform_blocked") or ""
                    status_val = "blocked_nudity" if "moderation" in plat_errors else "blocked_moderation"
                    state.setdefault("items", []).append(
                        {
                            **base_item,
                            "scheduled_for": _to_iso_utc(now),
                            "status": status_val,
                            "last_error": err_msg[:500],
                        }
                    )
                    by_platform[plat] = {"mode": status_val, "error": err_msg}
                    continue

                done_key = _PLATFORM_DONE_KEY.get(plat, "")
                if done_key and result.get(done_key):
                    self._record_last_published(state, profile.id, plat, now)

                if failed_plats:
                    failed_platforms.extend(failed_plats)
                    backoff_min = 120 if any("spam_risk_too_many_pending_share" in e for e in plat_errors.values()) else 30
                    retry_slot = now + timedelta(minutes=backoff_min)
                    state.setdefault("items", []).append(
                        {
                            **base_item,
                            "scheduled_for": _to_iso_utc(retry_slot),
                            "status": "pending",
                            "results": result,
                            "platform_errors": plat_errors,
                            "last_error": "; ".join(f"{k}: {v}" for k, v in plat_errors.items()),
                            "retry_count": 1,
                        }
                    )
                    by_platform[plat] = {
                        "mode": "failed_retry",
                        "scheduled_for": _to_iso_utc(retry_slot),
                        "scheduled_local": retry_slot.astimezone(tz).strftime("%d/%m às %H:%M"),
                        "error": plat_errors.get(plat, ""),
                    }
                elif result:
                    state.setdefault("items", []).append(
                        {
                            **base_item,
                            "scheduled_for": _to_iso_utc(now),
                            "published_at": _to_iso_utc(now),
                            "status": "published",
                            "results": result,
                        }
                    )
                    if force_immediate and done_key and result.get(done_key):
                        self._reflow_queue_after_advance(state, profile, plat, anchor_utc=now)
                    by_platform[plat] = {
                        "mode": "published_now",
                        "results": result,
                    }
            else:
                queued_count += 1
                queued_slots.append(slot_utc)
                local_str = slot_utc.astimezone(tz).strftime("%d/%m às %H:%M")
                state.setdefault("items", []).append(
                    {
                        **base_item,
                        "scheduled_for": _to_iso_utc(slot_utc),
                        "status": "pending",
                    }
                )
                by_platform[plat] = {
                    "mode": "queued",
                    "scheduled_for": _to_iso_utc(slot_utc),
                    "scheduled_local": local_str,
                }

        self._save_queue_state(state)

        if immediate_count > 0 and queued_count == 0:
            return {
                "mode": "published_now",
                "results": combined_results,
                "failed_platforms": failed_platforms,
                "by_platform": by_platform,
            }

        earliest_slot = min(queued_slots) if queued_slots else now
        earliest_local = earliest_slot.astimezone(tz).strftime("%d/%m às %H:%M")

        if queued_count > 0 and immediate_count == 0:
            return {
                "mode": "queued",
                "scheduled_for": _to_iso_utc(earliest_slot),
                "scheduled_local": earliest_local,
                "by_platform": by_platform,
            }

        return {
            "mode": "mixed",
            "results": combined_results,
            "failed_platforms": failed_platforms,
            "scheduled_for": _to_iso_utc(earliest_slot),
            "scheduled_local": earliest_local,
            "by_platform": by_platform,
        }

    def process_due_queue(
        self,
        profiles_by_id: Dict[str, ProfileConfig],
        processed_state: Dict[str, Any],
        work_dir: Path,
        now_utc: Optional[datetime] = None,
        force_next: bool = False,
        platform_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        now = now_utc or datetime.now(timezone.utc)
        state = self._load_queue_state()
        cuts_by_id = {c["cut_id"]: c for c in processed_state.get("cuts", [])}

        canon_filter = self._canonical_platform(platform_filter) if platform_filter else ""
        completed_reports: List[Dict[str, Any]] = []
        changed = False
        forced_queues: set[Tuple[str, str]] = set()
        min_spacing_by_queue: Dict[Tuple[str, str], datetime] = {}

        published_queue_items = [it for it in state.get("items", []) if it.get("status") == "published"]
        published_cuts = [
            c
            for c in processed_state.get("cuts", [])
            if c.get("status") == "published" or any(p.get("cut_id") == c.get("cut_id") for p in published_queue_items)
        ]

        # Ordena pendentes por horário agendado para sempre respeitar a fila de cada rede em ordem cronológica
        pending_ordered = sorted(
            [it for it in state.get("items", []) if it.get("status") == "pending"],
            key=lambda it: str(it.get("scheduled_for") or ""),
        )

        for item in pending_ordered:
            plat = self._item_platform(item)
            if canon_filter and plat != canon_filter:
                continue

            due_dt = _parse_iso(item["scheduled_for"])
            prof = profiles_by_id.get(item["profile_id"])
            queue_key = (str(item.get("profile_id") or ""), plat)
            can_force_now = force_next and queue_key not in forced_queues

            # Tolerância de 120s para jitter/atraso do cron do GitHub Actions
            if not can_force_now and (due_dt - now).total_seconds() > 120:
                continue

            cut_dict = cuts_by_id.get(item["cut_id"])
            if not cut_dict:
                recovered = self.store.recover_cut_metadata_by_id(str(item.get("cut_id") or ""))
                if recovered:
                    cut_dict = recovered
                    cuts_by_id[recovered["cut_id"]] = recovered
                    processed_state.setdefault("cuts", []).append(recovered)
            if not prof or not cut_dict:
                item["status"] = "skipped_missing"
                changed = True
                continue

            existing_res = dict(item.get("results") or {})
            # Barreira em tempo de publicação contra duplicatas herdadas da fila antiga
            # (mas nunca bloqueia outra rede social do MESMO cut_id que já publicou em outra rede!)
            if not existing_res:
                other_cuts = [c for c in published_cuts if c.get("cut_id") != item["cut_id"]]
                other_q_items = [q for q in published_queue_items if q.get("cut_id") != item["cut_id"]]
                dup = find_duplicate_cut(
                    cut_dict,
                    processed_state={"cuts": other_cuts},
                    queue_state={"items": other_q_items},
                )
                if dup.is_duplicate:
                    logger.warning(
                        "Corte pendente %s (%s) bloqueado na publicação (duplicata de %s): %s",
                        item["cut_id"],
                        plat or "unknown",
                        dup.matched_cut_id or "corte já publicado",
                        dup.reason,
                    )
                    item["status"] = "skipped_duplicate"
                    item["last_error"] = f"Duplicata bloqueada antes de postar ({dup.reason})"
                    changed = True
                    continue

            eff_minutes = self.get_effective_interval_minutes(prof, state, platform=plat or None)
            interval = timedelta(minutes=eff_minutes)
            min_gap = timedelta(minutes=max(10, eff_minutes - 5))
            quiet_hours = self.get_effective_quiet_hours(prof, state)
            tz = ZoneInfo(prof.publishing.timezone)

            # Evita rajada de posts na MESMA rede social em intervalo curto quando vários itens vencem juntos
            last_iso = self._get_last_published_iso(state, prof.id, platform=plat or None)
            last_dt = _parse_iso(last_iso) if last_iso else None
            if queue_key in min_spacing_by_queue:
                anchor = min_spacing_by_queue[queue_key] + interval
                next_slot = self._push_past_quiet_hours(max(now, anchor), quiet_hours, tz)
                min_spacing_by_queue[queue_key] = next_slot
                item["scheduled_for"] = _to_iso_utc(next_slot)
                changed = True
                continue
            if not can_force_now and last_dt is not None and (now - last_dt) < min_gap:
                next_slot = self._push_past_quiet_hours(last_dt + interval, quiet_hours, tz)
                if next_slot <= now:
                    next_slot = self._push_past_quiet_hours(now + timedelta(minutes=eff_minutes), quiet_hours, tz)
                min_spacing_by_queue[queue_key] = next_slot
                item["scheduled_for"] = _to_iso_utc(next_slot)
                changed = True
                continue

            if can_force_now:
                forced_queues.add(queue_key)

            try:
                if item.get("caption"):
                    cut_dict["caption"] = str(item["caption"])
                target_plats = [plat] if plat else (item.get("platforms") or [])
                res, failed_plats, plat_errors = self._execute_publish_resilient(
                    prof,
                    cut_dict,
                    target_plats,
                    work_dir,
                    existing_results=existing_res,
                )
                if "moderation" in plat_errors:
                    item["status"] = "blocked_nudity"
                    item["last_error"] = plat_errors["moderation"][:500]
                    changed = True
                    self._save_queue_state(state)
                    continue
                if "moderation_platform_blocked" in plat_errors:
                    item["status"] = "blocked_moderation"
                    item["last_error"] = plat_errors["moderation_platform_blocked"][:500]
                    changed = True
                    self._save_queue_state(state)
                    continue

                item["results"] = res
                done_key = _PLATFORM_DONE_KEY.get(plat, "")
                posted_this_plat = bool(done_key and res.get(done_key) and not existing_res.get(done_key))
                if posted_this_plat and plat:
                    self._record_last_published(state, prof.id, plat, now)
                    min_spacing_by_queue[queue_key] = now

                if not failed_plats:
                    item["status"] = "published"
                    item["published_at"] = _to_iso_utc(now)
                    item.pop("last_error", None)
                    item.pop("platform_errors", None)
                    cut_dict["status"] = "published"
                    published_cuts.append(cut_dict)
                    published_queue_items.append(item)
                    if posted_this_plat and plat:
                        self._reflow_queue_after_advance(state, prof, plat, anchor_utc=now)
                    completed_reports.append({"item": item, "cut": cut_dict, "platform": plat, "results": res})
                else:
                    item["platforms"] = failed_plats
                    item["platform_errors"] = plat_errors
                    item["last_error"] = "; ".join(f"{k}: {v}" for k, v in plat_errors.items())[:500]
                    is_rate_limit = any(
                        code in err
                        for err in plat_errors.values()
                        for code in ("spam_risk_too_many_pending_share", "quotaExceeded", "uploadLimitExceeded")
                    )
                    retry_count = int(item.get("retry_count", 0)) + 1
                    item["retry_count"] = retry_count
                    max_retries = 12 if is_rate_limit else 3
                    if retry_count >= max_retries:
                        item["status"] = "failed"
                    else:
                        delay_min = 120 if is_rate_limit else 30
                        item["scheduled_for"] = _to_iso_utc(now + timedelta(minutes=delay_min))
                    logger.warning(
                        "Falha na fila %s para %s (retry=%s): falharam=%s",
                        plat,
                        item["cut_id"],
                        retry_count,
                        failed_plats,
                    )
                changed = True
                self._save_queue_state(state)
            except Exception as exc:
                logger.error("Erro ao publicar item agendado %s (%s): %s", item["cut_id"], plat, exc)
                item["last_error"] = str(exc)
                retry_count = int(item.get("retry_count", 0)) + 1
                item["retry_count"] = retry_count
                if retry_count >= 3:
                    item["status"] = "failed"
                else:
                    item["scheduled_for"] = _to_iso_utc(now + timedelta(minutes=30))
                changed = True
                self._save_queue_state(state)

        if changed:
            self._save_queue_state(state)
        return completed_reports

    def _execute_publish(
        self,
        profile: ProfileConfig,
        cut_dict: Dict[str, Any],
        platforms: List[str],
        work_dir: Path,
    ) -> Dict[str, str]:
        results, failed, errors = self._execute_publish_resilient(profile, cut_dict, platforms, work_dir)
        if failed:
            first_p = failed[0]
            raise RuntimeError(f"Falha ao publicar em {first_p}: {errors.get(first_p, 'erro desconhecido')}")
        return results

    def _execute_publish_resilient(
        self,
        profile: ProfileConfig,
        cut_dict: Dict[str, Any],
        platforms: List[str],
        work_dir: Path,
        existing_results: Optional[Dict[str, str]] = None,
    ) -> tuple[Dict[str, str], List[str], Dict[str, str]]:
        work_dir.mkdir(parents=True, exist_ok=True)
        local_master = work_dir / f"{cut_dict['cut_id']}_publish_master.mp4"
        if not local_master.exists() or local_master.stat().st_size == 0:
            self.store.download_file_by_ref(cut_dict["master_file_id"], local_master)

        caption = str(cut_dict.get("caption", ""))
        requested_canon = [self._canonical_platform(p) for p in platforms]

        # Barreira de moderação visual real no arquivo baixado (reavalia itens antigos ou sem vision_checked)
        mod_info = cut_dict.get("moderation") if isinstance(cut_dict.get("moderation"), dict) else {}
        if not mod_info.get("vision_checked"):
            moderation = moderate_video_for_platforms(
                video_path=local_master,
                requested_platforms=list(SUPPORTED_PLATFORMS),
                title=f"{cut_dict.get('dominant_title') or ''} {caption}",
                transcript_text=str(cut_dict.get("transcript_snippet") or ""),
            )
            mod_info = moderation.to_dict()
            cut_dict["moderation"] = mod_info
            cut_dict["_dirty_moderation"] = True

        if mod_info.get("vision_checked"):
            allowed_list = mod_info.get("allowed_platforms")
            blocked_list = mod_info.get("blocked_platforms") or []
            reason = str(mod_info.get("reason") or "Conteúdo retido pela moderação.")
            if allowed_list is not None:
                allowed_canon = {self._canonical_platform(p) for p in allowed_list}
                blocked_canon = {self._canonical_platform(p) for p in blocked_list}
                if not allowed_canon:
                    logger.warning("Corte %s bloqueado por nudismo/decência antes de postar: %s", cut_dict["cut_id"], reason)
                    return dict(existing_results or {}), [], {"moderation": reason}
                filtered_req = [p for p in requested_canon if p in allowed_canon]
                if not filtered_req and any(p in blocked_canon for p in requested_canon):
                    return dict(existing_results or {}), [], {"moderation_platform_blocked": reason}
                requested_canon = filtered_req

        results: Dict[str, str] = dict(existing_results or {})
        failed_platforms: List[str] = []
        platform_errors: Dict[str, str] = {}

        # 1. Instagram Reels
        if "instagram" in requested_canon and profile.publishing.instagram.enabled:
            if results.get("instagram_reel_id"):
                logger.info("Instagram já publicado antes para %s; pulando para evitar duplicação.", cut_dict["cut_id"])
            else:
                try:
                    ig = InstagramReelsPublisher(profile)
                    if ig.is_ready:
                        video_url = str(cut_dict.get("download_url") or "")
                        if not video_url:
                            try:
                                video_url = self.store.ensure_public_download_url(cut_dict["master_file_id"])
                                if video_url:
                                    cut_dict["download_url"] = video_url
                            except Exception:
                                pass
                        results["instagram_reel_id"] = ig.publish_reel(
                            local_master,
                            caption,
                            video_url=video_url,
                            title_hint=str(cut_dict.get("teaser_headline") or cut_dict.get("dominant_title") or ""),
                            creator_name=str(cut_dict.get("creator_name") or ""),
                            game_name=str(cut_dict.get("game_name") or ""),
                        )
                    else:
                        results["instagram"] = "skipped_unconfigured"
                except Exception as exc:
                    logger.error("Erro Instagram para %s: %s", cut_dict["cut_id"], exc)
                    failed_platforms.append("instagram")
                    platform_errors["instagram"] = str(exc)

        # 2. Facebook Page Reels
        fb_cfg = getattr(profile.publishing, "facebook", None)
        fb_enabled = bool(fb_cfg.enabled) if fb_cfg is not None else True
        if "facebook" in requested_canon and fb_enabled:
            if results.get("facebook_reel_id"):
                logger.info("Facebook já publicado antes para %s; pulando para evitar duplicação.", cut_dict["cut_id"])
            else:
                try:
                    fb = FacebookReelsPublisher(profile)
                    if fb.is_ready and fb.has_publish_permission():
                        results["facebook_reel_id"] = fb.publish_reel(local_master, caption)
                    elif fb.is_ready:
                        logger.warning(
                            "Facebook Page detectado, mas token ainda sem permissão `pages_manage_posts`; defina FB_ACCESS_TOKEN."
                        )
                        results["facebook"] = "skipped_missing_pages_manage_posts"
                    else:
                        results["facebook"] = "skipped_unconfigured"
                except Exception as exc:
                    logger.error("Erro Facebook para %s: %s", cut_dict["cut_id"], exc)
                    failed_platforms.append("facebook")
                    platform_errors["facebook"] = str(exc)

        # 3. TikTok
        if "tiktok" in requested_canon and profile.publishing.tiktok.enabled:
            if results.get("tiktok_publish_id"):
                logger.info("TikTok já publicado antes para %s; pulando para evitar duplicação.", cut_dict["cut_id"])
            else:
                try:
                    tt = TikTokPublisher(profile)
                    if tt.is_ready:
                        results["tiktok_publish_id"] = tt.publish_video(local_master, caption)
                        if tt.last_publish_mode:
                            results["tiktok_mode"] = tt.last_publish_mode
                    else:
                        results["tiktok"] = "skipped_unconfigured"
                except Exception as exc:
                    logger.error("Erro TikTok para %s: %s", cut_dict["cut_id"], exc)
                    failed_platforms.append("tiktok")
                    platform_errors["tiktok"] = str(exc)

        # 4. YouTube Shorts
        if "youtube" in requested_canon and profile.publishing.youtube_shorts.enabled:
            if results.get("youtube_short_url"):
                logger.info("YouTube Shorts já publicado antes para %s; pulando para evitar duplicação.", cut_dict["cut_id"])
            else:
                try:
                    yt = YouTubeShortsPublisher(profile)
                    if yt.is_ready:
                        results["youtube_short_url"] = yt.publish_short(
                            local_master,
                            caption,
                            title_hint=str(cut_dict.get("teaser_headline") or ""),
                            creator_name=str(cut_dict.get("creator_name") or ""),
                        )
                    else:
                        results["youtube_shorts"] = "skipped_unconfigured"
                except Exception as exc:
                    logger.error("Erro YouTube Shorts para %s: %s", cut_dict["cut_id"], exc)
                    failed_platforms.append("youtube")
                    platform_errors["youtube"] = str(exc)

        return results, failed_platforms, platform_errors
