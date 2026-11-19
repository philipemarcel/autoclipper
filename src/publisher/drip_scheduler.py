from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.ai.deduplicator import find_duplicate_cut
from src.models import ProfileConfig
from src.publisher.facebook_reels import FacebookReelsPublisher
from src.publisher.instagram_reels import InstagramReelsPublisher
from src.publisher.tiktok_poster import TikTokPublisher
from src.publisher.youtube_shorts import YouTubeShortsPublisher
from src.storage.gdrive import DriveStore

logger = logging.getLogger(__name__)

DRIP_STATE_FILE = "drip_queue.json"


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
    Fila Espaçada Inteligente (Smart Drip Queue):
      - Se não houve post nos últimos `drip_interval_minutes` (ex: 75 min) E estamos fora da janela de
        madrugada (`quiet_hours_brt`), o 1º corte aprovado sobe IMEDIATAMENTE!
      - Cortes aprovados em sequência entram espaçados automaticamente a cada `drip_interval_minutes`,
        pulando horários mortos da madrugada para proteger o alcance no Reels e TikTok.
      - O botão `Postar Agora` (`force_immediate=True`) fura a fila e publica no ato a qualquer hora.
    """

    def __init__(self, store: DriveStore) -> None:
        self.store = store

    def _load_queue_state(self) -> Dict[str, Any]:
        return self.store.load_state_json(
            DRIP_STATE_FILE,
            default={"last_published_by_profile": {}, "items": []},
        )

    def _save_queue_state(self, state: Dict[str, Any]) -> None:
        self.store.save_state_json(DRIP_STATE_FILE, state)

    def get_effective_interval_minutes(
        self,
        profile: ProfileConfig,
        state: Optional[Dict[str, Any]] = None,
    ) -> int:
        q_state = state if state is not None else self._load_queue_state()
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

    def update_timer_override(
        self,
        profiles_by_id: Dict[str, ProfileConfig],
        interval_minutes: int,
        quiet_hours_str: str = "",
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Atualiza o intervalo da Fila Espaçada (em minutos) diretamente em `drip_queue.json`
        e recalcula os horários dos itens ainda `pending` na fila.
        """
        now = now_utc or datetime.now(timezone.utc)
        state = self._load_queue_state()
        new_interval = max(5, min(1440, int(interval_minutes)))
        state["drip_interval_override_minutes"] = new_interval

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

        delta = timedelta(minutes=new_interval)
        pending_items = [it for it in state.get("items", []) if it.get("status") == "pending"]
        pending_items.sort(key=lambda it: str(it.get("created_at") or it.get("scheduled_for") or ""))

        cursor_by_profile: Dict[str, datetime] = {}
        rescheduled = 0
        next_local = ""

        for item in pending_items:
            pid = str(item.get("profile_id") or "")
            prof = profiles_by_id.get(pid)
            if not prof:
                continue
            tz = ZoneInfo(prof.publishing.timezone)
            quiet_hours = self.get_effective_quiet_hours(prof, state)

            if pid not in cursor_by_profile:
                last_pub = state.get("last_published_by_profile", {}).get(pid)
                base_dt = max(now, _parse_iso(last_pub) + delta) if last_pub else now
            else:
                base_dt = max(now, cursor_by_profile[pid] + delta)

            slot = self._push_past_quiet_hours(base_dt, quiet_hours, tz)
            cursor_by_profile[pid] = slot
            item["scheduled_for"] = _to_iso_utc(slot)
            rescheduled += 1
            if not next_local:
                next_local = slot.astimezone(tz).strftime("%d/%m às %H:%M")

        self._save_queue_state(state)
        return {
            "interval_minutes": new_interval,
            "quiet_hours": state.get("quiet_hours_override", "default"),
            "rescheduled_count": rescheduled,
            "next_scheduled_local": next_local,
        }

    def compute_next_slot(
        self,
        profile: ProfileConfig,
        state: Dict[str, Any],
        now_utc: Optional[datetime] = None,
    ) -> Tuple[datetime, bool]:
        """
        Calcula `(scheduled_utc, is_immediate)`.
        """
        now = now_utc or datetime.now(timezone.utc)
        eff_minutes = self.get_effective_interval_minutes(profile, state)
        interval = timedelta(minutes=max(5, eff_minutes))
        quiet_hours = self.get_effective_quiet_hours(profile, state)
        tz = ZoneInfo(profile.publishing.timezone)

        last_pub_iso = state.get("last_published_by_profile", {}).get(profile.id)
        pending_for_profile = [
            _parse_iso(it["scheduled_for"])
            for it in state.get("items", [])
            if it.get("profile_id") == profile.id and it.get("status") == "pending"
        ]

        anchors: List[datetime] = []
        if last_pub_iso:
            anchors.append(_parse_iso(last_pub_iso) + interval)
        if pending_for_profile:
            anchors.append(max(pending_for_profile) + interval)

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

        # Normaliza nomes e remove plataformas já publicadas para o mesmo cut_id (evita duplicação!)
        already_done_plats: set[str] = set()
        for it in state.get("items", []):
            if it.get("cut_id") != cut_dict["cut_id"]:
                continue
            res = it.get("results") or {}
            if res.get("instagram_reel_id"):
                already_done_plats.add("instagram")
            if res.get("facebook_reel_id"):
                already_done_plats.add("facebook")
            if res.get("tiktok_publish_id"):
                already_done_plats.add("tiktok")
            if res.get("youtube_short_url"):
                already_done_plats.add("youtube")
            if it.get("status") == "published" and not res:
                already_done_plats.update(self._canonical_platform(p) for p in (it.get("platforms") or []))

        clean_platforms: List[str] = []
        seen_canon: set[str] = set()
        for p in platforms:
            canon = self._canonical_platform(p)
            if canon in already_done_plats or canon in seen_canon:
                continue
            seen_canon.add(canon)
            clean_platforms.append(p)

        if not clean_platforms:
            return {"mode": "already_published", "platforms": []}

        # Bloqueia cut_id novo cujo conteúdo/descrição/horário extraído duplique item já na fila ou publicado
        same_cut_exists = any(it.get("cut_id") == cut_dict.get("cut_id") for it in state.get("items", []))
        if not same_cut_exists:
            dup = find_duplicate_cut(cut_dict, processed_state=None, queue_state=state)
            if dup.is_duplicate:
                logger.warning(
                    "Bloqueando enfileiramento/publicação de %s (duplicata de %s): %s",
                    cut_dict.get("cut_id"),
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

        # Evita duplicar o mesmo cut_id pendente na fila
        existing = [
            it for it in state.get("items", []) if it.get("cut_id") == cut_dict["cut_id"] and it.get("status") == "pending"
        ]
        if existing and not force_immediate:
            return {"mode": "already_queued", "scheduled_for": existing[0]["scheduled_for"]}

        if force_immediate:
            # Remove eventual entrada pendente da fila pois vamos publicar agora
            state["items"] = [
                it for it in state.get("items", []) if not (it.get("cut_id") == cut_dict["cut_id"] and it.get("status") == "pending")
            ]
            result, failed_plats, plat_errors = self._execute_publish_resilient(
                profile, cut_dict, clean_platforms, work_dir
            )
            if result:
                state.setdefault("last_published_by_profile", {})[profile.id] = _to_iso_utc(now)
            if failed_plats:
                backoff_min = 120 if any("spam_risk_too_many_pending_share" in e for e in plat_errors.values()) else 30
                state.setdefault("items", []).append(
                    {
                        "cut_id": cut_dict["cut_id"],
                        "profile_id": profile.id,
                        "platforms": failed_plats,
                        "scheduled_for": _to_iso_utc(now + timedelta(minutes=backoff_min)),
                        "status": "pending",
                        "results": result,
                        "platform_errors": plat_errors,
                        "last_error": "; ".join(f"{k}: {v}" for k, v in plat_errors.items()),
                        "retry_count": 1,
                        "discord_message_id": cut_dict.get("discord_message_id", ""),
                        "created_at": _to_iso_utc(now),
                        **dedup_meta,
                    }
                )
            elif result:
                state.setdefault("items", []).append(
                    {
                        "cut_id": cut_dict["cut_id"],
                        "profile_id": profile.id,
                        "platforms": clean_platforms,
                        "scheduled_for": _to_iso_utc(now),
                        "published_at": _to_iso_utc(now),
                        "status": "published",
                        "results": result,
                        "discord_message_id": cut_dict.get("discord_message_id", ""),
                        "created_at": _to_iso_utc(now),
                        **dedup_meta,
                    }
                )
            self._save_queue_state(state)
            return {"mode": "published_now", "results": result, "failed_platforms": failed_plats}

        slot_utc, is_immediate = self.compute_next_slot(profile, state, now_utc=now)
        if is_immediate:
            result, failed_plats, plat_errors = self._execute_publish_resilient(
                profile, cut_dict, clean_platforms, work_dir
            )
            if result:
                state.setdefault("last_published_by_profile", {})[profile.id] = _to_iso_utc(now)
            if failed_plats:
                backoff_min = 120 if any("spam_risk_too_many_pending_share" in e for e in plat_errors.values()) else 30
                state.setdefault("items", []).append(
                    {
                        "cut_id": cut_dict["cut_id"],
                        "profile_id": profile.id,
                        "platforms": failed_plats,
                        "scheduled_for": _to_iso_utc(now + timedelta(minutes=backoff_min)),
                        "status": "pending",
                        "results": result,
                        "platform_errors": plat_errors,
                        "last_error": "; ".join(f"{k}: {v}" for k, v in plat_errors.items()),
                        "retry_count": 1,
                        "discord_message_id": cut_dict.get("discord_message_id", ""),
                        "created_at": _to_iso_utc(now),
                        **dedup_meta,
                    }
                )
            elif result:
                state.setdefault("items", []).append(
                    {
                        "cut_id": cut_dict["cut_id"],
                        "profile_id": profile.id,
                        "platforms": clean_platforms,
                        "scheduled_for": _to_iso_utc(now),
                        "published_at": _to_iso_utc(now),
                        "status": "published",
                        "results": result,
                        "discord_message_id": cut_dict.get("discord_message_id", ""),
                        "created_at": _to_iso_utc(now),
                        **dedup_meta,
                    }
                )
            self._save_queue_state(state)
            return {"mode": "published_now", "results": result, "failed_platforms": failed_plats}

        item = {
            "cut_id": cut_dict["cut_id"],
            "profile_id": profile.id,
            "platforms": clean_platforms,
            "scheduled_for": _to_iso_utc(slot_utc),
            "status": "pending",
            "discord_message_id": cut_dict.get("discord_message_id", ""),
            "created_at": _to_iso_utc(now),
            **dedup_meta,
        }
        state.setdefault("items", []).append(item)
        self._save_queue_state(state)

        tz = ZoneInfo(profile.publishing.timezone)
        local_str = slot_utc.astimezone(tz).strftime("%d/%m às %H:%M")
        return {
            "mode": "queued",
            "scheduled_for": item["scheduled_for"],
            "scheduled_local": local_str,
        }

    def process_due_queue(
        self,
        profiles_by_id: Dict[str, ProfileConfig],
        processed_state: Dict[str, Any],
        work_dir: Path,
        now_utc: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        now = now_utc or datetime.now(timezone.utc)
        state = self._load_queue_state()
        cuts_by_id = {c["cut_id"]: c for c in processed_state.get("cuts", [])}

        completed_reports: List[Dict[str, Any]] = []
        changed = False
        min_spacing_profiles: Dict[str, datetime] = {}

        # Ordena pendentes por horário agendado para sempre respeitar a fila em ordem cronológica
        pending_ordered = sorted(
            [it for it in state.get("items", []) if it.get("status") == "pending"],
            key=lambda it: str(it.get("scheduled_for") or ""),
        )

        for item in pending_ordered:
            due_dt = _parse_iso(item["scheduled_for"])
            if due_dt > now:
                continue

            prof = profiles_by_id.get(item["profile_id"])
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

            eff_minutes = self.get_effective_interval_minutes(prof, state)
            interval = timedelta(minutes=eff_minutes)
            min_gap = timedelta(minutes=max(10, eff_minutes - 5))
            quiet_hours = self.get_effective_quiet_hours(prof, state)
            tz = ZoneInfo(prof.publishing.timezone)

            # Evita rajada de posts em intervalo curto quando vários itens vencem juntos
            last_iso = state.get("last_published_by_profile", {}).get(prof.id)
            last_dt = _parse_iso(last_iso) if last_iso else None
            if prof.id in min_spacing_profiles:
                anchor = min_spacing_profiles[prof.id] + interval
                next_slot = self._push_past_quiet_hours(max(now, anchor), quiet_hours, tz)
                min_spacing_profiles[prof.id] = next_slot
                item["scheduled_for"] = _to_iso_utc(next_slot)
                changed = True
                continue
            if last_dt is not None and (now - last_dt) < min_gap:
                next_slot = self._push_past_quiet_hours(last_dt + interval, quiet_hours, tz)
                if next_slot <= now:
                    next_slot = self._push_past_quiet_hours(now + timedelta(minutes=eff_minutes), quiet_hours, tz)
                min_spacing_profiles[prof.id] = next_slot
                item["scheduled_for"] = _to_iso_utc(next_slot)
                changed = True
                continue

            try:
                if item.get("caption"):
                    cut_dict["caption"] = str(item["caption"])
                existing_res = dict(item.get("results") or {})
                res, failed_plats, plat_errors = self._execute_publish_resilient(
                    prof,
                    cut_dict,
                    item.get("platforms") or [],
                    work_dir,
                    existing_results=existing_res,
                )
                item["results"] = res
                posted_content = any(
                    res.get(k) for k in ("instagram_reel_id", "facebook_reel_id", "tiktok_publish_id", "youtube_short_url")
                )
                if posted_content:
                    state.setdefault("last_published_by_profile", {})[prof.id] = _to_iso_utc(now)
                    min_spacing_profiles[prof.id] = now

                if not failed_plats:
                    item["status"] = "published"
                    item["published_at"] = _to_iso_utc(now)
                    item.pop("last_error", None)
                    item.pop("platform_errors", None)
                    cut_dict["status"] = "published"
                    completed_reports.append({"item": item, "cut": cut_dict, "results": res})
                else:
                    # Preserva o sucesso parcial e NUNCA re-publica nas plataformas que já receberam o vídeo
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
                        "Entrega parcial/falha para %s (retry=%s): ok=%s | falharam=%s",
                        item["cut_id"],
                        retry_count,
                        list(res.keys()),
                        failed_plats,
                    )
                changed = True
                self._save_queue_state(state)
            except Exception as exc:
                logger.error("Erro ao publicar item agendado %s: %s", item["cut_id"], exc)
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

    @staticmethod
    def _canonical_platform(name: str) -> str:
        low = str(name or "").strip().lower()
        if low in ("youtube", "youtube_shorts"):
            return "youtube"
        return low

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
        self.store.download_file_by_ref(cut_dict["master_file_id"], local_master)

        caption = str(cut_dict.get("caption", ""))
        results: Dict[str, str] = dict(existing_results or {})
        failed_platforms: List[str] = []
        platform_errors: Dict[str, str] = {}
        requested_canon = [self._canonical_platform(p) for p in platforms]

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
