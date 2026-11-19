from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.models import ProfileConfig
from src.publisher.instagram_reels import InstagramReelsPublisher
from src.publisher.tiktok_poster import TikTokPublisher
from src.storage.gdrive import DriveStore

logger = logging.getLogger(__name__)

DRIP_STATE_FILE = "drip_queue.json"


def _parse_iso(ts: str) -> datetime:
    cleaned = ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


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
        platforms: List[str],  # ["instagram", "tiktok"]
        work_dir: Path,
        force_immediate: bool = False,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = now_utc or datetime.now(timezone.utc)
        state = self._load_queue_state()

        # Evita duplicar o mesmo cut_id na fila
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
            result = self._execute_publish(profile, cut_dict, platforms, work_dir)
            state.setdefault("last_published_by_profile", {})[profile.id] = _to_iso_utc(now)
            self._save_queue_state(state)
            return {"mode": "published_now", "results": result}

        slot_utc, is_immediate = self.compute_next_slot(profile, state, now_utc=now)
        if is_immediate:
            result = self._execute_publish(profile, cut_dict, platforms, work_dir)
            state.setdefault("last_published_by_profile", {})[profile.id] = _to_iso_utc(now)
            self._save_queue_state(state)
            return {"mode": "published_now", "results": result}

        item = {
            "cut_id": cut_dict["cut_id"],
            "profile_id": profile.id,
            "platforms": platforms,
            "scheduled_for": _to_iso_utc(slot_utc),
            "status": "pending",
            "discord_message_id": cut_dict.get("discord_message_id", ""),
            "created_at": _to_iso_utc(now),
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

        for item in state.get("items", []):
            if item.get("status") != "pending":
                continue
            due_dt = _parse_iso(item["scheduled_for"])
            if due_dt > now:
                continue

            prof = profiles_by_id.get(item["profile_id"])
            cut_dict = cuts_by_id.get(item["cut_id"])
            if not prof or not cut_dict:
                item["status"] = "skipped_missing"
                changed = True
                continue

            try:
                res = self._execute_publish(prof, cut_dict, item["platforms"], work_dir)
                item["status"] = "published"
                item["published_at"] = _to_iso_utc(now)
                item["results"] = res
                cut_dict["status"] = "published"
                state.setdefault("last_published_by_profile", {})[prof.id] = _to_iso_utc(now)
                completed_reports.append({"item": item, "cut": cut_dict, "results": res})
                changed = True
            except Exception as exc:
                logger.error("Erro ao publicar item agendado %s: %s", item["cut_id"], exc)
                item["last_error"] = str(exc)
                changed = True

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
        work_dir.mkdir(parents=True, exist_ok=True)
        local_master = work_dir / f"{cut_dict['cut_id']}_publish_master.mp4"
        self.store.download_file_by_ref(cut_dict["master_file_id"], local_master)

        caption = str(cut_dict.get("caption", ""))
        results: Dict[str, str] = {}

        if "instagram" in platforms and profile.publishing.instagram.enabled:
            ig = InstagramReelsPublisher(profile)
            if ig.is_ready:
                video_url = str(cut_dict.get("download_url") or "")
                if ig.is_instagram_login_token and not video_url:
                    video_url = self.store.ensure_public_download_url(cut_dict["master_file_id"])
                    if video_url:
                        cut_dict["download_url"] = video_url
                results["instagram_reel_id"] = ig.publish_reel(local_master, caption, video_url=video_url)
            else:
                results["instagram"] = "skipped_unconfigured"

        if "tiktok" in platforms and profile.publishing.tiktok.enabled:
            tt = TikTokPublisher(profile)
            if tt.is_ready:
                results["tiktok_publish_id"] = tt.publish_video(local_master, caption)
                if tt.last_publish_mode:
                    results["tiktok_mode"] = tt.last_publish_mode
            else:
                results["tiktok"] = "skipped_unconfigured"

        return results
