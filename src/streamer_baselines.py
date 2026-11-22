from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.models import ProfileConfig
from src.storage.gdrive import DriveStore

logger = logging.getLogger(__name__)

BASELINES_STATE_FILE = "streamer_baselines.json"
WEEKLY_REFRESH_DAYS = 7


def _parse_iso_utc(ts: str) -> Optional[datetime]:
    raw = str(ts or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _to_iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_robust_average(values: Iterable[float]) -> float:
    """
    Calcula a média de score/views de um streamer.
    Quando há >= 6 amostras, apara o maior outlier extremo (top 1) caso ele seja > 4x a mediana,
    evitando que 1 único clipe mega-viral de 15.000 pontos bloqueie todos os outros cortes fortes do canal.
    """
    clean = sorted(float(v) for v in values if v is not None and float(v) > 0)
    if not clean:
        return 0.0
    if len(clean) >= 6:
        median = clean[len(clean) // 2]
        if median > 0 and clean[-1] > median * 4.0:
            clean = clean[:-1]
    return round(sum(clean) / len(clean), 1)


def extract_streamer_historical_scores(
    processed_state: Optional[Dict[str, Any]],
    slug: str,
) -> List[float]:
    """Extrai os scores históricos reais de um streamer a partir de `processed_moments.json`."""
    target = str(slug or "").strip().lower().lstrip("@")
    if not target or not processed_state:
        return []

    scores: List[float] = []
    for cut in processed_state.get("cuts") or []:
        if not isinstance(cut, dict):
            continue
        login = str(cut.get("twitch_login") or "").strip().lower().lstrip("@")
        creator = str(cut.get("creator_name") or "").strip().lower().lstrip("@")
        if target not in {login, creator}:
            continue
        # Ignora cortes cegos de fallback sem engajamento real da comunidade
        if str(cut.get("mode") or "") in {"kick_live", "kick_vod"}:
            continue
        try:
            sc = float(cut.get("score") or 0.0)
        except (TypeError, ValueError):
            sc = 0.0
        if sc > 0:
            scores.append(sc)
    return scores


class StreamerBaselineManager:
    """
    Gerencia a nota de corte dinâmica (`min_views_single_clip_live` e `min_views_post_live`)
    baseada na média de score de cada streamer:
      - Calcula automaticamente para todos os streamers atuais e para qualquer novo streamer adicionado.
      - Recalcula e atualiza a média automaticamente a cada 7 dias (`WEEKLY_REFRESH_DAYS`).
    """

    def __init__(self, store: DriveStore, refresh_days: int = WEEKLY_REFRESH_DAYS) -> None:
        self.store = store
        self.refresh_days = max(1, int(refresh_days))

    def load_state(self) -> Dict[str, Any]:
        state = self.store.load_state_json(
            BASELINES_STATE_FILE,
            default={"refresh_interval_days": self.refresh_days, "streamers": {}},
        )
        if not isinstance(state.get("streamers"), dict):
            state["streamers"] = {}
        state.setdefault("refresh_interval_days", self.refresh_days)
        return state

    def save_state(self, state: Dict[str, Any]) -> None:
        self.store.save_state_json(BASELINES_STATE_FILE, state)

    def is_due_for_refresh(
        self,
        entry: Optional[Dict[str, Any]],
        now_utc: Optional[datetime] = None,
    ) -> bool:
        if not entry or not isinstance(entry, dict):
            return True
        now = now_utc or datetime.now(timezone.utc)
        next_dt = _parse_iso_utc(str(entry.get("next_refresh_at") or ""))
        if next_dt is not None:
            return now >= next_dt
        updated_dt = _parse_iso_utc(str(entry.get("updated_at") or ""))
        if updated_dt is None:
            return True
        return (now - updated_dt) >= timedelta(days=self.refresh_days)

    def compute_streamer_entry(
        self,
        slug: str,
        platform: str,
        processed_state: Optional[Dict[str, Any]] = None,
        live_clip_scores: Optional[List[float]] = None,
        default_floor: int = 80,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = now_utc or datetime.now(timezone.utc)
        clean_slug = str(slug or "").strip().lower().lstrip("@")
        hist_scores = extract_streamer_historical_scores(processed_state, clean_slug)
        api_scores = [float(x) for x in (live_clip_scores or []) if x is not None and float(x) > 0]

        # Combina histórico do streamer + amostra recente da API (priorizando dados reais disponíveis)
        combined = hist_scores + api_scores
        avg_score = compute_robust_average(combined)
        if avg_score <= 0:
            avg_score = float(max(50, int(default_floor)))

        cutoff_int = max(50, int(round(avg_score)))
        next_refresh = now + timedelta(days=self.refresh_days)

        return {
            "streamer": clean_slug,
            "platform": platform,
            "avg_score": avg_score,
            "min_views_single_clip_live": cutoff_int,
            "min_views_post_live": cutoff_int,
            "min_score_cutoff": avg_score,
            "historical_samples": len(hist_scores),
            "api_samples": len(api_scores),
            "sample_count": len(combined),
            "updated_at": _to_iso_utc(now),
            "next_refresh_at": _to_iso_utc(next_refresh),
        }

    def ensure_profiles_baselines(
        self,
        profiles: List[ProfileConfig],
        processed_state: Optional[Dict[str, Any]] = None,
        helix: Any = None,
        kick_clip_fetcher: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        force_refresh: bool = False,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Garante que todos os streamers (Twitch e Kick) dos perfis ativos tenham sua média calculada
        e atualizada semanalmente (ou imediatamente quando um novo streamer é adicionado).
        """
        from src.kick.analyzer import extract_kick_slug

        now = now_utc or datetime.now(timezone.utc)
        state = self.load_state()
        streamers_map: Dict[str, Dict[str, Any]] = state.setdefault("streamers", {})
        changed = False

        for prof in profiles:
            floor = min(
                int(getattr(prof.detection, "min_views_single_clip_live", 120) or 120),
                int(getattr(prof.detection, "min_views_post_live", 120) or 120),
            )

            # 1) Streamer Twitch do perfil
            tw_slug = str(prof.twitch_login or "").strip().lower().lstrip("@")
            if tw_slug and tw_slug not in {"seu_criador_aqui", ""}:
                existing = streamers_map.get(tw_slug)
                if force_refresh or self.is_due_for_refresh(existing, now_utc=now):
                    api_scores: List[float] = []
                    if helix is not None:
                        try:
                            user = helix.get_user_by_login(tw_slug)
                            if user:
                                start_7d = _to_iso_utc(now - timedelta(days=7))
                                end_now = _to_iso_utc(now)
                                clips = helix.get_clips_in_window(
                                    broadcaster_id=str(user["id"]),
                                    started_at_iso=start_7d,
                                    ended_at_iso=end_now,
                                    max_pages=2,
                                )
                                for c in clips:
                                    v = max(1, int(getattr(c, "view_count", 0) or 0))
                                    sc = 14.0 + (math.log1p(v) * 22.0) + (math.sqrt(v) * 4.5)
                                    api_scores.append(round(sc, 1))
                        except Exception as exc:
                            logger.warning("Aviso ao buscar clipes 7d da Twitch para média de %s: %s", tw_slug, exc)

                    entry = self.compute_streamer_entry(
                        slug=tw_slug,
                        platform="twitch",
                        processed_state=processed_state,
                        live_clip_scores=api_scores,
                        default_floor=floor,
                        now_utc=now,
                    )
                    streamers_map[tw_slug] = entry
                    changed = True
                    logger.info(
                        "Baseline semanal atualizada para Twitch @%s: avg_score=%.1f -> min_views=%d (amostras=%d)",
                        tw_slug,
                        entry["avg_score"],
                        entry["min_views_post_live"],
                        entry["sample_count"],
                    )

            # 2) Streamers da Kick configurados no perfil
            for k_ref in prof.kick_channels or []:
                k_slug = extract_kick_slug(k_ref)
                if not k_slug:
                    continue
                existing = streamers_map.get(k_slug)
                if force_refresh or self.is_due_for_refresh(existing, now_utc=now):
                    api_scores = []
                    if kick_clip_fetcher is not None:
                        try:
                            k_clips = kick_clip_fetcher(k_slug)
                            for c in k_clips or []:
                                v = int(c.get("views") or 0)
                                if v > 0:
                                    api_scores.append(float(v))
                        except Exception as exc:
                            logger.warning("Aviso ao buscar clipes 7d da Kick para média de %s: %s", k_slug, exc)

                    entry = self.compute_streamer_entry(
                        slug=k_slug,
                        platform="kick",
                        processed_state=processed_state,
                        live_clip_scores=api_scores,
                        default_floor=floor,
                        now_utc=now,
                    )
                    streamers_map[k_slug] = entry
                    changed = True
                    logger.info(
                        "Baseline semanal atualizada para Kick @%s: avg_score=%.1f -> min_views=%d (amostras=%d)",
                        k_slug,
                        entry["avg_score"],
                        entry["min_views_post_live"],
                        entry["sample_count"],
                    )

        if changed:
            state["updated_at"] = _to_iso_utc(now)
            self.save_state(state)

        return streamers_map
