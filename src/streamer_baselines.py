from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.models import ProfileConfig
from src.storage.gdrive import DriveStore

logger = logging.getLogger(__name__)

BASELINES_STATE_FILE = "streamer_baselines.json"
WEEKLY_REFRESH_DAYS = 7
DEFAULT_TARGET_CUTS_PER_LIVE = 3
DEFAULT_HOURS_PER_CUT = 2.0  # Limite proporcional ao tempo de live: 1 corte a cada 2h de live
DEFAULT_MAX_CUTS_HARD_CAP = 10  # Teto de segurança para maratonas de 20h+
DEFAULT_SELECTIVITY = 0.80  # Alvo seletivo (~80% da cota de tempo -> ~0.40 cortes/hora)
DEFAULT_LIVE_RATIO = 0.65  # Durante a live, exige 65% da nota de corte pós-live
MIN_FLOOR_POST_LIVE = 60
MIN_FLOOR_LIVE = 40


def _parse_iso_utc(ts: str) -> Optional[datetime]:
    raw = str(ts or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        if " " in raw and "T" not in raw:
            raw = raw.replace(" ", "T", 1)
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _to_iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _true_median(values: Sequence[float]) -> float:
    s = sorted(float(v) for v in values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def compute_stream_cut_limit(
    duration_seconds: Optional[float],
    hours_per_cut: float = DEFAULT_HOURS_PER_CUT,
    fallback_max_cuts: int = DEFAULT_TARGET_CUTS_PER_LIVE,
    hard_cap: int = 0,
) -> int:
    """
    Calcula o limite máximo de cortes proporcional ao tempo da live (`hours_per_cut`):
      - Ex. com `hours_per_cut=2.0` (1 corte a cada 2h de live):
        * 1h–2h59m -> 1 corte
        * 3h–4h59m -> 2 cortes
        * 5h–6h59m -> 3 cortes
        * 7h–8h59m -> 4 cortes
        * 9h–10h59m -> 5 cortes
        * 15h -> 8 cortes
      - Se `duration_seconds` não for informado ou `hours_per_cut <= 0`, usa `fallback_max_cuts`.
      - Se `hard_cap > 0`, aplica como teto máximo de segurança (ex: 10 cortes para lives de 20h+).
    """
    if hours_per_cut and float(hours_per_cut) > 0 and duration_seconds is not None and float(duration_seconds) > 0:
        hours = float(duration_seconds) / 3600.0
        limit = max(1, int(hours / float(hours_per_cut) + 0.5))
    else:
        limit = max(1, int(fallback_max_cuts))

    if hard_cap and int(hard_cap) > 0:
        limit = min(limit, int(hard_cap))
    return limit


def compute_robust_average(values: Iterable[float]) -> float:
    """
    Calcula uma tendência central robusta (média geométrica limitada pela mediana),
    imune a outliers extremos (ex: 1 clipe de 50.000 views não distorce o canal).
    """
    clean = sorted(float(v) for v in values if v is not None and float(v) > 0)
    if not clean:
        return 0.0
    geomean = math.exp(sum(math.log(v) for v in clean) / len(clean))
    med = _true_median(clean)
    return round(0.5 * geomean + 0.5 * med, 1)


def calibrate_streamer_thresholds(
    stream_view_pairs: Sequence[Tuple[str, float]],
    stream_durations_sec: Optional[Dict[str, float]] = None,
    hours_per_cut: float = DEFAULT_HOURS_PER_CUT,
    target_cuts_per_live: int = DEFAULT_TARGET_CUTS_PER_LIVE,
    selectivity: float = DEFAULT_SELECTIVITY,
    live_ratio: float = DEFAULT_LIVE_RATIO,
    floor_post: int = MIN_FLOOR_POST_LIVE,
    floor_live: int = MIN_FLOOR_LIVE,
    max_cuts_hard_cap: int = DEFAULT_MAX_CUTS_HARD_CAP,
) -> Dict[str, Any]:
    """
    Calibração inteligente por streamer baseada no TEMPO DE LIVE (`hours_per_cut`, padrão 1 corte a cada 2h):
      - Agrupa os clipes dos últimos 7 dias por live (`video_id` / `livestream_id` / `vod_id` / data).
      - Cada live recebe um teto proporcional à sua duração real (ex: 2h->1, 6h->3, 12h->6, 15h->8).
        Assim, 1 live explosiva nunca domina a calibração com dezenas de clipes, e lives longas
        têm espaço justo proporcional às horas transmitidas.
      - Escolhe a MAIOR nota de corte `T` em que a soma limitada por duração atinge `selectivity * soma_dos_tetos`
        (com `selectivity=0.80` e `hours_per_cut=2.0`, entrega ~0,40 cortes/hora de live com alta seletividade).
      - Define `min_views_single_clip_live = 65%` de `min_views_post_live` para capturar picos ainda ao vivo.
    """
    target = max(1, int(target_cuts_per_live))
    hpc = float(hours_per_cut or 0.0)
    dur_map = {str(k): float(v) for k, v in (stream_durations_sec or {}).items() if v is not None and float(v) > 0}

    by_stream: Dict[str, List[float]] = defaultdict(list)
    all_views: List[float] = []

    for sk, raw_v in stream_view_pairs:
        try:
            v = float(raw_v)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        key = str(sk or "default").strip() or "default"
        by_stream[key].append(v)
        all_views.append(v)

    if not all_views:
        post_cut = max(floor_post, 120)
        live_cut = max(floor_live, int(round(post_cut * live_ratio)))
        return {
            "min_views_post_live": post_cut,
            "min_views_single_clip_live": live_cut,
            "median_views": 0.0,
            "active_lives": 0,
            "total_live_hours": 0.0,
            "hours_per_cut": hpc,
            "estimated_cuts_per_hour": 0.0,
            "estimated_cuts_per_live": 0.0,
            "method": "default_fallback",
        }

    # Prefere lives com >= 2 clipes para não distorcer a calibração com dias isolados de 1 clipe
    active_items = [(sk, vs) for sk, vs in by_stream.items() if len(vs) >= 2]
    if not active_items:
        active_items = list(by_stream.items())

    # Duração padrão para lives antigas cujo VOD já não está listado: mediana das durações conhecidas do canal
    known_durations = [d for d in dur_map.values() if d >= 900.0]
    fallback_dur_sec = _true_median(known_durations) if known_durations else (
        hpc * target * 3600.0 if hpc > 0 else 0.0
    )

    stream_entries: List[Tuple[str, List[float], float, int]] = []
    for sk, vs in active_items:
        dur_sec = dur_map.get(sk, 0.0)
        eff_dur_sec = dur_sec if dur_sec > 0 else fallback_dur_sec
        cap = compute_stream_cut_limit(
            duration_seconds=eff_dur_sec if (hpc > 0 and eff_dur_sec > 0) else None,
            hours_per_cut=hpc,
            fallback_max_cuts=target,
            hard_cap=max_cuts_hard_cap,
        )
        stream_entries.append((sk, vs, eff_dur_sec / 3600.0 if eff_dur_sec > 0 else 0.0, cap))

    n_active = len(stream_entries)
    total_cap = sum(cap for _, _, _, cap in stream_entries)
    max_possible = sum(min(cap, len(vs)) for _, vs, _, cap in stream_entries)
    desired_total = min(max_possible, max(1, int(round(total_cap * selectivity))))

    active_views = [v for _, vs, _, _ in stream_entries for v in vs]
    candidates = sorted(set(active_views), reverse=True)
    chosen = candidates[-1]
    for threshold in candidates:
        capped_sum = sum(min(cap, sum(1 for v in vs if v >= threshold)) for _, vs, _, cap in stream_entries)
        if capped_sum >= desired_total:
            chosen = threshold
            break

    post_cut = max(floor_post, int(round(chosen)))
    live_cut = max(floor_live, int(round(post_cut * live_ratio)))
    actual_capped = sum(min(cap, sum(1 for v in vs if v >= post_cut)) for _, vs, _, cap in stream_entries)
    total_hours = sum(hr for _, _, hr, _ in stream_entries)

    method_name = f"duration_capped_per_{hpc:.1f}h" if (hpc > 0 and known_durations) else f"per_live_capped_top{target}"

    return {
        "min_views_post_live": post_cut,
        "min_views_single_clip_live": live_cut,
        "median_views": round(_true_median(all_views), 1),
        "active_lives": n_active,
        "total_live_hours": round(total_hours, 2),
        "hours_per_cut": hpc,
        "estimated_cuts_per_hour": round(actual_capped / total_hours, 2) if total_hours > 0 else 0.0,
        "estimated_cuts_per_live": round(actual_capped / max(1, n_active), 2),
        "method": method_name,
    }


def extract_streamer_historical_views(
    processed_state: Optional[Dict[str, Any]],
    slug: str,
    platform: str = "",
) -> List[Tuple[str, float]]:
    """
    Extrai pares `(stream_key, views_estimadas)` do histórico em `processed_moments.json`
    para fallback quando a API de clipes 7d não está disponível.
    Converte `score` da Kick (`views * 2 + bônus`) de volta para escala real de views.
    """
    target = str(slug or "").strip().lower().lstrip("@")
    if not target or not processed_state:
        return []

    pairs: List[Tuple[str, float]] = []
    for cut in processed_state.get("cuts") or []:
        if not isinstance(cut, dict):
            continue
        login = str(cut.get("twitch_login") or "").strip().lower().lstrip("@")
        creator = str(cut.get("creator_name") or "").strip().lower().lstrip("@")
        if target not in {login, creator}:
            continue
        mode = str(cut.get("mode") or "")
        if mode in {"kick_live", "kick_vod"}:
            continue

        skey = str(cut.get("stream_key") or cut.get("video_id") or str(cut.get("created_at") or "")[:10] or "hist")
        raw_max_views = cut.get("max_views")
        if raw_max_views is not None and float(raw_max_views or 0) > 0:
            est_views = float(raw_max_views)
        else:
            sc = float(cut.get("score") or 0.0)
            if sc <= 0:
                continue
            if platform == "kick" or mode == "kick_clip":
                # Na Kick, score = views * 2 + frame_bonus(60) + recency(0..30)
                est_views = max(10.0, (sc - 60.0) / 2.0)
            else:
                est_views = sc
        pairs.append((skey, est_views))
    return pairs


def extract_streamer_historical_scores(
    processed_state: Optional[Dict[str, Any]],
    slug: str,
) -> List[float]:
    """Compatibilidade: retorna lista de valores históricos positivos de um streamer."""
    return [v for _, v in extract_streamer_historical_views(processed_state, slug)]


def _extract_kick_video_durations(videos: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    from src.kick.analyzer import _ms_or_sec_to_seconds

    durations: Dict[str, float] = {}
    for v in videos or []:
        if not isinstance(v, dict):
            continue
        dur_sec = _ms_or_sec_to_seconds(v.get("duration"))
        if dur_sec <= 0:
            continue
        vid_obj = v.get("video") or {}
        for key in (
            v.get("id"),
            vid_obj.get("live_stream_id"),
            vid_obj.get("uuid"),
            v.get("uuid"),
        ):
            skey = str(key or "").strip()
            if skey:
                durations[skey] = max(durations.get(skey, 0.0), float(dur_sec))
    return durations


class StreamerBaselineManager:
    """
    Gerencia a nota de corte inteligente (`min_views_single_clip_live` e `min_views_post_live`)
    por streamer, calibrada por tempo de live (padrão: 1 corte a cada 2h de live):
      - Usa os clipes reais dos últimos 7 dias agrupados por live + duração real de cada VOD/live,
        com fallback para histórico convertido para views reais.
      - Calcula automaticamente para qualquer novo streamer adicionado.
      - Atualiza semanalmente (a cada 7 dias) em `_state/streamer_baselines.json`.
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
        live_stream_views: Optional[List[Tuple[str, float]]] = None,
        stream_durations_sec: Optional[Dict[str, float]] = None,
        hours_per_cut: float = DEFAULT_HOURS_PER_CUT,
        target_cuts_per_live: int = DEFAULT_TARGET_CUTS_PER_LIVE,
        max_cuts_hard_cap: int = DEFAULT_MAX_CUTS_HARD_CAP,
        default_floor: int = MIN_FLOOR_POST_LIVE,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = now_utc or datetime.now(timezone.utc)
        clean_slug = str(slug or "").strip().lower().lstrip("@")
        api_pairs = [(str(sk), float(v)) for sk, v in (live_stream_views or []) if v is not None and float(v) > 0]
        hist_pairs = extract_streamer_historical_views(processed_state, clean_slug, platform=platform)

        # Prioriza amostra limpa dos últimos 7 dias da API (reflete o momento atual do canal sem viés antigo);
        # usa histórico como complemento apenas quando há poucos clipes recentes na API (< 6).
        if len(api_pairs) >= 6:
            sample_pairs = api_pairs
        elif api_pairs or hist_pairs:
            sample_pairs = api_pairs + hist_pairs
        else:
            sample_pairs = []

        floor_post = max(MIN_FLOOR_POST_LIVE, min(120, int(default_floor))) if not sample_pairs else MIN_FLOOR_POST_LIVE
        cal = calibrate_streamer_thresholds(
            sample_pairs,
            stream_durations_sec=stream_durations_sec,
            hours_per_cut=hours_per_cut,
            target_cuts_per_live=target_cuts_per_live,
            floor_post=floor_post,
            floor_live=MIN_FLOOR_LIVE,
            max_cuts_hard_cap=max_cuts_hard_cap,
        )
        next_refresh = now + timedelta(days=self.refresh_days)

        return {
            "streamer": clean_slug,
            "platform": platform,
            "avg_score": cal["median_views"] or float(cal["min_views_post_live"]),
            "median_views": cal["median_views"],
            "min_views_single_clip_live": cal["min_views_single_clip_live"],
            "min_views_post_live": cal["min_views_post_live"],
            "min_score_cutoff": 0.0,
            "hours_per_cut": float(hours_per_cut),
            "target_cuts_per_live": int(target_cuts_per_live),
            "active_lives_7d": cal["active_lives"],
            "total_live_hours_7d": cal["total_live_hours"],
            "estimated_cuts_per_hour": cal["estimated_cuts_per_hour"],
            "estimated_cuts_per_live": cal["estimated_cuts_per_live"],
            "calibration_method": cal["method"],
            "historical_samples": len(hist_pairs),
            "api_samples": len(api_pairs),
            "sample_count": len(sample_pairs),
            "updated_at": _to_iso_utc(now),
            "next_refresh_at": _to_iso_utc(next_refresh),
        }

    def ensure_profiles_baselines(
        self,
        profiles: List[ProfileConfig],
        processed_state: Optional[Dict[str, Any]] = None,
        helix: Any = None,
        kick_clip_fetcher: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        kick_video_fetcher: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        force_refresh: bool = False,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Garante que todos os streamers (Twitch e Kick) dos perfis ativos tenham sua calibração inteligente
        por tempo de live calculada na hora (para novos streamers) e atualizada automaticamente a cada 7 dias.
        """
        from src.kick.analyzer import extract_kick_slug
        from src.twitch.client import parse_twitch_duration_seconds

        now = now_utc or datetime.now(timezone.utc)
        state = self.load_state()
        streamers_map: Dict[str, Dict[str, Any]] = state.setdefault("streamers", {})
        changed = False

        for prof in profiles:
            hours_per_cut = float(getattr(prof.detection, "hours_per_cut", DEFAULT_HOURS_PER_CUT) or 0.0)
            target_per_live = max(
                1,
                int(getattr(prof.detection, "target_cuts_per_live", DEFAULT_TARGET_CUTS_PER_LIVE) or DEFAULT_TARGET_CUTS_PER_LIVE),
            )
            hard_cap = max(
                1,
                int(getattr(prof.detection, "max_cuts_per_stream", DEFAULT_MAX_CUTS_HARD_CAP) or DEFAULT_MAX_CUTS_HARD_CAP),
            )
            floor = min(
                int(getattr(prof.detection, "min_views_single_clip_live", 120) or 120),
                int(getattr(prof.detection, "min_views_post_live", 120) or 120),
            )

            # 1) Streamer Twitch do perfil
            tw_slug = str(prof.twitch_login or "").strip().lower().lstrip("@")
            if tw_slug and tw_slug not in {"seu_criador_aqui", ""}:
                existing = streamers_map.get(tw_slug)
                if force_refresh or self.is_due_for_refresh(existing, now_utc=now):
                    api_pairs: List[Tuple[str, float]] = []
                    tw_durations: Dict[str, float] = {}
                    if helix is not None:
                        try:
                            user = helix.get_user_by_login(tw_slug)
                            if user:
                                bid = str(user["id"])
                                if hasattr(helix, "get_recent_vods"):
                                    try:
                                        for v in helix.get_recent_vods(bid, first=20) or []:
                                            vid = str(v.get("id") or "").strip()
                                            dsec = parse_twitch_duration_seconds(str(v.get("duration", "")))
                                            if vid and dsec > 0:
                                                tw_durations[vid] = float(dsec)
                                    except Exception as exc:
                                        logger.warning("Aviso ao buscar VODs Twitch para duração de %s: %s", tw_slug, exc)

                                if hasattr(helix, "get_active_stream"):
                                    try:
                                        active = helix.get_active_stream(bid)
                                        if active and active.get("started_at"):
                                            s_dt = _parse_iso_utc(str(active["started_at"]))
                                            if s_dt:
                                                elapsed = max(60.0, (now - s_dt).total_seconds())
                                                tw_durations[f"stream_{active['id']}"] = elapsed
                                    except Exception:
                                        pass

                                start_7d = _to_iso_utc(now - timedelta(days=7))
                                end_now = _to_iso_utc(now)
                                clips = helix.get_clips_in_window(
                                    broadcaster_id=bid,
                                    started_at_iso=start_7d,
                                    ended_at_iso=end_now,
                                    max_pages=2,
                                )
                                for c in clips:
                                    v = int(getattr(c, "view_count", 0) or 0)
                                    if v > 0:
                                        skey = str(getattr(c, "video_id", "") or str(getattr(c, "created_at", ""))[:10] or "tw")
                                        api_pairs.append((skey, float(v)))
                        except Exception as exc:
                            logger.warning("Aviso ao buscar clipes 7d da Twitch para calibração de %s: %s", tw_slug, exc)

                    entry = self.compute_streamer_entry(
                        slug=tw_slug,
                        platform="twitch",
                        processed_state=processed_state,
                        live_stream_views=api_pairs,
                        stream_durations_sec=tw_durations,
                        hours_per_cut=hours_per_cut,
                        target_cuts_per_live=target_per_live,
                        max_cuts_hard_cap=hard_cap,
                        default_floor=floor,
                        now_utc=now,
                    )
                    streamers_map[tw_slug] = entry
                    changed = True
                    logger.info(
                        "Baseline semanal atualizada para Twitch @%s: live>=%d | post_live>=%d (~%.2f cortes/h em %.1fh / %d lives)",
                        tw_slug,
                        entry["min_views_single_clip_live"],
                        entry["min_views_post_live"],
                        entry["estimated_cuts_per_hour"],
                        entry["total_live_hours_7d"],
                        entry["active_lives_7d"],
                    )

            # 2) Streamers da Kick configurados no perfil
            for k_ref in prof.kick_channels or []:
                k_slug = extract_kick_slug(k_ref)
                if not k_slug:
                    continue
                existing = streamers_map.get(k_slug)
                if force_refresh or self.is_due_for_refresh(existing, now_utc=now):
                    api_pairs = []
                    k_durations: Dict[str, float] = {}
                    if kick_video_fetcher is not None:
                        try:
                            k_durations = _extract_kick_video_durations(kick_video_fetcher(k_slug))
                        except Exception as exc:
                            logger.warning("Aviso ao buscar VODs da Kick para duração de %s: %s", k_slug, exc)
                    if kick_clip_fetcher is not None:
                        try:
                            k_clips = kick_clip_fetcher(k_slug)
                            for c in k_clips or []:
                                v = int(c.get("views") or 0)
                                if v > 0:
                                    ls_id = str(c.get("livestream_id") or "").strip()
                                    vod_id = str((c.get("vod") or {}).get("id") or "").strip()
                                    skey = ls_id or vod_id or str(c.get("created_at") or "")[:10] or "kick"
                                    api_pairs.append((skey, float(v)))
                        except Exception as exc:
                            logger.warning("Aviso ao buscar clipes 7d da Kick para calibração de %s: %s", k_slug, exc)

                    entry = self.compute_streamer_entry(
                        slug=k_slug,
                        platform="kick",
                        processed_state=processed_state,
                        live_stream_views=api_pairs,
                        stream_durations_sec=k_durations,
                        hours_per_cut=hours_per_cut,
                        target_cuts_per_live=target_per_live,
                        max_cuts_hard_cap=hard_cap,
                        default_floor=floor,
                        now_utc=now,
                    )
                    streamers_map[k_slug] = entry
                    changed = True
                    logger.info(
                        "Baseline semanal atualizada para Kick @%s: live>=%d | post_live>=%d (~%.2f cortes/h em %.1fh / %d lives)",
                        k_slug,
                        entry["min_views_single_clip_live"],
                        entry["min_views_post_live"],
                        entry["estimated_cuts_per_hour"],
                        entry["total_live_hours_7d"],
                        entry["active_lives_7d"],
                    )

        if changed:
            state["updated_at"] = _to_iso_utc(now)
            self.save_state(state)

        return streamers_map
