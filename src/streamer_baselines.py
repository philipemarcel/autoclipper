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
DEFAULT_SELECTIVITY = 0.80  # Alvo seletivo (~2.3–2.6 cortes/live em média, teto 3/live)
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
    target_cuts_per_live: int = DEFAULT_TARGET_CUTS_PER_LIVE,
    selectivity: float = DEFAULT_SELECTIVITY,
    live_ratio: float = DEFAULT_LIVE_RATIO,
    floor_post: int = MIN_FLOOR_POST_LIVE,
    floor_live: int = MIN_FLOOR_LIVE,
) -> Dict[str, Any]:
    """
    Calibração inteligente por streamer mirando ~`target_cuts_per_live` (padrão 3) bons momentos por live,
    mantendo alta seletividade e imunidade a outliers:
      - Agrupa os clipes dos últimos 7 dias por live (`video_id` / `vod_id` / data).
      - Aplica teto de `target_cuts_per_live` por live para que 1 única live explosiva (ex.: lançamento de jogo)
        NUNCA infle a nota de corte e bloqueie as outras lives normais da semana.
      - Escolhe a MAIOR nota de corte `T` em que a soma limitada por live atinge `selectivity * target * lives_ativas`
        (com `selectivity=0.80` e `target=3`, entrega ~2.3–2.6 cortes/live na média: lives fortes batem 3,
        lives fracas entregam 0–1).
      - Define `min_views_single_clip_live = 65%` de `min_views_post_live` para capturar picos ainda ao vivo.
    """
    target = max(1, int(target_cuts_per_live))
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
            "estimated_cuts_per_live": 0.0,
            "method": "default_fallback",
        }

    # Prefere lives com >= 2 clipes para não distorcer a média com dias isolados de 1 clipe
    active_streams = [vs for vs in by_stream.values() if len(vs) >= 2]
    if not active_streams:
        active_streams = list(by_stream.values())

    n_active = len(active_streams)
    max_possible = sum(min(target, len(vs)) for vs in active_streams)
    desired_total = min(max_possible, max(target, int(round(target * selectivity * n_active))))

    candidates = sorted(set(all_views), reverse=True)
    chosen = candidates[-1]
    for threshold in candidates:
        capped_sum = sum(min(target, sum(1 for v in vs if v >= threshold)) for vs in active_streams)
        if capped_sum >= desired_total:
            chosen = threshold
            break

    post_cut = max(floor_post, int(round(chosen)))
    live_cut = max(floor_live, int(round(post_cut * live_ratio)))
    actual_capped = sum(min(target, sum(1 for v in vs if v >= post_cut)) for vs in active_streams)

    return {
        "min_views_post_live": post_cut,
        "min_views_single_clip_live": live_cut,
        "median_views": round(_true_median(all_views), 1),
        "active_lives": n_active,
        "estimated_cuts_per_live": round(actual_capped / max(1, n_active), 2),
        "method": f"per_live_capped_top{target}",
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


class StreamerBaselineManager:
    """
    Gerencia a nota de corte inteligente (`min_views_single_clip_live` e `min_views_post_live`)
    por streamer, calibrada para ~3 melhores momentos por live:
      - Usa os clipes reais dos últimos 7 dias agrupados por live (Twitch Helix / Kick API),
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
        target_cuts_per_live: int = DEFAULT_TARGET_CUTS_PER_LIVE,
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
            target_cuts_per_live=target_cuts_per_live,
            floor_post=floor_post,
            floor_live=MIN_FLOOR_LIVE,
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
            "target_cuts_per_live": int(target_cuts_per_live),
            "active_lives_7d": cal["active_lives"],
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
        force_refresh: bool = False,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Garante que todos os streamers (Twitch e Kick) dos perfis ativos tenham sua calibração inteligente
        calculada na hora (para novos streamers) e atualizada automaticamente a cada 7 dias.
        """
        from src.kick.analyzer import extract_kick_slug

        now = now_utc or datetime.now(timezone.utc)
        state = self.load_state()
        streamers_map: Dict[str, Dict[str, Any]] = state.setdefault("streamers", {})
        changed = False

        for prof in profiles:
            target_per_live = max(1, int(getattr(prof.detection, "target_cuts_per_live", DEFAULT_TARGET_CUTS_PER_LIVE) or DEFAULT_TARGET_CUTS_PER_LIVE))
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
                        target_cuts_per_live=target_per_live,
                        default_floor=floor,
                        now_utc=now,
                    )
                    streamers_map[tw_slug] = entry
                    changed = True
                    logger.info(
                        "Baseline semanal atualizada para Twitch @%s: live>=%d | post_live>=%d (~%.2f cortes/live em %d lives)",
                        tw_slug,
                        entry["min_views_single_clip_live"],
                        entry["min_views_post_live"],
                        entry["estimated_cuts_per_live"],
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
                    if kick_clip_fetcher is not None:
                        try:
                            k_clips = kick_clip_fetcher(k_slug)
                            for c in k_clips or []:
                                v = int(c.get("views") or 0)
                                if v > 0:
                                    vod_id = str((c.get("vod") or {}).get("id") or "")
                                    skey = vod_id or str(c.get("created_at") or "")[:10] or "kick"
                                    api_pairs.append((skey, float(v)))
                        except Exception as exc:
                            logger.warning("Aviso ao buscar clipes 7d da Kick para calibração de %s: %s", k_slug, exc)

                    entry = self.compute_streamer_entry(
                        slug=k_slug,
                        platform="kick",
                        processed_state=processed_state,
                        live_stream_views=api_pairs,
                        target_cuts_per_live=target_per_live,
                        default_floor=floor,
                        now_utc=now,
                    )
                    streamers_map[k_slug] = entry
                    changed = True
                    logger.info(
                        "Baseline semanal atualizada para Kick @%s: live>=%d | post_live>=%d (~%.2f cortes/live em %d lives)",
                        k_slug,
                        entry["min_views_single_clip_live"],
                        entry["min_views_post_live"],
                        entry["estimated_cuts_per_live"],
                        entry["active_lives_7d"],
                    )

        if changed:
            state["updated_at"] = _to_iso_utc(now)
            self.save_state(state)

        return streamers_map
