"""Multi-signal video cut deduplication (context, description, and extracted timestamp)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


_STOPWORDS: Set[str] = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
    "em", "por", "para", "pra", "pro", "com", "sem", "sob", "sobre",
    "e", "ou", "mas", "que", "se", "como", "quando", "onde", "porque",
    "ele", "ela", "eles", "elas", "isso", "isto", "aquilo", "meu", "minha",
    "seu", "sua", "esse", "essa", "este", "esta", "muito", "muita", "mais",
    "ao", "aos", "pelo", "pela", "num", "numa", "ate", "ja", "foi", "era",
    "vai", "vou", "ta", "tava", "ser", "ter", "fazer", "fez", "disse", "falou",
    "cara", "mano", "velho", "galera", "gente", "aqui", "ali", "la", "ai",
    "kkk", "kkkk", "kkkkk", "haha", "hahaha", "rs", "ne", "entao", "tipo",
    "the", "and", "for", "with", "that", "this", "from", "you", "are", "was",
    "cortes", "clip", "clipe", "shorts", "reels", "viral", "live", "stream",
}


@dataclass(frozen=True)
class DuplicateMatch:
    """Result of comparing a candidate/cut against previously processed or queued items."""

    is_duplicate: bool
    matched_cut_id: str = ""
    reason: str = ""
    score: float = 0.0


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_for_dedup(text: str) -> str:
    """Normalize text for similarity checks by removing URLs, hashtags, and punctuation."""
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"https?://\S+", " ", raw)
    raw = re.sub(r"#[A-Za-z0-9_\u00C0-\u024F]+", " ", raw)
    raw = re.sub(r"@[A-Za-z0-9_.\u00C0-\u024F]+", " ", raw)
    raw = _strip_accents(raw)
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    tokens = [t for t in raw.split() if t and len(t) > 1]
    return " ".join(tokens)


def content_tokens(text: str) -> Set[str]:
    """Return meaningful non-stopword tokens from text."""
    norm = normalize_for_dedup(text)
    if not norm:
        return set()
    return {t for t in norm.split() if len(t) >= 3 and t not in _STOPWORDS}


def text_similarity_metrics(text_a: str, text_b: str) -> Tuple[float, float, int]:
    """Return (sequence_ratio, jaccard_content_overlap, shared_content_token_count)."""
    norm_a = normalize_for_dedup(text_a)
    norm_b = normalize_for_dedup(text_b)
    if not norm_a or not norm_b:
        return 0.0, 0.0, 0
    seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    tok_a = content_tokens(text_a)
    tok_b = content_tokens(text_b)
    if not tok_a or not tok_b:
        return seq_ratio, 0.0, 0
    shared = len(tok_a & tok_b)
    union = len(tok_a | tok_b)
    jaccard = shared / union if union else 0.0
    return seq_ratio, jaccard, shared


def _parse_iso_dt(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_creator(record: Dict[str, Any]) -> str:
    for key in ("twitch_login", "creator_name", "profile_id"):
        val = str(record.get(key) or "").strip().lower()
        if val:
            return val
    return ""


def _extract_vod_interval(record: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Compute extracted absolute source-video window [start_sec, end_sec] in VOD/video coordinates."""
    anchor = record.get("vod_anchor_sec")
    if anchor is None:
        return None
    try:
        anchor_val = float(anchor)
    except (TypeError, ValueError):
        return None

    director = record.get("director") if isinstance(record.get("director"), dict) else {}
    main_start = record.get("main_start", director.get("main_start"))
    main_end = record.get("main_end", director.get("main_end"))
    if main_start is not None and main_end is not None:
        try:
            ms = float(main_start)
            me = float(main_end)
            if me > ms:
                return anchor_val + ms, anchor_val + me
        except (TypeError, ValueError):
            pass

    dur_raw = record.get("total_duration") or record.get("duration") or 40.0
    try:
        dur = max(10.0, float(dur_raw))
    except (TypeError, ValueError):
        dur = 40.0
    return anchor_val, anchor_val + dur


def _same_source_stream(rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> bool:
    vid_a = str(rec_a.get("video_id") or "").strip()
    vid_b = str(rec_b.get("video_id") or "").strip()
    if vid_a and vid_b:
        return vid_a == vid_b
    sk_a = str(rec_a.get("stream_key") or "").strip()
    sk_b = str(rec_b.get("stream_key") or "").strip()
    if sk_a and sk_b:
        return sk_a == sk_b
    return False


def _interval_overlap(win_a: Tuple[float, float], win_b: Tuple[float, float]) -> Tuple[float, float]:
    """Return (overlap_seconds, overlap_ratio_of_shorter)."""
    overlap = max(0.0, min(win_a[1], win_b[1]) - max(win_a[0], win_b[0]))
    if overlap <= 0.0:
        return 0.0, 0.0
    len_a = max(1.0, win_a[1] - win_a[0])
    len_b = max(1.0, win_b[1] - win_b[0])
    return overlap, overlap / min(len_a, len_b)


def _iter_existing_records(
    processed_state: Optional[Dict[str, Any]],
    queue_state: Optional[Dict[str, Any]],
    exclude_cut_id: str = "",
) -> Iterable[Dict[str, Any]]:
    seen_ids: Set[str] = set()
    if processed_state and isinstance(processed_state.get("cuts"), list):
        for cut in processed_state["cuts"]:
            if not isinstance(cut, dict):
                continue
            cid = str(cut.get("cut_id") or "").strip()
            if exclude_cut_id and cid == exclude_cut_id:
                continue
            if cid:
                seen_ids.add(cid)
            yield cut

    if queue_state and isinstance(queue_state.get("items"), list):
        for item in queue_state["items"]:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().lower()
            if status in {"failed", "cancelled", "canceled"}:
                continue
            cid = str(item.get("cut_id") or "").strip()
            if exclude_cut_id and cid == exclude_cut_id:
                continue
            if cid and cid in seen_ids:
                continue
            yield item


def _check_clip_overlap(cand: Dict[str, Any], existing: Dict[str, Any]) -> Optional[DuplicateMatch]:
    cand_cids = {str(x).strip() for x in (cand.get("clip_ids") or []) if str(x).strip()}
    ext_cids = {str(x).strip() for x in (existing.get("clip_ids") or []) if str(x).strip()}
    if cand_cids and ext_cids and (cand_cids & ext_cids):
        overlap_id = sorted(cand_cids & ext_cids)[0]
        return DuplicateMatch(
            True,
            str(existing.get("cut_id") or ""),
            f"clip_id já processado ({overlap_id})",
            1.0,
        )

    cand_urls = {str(x).strip() for x in (cand.get("clip_urls") or []) if str(x).strip()}
    ext_urls = {str(x).strip() for x in (existing.get("clip_urls") or []) if str(x).strip()}
    if cand_urls and ext_urls and (cand_urls & ext_urls):
        return DuplicateMatch(
            True,
            str(existing.get("cut_id") or ""),
            "clip_url já processada anteriormente",
            1.0,
        )
    return None


def _check_extracted_timestamp_overlap(
    cand: Dict[str, Any],
    existing: Dict[str, Any],
    anchor_proximity_sec: float = 120.0,
    clip_proximity_sec: float = 150.0,
) -> Optional[DuplicateMatch]:
    same_stream = _same_source_stream(cand, existing)
    win_a = _extract_vod_interval(cand)
    win_b = _extract_vod_interval(existing)
    if same_stream and win_a and win_b:
        overlap_sec, overlap_ratio = _interval_overlap(win_a, win_b)
        if overlap_sec >= 8.0 or overlap_ratio >= 0.35:
            return DuplicateMatch(
                True,
                str(existing.get("cut_id") or ""),
                f"horário/trecho extraído do vídeo sobrepõe {overlap_sec:.1f}s ({overlap_ratio:.0%})",
                min(1.0, overlap_ratio),
            )
        anchor_a = cand.get("vod_anchor_sec")
        anchor_b = existing.get("vod_anchor_sec")
        if anchor_a is not None and anchor_b is not None:
            gap = abs(float(anchor_a) - float(anchor_b))
            if gap <= anchor_proximity_sec:
                return DuplicateMatch(
                    True,
                    str(existing.get("cut_id") or ""),
                    f"âncora VOD muito próxima ({gap:.0f}s <= {anchor_proximity_sec:.0f}s no mesmo vídeo/live)",
                    0.9,
                )

    same_creator = _normalize_creator(cand) and (_normalize_creator(cand) == _normalize_creator(existing))
    ts_a = _parse_iso_dt(cand.get("clip_created_at"))
    ts_b = _parse_iso_dt(existing.get("clip_created_at"))
    if same_creator and ts_a and ts_b:
        dt_gap = abs((ts_a - ts_b).total_seconds())
        if dt_gap <= 45.0:
            return DuplicateMatch(
                True,
                str(existing.get("cut_id") or ""),
                f"clipe criado no mesmo instante do mesmo canal ({dt_gap:.0f}s de diferença)",
                0.88,
            )
        if dt_gap <= clip_proximity_sec:
            title_seq, title_jac, _ = text_similarity_metrics(
                str(cand.get("dominant_title") or cand.get("caption") or ""),
                str(existing.get("dominant_title") or existing.get("caption") or ""),
            )
            tr_seq, tr_jac, _ = text_similarity_metrics(
                str(cand.get("transcript_snippet") or ""),
                str(existing.get("transcript_snippet") or ""),
            )
            has_transcript_both = bool(cand.get("transcript_snippet") and existing.get("transcript_snippet"))
            if not has_transcript_both or max(title_seq, tr_seq) >= 0.28 or max(title_jac, tr_jac) >= 0.15:
                return DuplicateMatch(
                    True,
                    str(existing.get("cut_id") or ""),
                    f"horário de criação próximo ({dt_gap:.0f}s) com mesmo contexto no canal",
                    max(0.8, title_seq, tr_seq),
                )
    return None


_WHOLE_BROADCAST_MODES: Set[str] = {
    "youtube_live",
    "youtube_auto",
    "youtube_manual",
    "kick_live",
    "kick_vod",
}

_GENERIC_TITLE_PREFIXES: Tuple[str, ...] = (
    "clipe de ",
    "clipe kick ",
    "live kick ",
    "live na kick",
    "ao vivo ",
    "youtube video",
)


def _should_compare_source_title(rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> bool:
    """Skip source-title-only matching when title describes an entire live/VOD broadcast rather than a clip."""
    mode_a = str(rec_a.get("mode") or "").strip().lower()
    mode_b = str(rec_b.get("mode") or "").strip().lower()
    if mode_a in _WHOLE_BROADCAST_MODES or mode_b in _WHOLE_BROADCAST_MODES:
        return False

    # Same video with clearly non-overlapping extracted windows shouldn't fail on shared title alone
    if _same_source_stream(rec_a, rec_b):
        win_a = _extract_vod_interval(rec_a)
        win_b = _extract_vod_interval(rec_b)
        if win_a and win_b:
            overlap_sec, _ = _interval_overlap(win_a, win_b)
            if overlap_sec <= 0.0:
                return False

    norm_a = normalize_for_dedup(str(rec_a.get("dominant_title") or rec_a.get("caption") or ""))
    norm_b = normalize_for_dedup(str(rec_b.get("dominant_title") or rec_b.get("caption") or ""))
    if not norm_a or not norm_b:
        return False
    if any(norm_a.startswith(p) or norm_b.startswith(p) for p in _GENERIC_TITLE_PREFIXES):
        return False
    return True


def find_duplicate_candidate(
    candidate: Dict[str, Any],
    processed_state: Optional[Dict[str, Any]],
    queue_state: Optional[Dict[str, Any]] = None,
    anchor_proximity_sec: float = 120.0,
    clip_proximity_sec: float = 150.0,
) -> DuplicateMatch:
    """Pre-download check comparing clip IDs, extracted VOD window/timestamp, and title context."""
    cand_title = str(candidate.get("dominant_title") or candidate.get("caption") or "").strip()
    cand_creator = _normalize_creator(candidate)

    for existing in _iter_existing_records(processed_state, queue_state, exclude_cut_id=str(candidate.get("cut_id") or "")):
        clip_hit = _check_clip_overlap(candidate, existing)
        if clip_hit:
            return clip_hit

        ts_hit = _check_extracted_timestamp_overlap(
            candidate,
            existing,
            anchor_proximity_sec=anchor_proximity_sec,
            clip_proximity_sec=clip_proximity_sec,
        )
        if ts_hit:
            return ts_hit

        ext_title = str(existing.get("dominant_title") or existing.get("caption") or "").strip()
        if cand_title and ext_title and _should_compare_source_title(candidate, existing):
            seq, jac, shared = text_similarity_metrics(cand_title, ext_title)
            same_creator = bool(cand_creator and cand_creator == _normalize_creator(existing))
            # Clip titles on Twitch/Kick are often copied or lightly edited for the same highlight
            if same_creator and ((seq >= 0.72 and shared >= 2) or (jac >= 0.65 and shared >= 3)):
                return DuplicateMatch(
                    True,
                    str(existing.get("cut_id") or ""),
                    f"título/contexto similar no mesmo canal (seq={seq:.2f}, jaccard={jac:.2f})",
                    max(seq, jac),
                )
    return DuplicateMatch(False)


def find_duplicate_cut(
    cut: Dict[str, Any],
    processed_state: Optional[Dict[str, Any]],
    queue_state: Optional[Dict[str, Any]] = None,
    anchor_proximity_sec: float = 120.0,
    clip_proximity_sec: float = 150.0,
) -> DuplicateMatch:
    """Comprehensive duplicate gate checking context (transcript/clips), proposed caption, and extracted window."""
    cut_id = str(cut.get("cut_id") or "").strip()
    cut_creator = _normalize_creator(cut)
    cut_caption = str(cut.get("caption") or cut.get("caption_text") or "").strip()
    cut_teaser = str(cut.get("teaser_headline") or "").strip()
    cut_transcript = str(cut.get("transcript_snippet") or "").strip()
    cut_title = str(cut.get("dominant_title") or "").strip()

    for existing in _iter_existing_records(processed_state, queue_state, exclude_cut_id=cut_id):
        clip_hit = _check_clip_overlap(cut, existing)
        if clip_hit:
            return clip_hit

        ts_hit = _check_extracted_timestamp_overlap(
            cut,
            existing,
            anchor_proximity_sec=anchor_proximity_sec,
            clip_proximity_sec=clip_proximity_sec,
        )
        if ts_hit:
            return ts_hit

        ext_creator = _normalize_creator(existing)
        same_creator = bool(cut_creator and ext_creator and cut_creator == ext_creator)

        # 1) Proposed description (caption) and teaser similarity
        ext_caption = str(existing.get("caption") or existing.get("caption_text") or "").strip()
        if cut_caption and ext_caption:
            cap_seq, cap_jac, cap_shared = text_similarity_metrics(cut_caption, ext_caption)
            if same_creator and ((cap_seq >= 0.62 and cap_shared >= 2) or (cap_jac >= 0.52 and cap_shared >= 3)):
                return DuplicateMatch(
                    True,
                    str(existing.get("cut_id") or ""),
                    f"descrição muito parecida no mesmo canal (seq={cap_seq:.2f}, jaccard={cap_jac:.2f})",
                    max(cap_seq, cap_jac),
                )
            if not same_creator and ((cap_seq >= 0.78 and cap_shared >= 3) or (cap_jac >= 0.70 and cap_shared >= 4)):
                return DuplicateMatch(
                    True,
                    str(existing.get("cut_id") or ""),
                    f"descrição praticamente idêntica (seq={cap_seq:.2f}, jaccard={cap_jac:.2f})",
                    max(cap_seq, cap_jac),
                )

        ext_teaser = str(existing.get("teaser_headline") or "").strip()
        if same_creator and cut_teaser and ext_teaser:
            t_seq, t_jac, t_shared = text_similarity_metrics(cut_teaser, ext_teaser)
            if (t_seq >= 0.72 and t_shared >= 2) or (t_jac >= 0.65 and t_shared >= 3):
                return DuplicateMatch(
                    True,
                    str(existing.get("cut_id") or ""),
                    f"headline/gancho duplicado no mesmo canal (seq={t_seq:.2f}, jaccard={t_jac:.2f})",
                    max(t_seq, t_jac),
                )

        # 2) Spoken video context (transcript snippet overlap)
        ext_transcript = str(existing.get("transcript_snippet") or "").strip()
        if same_creator and cut_transcript and ext_transcript:
            tr_seq, tr_jac, tr_shared = text_similarity_metrics(cut_transcript, ext_transcript)
            if (tr_jac >= 0.45 and tr_shared >= 5) or (tr_seq >= 0.58 and tr_shared >= 5):
                return DuplicateMatch(
                    True,
                    str(existing.get("cut_id") or ""),
                    f"contexto de áudio/transcrição idêntico no mesmo canal (seq={tr_seq:.2f}, jaccard={tr_jac:.2f})",
                    max(tr_seq, tr_jac),
                )
            # Combined moderate transcript + moderate caption overlap catches differently-cropped same speech
            if cut_caption and ext_caption:
                cap_seq, cap_jac, _ = text_similarity_metrics(cut_caption, ext_caption)
                if tr_jac >= 0.25 and tr_shared >= 3 and max(cap_seq, cap_jac) >= 0.42:
                    return DuplicateMatch(
                        True,
                        str(existing.get("cut_id") or ""),
                        f"combinação de transcrição (jaccard={tr_jac:.2f}) e descrição (seq={cap_seq:.2f}) similar",
                        (tr_jac + max(cap_seq, cap_jac)) / 2.0,
                    )

        # 3) Source title/context check when caption/transcript weren't enough
        ext_title = str(existing.get("dominant_title") or "").strip()
        if same_creator and cut_title and ext_title and _should_compare_source_title(cut, existing):
            title_seq, title_jac, title_shared = text_similarity_metrics(cut_title, ext_title)
            if (title_seq >= 0.78 and title_shared >= 3) or (title_jac >= 0.72 and title_shared >= 4):
                return DuplicateMatch(
                    True,
                    str(existing.get("cut_id") or ""),
                    f"título de origem duplicado no mesmo canal (seq={title_seq:.2f}, jaccard={title_jac:.2f})",
                    max(title_seq, title_jac),
                )

    return DuplicateMatch(False)
