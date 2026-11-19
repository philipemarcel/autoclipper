from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ai.audio_cleaner import AudioCleaner
from src.ai.content_moderator import moderate_video_for_platforms
from src.ai.director import GeminiDirector
from src.ai.transcriber import WordTranscriber
from src.approval.discord_notifier import DiscordNotifier
from src.config_loader import (
    PROJECT_ROOT,
    load_environment,
    load_profiles,
    load_subtitle_theme,
)
from src.kick.analyzer import KickMomentAnalyzer
from src.kick.downloader import KickDownloader
from src.models import CutRecord, MomentCluster, ProfileConfig
from src.publisher.drip_scheduler import DripScheduler
from src.storage.gdrive import DriveStore
from src.twitch.client import TwitchHelixClient
from src.twitch.detector import MomentDetector
from src.twitch.downloader import TwitchDownloader
from src.video.captions import CaptionGenerator
from src.video.layout_analyzer import LayoutAnalyzer
from src.video.renderer import VideoRenderer
from src.youtube.analyzer import YouTubeMomentAnalyzer
from src.youtube.downloader import YouTubeDownloader

PROCESSED_STATE_FILE = "processed_moments.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("autocliper")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def process_single_cluster(
    profile: ProfileConfig,
    cluster: MomentCluster,
    store: DriveStore,
    subtitle_theme: Dict[str, Any],
    processed_state: Dict[str, Any],
    work_root: Path,
) -> CutRecord:
    cut_work_dir = work_root / cluster.cluster_id
    cut_work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Download da mídia bruta (YouTube, Kick OU Twitch VOD/Clipe)
        if cluster.mode.startswith("youtube"):
            raw_media = YouTubeDownloader().download_moment(cluster, cut_work_dir)
        elif cluster.mode.startswith("kick"):
            raw_media = KickDownloader().download_moment(cluster, cut_work_dir)
        else:
            downloader = TwitchDownloader(profile.detection)
            raw_media = downloader.download_moment(cluster, cut_work_dir)

        # 2. Transcrição word-level (Groq Whisper na nuvem / faster-whisper GPU no Colab Ultra)
        transcriber = WordTranscriber(language=profile.language)
        words = transcriber.transcribe_video(raw_media.video_path, cut_work_dir)

        # 3. Diretor Gemini (Corte cirúrgico <=58s + Cold Open 2-3s + Headline + Copy minimalista)
        director = GeminiDirector(profile)
        decision = director.direct_cut(
            cluster=cluster,
            words=words,
            raw_duration_sec=raw_media.duration_sec,
            peak_hint_sec=raw_media.peak_Hint_sec_in_file,
        )

        # 4. Auto-Layout 9:16 inteligente (Split-Screen / Face-Tracking / Fundo Blur)
        analyzer = LayoutAnalyzer(profile.editing)
        layout = analyzer.analyze(
            video_path=raw_media.video_path,
            start_sec=decision.main_start,
            end_sec=decision.main_end,
        )

        # 5. Legendas dinâmicas OpusClip (.ass) + Backup (.srt)
        ass_path = cut_work_dir / "legenda.ass"
        srt_path = cut_work_dir / "legenda.srt"
        CaptionGenerator(subtitle_theme).generate_files(
            words,
            decision,
            ass_path,
            srt_path,
            layout_mode=layout.mode,
        )

        # 6. Renderização FFmpeg (CPU x264 no Actions / GPU NVENC no Colab Ultra)
        renderer = VideoRenderer()
        rendered = renderer.render_all(
            raw_video=raw_media.video_path,
            decision=decision,
            layout=layout,
            ass_file=ass_path,
            out_dir=cut_work_dir,
        )

        # 7. Salva textos auxiliares e sobe pacote completo nos 20 TB do Google Drive
        copy_path = cut_work_dir / "copy.txt"
        copy_path.write_text(decision.full_social_copy, encoding="utf-8")

        creator_slug = cluster.broadcaster_login or profile.twitch_login or profile.id
        creator_display = str(cluster.best_clip.broadcaster_name or creator_slug or profile.display_name).strip()
        date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stream_folder = f"{date_prefix}_{cluster.stream_or_vod_key}"
        anchor_label = f"vod{cluster.peak_offset}s" if cluster.peak_offset is not None else "clip"
        cut_folder_name = f"score{int(cluster.score)}_{anchor_label}_{cluster.cluster_id}"
        drive_parts = ["cortes", creator_slug, stream_folder, cut_folder_name]

        folder_id, folder_url = store.ensure_folder_path(drive_parts)
        master_id, master_url = store.upload_file(rendered.master_mp4, drive_parts, "master_1080p60.mp4")
        download_url = store.ensure_public_download_url(master_id)
        clean_id, _ = store.upload_file(rendered.clean_mp4, drive_parts, "limpo_1080p60.mp4")
        srt_id, _ = store.upload_file(srt_path, drive_parts, "legenda.srt")
        store.upload_file(ass_path, drive_parts, "legenda.ass")
        store.upload_file(copy_path, drive_parts, "copy.txt")

        record = CutRecord(
            cut_id=cluster.cluster_id,
            profile_id=profile.id,
            twitch_login=creator_slug,
            stream_key=cluster.stream_or_vod_key,
            mode=cluster.mode,
            score=cluster.score,
            source_type=raw_media.source_type,
            video_id=cluster.video_id,
            vod_anchor_sec=cluster.peak_offset,
            clip_urls=[c.url for c in cluster.clips[:5]],
            caption=decision.full_social_copy,
            teaser_headline=decision.teaser_headline,
            total_duration=round(decision.total_duration, 2),
            layout_used=layout.mode,
            drive_folder_id=folder_id,
            drive_folder_url=folder_url,
            master_file_id=master_id,
            master_file_url=master_url,
            clean_file_id=clean_id,
            srt_file_id=srt_id,
            created_at=_utc_now_iso(),
            creator_name=creator_display,
            download_url=download_url,
        )

        meta_path = cut_work_dir / "metadata.json"
        meta_dict = record.to_dict()
        meta_dict["director"] = {
            "main_start": decision.main_start,
            "main_end": decision.main_end,
            "cold_open_enabled": decision.cold_open_enabled,
            "teaser_start": decision.teaser_start,
            "teaser_end": decision.teaser_end,
            "virality_score": decision.virality_score,
            "reasoning": decision.reasoning,
        }
        meta_dict["drive_parts"] = drive_parts
        meta_dict["clip_ids"] = [c.id for c in cluster.clips]
        if cluster.best_clip.created_at:
            meta_dict["clip_created_at"] = cluster.best_clip.created_at

        # 7.5. Moderação automática visual + áudio por plataforma (Instagram vs YouTube Shorts vs TikTok)
        enabled_platforms: List[str] = []
        if profile.publishing.instagram.enabled:
            enabled_platforms.append("instagram")
        if profile.publishing.youtube_shorts.enabled:
            enabled_platforms.append("youtube_shorts")
        if profile.publishing.tiktok.enabled:
            enabled_platforms.append("tiktok")

        cut_transcript = " ".join(
            w.word for w in words if (decision.main_start - 0.5) <= w.start <= (decision.main_end + 0.5)
        )
        moderation = moderate_video_for_platforms(
            video_path=rendered.master_mp4,
            requested_platforms=enabled_platforms,
            title=f"{cluster.dominant_title or ''} {decision.caption_text}",
            transcript_text=cut_transcript,
        )
        meta_dict["moderation"] = moderation.to_dict()

        meta_path.write_text(json.dumps(meta_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        store.upload_file(meta_path, drive_parts, "metadata.json")

        # 8. Envia prévia (<9 MB) no Discord e aplica Auto-Postagem nas plataformas aprovadas pela moderação
        notifier = DiscordNotifier(profile)
        scheduler = DripScheduler(store)
        active_drip_min = scheduler.get_effective_interval_minutes(profile)
        msg_id = notifier.send_cut_for_approval(
            record,
            rendered.discord_preview_mp4,
            drip_interval_override=active_drip_min,
        )
        record.discord_message_id = msg_id
        meta_dict["discord_message_id"] = msg_id

        auto_publish_safe = os.getenv("AUTO_PUBLISH_SAFE", "true").strip().lower() not in ("0", "false", "no")
        if auto_publish_safe and enabled_platforms:
            if moderation.allowed_platforms:
                outcome = scheduler.enqueue_or_publish_cut(
                    profile=profile,
                    cut_dict=meta_dict,
                    platforms=moderation.allowed_platforms,
                    force_immediate=False,
                )
                meta_dict["auto_publish_outcome"] = outcome
                if moderation.blocked_platforms:
                    notifier.post_status_reply(
                        msg_id,
                        f"🤖 **Auto-Postagem por Plataforma (`{moderation.exposure_level}`)**: enviado/enfileirado automaticamente em `{', '.join(moderation.allowed_platforms)}` e filtrado de `{', '.join(moderation.blocked_platforms)}` ({moderation.reason}).",
                    )
                else:
                    notifier.post_status_reply(
                        msg_id,
                        f"🤖 **Auto-Postagem Aprovada (`{moderation.exposure_level}`)**: enviado/enfileirado automaticamente em `{', '.join(moderation.allowed_platforms)}`.",
                    )
            else:
                notifier.post_status_reply(
                    msg_id,
                    f"🚫 **Retido pela Moderação Automática (`{moderation.exposure_level}`)**: bloqueado em todas as plataformas ({moderation.reason}). Use os botões acima apenas se quiser aprovar manualmente.",
                )

        # 9. Salva imediatamente no estado global para evitar duplicação
        processed_state.setdefault("cuts", []).append(meta_dict)
        store.save_state_json(PROCESSED_STATE_FILE, processed_state)
        logger.info("Corte %s concluído e enviado para aprovação!", record.cut_id)
        return record
    finally:
        shutil.rmtree(cut_work_dir, ignore_errors=True)


def _is_unconfigured_ci() -> bool:
    return (
        os.getenv("GITHUB_ACTIONS", "").lower() == "true"
        and not os.getenv("GDRIVE_TOKEN_JSON", "").strip()
        and not os.getenv("GDRIVE_MOUNT_PATH", "").strip()
    )


def cmd_watch(profile_filter: Optional[str] = None, max_cuts_override: Optional[int] = None) -> int:
    if _is_unconfigured_ci():
        logger.info("Aguardando configuração inicial (`/setup iniciar` no Discord ou GDRIVE_TOKEN_JSON nos Secrets).")
        return 0

    profiles = load_profiles(only_enabled=True)
    if profile_filter:
        profiles = [p for p in profiles if p.id == profile_filter or p.twitch_login == profile_filter.lower()]
    if not profiles:
        logger.warning("Nenhum perfil habilitado encontrado para monitorar.")
        return 0

    active_profiles = [
        p
        for p in profiles
        if (p.twitch_login not in {"seu_criador_aqui", ""}) or bool(p.youtube_channels) or bool(p.kick_channels)
    ]
    if not active_profiles:
        logger.warning(
            "Nenhum canal configurado (Twitch/YouTube/Kick). Use `/canais adicionar` no Discord ou `config/profiles.yaml`."
        )
        return 0

    has_twitch_target = any(p.twitch_login not in {"seu_criador_aqui", ""} for p in active_profiles)
    helix: Optional[TwitchHelixClient] = None
    if has_twitch_target:
        if os.getenv("TWITCH_CLIENT_ID", "").strip() and os.getenv("TWITCH_CLIENT_SECRET", "").strip():
            helix = TwitchHelixClient()
        else:
            logger.warning(
                "Canal Twitch configurado, mas TWITCH_CLIENT_ID/SECRET estão ausentes. Pulando Twitch e mantendo YouTube/Kick."
            )

    store = DriveStore()
    theme = load_subtitle_theme()
    processed_state = store.load_state_json(PROCESSED_STATE_FILE, default={"cuts": []})
    work_root = PROJECT_ROOT / "work_tmp" / "watch"

    total_created = 0
    for prof in active_profiles:
        if max_cuts_override is not None:
            prof.detection.max_cuts_per_run = max_cuts_override

        if helix is not None and prof.twitch_login not in {"seu_criador_aqui", ""}:
            detector = MomentDetector(prof)
            candidates = detector.discover_candidates(helix, processed_state)
            logger.info(
                "Perfil `%s` (@%s): %d momento(s) candidato(s) encontrado(s).",
                prof.id,
                prof.twitch_login,
                len(candidates),
            )

            for cluster in candidates:
                try:
                    process_single_cluster(prof, cluster, store, theme, processed_state, work_root)
                    total_created += 1
                except Exception as exc:
                    logger.exception("Erro ao processar cluster %s: %s", cluster.cluster_id, exc)

        # Monitoramento automático 24/7 dos canais do YouTube configurados no perfil (vídeos novos + lives + repescagem)
        # Garante cota justa de 1 corte novo por canal antes de encerrar, evitando que o 1º canal consuma toda a cota.
        if prof.youtube_channels:
            yt_analyzer = YouTubeMomentAnalyzer(prof)
            yt_budget = max(len(prof.youtube_channels), 2) if max_cuts_override is None else max(1, min(3, max_cuts_override))
            yt_created = 0
            for ch_url in prof.youtube_channels:
                if yt_created >= yt_budget:
                    break
                recent_urls = yt_analyzer.discover_channel_recent_urls(
                    ch_url, max_items=1, work_dir=work_root / "yt_meta"
                )
                channel_done = False
                for v_url in recent_urls:
                    if yt_created >= yt_budget or channel_done:
                        break
                    # Evita reanalisar um vídeo/VOD do YouTube que já gerou 2+ cortes
                    vid_match = __import__("re").search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", v_url)
                    yt_key = f"yt_{vid_match.group(1)}" if vid_match else ""
                    if yt_key and sum(1 for c in processed_state.get("cuts", []) if c.get("stream_key") == yt_key) >= 2:
                        continue
                    try:
                        _, yt_clusters = yt_analyzer.discover_moments(
                            url=v_url,
                            max_cuts=1,
                            work_dir=work_root / "yt_meta",
                            processed_state=processed_state,
                            max_age_hours=prof.detection.recap_hours,
                        )
                        for ytc in yt_clusters:
                            process_single_cluster(prof, ytc, store, theme, processed_state, work_root)
                            total_created += 1
                            yt_created += 1
                            channel_done = True
                            break
                    except Exception as exc:
                        logger.warning("Aviso ao monitorar vídeo YouTube %s: %s", v_url, exc)

        # Monitoramento automático 24/7 dos canais da Kick configurados no perfil (clipes + lives + VODs)
        if prof.kick_channels:
            kick_analyzer = KickMomentAnalyzer(prof)
            kick_budget = max(len(prof.kick_channels), 2) if max_cuts_override is None else max(1, min(3, max_cuts_override))
            kick_created = 0
            for k_ref in prof.kick_channels:
                if kick_created >= kick_budget:
                    break
                try:
                    k_clusters = kick_analyzer.discover_channel_moments(
                        k_ref,
                        max_cuts=1,
                        processed_state=processed_state,
                    )
                    for kc in k_clusters:
                        process_single_cluster(prof, kc, store, theme, processed_state, work_root)
                        total_created += 1
                        kick_created += 1
                        break
                except Exception as exc:
                    logger.warning("Aviso ao monitorar canal Kick %s: %s", k_ref, exc)

    retention_days = max((int(p.publishing.drive_retention_days) for p in active_profiles), default=10)
    _run_drive_retention_cleanup(store, processed_state, retention_days=retention_days)
    return total_created


def _run_drive_retention_cleanup(
    store: DriveStore,
    processed_state: Dict[str, Any],
    retention_days: int = 10,
) -> Dict[str, Any]:
    queue_state = store.load_state_json("drip_queue.json", default={"items": []})
    pending_ids = {
        str(it.get("cut_id", ""))
        for it in queue_state.get("items", [])
        if it.get("status") == "pending" and it.get("cut_id")
    }
    report = store.cleanup_expired_cuts(
        processed_state=processed_state,
        retention_days=retention_days,
        active_queue_cut_ids=pending_ids,
    )
    if report["changed"]:
        store.save_state_json(PROCESSED_STATE_FILE, processed_state)
        logger.info(
            "Limpeza automática do Drive (>%dd): %d corte(s) expirado(s) excluído(s).",
            retention_days,
            report["deleted"],
        )
    return report


def cmd_cleanup(retention_days: Optional[int] = None) -> int:
    profiles = load_profiles(only_enabled=False)
    default_days = max((int(p.publishing.drive_retention_days) for p in profiles), default=10)
    days = retention_days if retention_days is not None else default_days
    store = DriveStore()
    processed_state = store.load_state_json(PROCESSED_STATE_FILE, default={"cuts": []})
    report = _run_drive_retention_cleanup(store, processed_state, retention_days=days)
    return int(report["deleted"])


def cmd_kick(
    url: str,
    max_cuts: int = 3,
    time_range: str = "",
    profile_filter: Optional[str] = None,
) -> int:
    """Processa manualmente um link da Kick (clipe, VOD ou live/canal) via `/cortar` ou CLI."""
    profiles = load_profiles(only_enabled=True) or load_profiles(only_enabled=False)
    if profile_filter:
        matched = [p for p in profiles if p.id == profile_filter or p.twitch_login == profile_filter.lower()]
        if matched:
            profiles = matched
    profile = profiles[0]

    store = DriveStore()
    theme = load_subtitle_theme()
    processed_state = store.load_state_json(PROCESSED_STATE_FILE, default={"cuts": []})
    work_root = PROJECT_ROOT / "work_tmp" / "kick"
    notifier = DiscordNotifier(profile)

    analyzer = KickMomentAnalyzer(profile)
    try:
        clusters = analyzer.discover_manual_url(
            url=url.strip(),
            max_cuts=max(1, min(10, int(max_cuts))),
            time_range_str=time_range,
            processed_state=processed_state,
        )
    except Exception as exc:
        logger.exception("Erro ao analisar URL da Kick (%s): %s", url, exc)
        notifier.post_status_reply("", f"❌ Não consegui analisar o link da Kick <{url}>: `{str(exc)[:300]}`")
        return 0

    if not clusters:
        notifier.post_status_reply(
            "",
            f"⚠️ Não encontrei momentos novos para cortar em <{url}> na Kick (ou já foram processados).",
        )
        return 0

    created = 0
    for cluster in clusters:
        try:
            process_single_cluster(profile, cluster, store, theme, processed_state, work_root)
            created += 1
        except Exception as exc:
            logger.exception("Erro ao gerar corte da Kick (%s): %s", cluster.cluster_id, exc)
            notifier.post_status_reply("", f"❌ Erro ao processar trecho da Kick <{url}>: `{exc}`")

    return created


def cmd_youtube(
    url: str,
    max_cuts: int = 3,
    time_range: str = "",
    profile_filter: Optional[str] = None,
) -> int:
    """
    Processa qualquer URL do YouTube (vídeo pronto, VOD ou Live em andamento) ou da Kick (`kick.com/...`)
    enviada via comando `/cortar` no Discord ou via CLI.
    """
    if "kick.com" in url.lower():
        return cmd_kick(url=url, max_cuts=max_cuts, time_range=time_range, profile_filter=profile_filter)

    profiles = load_profiles(only_enabled=True) or load_profiles(only_enabled=False)
    if profile_filter:
        matched = [p for p in profiles if p.id == profile_filter or p.twitch_login == profile_filter.lower()]
        if matched:
            profiles = matched
    profile = profiles[0]

    store = DriveStore()
    theme = load_subtitle_theme()
    processed_state = store.load_state_json(PROCESSED_STATE_FILE, default={"cuts": []})
    work_root = PROJECT_ROOT / "work_tmp" / "youtube"
    notifier = DiscordNotifier(profile)

    analyzer = YouTubeMomentAnalyzer(profile)
    try:
        info, clusters = analyzer.discover_moments(
            url=url.strip(),
            max_cuts=max(1, min(10, int(max_cuts))),
            work_dir=work_root / "meta",
            time_range_str=time_range,
            processed_state=processed_state,
        )
    except Exception as exc:
        logger.exception("Erro ao analisar URL do YouTube (%s): %s", url, exc)
        notifier.post_status_reply(
            "",
            f"❌ Não consegui analisar o vídeo <{url}>: `{str(exc)[:300]}`",
        )
        return 0

    if not clusters:
        notifier.post_status_reply(
            "",
            f"⚠️ Não encontrei momentos novos para cortar em <{url}> (ou todos os picos já foram cortados).",
        )
        return 0

    created = 0
    for cluster in clusters:
        try:
            process_single_cluster(profile, cluster, store, theme, processed_state, work_root)
            created += 1
        except Exception as exc:
            logger.exception("Erro ao gerar corte do YouTube (%s): %s", cluster.cluster_id, exc)
            notifier.post_status_reply(
                "",
                f"❌ Erro ao processar trecho de `{info.get('title', url)}`: `{exc}`",
            )

    return created


def cmd_discord_action(
    action: str,
    cut_id: str,
    discord_message_id: str = "",
    youtube_url: str = "",
    max_cuts: int = 3,
    time_range: str = "",
) -> None:
    if action == "youtube_cut":
        if not youtube_url.strip():
            raise ValueError("URL obrigatória para a ação youtube_cut.")
        cmd_youtube(url=youtube_url.strip(), max_cuts=max_cuts, time_range=time_range)
        return

    if action == "cleanup":
        deleted = cmd_cleanup()
        default_prof = next(iter(load_profiles(only_enabled=False)), None)
        if default_prof:
            DiscordNotifier(default_prof).post_status_reply(
                discord_message_id,
                f"🧹 Limpeza concluída no Google Drive: `{deleted}` corte(s) expirado(s) removido(s).",
            )
        return

    store = DriveStore()
    profiles = {p.id: p for p in load_profiles(only_enabled=False)}

    if action == "set_drip_interval":
        new_minutes = max(5, min(1440, int(max_cuts)))
        report = DripScheduler(store).update_timer_override(
            profiles_by_id=profiles,
            interval_minutes=new_minutes,
            quiet_hours_str=time_range,
        )
        default_prof = next(iter(profiles.values()), None)
        if default_prof:
            quiet_info = (
                "desativado"
                if report["quiet_hours"] == []
                else (f"{report['quiet_hours'][0]:02d}h–{report['quiet_hours'][1]:02d}h BRT" if isinstance(report["quiet_hours"], list) and len(report["quiet_hours"]) == 2 else "padrão do perfil")
            )
            recalc_note = (
                f" • `{report['rescheduled_count']}` corte(s) na fila recalculado(s) (próximo: **{report['next_scheduled_local']} BRT**)"
                if report["rescheduled_count"] > 0
                else ""
            )
            DiscordNotifier(default_prof).post_status_reply(
                discord_message_id,
                f"⏱️ **Timer da Fila Espaçada salvo em `{report['interval_minutes']} min`!** (Silêncio madrugada: `{quiet_info}`){recalc_note}",
            )
        return

    processed_state = store.load_state_json(PROCESSED_STATE_FILE, default={"cuts": []})

    cut_dict = next((c for c in processed_state.get("cuts", []) if c.get("cut_id") == cut_id), None)
    if not cut_dict:
        raise RuntimeError(f"Corte `{cut_id}` não encontrado em {PROCESSED_STATE_FILE}.")

    profile = profiles.get(cut_dict["profile_id"]) or next(iter(profiles.values()), None)
    if not profile:
        raise RuntimeError(f"Perfil `{cut_dict['profile_id']}` não encontrado na configuração.")

    notifier = DiscordNotifier(profile)
    msg_ref = discord_message_id or str(cut_dict.get("discord_message_id", ""))
    work_dir = PROJECT_ROOT / "work_tmp" / f"action_{cut_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if action == "reject":
            cut_dict["status"] = "rejected"
            store.save_state_json(PROCESSED_STATE_FILE, processed_state)
            notifier.post_status_reply(msg_ref, f"🗑️ Corte `{cut_id}` descartado e marcado para não repetir.")
            return

        if action == "clean_audio":
            local_master = work_dir / "master_in.mp4"
            local_clean = work_dir / "clean_in.mp4"
            store.download_file_by_ref(cut_dict["master_file_id"], local_master)
            store.download_file_by_ref(cut_dict["clean_file_id"], local_clean)

            cleaner = AudioCleaner()
            out_master = work_dir / "master_1080p60.mp4"
            out_clean = work_dir / "limpo_1080p60.mp4"
            cleaner.clean_video_music(local_master, out_master, work_dir / "dm_master")
            cleaner.clean_video_music(local_clean, out_clean, work_dir / "dm_clean")

            drive_parts: List[str] = cut_dict.get("drive_parts") or ["cortes", profile.twitch_login, f"cleaned_{cut_id}"]
            master_id, master_url = store.upload_file(out_master, drive_parts, "master_1080p60.mp4")
            download_url = store.ensure_public_download_url(master_id)
            clean_id, _ = store.upload_file(out_clean, drive_parts, "limpo_1080p60.mp4")

            preview_mp4 = work_dir / "preview_discord.mp4"
            VideoRenderer().render_discord_preview(out_master, preview_mp4, float(cut_dict.get("total_duration", 45.0)))

            cut_dict["master_file_id"] = master_id
            cut_dict["master_file_url"] = master_url
            cut_dict["download_url"] = download_url
            cut_dict["clean_file_id"] = clean_id
            cut_dict["audio_cleaned"] = True
            cut_dict["status"] = "pending_approval"

            rec_fields = {k: v for k, v in cut_dict.items() if k in CutRecord.__dataclass_fields__}
            rec = CutRecord(**rec_fields)
            active_drip_min = DripScheduler(store).get_effective_interval_minutes(profile)
            new_msg_id = notifier.send_cut_for_approval(rec, preview_mp4, drip_interval_override=active_drip_min)
            cut_dict["discord_message_id"] = new_msg_id
            store.save_state_json(PROCESSED_STATE_FILE, processed_state)
            return

        # Ações de aprovação e publicação
        platform_map = {
            "approve_both": (["instagram", "tiktok", "youtube"], False),
            "now_both": (["instagram", "tiktok", "youtube"], True),
            "approve_reels": (["instagram"], False),
            "approve_tiktok": (["tiktok"], False),
            "now_tiktok": (["tiktok"], True),
            "approve_youtube": (["youtube"], False),
            "now_youtube": (["youtube"], True),
        }
        if action not in platform_map:
            raise ValueError(f"Ação desconhecida recebida do Discord: {action}")

        platforms, force_now = platform_map[action]
        scheduler = DripScheduler(store)
        outcome = scheduler.enqueue_or_publish_cut(
            profile=profile,
            cut_dict=cut_dict,
            platforms=platforms,
            work_dir=work_dir,
            force_immediate=force_now,
        )

        if outcome["mode"] == "published_now":
            cut_dict["status"] = "published"
            store.save_state_json(PROCESSED_STATE_FILE, processed_state)
            notifier.post_status_reply(
                msg_ref,
                f"🚀 **Publicado no ar agora!** (`{cut_id}`) → `{json.dumps(outcome['results'], ensure_ascii=False)}`",
            )
        elif outcome["mode"] == "queued":
            cut_dict["status"] = "queued"
            store.save_state_json(PROCESSED_STATE_FILE, processed_state)
            notifier.post_status_reply(
                msg_ref,
                f"⏰ **Na Fila Espaçada!** (`{cut_id}`) sairá automaticamente em **{outcome['scheduled_local']} BRT** ({', '.join(platforms)}).",
            )
        else:
            notifier.post_status_reply(msg_ref, f"ℹ️ Corte `{cut_id}` já estava agendado na fila.")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def cmd_drip_publish() -> int:
    if _is_unconfigured_ci():
        logger.info("Aguardando configuração inicial (`/setup iniciar` ou GDRIVE_TOKEN_JSON nos Secrets).")
        return 0
    store = DriveStore()
    profiles = {p.id: p for p in load_profiles(only_enabled=False)}
    processed_state = store.load_state_json(PROCESSED_STATE_FILE, default={"cuts": []})
    work_dir = PROJECT_ROOT / "work_tmp" / "drip"

    try:
        scheduler = DripScheduler(store)
        reports = scheduler.process_due_queue(profiles, processed_state, work_dir)
        if reports:
            store.save_state_json(PROCESSED_STATE_FILE, processed_state)
            for rep in reports:
                cut = rep["cut"]
                prof = profiles.get(cut["profile_id"])
                if prof:
                    DiscordNotifier(prof).post_status_reply(
                        str(cut.get("discord_message_id", "")),
                        f"✅ **Fila Espaçada disparada!** Corte `{cut['cut_id']}` publicado: `{json.dumps(rep['results'], ensure_ascii=False)}`",
                    )
        logger.info("Verificação da Fila Espaçada concluída: %d post(s) disparado(s).", len(reports))
        retention_days = max((int(p.publishing.drive_retention_days) for p in profiles.values()), default=10)
        _run_drive_retention_cleanup(store, processed_state, retention_days=retention_days)
        return len(reports)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def cmd_colab_loop(interval_minutes: float = 10.0, max_hours: float = 23.5, profile_filter: Optional[str] = None) -> None:
    """
    Modo Turbo Contínuo para o Google Colab Ultra (Background Execution até 24h com GPU L4/A100).
    """
    deadline = time.time() + (max_hours * 3600.0)
    cycle = 0
    logger.info(
        "Iniciando Colab Ultra GPU Loop (intervalo=%.1f min, teto=%.1f h)...",
        interval_minutes,
        max_hours,
    )
    while time.time() < deadline:
        cycle += 1
        logger.info("=== Ciclo Colab GPU #%d ===", cycle)
        try:
            cmd_watch(profile_filter=profile_filter)
            cmd_drip_publish()
        except Exception as exc:
            logger.exception("Erro no ciclo #%d do Colab Loop: %s", cycle, exc)
        sleep_sec = max(60.0, interval_minutes * 60.0)
        if time.time() + sleep_sec >= deadline:
            break
        time.sleep(sleep_sec)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Twitch AutoClipper 24/7 — Orquestrador Principal")
    sub = parser.add_subparsers(dest="command", required=True)

    p_watch = sub.add_parser("watch", help="Monitora a Twitch (Modo Duplo) e gera novos cortes.")
    p_watch.add_argument("--profile", type=str, default=None, help="Filtrar por ID ou login do perfil.")
    p_watch.add_argument("--max-cuts", type=int, default=None, help="Sobrescrever limite de cortes nesta rodada.")

    p_act = sub.add_parser("discord-action", help="Processa botão ou comando /cortar do Discord.")
    p_act.add_argument("--action", type=str, required=True)
    p_act.add_argument("--cut-id", type=str, default="")
    p_act.add_argument("--message-id", type=str, default="")
    p_act.add_argument("--url", type=str, default="")
    p_act.add_argument("--max-cuts", type=int, default=3)
    p_act.add_argument("--time-range", type=str, default="")

    p_yt = sub.add_parser("youtube", help="Gera cortes automáticos de um vídeo ou live do YouTube.")
    p_yt.add_argument("--url", type=str, required=True, help="Link do vídeo ou live do YouTube.")
    p_yt.add_argument("--max-cuts", type=int, default=3, help="Quantidade de cortes (1 a 10).")
    p_yt.add_argument("--range", type=str, default="", help="Trecho opcional (ex: 12:30-15:00 ou 25:00).")
    p_yt.add_argument("--profile", type=str, default=None)

    p_kick = sub.add_parser("kick", help="Gera cortes automáticos de um clipe, VOD ou live da Kick.")
    p_kick.add_argument("--url", type=str, required=True, help="Link ou slug da Kick.")
    p_kick.add_argument("--max-cuts", type=int, default=3, help="Quantidade de cortes (1 a 10).")
    p_kick.add_argument("--range", type=str, default="", help="Trecho opcional (ex: 12:30-15:00).")
    p_kick.add_argument("--profile", type=str, default=None)

    sub.add_parser("drip-publish", help="Publica cortes vencidos na Fila Espaçada Inteligente.")

    p_clean = sub.add_parser("cleanup", help="Remove do Google Drive vídeos mais antigos que N dias.")
    p_clean.add_argument("--days", type=int, default=None, help="Dias de retenção (padrão: config de profiles.yaml).")

    p_colab = sub.add_parser("colab-loop", help="Loop contínuo com GPU para Google Colab Ultra (até 24h).")
    p_colab.add_argument("--interval-minutes", type=float, default=10.0)
    p_colab.add_argument("--max-hours", type=float, default=23.5)
    p_colab.add_argument("--profile", type=str, default=None)

    return parser


def main() -> None:
    load_environment()
    parser = build_cli()
    args = parser.parse_args()

    if args.command == "watch":
        cmd_watch(profile_filter=args.profile, max_cuts_override=args.max_cuts)
    elif args.command == "youtube":
        cmd_youtube(
            url=args.url,
            max_cuts=args.max_cuts,
            time_range=args.range,
            profile_filter=args.profile,
        )
    elif args.command == "kick":
        cmd_kick(
            url=args.url,
            max_cuts=args.max_cuts,
            time_range=args.range,
            profile_filter=args.profile,
        )
    elif args.command == "discord-action":
        cmd_discord_action(
            action=args.action,
            cut_id=args.cut_id,
            discord_message_id=args.message_id,
            youtube_url=args.url,
            max_cuts=args.max_cuts,
            time_range=args.time_range,
        )
    elif args.command == "drip-publish":
        cmd_drip_publish()
    elif args.command == "cleanup":
        cmd_cleanup(retention_days=args.days)
    elif args.command == "colab-loop":
        cmd_colab_loop(
            interval_minutes=args.interval_minutes,
            max_hours=args.max_hours,
            profile_filter=args.profile,
        )


if __name__ == "__main__":
    main()
