from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.ai.director import GeminiDirector
from src.config_loader import load_profiles, load_subtitle_theme
from src.ffmpeg_utils import get_ffmpeg_bin
from src.models import (
    DetectionConfig,
    DirectorDecision,
    EditingConfig,
    MomentCluster,
    ProfileConfig,
    TwitchClip,
    WordTimestamp,
)
from src.publisher.drip_scheduler import DripScheduler
from src.storage.gdrive import DriveStore
from src.twitch.client import parse_twitch_duration_seconds
from src.twitch.detector import MomentDetector
from src.video.captions import CaptionGenerator, hex_to_ass_color
from src.video.layout_analyzer import CropRect, LayoutAnalyzer, LayoutSpec
from src.video.renderer import VideoRenderer


def _make_profile() -> ProfileConfig:
    return ProfileConfig(
        id="test_prof",
        twitch_login="streamerbr",
        display_name="StreamerBR",
        detection=DetectionConfig(
            cluster_window_seconds=75,
            min_clips_live_cluster=3,
            min_views_single_clip_live=80,
            min_views_post_live=120,
            max_cuts_per_stream=12,
            max_cuts_per_run=5,
            dedup_overlap_seconds=50,
        ),
        editing=EditingConfig(
            max_duration_seconds=58.0,
            min_duration_seconds=16.0,
            cold_open_enabled=True,
            cold_open_min_main_duration=26.0,
        ),
    )


def test_config_and_theme_load():
    profiles = load_profiles(only_enabled=False)
    assert len(profiles) >= 1
    theme = load_subtitle_theme()
    assert theme["canvas_width"] == 1080
    assert theme["canvas_height"] == 1920


def test_parse_twitch_duration():
    assert parse_twitch_duration_seconds("6h24m12s") == 6 * 3600 + 24 * 60 + 12
    assert parse_twitch_duration_seconds("45m10s") == 45 * 60 + 10
    assert parse_twitch_duration_seconds("58s") == 58
    assert parse_twitch_duration_seconds("") == 0


def test_moment_clustering_and_deduplication():
    prof = _make_profile()
    detector = MomentDetector(prof)
    now = datetime(2026, 11, 18, 3, 0, 0, tzinfo=timezone.utc)

    # 4 espectadores diferentes clipando entre vod_offset 1200s e 1245s -> deve formar 1 cluster forte!
    clips = [
        TwitchClip("c1", "https://clips.twitch.tv/c1", "QUE ISSO KKKKK", 25, "2026-11-18T02:55:00Z", 30.0, "userA", "99", "StreamerBR", video_id="v555", vod_offset=1200),
        TwitchClip("c2", "https://clips.twitch.tv/c2", "ELE CAIU", 40, "2026-11-18T02:55:10Z", 28.0, "userB", "99", "StreamerBR", video_id="v555", vod_offset=1215),
        TwitchClip("c3", "https://clips.twitch.tv/c3", "INTANCAVEL", 15, "2026-11-18T02:55:20Z", 35.0, "userC", "99", "StreamerBR", video_id="v555", vod_offset=1240),
        # Segundo momento isolado mas com explosão individual de views (190 views >= 80)
        TwitchClip("c4", "https://clips.twitch.tv/c4", "CLUTCH 1V5", 190, "2026-11-18T02:40:00Z", 45.0, "userD", "99", "StreamerBR", video_id="v555", vod_offset=3600),
        # Clipe fraco isolado (1 pessoa, 5 views) -> deve ser ignorado!
        TwitchClip("c5", "https://clips.twitch.tv/c5", "teste", 5, "2026-11-18T02:30:00Z", 20.0, "userE", "99", "StreamerBR", video_id="v555", vod_offset=500),
    ]

    clusters = detector.cluster_clips(clips, stream_key="v555", mode="live_hot", now_utc=now)
    assert len(clusters) == 2

    # Testa deduplicação caso o momento em 1220s já tenha sido processado
    processed_state = {
        "cuts": [
            {
                "profile_id": "test_prof",
                "stream_key": "v555",
                "vod_anchor_sec": 1220,
                "clip_ids": ["old_clip"],
            }
        ]
    }
    filtered = detector._filter_and_deduplicate(
        clusters=clusters,
        processed_state=processed_state,
        stream_key="v555",
        alt_stream_key="",
        remaining_slots=10,
    )
    assert len(filtered) == 1
    assert filtered[0].peak_offset == 3600


def test_director_hard_58s_ceiling_and_cold_open():
    prof = _make_profile()
    director = GeminiDirector(prof, api_key="")

    words = [
        WordTimestamp(word=f"palavra{i}", start=float(i), end=float(i) + 0.7)
        for i in range(100)
    ]

    # Simula retorno onde o modelo pediu 75s de corte principal + 3s de teaser -> deve cravar <= 58.0s no total!
    decision = director.enforce_constraints(
        raw={
            "main_start": 10.0,
            "main_end": 85.0,
            "cold_open_enabled": True,
            "teaser_start": 48.0,
            "teaser_end": 50.8,
            "teaser_headline": "DEU TUDO ERRADO",
            "caption_text": "simplesmente perdeu tudo em 2 segundos",
            "hashtags": ["streamerbr", "#cortes", "#live", "#gamer", "#extra_ignorada"],
            "virality_score": 92,
        },
        words=words,
        raw_duration_sec=110.0,
        peak_hint_sec=50.0,
    )

    assert decision.cold_open_enabled is True
    assert 2.0 <= decision.teaser_duration <= 3.2
    assert decision.total_duration <= 58.001
    assert len(decision.hashtags) == 4


def test_opus_ass_and_srt_generation(tmp_path: Path):
    theme = load_subtitle_theme()
    assert hex_to_ass_color("#FFE600", 0) == "&H0000E6FF"

    words = [
        WordTimestamp("mano", 10.0, 10.4),
        WordTimestamp("olha", 10.45, 10.8),
        WordTimestamp("isso", 10.85, 11.3),
        WordTimestamp("deu", 30.2, 30.5),
        WordTimestamp("ruim!", 30.55, 31.1),
    ]
    decision = DirectorDecision(
        main_start=10.0,
        main_end=42.0,
        cold_open_enabled=True,
        teaser_start=30.0,
        teaser_end=32.5,
        teaser_headline="OLHA ESSE FINAL 💀",
        caption_text="intancável demais",
        hashtags=["#cortes", "#twitchbr", "#live"],
        virality_score=90,
    )

    ass_path = tmp_path / "legenda.ass"
    srt_path = tmp_path / "legenda.srt"
    CaptionGenerator(theme).generate_files(words, decision, ass_path, srt_path)

    ass_txt = ass_path.read_text(encoding="utf-8-sig")
    srt_txt = srt_path.read_text(encoding="utf-8")

    # Headline exclusiva em 2 cores nos primeiros ~2.5s do Cold Open
    assert "Style: TeaserHook" in ass_txt
    assert "0:00:00.00,0:00:02.44,TeaserHook" in ass_txt
    assert "OLHA ESSE" in ass_txt and "FINAL" in ass_txt
    # Destaque de cor ativo palavra a palavra (Amarelo Ouro / Coral / Verde Neon)
    assert "MANO" in ass_txt
    assert "1\n00:00:02,500" in srt_txt


def test_smart_drip_queue_spacing_and_quiet_hours(tmp_path: Path):
    store = DriveStore(mount_path=str(tmp_path / "drive"))
    prof = _make_profile()
    prof.publishing.drip_interval_minutes = 75
    prof.publishing.quiet_hours_brt = [3, 9]

    scheduler = DripScheduler(store)
    # 18:00 UTC = 15:00 BRT (fora do horário de silêncio)
    t0 = datetime(2026, 11, 18, 18, 0, 0, tzinfo=timezone.utc)
    state = {"last_published_by_profile": {}, "items": []}

    slot1, immediate1 = scheduler.compute_next_slot(prof, state, now_utc=t0)
    assert immediate1 is True
    assert slot1 == t0

    # Simula que o 1º acabou de ser publicado às 18:00 UTC -> próximo deve cair às 19:15 UTC (+75m)
    state["last_published_by_profile"][prof.id] = "2026-11-18T18:00:00Z"
    slot2, immediate2 = scheduler.compute_next_slot(prof, state, now_utc=t0 + timedelta(minutes=2))
    assert immediate2 is False
    assert slot2 == datetime(2026, 11, 18, 19, 15, 0, tzinfo=timezone.utc)

    # Simula aprovação na madrugada (07:30 UTC = 04:30 BRT, dentro de [3, 9]) -> empurra para 09:05 BRT (12:05 UTC)
    madrugada_utc = datetime(2026, 11, 19, 7, 30, 0, tzinfo=timezone.utc)
    state_madrugada = {"last_published_by_profile": {}, "items": []}
    slot_mad, imm_mad = scheduler.compute_next_slot(prof, state_madrugada, now_utc=madrugada_utc)
    assert imm_mad is False
    assert slot_mad == datetime(2026, 11, 19, 12, 5, 0, tzinfo=timezone.utc)


@pytest.mark.skipif(shutil.which(get_ffmpeg_bin()) is None and not Path(get_ffmpeg_bin()).exists(), reason="FFmpeg indisponível")
def test_end_to_end_ffmpeg_render_and_drive_store(tmp_path: Path):
    raw_mp4 = tmp_path / "synthetic_16x9.mp4"
    # Gera um vídeo horizontal 1280x720 de 8s com áudio senoidal via FFmpeg
    subprocess.run(
        [
            get_ffmpeg_bin(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=8:size=1280x720:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(raw_mp4),
        ],
        check=True,
        capture_output=True,
    )

    prof = _make_profile()
    layout = LayoutAnalyzer(prof.editing).analyze(raw_mp4, 0.5, 7.5, num_samples=6)
    assert layout.mode in {"blur_center", "split_screen", "face_track"}

    decision = DirectorDecision(
        main_start=0.5,
        main_end=7.0,
        cold_open_enabled=True,
        teaser_start=4.5,
        teaser_end=6.6,
        teaser_headline="TESTE DE TEASER",
        caption_text="teste de corte automatizado",
        hashtags=["#cortes", "#twitchbr", "#live"],
        virality_score=88,
    )
    words = [
        WordTimestamp("teste", 1.0, 1.5),
        WordTimestamp("automatizado", 1.6, 2.3),
        WordTimestamp("explosivo", 4.7, 5.5),
    ]

    ass_path = tmp_path / "legenda.ass"
    srt_path = tmp_path / "legenda.srt"
    CaptionGenerator(load_subtitle_theme()).generate_files(words, decision, ass_path, srt_path)

    rendered = VideoRenderer(force_cpu=True).render_all(
        raw_video=raw_mp4,
        decision=decision,
        layout=layout,
        ass_file=ass_path,
        out_dir=tmp_path / "out",
    )

    assert rendered.master_mp4.exists() and rendered.master_mp4.stat().st_size > 10_000
    assert rendered.clean_mp4.exists() and rendered.clean_mp4.stat().st_size > 10_000
    assert rendered.discord_preview_mp4.exists()
    assert rendered.discord_preview_mp4.stat().st_size < 9 * 1024 * 1024

    store = DriveStore(mount_path=str(tmp_path / "drive_mount"))
    fid, _ = store.upload_file(rendered.master_mp4, ["cortes", "streamerbr", "cut01"], "master_1080p60.mp4")
    assert Path(fid).exists()

    # Valida também o grafo FFmpeg do `split_screen` (Webcam topo + Gameplay base)
    split_layout = LayoutSpec(
        mode="split_screen",
        src_width=1280,
        src_height=720,
        webcam_crop=CropRect(x=20, y=20, w=320, h=200),
        gameplay_crop=CropRect(x=360, y=0, w=626, h=720),
    )
    rendered_split = VideoRenderer(force_cpu=True).render_all(
        raw_video=raw_mp4,
        decision=decision,
        layout=split_layout,
        ass_file=ass_path,
        out_dir=tmp_path / "out_split",
    )
    assert rendered_split.master_mp4.exists() and rendered_split.master_mp4.stat().st_size > 10_000
