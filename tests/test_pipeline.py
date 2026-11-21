from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.ai.deduplicator import find_duplicate_candidate, find_duplicate_cut
from src.ai.director import GeminiDirector
from src.config_loader import load_profiles, load_subtitle_theme
from src.ffmpeg_utils import get_ffmpeg_bin
from src.models import (
    CutRecord,
    DetectionConfig,
    DirectorDecision,
    EditingConfig,
    MomentCluster,
    ProfileConfig,
    TikTokPublishConfig,
    TwitchClip,
    WordTimestamp,
)
from src.publisher.drip_scheduler import DripScheduler
from src.publisher.tiktok_poster import TikTokPublisher
from src.kick.analyzer import KickMomentAnalyzer, extract_kick_slug, parse_kick_url
from src.storage.gdrive import DriveStore
from src.twitch.client import parse_twitch_duration_seconds
from src.twitch.detector import MomentDetector
from src.video.captions import CaptionGenerator, hex_to_ass_color
from src.video.layout_analyzer import CropRect, LayoutAnalyzer, LayoutSpec
from src.video.renderer import VideoRenderer
from src.youtube.analyzer import YouTubeMomentAnalyzer, parse_time_range


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


def test_youtube_heatmap_and_time_range_parser():
    assert parse_time_range("12:30-14:00") == (750, 840)
    assert parse_time_range("10:00") == (585, 650)

    heatmap = [
        {"start_time": 100.0, "end_time": 110.0, "value": 0.30},
        {"start_time": 300.0, "end_time": 310.0, "value": 0.95},
        {"start_time": 312.0, "end_time": 320.0, "value": 0.85},  # perto de 305 -> deduplicado (<65s)
        {"start_time": 600.0, "end_time": 610.0, "value": 0.80},
    ]
    peaks = YouTubeMomentAnalyzer._extract_heatmap_peaks(heatmap, 0, 1000, max_cuts=2, chapters=[])
    assert len(peaks) >= 2
    # O pico #1 deve ser o de value=0.95 (~305s -> janela começando em 281s com 60s de contexto)
    assert peaks[0][0] == 281
    assert peaks[1][0] == 581


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

    # Garante matematicamente que NENHUMA legenda OpusCap sobreponha a seguinte no tempo
    cap_lines = [ln for ln in ass_txt.splitlines() if ln.startswith("Dialogue: 2,")]
    def _to_sec(ts: str) -> float:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    parsed_ranges = [(_to_sec(ln.split(",")[1]), _to_sec(ln.split(",")[2])) for ln in cap_lines]
    for idx in range(len(parsed_ranges) - 1):
        assert parsed_ranges[idx][1] <= parsed_ranges[idx + 1][0] + 1e-6


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


def test_drive_retention_cleanup_10_days_and_public_download_url(tmp_path: Path):
    store = DriveStore(mount_path=str(tmp_path / "drive_cleanup"))
    now = datetime(2026, 11, 18, 18, 0, 0, tzinfo=timezone.utc)

    old_mp4 = tmp_path / "old_src.mp4"
    old_mp4.write_bytes(b"old-video-bytes")
    new_mp4 = tmp_path / "new_src.mp4"
    new_mp4.write_bytes(b"new-video-bytes")
    queued_mp4 = tmp_path / "queued_src.mp4"
    queued_mp4.write_bytes(b"queued-video-bytes")

    old_folder_id, _ = store.ensure_folder_path(["cortes", "alanzoka", "old_cut"])
    old_master_id, _ = store.upload_file(old_mp4, ["cortes", "alanzoka", "old_cut"], "master_1080p60.mp4")

    new_folder_id, _ = store.ensure_folder_path(["cortes", "alanzoka", "new_cut"])
    new_master_id, _ = store.upload_file(new_mp4, ["cortes", "alanzoka", "new_cut"], "master_1080p60.mp4")

    queued_folder_id, _ = store.ensure_folder_path(["cortes", "alanzoka", "queued_cut"])
    store.upload_file(queued_mp4, ["cortes", "alanzoka", "queued_cut"], "master_1080p60.mp4")

    dl_uri = store.ensure_public_download_url(new_master_id)
    assert dl_uri.startswith("file://")

    state = {
        "cuts": [
            {
                "cut_id": "cut_old_11d",
                "profile_id": "criador_01",
                "stream_key": "vod_1",
                "video_id": "vod_1",
                "vod_anchor_sec": 120,
                "clip_ids": ["clip_old"],
                "created_at": (now - timedelta(days=11)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "drive_folder_id": old_folder_id,
                "drive_folder_url": "https://drive.google.com/drive/folders/old",
                "master_file_id": old_master_id,
                "master_file_url": "https://drive.google.com/file/d/old/view",
                "download_url": "https://drive.google.com/uc?export=download&id=old&confirm=t",
                "clean_file_id": "",
                "srt_file_id": "",
                "status": "published",
            },
            {
                "cut_id": "cut_recent_3d",
                "profile_id": "criador_01",
                "stream_key": "vod_2",
                "video_id": "vod_2",
                "vod_anchor_sec": 300,
                "clip_ids": ["clip_new"],
                "created_at": (now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "drive_folder_id": new_folder_id,
                "master_file_id": new_master_id,
                "status": "pending_approval",
            },
            {
                "cut_id": "cut_old_but_queued",
                "profile_id": "criador_01",
                "stream_key": "vod_3",
                "video_id": "vod_3",
                "vod_anchor_sec": 500,
                "created_at": (now - timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "drive_folder_id": queued_folder_id,
                "status": "queued",
            },
        ]
    }

    report = store.cleanup_expired_cuts(
        state,
        retention_days=10,
        now_utc=now,
        active_queue_cut_ids={"cut_old_but_queued"},
    )
    assert report["changed"] is True
    assert report["deleted"] == 1
    assert report["deleted_cut_ids"] == ["cut_old_11d"]

    # Pasta antiga removida do Drive, mas metadados de deduplicação preservados
    assert not Path(old_folder_id).exists()
    assert state["cuts"][0]["drive_deleted"] is True
    assert state["cuts"][0]["vod_anchor_sec"] == 120
    assert state["cuts"][0]["clip_ids"] == ["clip_old"]
    assert state["cuts"][0]["download_url"] == ""

    # Corte recente (<10d) e corte protegido na fila continuam intactos
    assert Path(new_folder_id).exists()
    assert Path(queued_folder_id).exists()


def test_tiktok_direct_self_only_and_unaudited_inbox_fallback(tmp_path: Path, monkeypatch):
    video = tmp_path / "cut.mp4"
    video.write_bytes(b"0" * 1024)

    class FakeResp:
        def __init__(self, status_code: int, body: dict):
            self.status_code = status_code
            self._body = body
            self.ok = 200 <= status_code < 300

        def json(self):
            return self._body

        def raise_for_status(self):
            if not self.ok:
                raise RuntimeError(f"HTTP {self.status_code}: {self._body}")

    # Cenário 1: Direct Post pré-aprovação enviando SELF_ONLY com legenda/caption (conta privada)
    calls = []

    def fake_post_direct(url, headers=None, json=None, data=None, timeout=30):
        calls.append((url, json))
        if url.endswith("/post/publish/video/init/"):
            return FakeResp(200, {"data": {"publish_id": "pub_direct_1", "upload_url": "https://upload.tiktok/1"}})
        return FakeResp(404, {})

    def fake_put(url, headers=None, data=None, timeout=240):
        return FakeResp(200, {})

    monkeypatch.setattr("src.publisher.tiktok_poster.requests.post", fake_post_direct)
    monkeypatch.setattr("src.publisher.tiktok_poster.requests.put", fake_put)

    prof = _make_profile()
    prof.publishing.tiktok = TikTokPublishConfig(
        enabled=True,
        mode="direct",
        privacy_level="SELF_ONLY",
        fallback_to_inbox=True,
    )
    pub = TikTokPublisher(prof, access_token="tok_123")
    pid = pub.publish_video(video, "minha descrição braba\n\n#cortes #live")
    assert pid == "pub_direct_1"
    assert pub.last_publish_mode == "direct:SELF_ONLY"
    assert calls[0][1]["post_info"]["privacy_level"] == "SELF_ONLY"
    assert "minha descrição braba" in calls[0][1]["post_info"]["title"]

    # Cenário 2: Conta pública pré-aprovação retorna 403 unaudited_client -> fallback automático para inbox
    calls_fb = []

    def fake_post_fallback(url, headers=None, json=None, data=None, timeout=30):
        calls_fb.append(url)
        if url.endswith("/post/publish/video/init/"):
            return FakeResp(403, {"error": {"code": "unaudited_client_can_only_post_to_private_accounts"}})
        if url.endswith("/post/publish/inbox/video/init/"):
            return FakeResp(200, {"data": {"publish_id": "pub_inbox_2", "upload_url": "https://upload.tiktok/2"}})
        return FakeResp(404, {})

    monkeypatch.setattr("src.publisher.tiktok_poster.requests.post", fake_post_fallback)
    pub_fb = TikTokPublisher(prof, access_token="tok_123")
    pid_fb = pub_fb.publish_video(video, "caption test")
    assert pid_fb == "pub_inbox_2"
    assert pub_fb.last_publish_mode == "inbox_fallback"
    assert any(u.endswith("/post/publish/inbox/video/init/") for u in calls_fb)


def test_discord_notifier_embed_has_independent_download_and_copy_button():
    from src.approval.discord_notifier import DiscordNotifier

    cut = CutRecord(
        cut_id="c_dl_1",
        profile_id="criador_01",
        twitch_login="alanzoka",
        stream_key="v1",
        mode="live_hot",
        score=95.0,
        source_type="vod",
        video_id="v1",
        vod_anchor_sec=100,
        clip_urls=[],
        caption="que susto KKKKK\n\n#cortes #live",
        teaser_headline="",
        total_duration=42.0,
        layout_used="split_screen",
        drive_folder_id="fld1",
        drive_folder_url="https://drive.google.com/drive/folders/fld1",
        master_file_id="mst1",
        master_file_url="https://drive.google.com/file/d/mst1/view",
        clean_file_id="cln1",
        srt_file_id="srt1",
        created_at="2026-11-18T17:00:00Z",
        download_url="https://drive.google.com/uc?export=download&id=mst1&confirm=t",
    )
    assert "export=download" in cut.download_url
    rows = DiscordNotifier._build_action_rows("c_dl_1", drip_min=75, audio_cleaned=False, ig_enabled=False, tt_enabled=True)
    custom_ids = [btn["custom_id"] for r in rows for btn in r["components"]]
    assert "copy_caption:c_dl_1" in custom_ids
    assert "approve_tiktok:c_dl_1" in custom_ids


def test_youtube_director_trims_context_and_long_transcript_sampling(monkeypatch):
    # 1. Transcrição longa (>24k chars) deve preservar trechos do início, meio e fim
    long_lines = "\n".join(f"[{i * 10}s] fala numero {i} acontecendo na live agora" for i in range(900))
    assert len(long_lines) > 35000
    sampled = YouTubeMomentAnalyzer._sample_transcript_for_ai(long_lines, max_chars=8000)
    assert len(sampled) <= 8100
    assert "[0s]" in sampled
    assert "[45" in sampled  # meio da live (~4500s)
    assert "[67" in sampled or "[68" in sampled  # trecho final da live

    # YouTube auto-repescagem deve ignorar vídeos/VODs mais antigos que `max_age_hours` (36h)
    yt_an = YouTubeMomentAnalyzer(_make_profile())
    old_ts = datetime.now(timezone.utc).timestamp() - (72 * 3600)
    monkeypatch.setattr(
        yt_an,
        "fetch_metadata",
        lambda url, wd: {"id": "yt_old_1", "title": "Live antiga", "is_live": False, "duration": 1800, "timestamp": old_ts},
    )
    _, skipped_old = yt_an.discover_moments("https://youtu.be/old", max_cuts=2, work_dir=Path("work_tmp"), max_age_hours=36)
    assert skipped_old == []

    # 2. YouTube (`preserve_full_clip=False`) deve respeitar o corte cirúrgico dentro do buffer
    prof = _make_profile()
    prof.editing.max_duration_seconds = 65.0
    prof.editing.cold_open_enabled = False
    director = GeminiDirector(prof, api_key="", groq_api_key="dummy")

    fake_raw = {
        "main_start": 14.0,
        "main_end": 53.0,
        "cold_open_enabled": False,
        "teaser_start": 0.0,
        "teaser_end": 0.0,
        "teaser_headline": "",
        "caption_text": "ele achou que ia dar bom",
        "hashtags": ["#gameplayrj", "#cortes", "#youtube"],
        "virality_score": 91,
    }
    monkeypatch.setattr(director, "_call_groq_llm", lambda prompt: fake_raw)

    words = [WordTimestamp(f"w{i}", float(i), float(i) + 0.5) for i in range(60)]
    yt_cluster = MomentCluster(
        cluster_id="yt_c1",
        profile_id=prof.id,
        broadcaster_login="gameplayrj",
        stream_or_vod_key="yt_abc",
        mode="youtube_auto",
        clips=[TwitchClip("c1", "https://youtube.com/watch?v=abc", "historia completa", 100, "", 60.0, "Davy", "yt", "Gameplayrj")],
        video_id="yt_abc",
        vod_start_offset=100,
        vod_end_offset=160,
        peak_offset=130,
        unique_clippers=1,
        total_views=100,
        max_views=100,
        score=90.0,
        dominant_title="historia completa",
    )
    yt_decision = director.direct_cut(yt_cluster, words, raw_duration_sec=60.0, peak_hint_sec=32.0)
    assert 13.0 <= yt_decision.main_start <= 15.0
    assert 52.0 <= yt_decision.main_end <= 54.0

    # Já um clipe da Twitch `<= 65s` continua 100% íntegro (0.0 -> 60.0)
    tw_cluster = MomentCluster(**{**yt_cluster.__dict__, "cluster_id": "tw_c1", "mode": "live_hot"})
    tw_decision = director.direct_cut(tw_cluster, words, raw_duration_sec=60.0, peak_hint_sec=32.0)
    assert tw_decision.main_start == 0.0
    assert tw_decision.main_end == 60.0


def test_kick_url_parsing_moment_discovery_and_director(monkeypatch):
    assert extract_kick_slug("https://kick.com/JonVlogs") == "jonvlogs"
    assert extract_kick_slug("@paulinho") == "paulinho"
    assert parse_kick_url("https://kick.com/jonvlogs/clips/clip_123") == {
        "kind": "clip",
        "slug": "jonvlogs",
        "clip_id": "clip_123",
    }
    assert parse_kick_url("https://kick.com/jonvlogs/videos/vod_abc") == {
        "kind": "vod",
        "slug": "jonvlogs",
        "vod_id": "vod_abc",
    }
    assert parse_kick_url("https://kick.com/jonvlogs?clip=clip_q1") == {
        "kind": "clip",
        "slug": "jonvlogs",
        "clip_id": "clip_q1",
    }

    prof = _make_profile()
    prof.editing.max_duration_seconds = 65.0
    prof.editing.cold_open_enabled = False
    analyzer = KickMomentAnalyzer(prof)

    monkeypatch.setattr(
        analyzer,
        "fetch_channel_info",
        lambda slug: {"slug": slug, "user": {"username": "JonVlogs"}, "livestream": {"id": 999, "session_title": "IRL"}, "playback_url": "https://stream.kick.com/live.m3u8"},
    )
    monkeypatch.setattr(
        analyzer,
        "fetch_channel_clips",
        lambda slug, time_window="day", sort="view": [
            {"id": "k_seen", "title": "antigo", "views": 500, "duration": 40, "clip_url": "https://clips.kick.com/old.m3u8"},
            {"id": "k_hot", "title": "quase bateu o carro", "views": 210, "duration": 42, "clip_url": "https://clips.kick.com/hot.m3u8"},
            {"id": "k_low", "title": "baixo", "views": 12, "duration": 30, "clip_url": "https://clips.kick.com/low.m3u8"},
        ],
    )
    monkeypatch.setattr(analyzer, "fetch_channel_videos", lambda slug: [])

    state = {"cuts": [{"clip_ids": ["k_seen"], "cut_id": "kick_old"}]}
    clusters = analyzer.discover_channel_moments("https://kick.com/jonvlogs", max_cuts=1, processed_state=state)
    assert len(clusters) == 1
    assert clusters[0].mode == "kick_clip"
    assert clusters[0].best_clip.id == "k_hot"
    assert clusters[0].broadcaster_login == "jonvlogs"

    # Repescagem: quando clipe de `day` já foi visto, deve buscar clipe elegível da janela `week`
    def fake_clips_day_vs_week(slug, time_window="day", sort="view"):
        if time_window == "day":
            return [{"id": "k_hot", "title": "visto", "views": 210, "duration": 42, "clip_url": "https://clips.kick.com/hot.m3u8"}]
        return [
            {"id": "k_hot", "title": "visto", "views": 210, "duration": 42, "clip_url": "https://clips.kick.com/hot.m3u8"},
            {"id": "k_week_1", "title": "repescagem semanal", "views": 350, "duration": 48, "clip_url": "https://clips.kick.com/w1.m3u8"},
        ]

    monkeypatch.setattr(analyzer, "fetch_channel_clips", fake_clips_day_vs_week)
    week_clusters = analyzer.discover_channel_moments("https://kick.com/jonvlogs", max_cuts=1, processed_state={"cuts": [{"clip_ids": ["k_hot"]}]})
    assert len(week_clusters) == 1
    assert week_clusters[0].best_clip.id == "k_week_1"

    # Clipe bem enquadrado (48s) deve ranquear acima de recorte gigante (180s) com mesmas views e deduplicar <=150s
    fixed_now = datetime(2026, 11, 18, 20, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        analyzer,
        "fetch_channel_clips",
        lambda slug, time_window="day", sort="view": [
            {"id": "k_dump", "title": "180s dump", "views": 220, "duration": 180, "created_at": "2026-11-18T19:00:00Z", "clip_url": "https://clips.kick.com/dump.m3u8"},
            {"id": "k_tight", "title": "48s bom", "views": 220, "duration": 48, "created_at": "2026-11-18T18:00:00Z", "clip_url": "https://clips.kick.com/tight.m3u8"},
            {"id": "k_old_5d", "title": "antigo 5d", "views": 9000, "duration": 45, "created_at": "2026-11-13T18:00:00Z", "clip_url": "https://clips.kick.com/old5d.m3u8"},
        ] if time_window == "day" else [],
    )
    ranked_both = analyzer.discover_channel_moments("https://kick.com/davyjones", max_cuts=3, processed_state={"cuts": []}, now_utc=fixed_now)
    assert [c.best_clip.id for c in ranked_both] == ["k_tight", "k_dump"]

    ranked = analyzer.discover_channel_moments(
        "https://kick.com/davyjones",
        max_cuts=2,
        processed_state={"cuts": [{"twitch_login": "davyjones", "clip_created_at": "2026-11-18T19:01:00Z"}]},
        now_utc=fixed_now,
    )
    assert len(ranked) == 1
    assert ranked[0].best_clip.id == "k_tight"

    # Se offline sem clipes elegíveis, repescagem pega apenas VOD recente (<= 36h) e ignora VOD antigo
    monkeypatch.setattr(analyzer, "fetch_channel_info", lambda slug: {"slug": slug, "user": {"username": "JonVlogs"}, "livestream": None})
    monkeypatch.setattr(analyzer, "fetch_channel_clips", lambda slug, time_window="day", sort="view": [])
    monkeypatch.setattr(
        analyzer,
        "fetch_channel_videos",
        lambda slug: [
            {"id": "vod_old", "uuid": "vod_old", "duration": 3600000, "created_at": "2026-11-14 12:00:00", "source": "https://stream.kick.com/old.m3u8"},
            {"id": "vod_recent", "uuid": "vod_recent", "duration": 3600000, "created_at": "2026-11-18 10:00:00", "source": "https://stream.kick.com/recent.m3u8"},
        ],
    )
    vod_recap = analyzer.discover_channel_moments("jonvlogs", max_cuts=1, processed_state={"cuts": []}, now_utc=fixed_now)
    assert len(vod_recap) == 1
    assert vod_recap[0].mode == "kick_vod"
    assert "vod_recent" in vod_recap[0].stream_or_vod_key

    # Restaura canal live para teste de `kick_live`
    monkeypatch.setattr(
        analyzer,
        "fetch_channel_info",
        lambda slug: {"slug": slug, "user": {"username": "JonVlogs"}, "livestream": {"id": 999, "session_title": "IRL"}, "playback_url": "https://stream.kick.com/live.m3u8"},
    )
    monkeypatch.setattr(analyzer, "fetch_channel_videos", lambda slug: [])

    # Se não há clipes fortes e canal está ao vivo, cai para captura ao vivo HLS (`kick_live`)
    monkeypatch.setattr(analyzer, "fetch_channel_clips", lambda slug, time_window="day", sort="view": [])
    live_clusters = analyzer.discover_channel_moments("jonvlogs", max_cuts=1, processed_state={"cuts": []})
    assert len(live_clusters) == 1
    assert live_clusters[0].mode == "kick_live"
    assert live_clusters[0].best_clip.url == "https://stream.kick.com/live.m3u8"

    # Diretor preserva `kick_clip` <=65s por inteiro e apara `kick_vod` cirurgicamente
    director = GeminiDirector(prof, api_key="", groq_api_key="dummy")
    monkeypatch.setattr(
        director,
        "_call_groq_llm",
        lambda prompt: {
            "main_start": 15.0,
            "main_end": 50.0,
            "cold_open_enabled": False,
            "teaser_start": 0.0,
            "teaser_end": 0.0,
            "teaser_headline": "",
            "caption_text": "ele nao esperava isso",
            "hashtags": ["#jonvlogs", "#cortes", "#kick"],
            "virality_score": 88,
        },
    )
    words = [WordTimestamp(f"w{i}", float(i), float(i) + 0.5) for i in range(60)]
    d_clip = director.direct_cut(clusters[0], words, raw_duration_sec=42.0, peak_hint_sec=21.0)
    assert d_clip.main_start == 0.0 and d_clip.main_end == 42.0

    # `kick_live` <= 65s também preserva o bloco completo sem cortar os 20s finais
    d_live = director.direct_cut(live_clusters[0], words, raw_duration_sec=58.0, peak_hint_sec=29.0)
    assert d_live.main_start == 0.0 and d_live.main_end == 58.0

    vod_cluster = MomentCluster(**{**live_clusters[0].__dict__, "mode": "kick_vod"})
    d_vod = director.direct_cut(vod_cluster, words, raw_duration_sec=60.0, peak_hint_sec=30.0)
    assert 14.0 <= d_vod.main_start <= 16.0 and 50.0 <= d_vod.main_end <= 52.0

    # Clipe longo da Kick (> 65s) onde o modelo tentou cortar cedo (60s -> 77s de 92s) deve preservar o final até 92s
    monkeypatch.setattr(
        director,
        "_call_groq_llm",
        lambda prompt: {
            "main_start": 60.1,
            "main_end": 77.18,
            "cold_open_enabled": False,
            "teaser_start": 0.0,
            "teaser_end": 0.0,
            "caption_text": "davy jones reagindo ate o final",
            "hashtags": ["#davyjones", "#kick", "#fy", "#viral"],
            "virality_score": 85,
        },
    )
    words_92 = [WordTimestamp(f"w{i}", float(i), float(i) + 0.5) for i in range(90)]
    d_long_clip = director.direct_cut(clusters[0], words_92, raw_duration_sec=92.0, peak_hint_sec=72.0)
    assert d_long_clip.main_end == 92.0
    assert d_long_clip.total_duration == 65.0


def test_dynamic_trending_hashtags_and_director_prompt(monkeypatch):
    from src.ai import trending_hashtags as th

    monkeypatch.setattr(th, "fetch_brazil_realtime_trends", lambda limit=20: ["kscerato", "cartola", "trailer novo"])
    monkeypatch.setattr("src.ai.director.fetch_brazil_realtime_trends", lambda limit=15: ["kscerato", "cartola"])
    monkeypatch.setattr(
        "src.ai.director.fetch_web_clip_benchmarks",
        lambda creator, game="", limit=6: {
            "titles": ["DAVY JONES PIROU NO CS2!", "KSCERATO FEZ O IMPOSSIVEL"],
            "hashtags": ["#cortesdegames", "#cs2brasil", "#fyp"],
            "search_terms": ["davy jones cs2", "kscerato clutch"],
        },
    )

    tags = th.suggest_dynamic_hashtags(
        creator_handle="davyjones",
        platform_mode="youtube_auto",
        game_name="Counter-Strike 2",
        title_text="Kscerato deu aula nesse clutch absurdo",
        transcript_snippet="olha o kscerato jogando",
        max_tags=5,
        base_discovery_tags=["#fy", "#fyp", "#viral"],
        web_benchmark_tags=["#cortesdegames", "#cs2brasil"],
    )
    assert tags[0] == "#davyjones"
    assert "#cs2" in tags
    assert "#kscerato" in tags
    assert "#cortesdegames" in tags
    assert "#twitchbr" not in tags

    prof = _make_profile()
    director = GeminiDirector(prof, api_key="")
    cl = MomentCluster(
        cluster_id="c_tr",
        profile_id=prof.id,
        broadcaster_login="davyjones",
        stream_or_vod_key="yt_1",
        mode="youtube_auto",
        clips=[TwitchClip("c1", "https://youtu.be/x", "kscerato clip", 100, "", 40.0, "davy", "yt", "Davy")],
        video_id="yt_1",
        vod_start_offset=10,
        vod_end_offset=50,
        peak_offset=30,
        unique_clippers=1,
        total_views=100,
        max_views=100,
        score=90.0,
        dominant_title="kscerato",
        game_name="Counter-Strike",
    )
    prompt = director._build_prompt(cl, [WordTimestamp("bora", 1.0, 1.5)], 40.0, 20.0)
    assert "kscerato" in prompt
    assert "Google Trends tempo real" in prompt
    assert "#cortesdegames" in prompt
    assert "LEGENDA NATURAL + SEO BASEADA NO QUE O STREAMER FALOU" in prompt
    assert "1 única linha curta em caixa baixa" in prompt
    assert "PROIBIDO usar `ele gritou" in prompt
    assert "REFERÊNCIAS REAIS DE LEGENDAS DE CLIPS DE JOGOS NA WEB" not in prompt

    # Sem falas claras (<3 palavras), fallback usa título em caixa baixa
    fb_dec = director.direct_cut(cl, [WordTimestamp("bora", 1.0, 1.5)], 40.0, 20.0)
    assert fb_dec.caption_text == "kscerato"

    # Com fala marcante perto do pico, fallback extrai citação real do streamer
    spoken_words = [
        WordTimestamp("cara", 17.0, 17.3),
        WordTimestamp("eu", 17.3, 17.5),
        WordTimestamp("não", 17.5, 17.7),
        WordTimestamp("tenho", 17.7, 18.0),
        WordTimestamp("mões!", 18.0, 18.4),
    ]
    fb_spoken = director.direct_cut(cl, spoken_words, 40.0, 18.0)
    assert "não tenho mões" in fb_spoken.caption_text

    # Garante limpeza de perguntas forçadas no modelo minimalista
    dec = director.enforce_constraints(
        {
            "main_start": 0.0,
            "main_end": 35.0,
            "cold_open_enabled": False,
            "teaser_start": 0.0,
            "teaser_end": 0.0,
            "caption_text": "ele reclamando dos 30 fps no meio da partida kkkkk! O que vocês acharam?",
            "hashtags": ["#davyjones", "#cs2", "#fy", "#viral"],
            "virality_score": 90,
        },
        words=[WordTimestamp("oi", 1.0, 1.5)],
        raw_duration_sec=35.0,
        peak_hint_sec=15.0,
    )
    assert "O que vocês acharam" not in dec.caption_text
    assert "reclamando dos 30 fps" in dec.caption_text


def test_drip_timer_override_and_discord_select_menu(tmp_path: Path):
    from src.approval.discord_notifier import DiscordNotifier

    store = DriveStore(mount_path=str(tmp_path / "drive_timer"))
    prof = _make_profile()
    prof.publishing.drip_interval_minutes = 75
    prof.publishing.quiet_hours_brt = [3, 9]
    scheduler = DripScheduler(store)

    now = datetime(2026, 11, 18, 18, 0, 0, tzinfo=timezone.utc)
    state = {
        "last_published_by_profile": {prof.id: "2026-11-18T17:50:00Z"},
        "items": [
            {"cut_id": "cut_a", "profile_id": prof.id, "platforms": ["tiktok"], "scheduled_for": "2026-11-18T19:05:00Z", "status": "pending", "created_at": "2026-11-18T17:55:00Z"},
            {"cut_id": "cut_b", "profile_id": prof.id, "platforms": ["tiktok"], "scheduled_for": "2026-11-18T20:20:00Z", "status": "pending", "created_at": "2026-11-18T17:58:00Z"},
        ],
    }
    store.save_state_json("drip_queue.json", state)

    report = scheduler.update_timer_override(
        profiles_by_id={prof.id: prof},
        interval_minutes=20,
        quiet_hours_str="desativar",
        now_utc=now,
    )
    assert report["interval_minutes"] == 20
    assert report["rescheduled_count"] == 2

    reloaded = store.load_state_json("drip_queue.json", default={})
    assert reloaded["drip_interval_override_minutes"] == 20
    # last_pub=17:50 + 20m = 18:10 UTC; item 2 = 18:30 UTC
    assert reloaded["items"][0]["scheduled_for"] == "2026-11-18T18:10:00Z"
    assert reloaded["items"][1]["scheduled_for"] == "2026-11-18T18:30:00Z"

    # Próximo slot também respeita 20m mesmo em horário de madrugada pois silêncio foi desativado
    madrugada_utc = datetime(2026, 11, 19, 7, 30, 0, tzinfo=timezone.utc)
    slot_mad, imm_mad = scheduler.compute_next_slot(prof, {"last_published_by_profile": {}, "items": [], "drip_interval_override_minutes": 20, "quiet_hours_override": []}, now_utc=madrugada_utc)
    assert imm_mad is True
    assert slot_mad == madrugada_utc

    rows = DiscordNotifier._build_action_rows("cut_a", drip_min=20, ig_enabled=False, tt_enabled=True)
    assert len(rows) == 3
    assert "Fila 20m" in rows[0]["components"][0]["label"]
    assert rows[2]["components"][0]["custom_id"] == "set_timer:cut_a"
    assert "atual: 20m" in rows[2]["components"][0]["placeholder"]


def test_multitenant_env_profiles_relay_and_ci_guard(tmp_path: Path, monkeypatch):
    from src.approval.discord_notifier import DiscordNotifier
    from src.config_loader import load_profiles
    from src.main import cmd_watch

    payload = {
        "profiles": [
            {
                "id": "perfil_01",
                "twitch_login": "",
                "display_name": "Cortes do Joao",
                "kick_channels": ["davyjones"],
                "discord": {"channel_id": "999888", "relay_tenant": "ch:999888"},
                "publishing": {"drip_interval_minutes": 45, "quiet_hours_brt": []},
            }
        ]
    }
    monkeypatch.setenv("AUTOCLIPPER_PROFILES_JSON", json.dumps(payload))
    profs = load_profiles(only_enabled=True)
    assert len(profs) == 1
    assert profs[0].twitch_login == ""
    assert profs[0].kick_channels == ["davyjones"]
    assert profs[0].discord.channel_id == "999888"
    assert profs[0].discord.relay_tenant == "ch:999888"

    # DiscordNotifier em modo Relay envia para /send e /reply com X-Relay-Tenant sem vazar BOT_TOKEN
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "")
    monkeypatch.setenv("DISCORD_CHANNEL_ID", "")
    monkeypatch.setenv("DISCORD_RELAY_URL", "https://bot.example.workers.dev/api/relay")
    monkeypatch.setenv("DISCORD_RELAY_TOKEN", "relay_tok_123")

    calls = []

    class FakeResp:
        ok = True
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "msg_relay_42"}

    def fake_post(url, headers=None, data=None, files=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers or {}, "json": json, "has_files": bool(files)})
        return FakeResp()

    monkeypatch.setattr("src.approval.discord_notifier.requests.post", fake_post)
    notifier = DiscordNotifier(profs[0])
    assert notifier.use_relay is True
    assert notifier.is_configured is True

    dummy_mp4 = tmp_path / "prev.mp4"
    dummy_mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    rec = CutRecord(
        cut_id="cut_rel_1",
        profile_id="perfil_01",
        twitch_login="davyjones",
        stream_key="kick_1",
        mode="kick_clip",
        score=95.0,
        source_type="kick_clip",
        video_id="k1",
        vod_anchor_sec=None,
        clip_urls=[],
        caption="Teste relay!",
        teaser_headline="",
        total_duration=25.0,
        layout_used="split_screen",
        drive_folder_id="f1",
        drive_folder_url="https://drive.google.com/f1",
        master_file_id="m1",
        master_file_url="https://drive.google.com/m1",
        clean_file_id="c1",
        srt_file_id="s1",
        created_at="2026-11-18T18:00:00Z",
    )
    msg_id = notifier.send_cut_for_approval(rec, dummy_mp4)
    assert msg_id == "msg_relay_42"
    notifier.post_status_reply("msg_relay_42", "Publicado!")

    assert calls[0]["url"] == "https://bot.example.workers.dev/api/relay/send"
    assert calls[0]["headers"]["Authorization"] == "Bearer relay_tok_123"
    assert calls[0]["headers"]["X-Relay-Tenant"] == "ch:999888"
    assert calls[0]["has_files"] is True
    assert calls[1]["url"] == "https://bot.example.workers.dev/api/relay/reply"

    # Fresh Open Template fork em GitHub Actions sem GDRIVE_TOKEN_JSON sai limpo (codigo 0)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GDRIVE_TOKEN_JSON", "")
    monkeypatch.setenv("GDRIVE_MOUNT_PATH", "")
    assert cmd_watch() == 0

    # Perfil somente Kick (twitch_login vazio) nao falha mesmo sem TWITCH_CLIENT_ID
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    monkeypatch.setenv("TWITCH_CLIENT_ID", "")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "")
    monkeypatch.setenv("GDRIVE_MOUNT_PATH", str(tmp_path / "mirror_watch"))
    monkeypatch.setattr("src.kick.analyzer.KickMomentAnalyzer.discover_channel_moments", lambda *a, **kw: [])
    assert cmd_watch() == 0


def test_instagram_reels_dual_login_and_placeholder_guard(tmp_path: Path, monkeypatch):
    from src.publisher.instagram_reels import InstagramReelsPublisher

    prof = _make_profile()
    prof.publishing.instagram.enabled = True

    # Placeholders iniciais do .env nao podem passar como configurados
    ph = InstagramReelsPublisher(
        prof,
        account_id="id_conta_instagram_creator_ou_business",
        access_token="token_longa_duracao_meta_graph_api",
    )
    assert ph.is_ready is False

    # Token IG... (Instagram Login sem Página Facebook) usa graph.instagram.com + video_url
    calls = []

    class FakeResp:
        ok = True
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, data=None, headers=None, timeout=None):
        calls.append(("POST", url, data or {}))
        if url.endswith("/media"):
            return FakeResp({"id": "cnt_ig_1", "uri": "https://rupload.facebook.com/ig-api-upload/v21.0/cnt_ig_1"})
        if url.endswith("/media_publish"):
            return FakeResp({"id": "media_pub_99"})
        return FakeResp({})

    def fake_get(url, params=None, timeout=None):
        calls.append(("GET", url, params or {}))
        if "graph.instagram.com" in url and url.endswith("/me"):
            return FakeResp({"user_id": "178414001", "username": "cortesteste", "account_type": "CREATOR"})
        return FakeResp({"status_code": "FINISHED"})

    monkeypatch.setattr("src.publisher.instagram_reels.requests.post", fake_post)
    monkeypatch.setattr("src.publisher.instagram_reels.requests.get", fake_get)
    monkeypatch.setattr("scripts.setup_instagram.requests.get", fake_get)

    from scripts.setup_instagram import discover_instagram_account

    disc = discover_instagram_account("IGAA_test_token_123")
    assert disc["accounts"][0]["ig_account_id"] == "178414001"
    assert disc["accounts"][0]["ig_username"] == "cortesteste"

    dummy_mp4 = tmp_path / "ig.mp4"
    dummy_mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    ig_pub = InstagramReelsPublisher(prof, account_id="178414001", access_token="IGAA_test_token_123")
    assert ig_pub.is_ready is True
    assert ig_pub.is_instagram_login_token is True

    media_id = ig_pub.publish_reel(
        dummy_mp4,
        caption="legenda teste #cortes",
        video_url="https://drive.google.com/uc?export=download&confirm=t&id=file_123",
    )
    assert media_id == "media_pub_99"
    assert calls[1][1] == "https://graph.instagram.com/v21.0/178414001/media"
    assert calls[1][2]["video_url"].startswith("https://drive.google.com/")
    assert calls[3][1] == "https://graph.instagram.com/v21.0/178414001/media_publish"


def test_non_twitch_creator_name_does_not_leak_profile_alanzoka(tmp_path: Path, monkeypatch):
    from src.approval.discord_notifier import DiscordNotifier

    prof = _make_profile()
    prof.twitch_login = "alanzoka"
    prof.display_name = "Alanzoka"

    # Director com cluster Kick do Davy Jones nunca deve usar Alanzoka no fallback/tags
    director = GeminiDirector(prof, api_key="", groq_api_key="")
    cl = MomentCluster(
        cluster_id="c_davy",
        profile_id=prof.id,
        broadcaster_login="davyjones",
        stream_or_vod_key="kick_1",
        mode="kick_clip",
        clips=[TwitchClip("k1", "https://kick.com/davyjones", "davy clip", 200, "", 35.0, "viewer1", "kick", "Davy Jones")],
        video_id="k1",
        vod_start_offset=0,
        vod_end_offset=35,
        peak_offset=18,
        unique_clippers=1,
        total_views=200,
        max_views=200,
        score=95.0,
        dominant_title="",
        game_name="GTA RP",
    )
    dec = director.enforce_constraints(
        {"main_start": 0.0, "main_end": 35.0, "cold_open_enabled": False, "caption_text": "", "hashtags": []},
        words=[],
        raw_duration_sec=35.0,
        peak_hint_sec=15.0,
        creator_handle="davyjones",
        creator_name="Davy Jones",
    )
    assert "alanzoka" not in dec.caption_text.lower()
    assert "davy jones" in dec.caption_text.lower()
    assert "#davyjones" in dec.hashtags
    assert "#alanzoka" not in dec.hashtags

    # DiscordNotifier deve anunciar Davy Jones em vez de profile.display_name (Alanzoka)
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "m_davy"}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        captured["payload"] = json.loads(data["payload_json"])
        return FakeResp()

    monkeypatch.setattr("src.approval.discord_notifier.requests.post", fake_post)
    dummy_mp4 = tmp_path / "davy.mp4"
    dummy_mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    rec = CutRecord(
        cut_id="cut_davy_1",
        profile_id=prof.id,
        twitch_login="davyjones",
        stream_key="kick_1",
        mode="kick_clip",
        score=95.0,
        source_type="kick_clip",
        video_id="k1",
        vod_anchor_sec=None,
        clip_urls=[],
        caption="legenda",
        teaser_headline="",
        total_duration=35.0,
        layout_used="split_screen",
        drive_folder_id="f1",
        drive_folder_url="https://drive.google.com/f1",
        master_file_id="m1",
        master_file_url="https://drive.google.com/m1",
        clean_file_id="c1",
        srt_file_id="s1",
        created_at="2026-11-18T18:00:00Z",
        creator_name="Davy Jones",
    )
    DiscordNotifier(prof, bot_token="bot_x", channel_id="chan_x").send_cut_for_approval(rec, dummy_mp4)
    assert "Davy Jones" in captured["payload"]["content"]
    assert "Alanzoka" not in captured["payload"]["content"]


def test_youtube_shorts_publisher_metadata_and_discord_buttons(tmp_path: Path, monkeypatch):
    from zoneinfo import ZoneInfo

    from src.approval.discord_notifier import DiscordNotifier
    from src.publisher.youtube_shorts import YouTubeShortsPublisher, compute_youtube_publish_at

    prof = _make_profile()
    prof.publishing.youtube_shorts.enabled = True

    empty_pub = YouTubeShortsPublisher(prof, token_json_str="")
    assert empty_pub.is_ready is False

    valid_tok = json.dumps({"token": "ya29.test", "refresh_token": "1//ref", "client_id": "cid", "client_secret": "csec"})
    yt_pub = YouTubeShortsPublisher(prof, token_json_str=valid_tok)
    assert yt_pub.is_ready is True

    meta = yt_pub.build_video_metadata(
        caption="davy jones impressionado com golfinhos no gta 6\n\n#davyjones #gta6 #cortesdegames",
        title_hint="GOLFINHOS NO GTA 6",
        creator_name="Davy Jones",
    )
    assert len(meta["snippet"]["title"]) <= 58
    assert meta["snippet"]["title"].startswith("DAVY JONES IMPRESSIONADO COM")
    assert meta["snippet"]["title"].count("#") == 1
    assert "#Shorts" in meta["snippet"]["title"]
    assert "Palavras-chave:" in meta["snippet"]["description"]
    assert "#Shorts" in meta["snippet"]["description"]
    assert meta["snippet"]["categoryId"] == "20"
    assert len(meta["snippet"]["tags"]) >= 8
    assert "davyjones" in meta["snippet"]["tags"]
    assert "Shorts" in meta["snippet"]["tags"]
    assert meta["status"]["privacyStatus"] == "public"

    # Dica do vídeo: aguardar 25 min para processamento HD + evitar tarde (12h-18h BRT -> 19h+ BRT)
    afternoon_utc = datetime(2026, 11, 19, 16, 10, 0, tzinfo=timezone.utc)  # 13:10 BRT
    pub_at_afternoon = compute_youtube_publish_at(now_utc=afternoon_utc)
    dt_brt = datetime.fromisoformat(pub_at_afternoon.replace("Z", "+00:00")).astimezone(ZoneInfo("America/Sao_Paulo"))
    assert dt_brt.hour >= 19

    night_utc = datetime(2026, 11, 19, 23, 0, 0, tzinfo=timezone.utc)  # 20:00 BRT
    pub_at_night = compute_youtube_publish_at(now_utc=night_utc)
    dt_night = datetime.fromisoformat(pub_at_night.replace("Z", "+00:00"))
    assert (dt_night - night_utc).total_seconds() >= 25 * 60

    sched_meta = yt_pub.build_video_metadata(caption="clip #cortes", publish_at=pub_at_night)
    assert sched_meta["status"]["privacyStatus"] == "private"
    assert sched_meta["status"]["publishAt"] == pub_at_night

    long_meta = yt_pub.build_video_metadata(
        caption="essa frase propositalmente bem mais longa serve para testar o corte antes da truncagem mobile",
    )
    assert len(long_meta["snippet"]["title"]) <= 58
    assert "#Shorts" in long_meta["snippet"]["description"]

    # Mock resumable upload + thumbnail set via Google API
    uploaded = {}

    class FakeExec:
        def execute(self):
            return {"items": []}

    class FakeInsertReq:
        def next_chunk(self):
            return None, {"id": "ytShort123"}

    class FakeVideos:
        def insert(self, part=None, body=None, media_body=None):
            uploaded["body"] = body
            return FakeInsertReq()

    class FakeThumbnails:
        def set(self, videoId=None, media_body=None):
            uploaded["thumb_video_id"] = videoId
            return FakeExec()

    class FakeService:
        def videos(self):
            return FakeVideos()

        def thumbnails(self):
            return FakeThumbnails()

    thumb_dummy = tmp_path / "dummy.thumb.jpg"
    thumb_dummy.write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr("src.publisher.youtube_shorts.extract_striking_thumbnail", lambda vp, out: thumb_dummy)
    monkeypatch.setattr(yt_pub, "_build_youtube_service", lambda: FakeService())
    dummy_mp4 = tmp_path / "short.mp4"
    dummy_mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    url = yt_pub.publish_short(dummy_mp4, caption="clip brabo #cortes", creator_name="Alanzoka")
    assert url == "https://youtube.com/shorts/ytShort123"
    assert uploaded["body"]["snippet"]["categoryId"] == "20"
    assert uploaded["body"]["status"]["privacyStatus"] == "private"
    assert "publishAt" in uploaded["body"]["status"]
    assert uploaded["thumb_video_id"] == "ytShort123"

    # Discord Row 1 com Reels + TikTok + Shorts cabe perfeitamente nos 5 botões da API do Discord
    rows = DiscordNotifier._build_action_rows("cut_yt_1", drip_min=60, ig_enabled=True, tt_enabled=True, yt_enabled=True)
    assert len(rows[0]["components"]) == 5
    custom_ids = [b["custom_id"] for b in rows[0]["components"]]
    assert "approve_both:cut_yt_1" in custom_ids
    assert "approve_youtube:cut_yt_1" in custom_ids


def test_resilient_per_platform_no_duplicate_on_partial_failure(tmp_path: Path, monkeypatch):
    store = DriveStore(fallback_local_dir=tmp_path / "drive_resilient")
    prof = _make_profile()
    prof.publishing.instagram.enabled = True
    prof.publishing.facebook.enabled = False
    prof.publishing.tiktok.enabled = True
    prof.publishing.youtube_shorts.enabled = True

    master = tmp_path / "master.mp4"
    master.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    mid, _ = store.upload_file(master, ["cortes", "criador_01", "cut_part_1"], "master_1080p60.mp4")

    processed_state = {
        "cuts": [
            {
                "cut_id": "cut_part_1",
                "profile_id": prof.id,
                "caption": "legenda teste #cortes",
                "master_file_id": mid,
                "download_url": "https://example.com/v.mp4",
                "teaser_headline": "TESTE",
            }
        ]
    }
    now = datetime(2026, 11, 19, 15, 0, 0, tzinfo=timezone.utc)
    store.save_state_json(
        "drip_queue.json",
        {
            "last_published_by_profile": {},
            "items": [
                {
                    "cut_id": "cut_part_1",
                    "profile_id": prof.id,
                    "platforms": ["instagram", "youtube_shorts", "tiktok"],
                    "scheduled_for": "2026-11-19T14:55:00Z",
                    "status": "pending",
                    "created_at": "2026-11-19T14:00:00Z",
                }
            ],
        },
    )

    ig_calls = 0
    yt_calls = 0
    tt_calls = 0

    class FakeIG:
        is_ready = True
        is_instagram_login_token = False

        def __init__(self, *a, **kw):
            pass

        def publish_reel(self, *a, **kw):
            nonlocal ig_calls
            ig_calls += 1
            return "ig_media_111"

    class FakeYT:
        is_ready = True

        def __init__(self, *a, **kw):
            pass

        def publish_short(self, *a, **kw):
            nonlocal yt_calls
            yt_calls += 1
            return "https://youtube.com/shorts/yt111"

    class FakeTT:
        is_ready = True
        last_publish_mode = ""

        def __init__(self, *a, **kw):
            pass

        def publish_video(self, *a, **kw):
            nonlocal tt_calls
            tt_calls += 1
            if tt_calls == 1:
                raise RuntimeError("TikTok Inbox cheia (spam_risk_too_many_pending_share)")
            self.last_publish_mode = "inbox_fallback"
            return "tt_pub_111"

    monkeypatch.setattr("src.publisher.drip_scheduler.InstagramReelsPublisher", FakeIG)
    monkeypatch.setattr("src.publisher.drip_scheduler.YouTubeShortsPublisher", FakeYT)
    monkeypatch.setattr("src.publisher.drip_scheduler.TikTokPublisher", FakeTT)

    scheduler = DripScheduler(store)
    scheduler.process_due_queue({prof.id: prof}, processed_state, tmp_path / "work", now_utc=now)

    # IG and YT both succeeded in their own queues; TikTok failed in its own queue -> independent items!
    st = store.load_state_json("drip_queue.json", {})
    by_plat = {it["platform"]: it for it in st["items"]}
    assert ig_calls == 1
    assert yt_calls == 1
    assert tt_calls == 1
    assert by_plat["instagram"]["status"] == "published"
    assert by_plat["instagram"]["results"]["instagram_reel_id"] == "ig_media_111"
    assert by_plat["youtube"]["status"] == "published"
    assert by_plat["youtube"]["results"]["youtube_short_url"] == "https://youtube.com/shorts/yt111"
    assert by_plat["tiktok"]["status"] == "pending"
    assert by_plat["tiktok"]["platforms"] == ["tiktok"]

    # Retry run 2 hours later: IG and YT must NEVER be called again!
    later = now + timedelta(hours=2, minutes=5)
    reports = scheduler.process_due_queue({prof.id: prof}, processed_state, tmp_path / "work", now_utc=later)
    assert len(reports) == 1
    assert ig_calls == 1
    assert yt_calls == 1
    assert tt_calls == 2
    st2 = store.load_state_json("drip_queue.json", {})
    by_plat2 = {it["platform"]: it for it in st2["items"]}
    assert by_plat2["tiktok"]["status"] == "published"
    assert by_plat2["tiktok"]["results"]["tiktok_publish_id"] == "tt_pub_111"


def test_facebook_reels_publisher_flow(tmp_path: Path, monkeypatch):
    from src.publisher.facebook_reels import FacebookReelsPublisher

    prof = _make_profile()
    prof.publishing.facebook.enabled = True

    class FakeResp:
        ok = True
        status_code = 200
        text = "{}"

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    posts = []

    def fake_post(url, data=None, headers=None, files=None, timeout=None):
        posts.append((url, data))
        if url.endswith("/video_reels") and (data or {}).get("upload_phase") == "start":
            return FakeResp({"video_id": "fb_vid_77", "upload_url": "https://rupload.facebook.com/video-upload/v21.0/fb_vid_77"})
        if "rupload.facebook.com/video-upload" in url:
            assert headers.get("Content-Length") == str(len(b"\x00\x00\x00\x20ftypmp42"))
            return FakeResp({"success": True})
        return FakeResp({"success": True})

    monkeypatch.setattr("src.publisher.facebook_reels.requests.post", fake_post)
    fb = FacebookReelsPublisher(prof, page_id="1516904708162317", access_token="EAA_fb_page_tok")
    assert fb.is_ready is True
    mp4 = tmp_path / "fb.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    vid_id = fb.publish_reel(mp4, caption="clip brabo #cortes")
    assert vid_id == "fb_vid_77"
    assert len(posts) == 3


def test_process_due_queue_enforces_spacing_on_backlog(tmp_path: Path, monkeypatch):
    store = DriveStore(fallback_local_dir=tmp_path / "drive_spacing")
    prof = _make_profile()
    prof.publishing.drip_interval_minutes = 60
    prof.publishing.quiet_hours_brt = [3, 9]
    prof.publishing.instagram.enabled = True
    prof.publishing.facebook.enabled = False
    prof.publishing.tiktok.enabled = False
    prof.publishing.youtube_shorts.enabled = False

    master = tmp_path / "m_sp.mp4"
    master.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    mid1, _ = store.upload_file(master, ["cortes", "criador_01", "sp_1"], "master_1080p60.mp4")
    mid2, _ = store.upload_file(master, ["cortes", "criador_01", "sp_2"], "master_1080p60.mp4")

    processed_state = {
        "cuts": [
            {"cut_id": "sp_1", "profile_id": prof.id, "caption": "um", "master_file_id": mid1},
            {"cut_id": "sp_2", "profile_id": prof.id, "caption": "dois", "master_file_id": mid2},
        ]
    }
    now = datetime(2026, 11, 19, 15, 0, 0, tzinfo=timezone.utc)
    store.save_state_json(
        "drip_queue.json",
        {
            "last_published_by_profile": {},
            "items": [
                {"cut_id": "sp_1", "profile_id": prof.id, "platforms": ["instagram"], "scheduled_for": "2026-11-19T14:00:00Z", "status": "pending"},
                {"cut_id": "sp_2", "profile_id": prof.id, "platforms": ["instagram"], "scheduled_for": "2026-11-19T14:30:00Z", "status": "pending"},
            ],
        },
    )

    ig_posts = []

    class FakeIG:
        is_ready = True
        is_instagram_login_token = False

        def __init__(self, *a, **kw):
            pass

        def publish_reel(self, video_path, caption, **kw):
            ig_posts.append(caption)
            return f"ig_{len(ig_posts)}"

    monkeypatch.setattr("src.publisher.drip_scheduler.InstagramReelsPublisher", FakeIG)

    scheduler = DripScheduler(store)
    reports = scheduler.process_due_queue({prof.id: prof}, processed_state, tmp_path / "work", now_utc=now)
    assert len(reports) == 1
    assert ig_posts == ["um"]

    st = store.load_state_json("drip_queue.json", {})
    assert st["items"][0]["status"] == "published"
    assert st["items"][1]["status"] == "pending"
    assert st["items"][1]["scheduled_for"] == "2026-11-19T16:00:00Z"


def test_deduplicator_checks_context_caption_and_extracted_timestamp():
    processed_state = {
        "cuts": [
            {
                "cut_id": "cut_orig",
                "profile_id": "prof_1",
                "twitch_login": "gaules",
                "stream_key": "yt_abc123",
                "video_id": "yt_abc123",
                "mode": "youtube_auto",
                "vod_anchor_sec": 1200,
                "clip_ids": ["clip_10"],
                "clip_created_at": "2026-11-19T15:00:00Z",
                "dominant_title": "GAULES REAGINDO AO CLUTCH INSANO",
                "caption": "Simplesmente impossivel acreditar nesse clutch absurdo na Mirage #cs2 #gaules #cortes",
                "teaser_headline": "CLUTCH IMPOSSIVEL",
                "transcript_snippet": "meu deus olha o que ele fez na mirage quatro abates seguidos sem perder vida nenhuma absurdo",
                "director": {"main_start": 10.0, "main_end": 52.0},
            }
        ]
    }

    # (a) Janela extraída do vídeo sobreposta no mesmo VOD/vídeo (1230+5=1235 até 1270 vs 1210..1252)
    dup_window = find_duplicate_cut(
        {
            "cut_id": "cut_window",
            "profile_id": "prof_1",
            "twitch_login": "gaules",
            "stream_key": "yt_abc123",
            "video_id": "yt_abc123",
            "mode": "youtube_auto",
            "vod_anchor_sec": 1230,
            "caption": "Momento totalmente diferente aqui",
            "transcript_snippet": "outra conversa qualquer bem distinta",
            "director": {"main_start": 5.0, "main_end": 40.0},
        },
        processed_state,
    )
    assert dup_window.is_duplicate is True

    # (b) Descrição muito parecida (mudando só hashtags/pontuação) mesmo em vídeo/horário diferente
    dup_caption = find_duplicate_cut(
        {
            "cut_id": "cut_caption",
            "profile_id": "prof_1",
            "twitch_login": "gaules",
            "stream_key": "kick_other",
            "video_id": "kick_other",
            "vod_anchor_sec": 100,
            "caption": "Inacreditavel esse clutch absurdo na Mirage!! #reels #viral #fyp",
            "transcript_snippet": "texto diferente curto",
            "director": {"main_start": 0.0, "main_end": 35.0},
        },
        processed_state,
    )
    assert dup_caption.is_duplicate is True

    # (c) Contexto de áudio/transcrição igual no mesmo canal com cut_id/clip_id diferente
    dup_transcript = find_duplicate_cut(
        {
            "cut_id": "cut_audio",
            "profile_id": "prof_1",
            "twitch_login": "gaules",
            "stream_key": "twitch_live_999",
            "video_id": "",
            "clip_ids": ["clip_repost_77"],
            "clip_created_at": "2026-11-19T18:30:00Z",
            "caption": "Reacao épica na transmissão de hoje",
            "transcript_snippet": "olha o que ele fez na mirage quatro abates seguidos sem perder vida nenhuma cara",
            "director": {"main_start": 0.0, "main_end": 32.0},
        },
        processed_state,
    )
    assert dup_transcript.is_duplicate is True

    # (d) Segundo pico no MESMO vídeo do YouTube, fora da janela já cortada e com assunto distinto -> PERMITIDO!
    distinct_peak = find_duplicate_candidate(
        {
            "cut_id": "cut_second_peak",
            "profile_id": "prof_1",
            "twitch_login": "gaules",
            "stream_key": "yt_abc123",
            "video_id": "yt_abc123",
            "mode": "youtube_auto",
            "vod_anchor_sec": 2600,
            "total_duration": 42.0,
            "dominant_title": "GAULES REAGINDO AO CLUTCH INSANO",
        },
        processed_state,
    )
    assert distinct_peak.is_duplicate is False


def test_drip_scheduler_blocks_duplicate_cut_across_ids(tmp_path: Path):
    store = DriveStore(fallback_local_dir=tmp_path / "drive_dedup_q")
    prof = _make_profile()
    prof.publishing.quiet_hours_brt = [3, 9]
    prof.publishing.drip_interval_minutes = 75
    now = datetime(2026, 11, 19, 15, 0, 0, tzinfo=timezone.utc)
    # Garante agendamento na fila sem depender de tokens/chamadas externas de rede
    store.save_state_json(
        "drip_queue.json",
        {
            "last_published_by_profile": {prof.id: (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")},
            "items": [],
        },
    )
    scheduler = DripScheduler(store)

    # Primeiro corte entra na fila
    first_outcome = scheduler.enqueue_or_publish_cut(
        profile=prof,
        cut_dict={
            "cut_id": "cut_a",
            "profile_id": prof.id,
            "twitch_login": prof.twitch_login,
            "stream_key": "vod_900",
            "vod_anchor_sec": 600,
            "caption": "Ele dropou a faca rara ao vivo na frente de todo mundo #cs2 #cortes",
            "transcript_snippet": "nao acredito que ele tirou uma faca rara na caixa ao vivo mano",
            "director": {"main_start": 2.0, "main_end": 34.0},
        },
        platforms=["instagram"],
        work_dir=tmp_path / "w1",
        force_immediate=False,
        now_utc=now,
    )
    assert first_outcome["mode"] == "queued"

    # Simula corte duplicado com outro cut_id e descrição quase idêntica
    second_outcome = scheduler.enqueue_or_publish_cut(
        profile=prof,
        cut_dict={
            "cut_id": "cut_b_dup",
            "profile_id": prof.id,
            "twitch_login": prof.twitch_login,
            "stream_key": "clip_alt_901",
            "vod_anchor_sec": 50,
            "caption": "Dropou uma faca rara ao vivo na frente de todo mundo!! #reels #viral",
            "transcript_snippet": "totalmente diferente ou curto",
            "director": {"main_start": 0.0, "main_end": 30.0},
        },
        platforms=["instagram", "facebook"],
        work_dir=tmp_path / "w2",
        force_immediate=False,
        now_utc=now + timedelta(minutes=5),
    )
    assert second_outcome["mode"] == "skipped_duplicate"
    assert second_outcome["matched_cut_id"] == "cut_a"


def test_instagram_reels_seo_sends_cta_thumb_and_niche_hashtags(tmp_path: Path, monkeypatch):
    from src.publisher.instagram_reels import InstagramReelsPublisher

    prof = _make_profile()
    prof.publishing.instagram.enabled = True
    ig = InstagramReelsPublisher(prof, account_id="178414999", access_token="EAA_valid_meta_tok")

    meta = ig.build_reels_metadata(
        caption="ele descobriu o segredo escondido no mapa! O que vocês acharam? #fyp #viral #Shorts #cortes #gtav",
        title_hint="SEGREDO NO MAPA",
        creator_name="Davy Jones",
        game_name="Grand Theft Auto VI",
    )
    assert len(meta["hook_line"]) <= 98
    assert "O que vocês acharam" not in meta["hook_line"]
    assert 3 <= len(meta["hashtags"]) <= 5
    assert "#fyp" not in meta["hashtags"]
    assert "#viral" not in meta["hashtags"]
    assert "#shorts" not in meta["hashtags"]
    assert "#davyjones" in meta["hashtags"]
    assert any(k in meta["engagement_line"].lower() for k in ("manda ", "envia ", "salva "))
    assert "Davy Jones" in meta["engagement_line"]

    # Testa envio de container com thumb_offset, audio_name, share_to_feed, trial_params e fallback em 400
    calls = []

    class FakeResp:
        def __init__(self, payload, ok=True, status_code=200, text="{}"):
            self._payload = payload
            self.ok = ok
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if not self.ok:
                raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

        def json(self):
            return self._payload

    container_attempts = 0

    def fake_post(url, data=None, headers=None, timeout=None):
        nonlocal container_attempts
        calls.append((url, dict(data) if isinstance(data, dict) else data, dict(headers or {})))
        if url.endswith("/media"):
            container_attempts += 1
            if container_attempts == 1:
                assert (data or {}).get("thumb_offset") == "2200"
                assert (data or {}).get("audio_name") == "Som original - Davy Jones"
                assert (data or {}).get("share_to_feed") == "true"
                assert "SS_PERFORMANCE" in str((data or {}).get("trial_params") or "")
                # Simula conta sem Trial Reels -> fallback mantém thumb_offset no 2º init!
                return FakeResp({}, ok=False, status_code=400, text="trial_params unsupported")
            if container_attempts == 2:
                assert (data or {}).get("thumb_offset") == "2200"
                assert "trial_params" not in (data or {})
                return FakeResp({"id": "cont_ig_77", "uri": "https://rupload.facebook.com/ig-api-upload/v21.0/cont_ig_77"})
        if "rupload.facebook.com" in url:
            return FakeResp({"success": True})
        if url.endswith("/media_publish"):
            return FakeResp({"id": "reel_pub_77"})
        return FakeResp({"success": True})

    def fake_get(url, params=None, timeout=None):
        return FakeResp({"status_code": "FINISHED"})

    monkeypatch.setattr("src.publisher.instagram_reels.requests.post", fake_post)
    monkeypatch.setattr("src.publisher.instagram_reels.requests.get", fake_get)

    dummy_mp4 = tmp_path / "reel_opt.mp4"
    dummy_mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    media_id = ig.publish_reel(
        dummy_mp4,
        caption="ele reagindo ao bug insano #fyp #gta6 #cortes",
        creator_name="Davy Jones",
        game_name="GTA 6",
        thumb_offset_ms=2200,
        trial_mode="SS_PERFORMANCE",
    )
    assert media_id == "reel_pub_77"
    assert container_attempts == 2


@pytest.mark.skipif(shutil.which(get_ffmpeg_bin()) is None and not Path(get_ffmpeg_bin()).exists(), reason="FFmpeg indisponível")
def test_instagram_reels_compute_best_thumb_offset_ms(tmp_path: Path):
    from src.publisher.instagram_reels import compute_best_thumb_offset_ms

    mp4 = tmp_path / "sample_thumb.mp4"
    subprocess.run(
        [
            get_ffmpeg_bin(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x568:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    offset_ms = compute_best_thumb_offset_ms(mp4)
    assert offset_ms is not None
    assert 200 <= offset_ms <= 2600


def test_tiktok_inbox_full_diagnostic_and_extended_rate_limit_retries(tmp_path: Path, monkeypatch):
    from src.publisher.tiktok_poster import CHUNK_SIZE

    assert CHUNK_SIZE == 60 * 1024 * 1024

    class FakeResp:
        def __init__(self, status_code: int, body: dict):
            self.status_code = status_code
            self._body = body
            self.ok = 200 <= status_code < 300
            self.text = json.dumps(body)

        def json(self):
            return self._body

        def raise_for_status(self):
            if not self.ok:
                raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def fake_post(url, headers=None, json=None, data=None, timeout=30):
        if url.endswith("/post/publish/video/init/"):
            return FakeResp(403, {"error": {"code": "unaudited_client_can_only_post_to_private_accounts"}})
        if url.endswith("/post/publish/inbox/video/init/"):
            return FakeResp(403, {"error": {"code": "spam_risk_too_many_pending_share"}})
        if url.endswith("/post/publish/creator_info/query/"):
            return FakeResp(200, {"data": {"creator_username": "phiuphiufiu"}})
        return FakeResp(404, {})

    monkeypatch.setattr("src.publisher.tiktok_poster.requests.post", fake_post)

    prof = _make_profile()
    prof.publishing.tiktok = TikTokPublishConfig(
        enabled=True,
        mode="direct",
        privacy_level="SELF_ONLY",
        fallback_to_inbox=True,
    )
    video = tmp_path / "tt_test.mp4"
    video.write_bytes(b"\x00" * 2048)

    pub = TikTokPublisher(prof, access_token="tok_xyz")
    with pytest.raises(RuntimeError) as exc_info:
        pub.publish_video(video, "clip teste #cortes")
    assert "@phiuphiufiu" in str(exc_info.value)
    assert "spam_risk_too_many_pending_share" in str(exc_info.value)

    # DripScheduler deve manter retry em `pending` mesmo no 4º retry quando erro for limite de Inbox cheia
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok_xyz")
    monkeypatch.setenv("TIKTOK_REFRESH_TOKEN", "")
    store = DriveStore(fallback_local_dir=tmp_path / "drive_tt_retry")
    prof.publishing.instagram.enabled = False
    prof.publishing.facebook.enabled = False
    prof.publishing.youtube_shorts.enabled = False
    mid, _ = store.upload_file(video, ["cortes", "criador_01", "tt_retry_1"], "master_1080p60.mp4")
    processed_state = {"cuts": [{"cut_id": "tt_retry_1", "profile_id": prof.id, "caption": "teste", "master_file_id": mid}]}
    now = datetime(2026, 11, 19, 16, 40, 0, tzinfo=timezone.utc)
    store.save_state_json(
        "drip_queue.json",
        {
            "last_published_by_profile": {},
            "items": [
                {
                    "cut_id": "tt_retry_1",
                    "profile_id": prof.id,
                    "platforms": ["tiktok"],
                    "scheduled_for": "2026-11-19T16:30:00Z",
                    "status": "pending",
                    "retry_count": 3,
                }
            ],
        },
    )

    scheduler = DripScheduler(store)
    scheduler.process_due_queue({prof.id: prof}, processed_state, tmp_path / "work_tt", now_utc=now)
    st = store.load_state_json("drip_queue.json", {})
    it = st["items"][0]
    assert it["status"] == "pending"
    assert it["retry_count"] == 4
    assert "@phiuphiufiu" in it["last_error"]
    assert it["scheduled_for"] == "2026-11-19T18:40:00Z"


def test_cron_grace_window_and_tiktok_catchup_does_not_delay_feed_posts(tmp_path: Path, monkeypatch):
    store = DriveStore(fallback_local_dir=tmp_path / "drive_grace")
    prof = _make_profile()
    prof.publishing.drip_interval_minutes = 60
    prof.publishing.quiet_hours_brt = [3, 9]
    prof.publishing.instagram.enabled = True
    prof.publishing.facebook.enabled = False
    prof.publishing.tiktok.enabled = True
    prof.publishing.youtube_shorts.enabled = False

    mp4 = tmp_path / "g.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    m1, _ = store.upload_file(mp4, ["cortes", "criador_01", "g1"], "master_1080p60.mp4")
    m2, _ = store.upload_file(mp4, ["cortes", "criador_01", "g2"], "master_1080p60.mp4")

    processed_state = {
        "cuts": [
            {"cut_id": "g_catchup", "profile_id": prof.id, "caption": "um", "master_file_id": m1},
            {"cut_id": "g_fresh", "profile_id": prof.id, "caption": "dois", "master_file_id": m2},
        ]
    }
    now = datetime(2026, 11, 19, 18, 0, 35, tzinfo=timezone.utc)
    store.save_state_json(
        "drip_queue.json",
        {
            "last_published_by_profile": {prof.id: "2026-11-19T16:30:00Z"},
            "items": [
                {
                    "cut_id": "g_catchup",
                    "profile_id": prof.id,
                    "platforms": ["tiktok"],
                    "scheduled_for": "2026-11-19T17:58:00Z",
                    "status": "pending",
                    "results": {"instagram_reel_id": "ig_old_1"},
                    "retry_count": 1,
                },
                {
                    "cut_id": "g_fresh",
                    "profile_id": prof.id,
                    "platforms": ["instagram"],
                    "scheduled_for": "2026-11-19T18:01:10Z",  # 35s no futuro -> dentro da tolerância cron 120s!
                    "status": "pending",
                },
            ],
        },
    )

    ig_calls = []
    tt_calls = []

    class FakeIG:
        is_ready = True

        def __init__(self, *a, **kw):
            pass

        def publish_reel(self, *a, **kw):
            ig_calls.append(1)
            return "ig_new_2"

    class FakeTT:
        is_ready = True
        last_publish_mode = "inbox_fallback"

        def __init__(self, *a, **kw):
            pass

        def publish_video(self, *a, **kw):
            tt_calls.append(1)
            return "tt_catchup_1"

    monkeypatch.setattr("src.publisher.drip_scheduler.InstagramReelsPublisher", FakeIG)
    monkeypatch.setattr("src.publisher.drip_scheduler.TikTokPublisher", FakeTT)

    scheduler = DripScheduler(store)
    reports = scheduler.process_due_queue({prof.id: prof}, processed_state, tmp_path / "work_g", now_utc=now)
    assert len(reports) == 2
    assert len(tt_calls) == 1
    assert len(ig_calls) == 1


def test_content_moderator_batches_max_3_images_and_blocks_nudity(tmp_path: Path, monkeypatch):
    from src.ai import content_moderator as cm

    dummy = tmp_path / "vid.mp4"
    dummy.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    monkeypatch.setattr(cm, "extract_sample_frames_b64", lambda vp, max_frames=6: [f"img{i}" for i in range(6)])

    batch_sizes = []

    class FakeResp:
        def __init__(self, payload: dict):
            self.status_code = 200
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    call_idx = 0

    def fake_post(url, headers=None, json=None, timeout=None):
        nonlocal call_idx
        call_idx += 1
        msgs = json["messages"][0]["content"]
        imgs = [m for m in msgs if m.get("type") == "image_url"]
        batch_sizes.append(len(imgs))
        if call_idx == 1:
            return FakeResp({"choices": [{"message": {"content": '{"exposure_level": "clean", "instagram_safe": true, "youtube_shorts_safe": true, "tiktok_safe": true, "reason": "ok"}'}}]})
        return FakeResp({"choices": [{"message": {"content": '{"exposure_level": "explicit_indecent", "instagram_safe": false, "youtube_shorts_safe": false, "tiktok_safe": false, "reason": "Nudez detectada no frame 4"}'}}]})

    monkeypatch.setattr("src.ai.content_moderator.requests.post", fake_post)
    dec = cm.moderate_video_for_platforms(
        dummy,
        requested_platforms=["instagram", "facebook", "youtube_shorts", "tiktok"],
        title="gameplay normal",
        transcript_text="olha isso mano",
        groq_api_key="gsk_test",
    )
    assert batch_sizes == [3, 3]  # Nunca ultrapassa limite de 3 imagens da Groq Vision!
    assert dec.vision_checked is True
    assert dec.exposure_level == "explicit_indecent"
    assert dec.allowed_platforms == []
    assert set(dec.blocked_platforms) == {"instagram", "facebook", "youtube_shorts", "tiktok"}


def test_drip_scheduler_publish_time_dedup_and_nudity_gate(tmp_path: Path, monkeypatch):
    from src.ai.content_moderator import PlatformModerationDecision

    store = DriveStore(fallback_local_dir=tmp_path / "drive_pub_gate")
    prof = _make_profile()
    prof.publishing.instagram.enabled = True
    prof.publishing.facebook.enabled = False
    prof.publishing.tiktok.enabled = False
    prof.publishing.youtube_shorts.enabled = False

    mp4 = tmp_path / "pub_gate.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    m_dup, _ = store.upload_file(mp4, ["cortes", "criador_01", "dup_old"], "master_1080p60.mp4")
    m_nude, _ = store.upload_file(mp4, ["cortes", "criador_01", "nude_old"], "master_1080p60.mp4")

    processed_state = {
        "cuts": [
            {
                "cut_id": "cut_pub_already",
                "profile_id": prof.id,
                "twitch_login": prof.twitch_login,
                "caption": "ele dropou a faca mais rara do jogo no meio da live",
                "status": "published",
            },
            {
                "cut_id": "cut_pending_dup",
                "profile_id": prof.id,
                "twitch_login": prof.twitch_login,
                "caption": "dropou uma faca mais rara do jogo no meio da live #cortes",
                "master_file_id": m_dup,
            },
            {
                "cut_id": "cut_pending_nude",
                "profile_id": prof.id,
                "twitch_login": prof.twitch_login,
                "caption": "clipe normal sem palavras ruins no texto",
                "master_file_id": m_nude,
            },
        ]
    }
    now = datetime(2026, 11, 19, 19, 0, 0, tzinfo=timezone.utc)
    store.save_state_json(
        "drip_queue.json",
        {
            "last_published_by_profile": {},
            "items": [
                {
                    "cut_id": "cut_pub_already",
                    "profile_id": prof.id,
                    "platforms": ["instagram"],
                    "scheduled_for": "2026-11-19T12:00:00Z",
                    "published_at": "2026-11-19T12:00:00Z",
                    "status": "published",
                    "caption": "ele dropou a faca mais rara do jogo no meio da live",
                },
                {
                    "cut_id": "cut_pending_dup",
                    "profile_id": prof.id,
                    "platforms": ["instagram"],
                    "scheduled_for": "2026-11-19T18:00:00Z",
                    "status": "pending",
                },
                {
                    "cut_id": "cut_pending_nude",
                    "profile_id": prof.id,
                    "platforms": ["instagram"],
                    "scheduled_for": "2026-11-19T18:30:00Z",
                    "status": "pending",
                },
            ],
        },
    )

    ig_called = 0

    class FakeIG:
        is_ready = True

        def __init__(self, *a, **kw):
            pass

        def publish_reel(self, *a, **kw):
            nonlocal ig_called
            ig_called += 1
            return "ig_should_not_post"

    monkeypatch.setattr("src.publisher.drip_scheduler.InstagramReelsPublisher", FakeIG)
    monkeypatch.setattr(
        "src.publisher.drip_scheduler.moderate_video_for_platforms",
        lambda **kw: PlatformModerationDecision(
            exposure_level="explicit_indecent",
            instagram_safe=False,
            youtube_shorts_safe=False,
            tiktok_safe=False,
            allowed_platforms=[],
            blocked_platforms=["instagram"],
            reason="Nudez visual detectada",
            vision_checked=True,
        ),
    )

    scheduler = DripScheduler(store)
    scheduler.process_due_queue({prof.id: prof}, processed_state, tmp_path / "work_gate", now_utc=now)
    assert ig_called == 0
    st = store.load_state_json("drip_queue.json", {})
    by_id = {it["cut_id"]: it for it in st["items"]}
    assert by_id["cut_pending_dup"]["status"] == "skipped_duplicate"
    assert by_id["cut_pending_nude"]["status"] == "blocked_nudity"


def test_stagger_network_queues_offsets_between_networks(tmp_path: Path):
    store = DriveStore(fallback_local_dir=tmp_path / "drive_stagger")
    prof = _make_profile()
    prof.publishing.drip_interval_minutes = 60
    prof.publishing.quiet_hours_brt = None
    now = datetime(2026, 11, 20, 18, 0, 0, tzinfo=timezone.utc)

    def _it(cid, plat, when):
        return {"id": f"{cid}::{plat}", "cut_id": cid, "profile_id": prof.id, "platform": plat,
                "platforms": [plat], "scheduled_for": when, "status": "pending", "created_at": when}

    items = []
    for plat in ("tiktok", "instagram", "youtube"):
        items += [_it("a", plat, "2026-11-20T18:00:00Z"), _it("b", plat, "2026-11-20T19:00:00Z")]
    store.save_state_json("drip_queue.json", {"last_published_by_profile": {}, "items": items})

    scheduler = DripScheduler(store)
    rep = scheduler.stagger_network_queues(
        {prof.id: prof}, {"tiktok": 0, "instagram": 30, "youtube": 60}, now_utc=now
    )
    assert rep["rescheduled_count"] > 0
    st = store.load_state_json("drip_queue.json", {})
    slot = {i["id"]: i["scheduled_for"] for i in st["items"]}
    assert slot["a::tiktok"] == "2026-11-20T18:00:00Z"
    assert slot["a::instagram"] == "2026-11-20T18:30:00Z"
    assert slot["a::youtube"] == "2026-11-20T19:00:00Z"
    # Cada fila mantém seu próprio intervalo de 60 min
    assert slot["b::instagram"] == "2026-11-20T19:30:00Z"


def test_separate_queues_per_social_network_independence(tmp_path: Path, monkeypatch):
    store = DriveStore(fallback_local_dir=tmp_path / "drive_sep_queues")
    prof = _make_profile()
    prof.publishing.drip_interval_minutes = 60
    prof.publishing.quiet_hours_brt = [3, 9]
    prof.publishing.instagram.enabled = True
    prof.publishing.facebook.enabled = False
    prof.publishing.tiktok.enabled = True
    prof.publishing.youtube_shorts.enabled = True

    mp4 = tmp_path / "sep.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    m1, _ = store.upload_file(mp4, ["cortes", "criador_01", "sep_1"], "master_1080p60.mp4")
    m2, _ = store.upload_file(mp4, ["cortes", "criador_01", "sep_2"], "master_1080p60.mp4")

    processed_state = {
        "cuts": [
            {"cut_id": "sep_1", "profile_id": prof.id, "caption": "primeiro momento epico #cs2", "master_file_id": m1},
            {"cut_id": "sep_2", "profile_id": prof.id, "caption": "segundo momento engracado #lol", "master_file_id": m2},
        ]
    }

    now = datetime(2026, 11, 19, 15, 0, 0, tzinfo=timezone.utc)
    # Instagram já publicou há 10 min; TikTok e YouTube não publicaram recentemente
    store.save_state_json(
        "drip_queue.json",
        {
            "last_published_by_profile": {},
            "last_published_by_platform": {
                prof.id: {
                    "instagram": (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
            "items": [],
        },
    )

    tt_calls = []
    ig_calls = []

    class FakeIG:
        is_ready = True
        is_instagram_login_token = False

        def __init__(self, *a, **kw):
            pass

        def publish_reel(self, *a, **kw):
            ig_calls.append(1)
            return f"ig_{len(ig_calls)}"

    class FakeTT:
        is_ready = True
        last_publish_mode = "direct"

        def __init__(self, *a, **kw):
            pass

        def publish_video(self, *a, **kw):
            tt_calls.append(1)
            return f"tt_{len(tt_calls)}"

    monkeypatch.setattr("src.publisher.drip_scheduler.InstagramReelsPublisher", FakeIG)
    monkeypatch.setattr("src.publisher.drip_scheduler.TikTokPublisher", FakeTT)

    scheduler = DripScheduler(store)

    # 1) Enfileira sep_1 para Instagram e TikTok: TikTok sai IMEDIATAMENTE, Instagram entra na fila (+50m)
    out1 = scheduler.enqueue_or_publish_cut(
        profile=prof,
        cut_dict={**processed_state["cuts"][0], "moderation": {"vision_checked": True, "allowed_platforms": ["instagram", "tiktok"], "blocked_platforms": []}},
        platforms=["instagram", "tiktok"],
        work_dir=tmp_path / "w_sep1",
        now_utc=now,
    )
    assert out1["mode"] == "mixed"
    assert out1["by_platform"]["tiktok"]["mode"] == "published_now"
    assert out1["by_platform"]["instagram"]["mode"] == "queued"
    assert out1["by_platform"]["instagram"]["scheduled_for"] == "2026-11-19T15:50:00Z"
    assert len(tt_calls) == 1
    assert len(ig_calls) == 0

    # 2) Enfileirar YouTube depois para o MESMO cut_id (sep_1) funciona sem dar already_queued!
    out_yt = scheduler.enqueue_or_publish_cut(
        profile=prof,
        cut_dict={**processed_state["cuts"][0], "moderation": {"vision_checked": True, "allowed_platforms": ["instagram", "tiktok", "youtube_shorts"], "blocked_platforms": []}},
        platforms=["youtube_shorts"],
        work_dir=tmp_path / "w_sep1_yt",
        now_utc=now + timedelta(minutes=1),
    )
    # YouTubeShortsPublisher não configurado -> skipped_unconfigured, marcado como published na fila do YouTube
    assert out_yt["mode"] == "published_now"

    # 3) Enfileira sep_2 para Instagram e TikTok -> cada fila calcula seu próprio slot independente
    out2 = scheduler.enqueue_or_publish_cut(
        profile=prof,
        cut_dict={**processed_state["cuts"][1], "moderation": {"vision_checked": True, "allowed_platforms": ["instagram", "tiktok"], "blocked_platforms": []}},
        platforms=["instagram", "tiktok"],
        work_dir=tmp_path / "w_sep2",
        now_utc=now + timedelta(minutes=5),
    )
    assert out2["mode"] == "queued"
    assert out2["by_platform"]["instagram"]["scheduled_for"] == "2026-11-19T16:50:00Z"
    assert out2["by_platform"]["tiktok"]["scheduled_for"] == "2026-11-19T16:00:00Z"

    # 4) Filtro por plataforma em process_due_queue (ex: só TikTok às 16:05 -> publica sep_2 no TikTok sem tocar no Instagram!)
    due_tt = scheduler.process_due_queue(
        {prof.id: prof},
        processed_state,
        tmp_path / "w_due_tt",
        now_utc=now + timedelta(minutes=65),
        platform_filter="tiktok",
    )
    assert len(due_tt) == 1
    assert due_tt[0]["platform"] == "tiktok"
    assert len(tt_calls) == 2
    assert len(ig_calls) == 0

    # 5) Agora processa fila do Instagram (itens de 15:50 e 16:50 vencidos às 16:05 -> só 15:50 sai, 16:50 espaça!)
    due_ig = scheduler.process_due_queue(
        {prof.id: prof},
        processed_state,
        tmp_path / "w_due_ig",
        now_utc=now + timedelta(minutes=65),
        platform_filter="instagram",
    )
    assert len(due_ig) == 1
    assert due_ig[0]["platform"] == "instagram"
    assert len(ig_calls) == 1


def test_advancing_post_pulls_subsequent_queue_slots_to_interval(tmp_path: Path, monkeypatch):
    store = DriveStore(fallback_local_dir=tmp_path / "drive_advance_reflow")
    prof = _make_profile()
    prof.publishing.drip_interval_minutes = 60
    prof.publishing.quiet_hours_brt = [3, 9]
    prof.publishing.instagram.enabled = True
    prof.publishing.facebook.enabled = False
    prof.publishing.tiktok.enabled = True
    prof.publishing.youtube_shorts.enabled = False

    mp4 = tmp_path / "adv.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x20ftypmp42")
    m1, _ = store.upload_file(mp4, ["cortes", "criador_01", "adv_1"], "master_1080p60.mp4")
    m2, _ = store.upload_file(mp4, ["cortes", "criador_01", "adv_2"], "master_1080p60.mp4")
    m3, _ = store.upload_file(mp4, ["cortes", "criador_01", "adv_3"], "master_1080p60.mp4")

    processed_state = {
        "cuts": [
            {"cut_id": "adv_1", "profile_id": prof.id, "caption": "video um", "master_file_id": m1},
            {"cut_id": "adv_2", "profile_id": prof.id, "caption": "video dois", "master_file_id": m2},
            {"cut_id": "adv_3", "profile_id": prof.id, "caption": "video tres", "master_file_id": m3},
        ]
    }

    # Agora = 19:37 BRT (22:37 UTC). Video 1 faltava 23 min (23:00 UTC), Video 2 às 00:00 UTC, Video 3 às 01:00 UTC.
    now = datetime(2026, 11, 19, 22, 37, 0, tzinfo=timezone.utc)
    store.save_state_json(
        "drip_queue.json",
        {
            "last_published_by_profile": {},
            "last_published_by_platform": {},
            "items": [
                {"id": "adv_1::instagram", "cut_id": "adv_1", "profile_id": prof.id, "platform": "instagram", "platforms": ["instagram"], "scheduled_for": "2026-11-19T23:00:00Z", "status": "pending", "created_at": "2026-11-19T22:00:00Z"},
                {"id": "adv_2::instagram", "cut_id": "adv_2", "profile_id": prof.id, "platform": "instagram", "platforms": ["instagram"], "scheduled_for": "2026-11-20T00:00:00Z", "status": "pending", "created_at": "2026-11-19T22:01:00Z"},
                {"id": "adv_3::instagram", "cut_id": "adv_3", "profile_id": prof.id, "platform": "instagram", "platforms": ["instagram"], "scheduled_for": "2026-11-20T01:00:00Z", "status": "pending", "created_at": "2026-11-19T22:02:00Z"},
                {"id": "adv_2::tiktok", "cut_id": "adv_2", "profile_id": prof.id, "platform": "tiktok", "platforms": ["tiktok"], "scheduled_for": "2026-11-20T00:10:00Z", "status": "pending", "created_at": "2026-11-19T22:01:00Z"},
            ],
        },
    )

    class FakeIG:
        is_ready = True
        is_instagram_login_token = False

        def __init__(self, *a, **kw):
            pass

        def publish_reel(self, *a, **kw):
            return "ig_adv_1"

    monkeypatch.setattr("src.publisher.drip_scheduler.InstagramReelsPublisher", FakeIG)

    scheduler = DripScheduler(store)
    reports = scheduler.process_due_queue(
        {prof.id: prof},
        processed_state,
        tmp_path / "work_adv",
        now_utc=now,
        force_next=True,
        platform_filter="instagram",
    )
    assert len(reports) == 1
    assert reports[0]["cut"]["cut_id"] == "adv_1"

    st = store.load_state_json("drip_queue.json", {})
    by_id = {it["id"]: it for it in st["items"]}
    assert by_id["adv_1::instagram"]["status"] == "published"
    # Video 2 e Video 3 devem ser puxados para 22:37 + 60m = 23:37 UTC e 00:37 UTC!
    assert by_id["adv_2::instagram"]["scheduled_for"] == "2026-11-19T23:37:00Z"
    assert by_id["adv_3::instagram"]["scheduled_for"] == "2026-11-20T00:37:00Z"
    # Fila do TikTok permanece intocada!
    assert by_id["adv_2::tiktok"]["scheduled_for"] == "2026-11-20T00:10:00Z"

    # Também testa force_immediate=True (Postar Agora no adv_2 em 23:00 UTC -> adv_3 vai para 00:00 UTC)
    later = now + timedelta(minutes=23)
    out = scheduler.enqueue_or_publish_cut(
        profile=prof,
        cut_dict=processed_state["cuts"][1],
        platforms=["instagram"],
        work_dir=tmp_path / "work_adv2",
        force_immediate=True,
        now_utc=later,
    )
    assert out["mode"] == "published_now"
    st2 = store.load_state_json("drip_queue.json", {})
    by_id2 = {it["id"]: it for it in st2["items"]}
    assert by_id2["adv_3::instagram"]["scheduled_for"] == "2026-11-20T00:00:00Z"


