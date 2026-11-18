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
