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
    monkeypatch.setattr(
        analyzer,
        "fetch_channel_clips",
        lambda slug, time_window="day", sort="view": [
            {"id": "k_dump", "title": "180s dump", "views": 220, "duration": 180, "created_at": "2026-11-18T19:00:00Z", "clip_url": "https://clips.kick.com/dump.m3u8"},
            {"id": "k_tight", "title": "48s bom", "views": 220, "duration": 48, "created_at": "2026-11-18T18:00:00Z", "clip_url": "https://clips.kick.com/tight.m3u8"},
        ] if time_window == "day" else [],
    )
    ranked_both = analyzer.discover_channel_moments("https://kick.com/davyjones", max_cuts=2, processed_state={"cuts": []})
    assert [c.best_clip.id for c in ranked_both] == ["k_tight", "k_dump"]

    ranked = analyzer.discover_channel_moments(
        "https://kick.com/davyjones",
        max_cuts=2,
        processed_state={"cuts": [{"twitch_login": "davyjones", "clip_created_at": "2026-11-18T19:01:00Z"}]},
    )
    assert len(ranked) == 1
    assert ranked[0].best_clip.id == "k_tight"

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

    tags = th.suggest_dynamic_hashtags(
        creator_handle="davyjones",
        platform_mode="youtube_auto",
        game_name="Counter-Strike 2",
        title_text="Kscerato deu aula nesse clutch absurdo",
        transcript_snippet="olha o kscerato jogando",
        max_tags=5,
        base_discovery_tags=["#fy", "#fyp", "#viral"],
    )
    assert tags[0] == "#davyjones"
    assert "#cs2" in tags
    assert "#kscerato" in tags
    assert "#fy" in tags
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
    assert "DESCRIÇÃO COMPACTA, DESCRITIVA E COM GANCHO" in prompt

    # Garante que chamadas robóticas de engajamento sejam limpas sem amputar frases narrativas reais
    dec = director.enforce_constraints(
        {
            "main_start": 0.0,
            "main_end": 35.0,
            "cold_open_enabled": False,
            "teaser_start": 0.0,
            "teaser_end": 0.0,
            "caption_text": "Davy Jones tenta dar uma de encanador no meio do serviço e acaba alagando tudo! O que vocês acharam?",
            "hashtags": ["#davyjones", "#cs2", "#fy", "#viral"],
            "virality_score": 90,
        },
        words=[WordTimestamp("oi", 1.0, 1.5)],
        raw_duration_sec=35.0,
        peak_hint_sec=15.0,
    )
    assert "O que vocês acharam" not in dec.caption_text
    assert "alagando tudo!" in dec.caption_text


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
