from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetectionConfig:
    cluster_window_seconds: int = 75
    min_clips_live_cluster: int = 3
    min_views_single_clip_live: int = 80
    min_views_post_live: int = 120
    max_cuts_per_stream: int = 15
    max_cuts_per_run: int = 4
    dedup_overlap_seconds: int = 50
    vod_padding_before_seconds: int = 45
    vod_padding_after_seconds: int = 45
    recap_hours: int = 36
    min_virality_score: int = 75
    allow_blind_kick_fallback: bool = True
    target_cuts_per_live: int = 3


@dataclass
class EditingConfig:
    max_duration_seconds: float = 58.0
    min_duration_seconds: float = 16.0
    cold_open_enabled: bool = True
    cold_open_min_main_duration: float = 22.0
    cold_open_min_seconds: float = 2.0
    cold_open_max_seconds: float = 3.2
    layout_mode: str = "auto"  # auto | split_screen | face_track | blur_center
    fixed_webcam_box: Optional[List[float]] = None
    watermark_text: str = ""


@dataclass
class CopyConfig:
    style: str = "minimal_meme"
    max_hashtags: int = 4
    base_hashtags: List[str] = field(default_factory=lambda: ["#cortes", "#live", "#twitchbr"])


@dataclass
class DiscordProfileConfig:
    bot_token_env: str = "DISCORD_BOT_TOKEN"
    channel_id_env: str = "DISCORD_CHANNEL_ID"
    channel_id: str = ""
    relay_tenant: str = ""


@dataclass
class InstagramPublishConfig:
    enabled: bool = True
    account_id_env: str = "IG_ACCOUNT_ID"
    access_token_env: str = "IG_ACCESS_TOKEN"
    share_to_feed: bool = True
    trial_mode: str = ""  # "" | "SS_PERFORMANCE" | "MANUAL"
    drip_interval_minutes: Optional[int] = None


@dataclass
class TikTokPublishConfig:
    enabled: bool = True
    mode: str = "inbox"  # inbox | direct
    privacy_level: str = "PUBLIC_TO_EVERYONE"  # SELF_ONLY | PUBLIC_TO_EVERYONE | MUTUAL_FOLLOW_FRIENDS | FOLLOWER_OF_CREATOR
    fallback_to_inbox: bool = True
    access_token_env: str = "TIKTOK_ACCESS_TOKEN"
    drip_interval_minutes: Optional[int] = None


@dataclass
class YouTubeShortsPublishConfig:
    enabled: bool = False
    privacy_status: str = "public"  # public | unlisted | private
    category_id: str = "20"  # 20 = Gaming
    token_json_env: str = "YOUTUBE_TOKEN_JSON"
    drip_interval_minutes: Optional[int] = None


@dataclass
class FacebookPublishConfig:
    enabled: bool = True
    page_id_env: str = "FB_PAGE_ID"
    access_token_env: str = "FB_ACCESS_TOKEN"
    drip_interval_minutes: Optional[int] = None


@dataclass
class PublishingConfig:
    drip_interval_minutes: int = 75
    quiet_hours_brt: Optional[List[int]] = field(default_factory=lambda: [3, 9])
    timezone: str = "America/Sao_Paulo"
    drive_retention_days: int = 10
    instagram: InstagramPublishConfig = field(default_factory=InstagramPublishConfig)
    facebook: FacebookPublishConfig = field(default_factory=FacebookPublishConfig)
    tiktok: TikTokPublishConfig = field(default_factory=TikTokPublishConfig)
    youtube_shorts: YouTubeShortsPublishConfig = field(default_factory=YouTubeShortsPublishConfig)


@dataclass
class ProfileConfig:
    id: str
    twitch_login: str
    display_name: str = ""
    enabled: bool = True
    language: str = "pt"
    youtube_channels: List[str] = field(default_factory=list)
    kick_channels: List[str] = field(default_factory=list)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    editing: EditingConfig = field(default_factory=EditingConfig)
    copy: CopyConfig = field(default_factory=CopyConfig)
    discord: DiscordProfileConfig = field(default_factory=DiscordProfileConfig)
    publishing: PublishingConfig = field(default_factory=PublishingConfig)


@dataclass
class TwitchClip:
    id: str
    url: str
    title: str
    view_count: int
    created_at: str
    duration: float
    creator_name: str
    broadcaster_id: str
    broadcaster_name: str
    game_id: str = ""
    video_id: str = ""
    vod_offset: Optional[int] = None


@dataclass
class MomentCluster:
    cluster_id: str
    profile_id: str
    broadcaster_login: str
    stream_or_vod_key: str
    mode: str  # "live_hot" | "post_live"
    clips: List[TwitchClip]
    video_id: str
    vod_start_offset: Optional[int]
    vod_end_offset: Optional[int]
    peak_offset: Optional[int]
    unique_clippers: int
    total_views: int
    max_views: int
    score: float
    dominant_title: str
    game_name: str = ""
    stream_title: str = ""

    @property
    def best_clip(self) -> TwitchClip:
        return max(self.clips, key=lambda c: (c.view_count, c.duration))


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class DirectorDecision:
    main_start: float
    main_end: float
    cold_open_enabled: bool
    teaser_start: float
    teaser_end: float
    teaser_headline: str
    caption_text: str
    hashtags: List[str]
    virality_score: int
    reasoning: str = ""

    @property
    def teaser_duration(self) -> float:
        if not self.cold_open_enabled:
            return 0.0
        return max(0.0, self.teaser_end - self.teaser_start)

    @property
    def main_duration(self) -> float:
        return max(0.0, self.main_end - self.main_start)

    @property
    def total_duration(self) -> float:
        return self.teaser_duration + self.main_duration

    @property
    def full_social_copy(self) -> str:
        tags = " ".join(
            t if t.startswith("#") else f"#{t}"
            for t in self.hashtags
            if t.strip()
        )
        if tags:
            return f"{self.caption_text.strip()}\n\n{tags}".strip()
        return self.caption_text.strip()


@dataclass
class CutRecord:
    cut_id: str
    profile_id: str
    twitch_login: str
    stream_key: str
    mode: str
    score: float
    source_type: str  # "vod" | "clip_fallback"
    video_id: str
    vod_anchor_sec: Optional[int]
    clip_urls: List[str]
    caption: str
    teaser_headline: str
    total_duration: float
    layout_used: str
    drive_folder_id: str
    drive_folder_url: str
    master_file_id: str
    master_file_url: str
    clean_file_id: str
    srt_file_id: str
    created_at: str
    creator_name: str = ""
    download_url: str = ""
    status: str = "pending_approval"  # pending_approval | queued | published | rejected
    audio_cleaned: bool = False
    discord_message_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
