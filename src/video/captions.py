from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.models import DirectorDecision, WordTimestamp

HYPE_WORDS = {
    "NÃO",
    "NAO",
    "MANO",
    "NOSSA",
    "CARACA",
    "OLHA",
    "DEUS",
    "CALMA",
    "VERDADE",
    "MENTIRA",
    "BUGOU",
    "MORREU",
    "CORRE",
    "SOCORRO",
    "PUTA",
    "CARALHO",
    "PORRA",
    "NUNCA",
    "SEMPRE",
    "AGORA",
    "LINDO",
    "ABSURDO",
}


def hex_to_ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """
    Converte `#RRGGBB` para o padrão SubStation Alpha `&HAABBGGRR`
    (`alpha=0` é totalmente opaco, `alpha=255` é invisível).
    """
    clean = hex_rgb.strip().lstrip("#")
    if len(clean) != 6:
        clean = "FFFFFF"
    r = int(clean[0:2], 16)
    g = int(clean[2:4], 16)
    b = int(clean[4:6], 16)
    a = max(0, min(255, int(alpha)))
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


def format_ass_time(seconds: float) -> str:
    sec = max(0.0, seconds)
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    rem = sec % 60
    return f"{hours}:{minutes:02d}:{rem:05.2f}"


def format_srt_time(seconds: float) -> str:
    sec = max(0.0, seconds)
    total_ms = int(round(sec * 1000))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    ms = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def escape_ass_text(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def remap_words_to_final_timeline(
    words: List[WordTimestamp],
    decision: DirectorDecision,
) -> List[WordTimestamp]:
    """
    Mapeia as palavras faladas na janela bruta para a linha do tempo final do corte:
      - Quando o Cold Open tem `teaser_headline`, deixamos os primeiros 2-3s limpos APENAS com a Headline
        de impacto (sem empilhar duas legendas ao mesmo tempo!) e iniciamos o karaokê dinâmico exatamente
        após o flash do Cold Open (`offset = decision.teaser_duration`).
    """
    timeline_words: List[WordTimestamp] = []
    offset = 0.0

    if decision.cold_open_enabled and decision.teaser_duration > 0.2:
        if not decision.teaser_headline.strip():
            t0, t1 = decision.teaser_start, decision.teaser_end
            for w in words:
                if w.end > (t0 + 0.04) and w.start < (t1 - 0.04):
                    ns = max(0.0, w.start - t0)
                    ne = min(decision.teaser_duration, w.end - t0)
                    if ne - ns >= 0.05:
                        timeline_words.append(WordTimestamp(word=w.word, start=round(ns, 3), end=round(ne, 3)))
        offset = decision.teaser_duration

    m0, m1 = decision.main_start, decision.main_end
    main_dur = decision.main_duration
    for w in words:
        if w.end > (m0 + 0.04) and w.start < (m1 - 0.04):
            ns = offset + max(0.0, w.start - m0)
            ne = offset + min(main_dur, w.end - m0)
            if ne - ns >= 0.05:
                timeline_words.append(WordTimestamp(word=w.word, start=round(ns, 3), end=round(ne, 3)))

    return timeline_words


def chunk_words(
    words: List[WordTimestamp],
    max_words: int = 2,
    max_chars: int = 16,
    pause_threshold: float = 0.24,
    split_boundary_sec: float = 0.0,
) -> List[List[WordTimestamp]]:
    if not words:
        return []

    chunks: List[List[WordTimestamp]] = []
    current: List[WordTimestamp] = [words[0]]

    for w in words[1:]:
        prev = current[-1]
        crossed_teaser = (
            split_boundary_sec > 0.0
            and prev.end <= split_boundary_sec + 0.02
            and w.start >= split_boundary_sec - 0.02
        )
        pause = w.start - prev.end
        punct_break = bool(re.search(r"[.!?,;:]$", prev.word))
        projected_chars = sum(len(x.word) for x in current) + len(current) + len(w.word)

        if (
            crossed_teaser
            or len(current) >= max_words
            or projected_chars > max_chars
            or pause > pause_threshold
            or punct_break
        ):
            chunks.append(current)
            current = [w]
        else:
            current.append(w)

    if current:
        chunks.append(current)
    return chunks


class CaptionGenerator:
    """
    Gerador Submagic / OpusClip Pro:
      - Blocos curtos e ultra-dinâmicos (1 a 2 palavras por batida).
      - Posicionamento inteligente: exatamente na linha divisória central (`Y=685`) em `split_screen`
        (nunca tampa o rosto nem a gameplay) ou na zona segura (`Y=1340`) em tela cheia.
      - Efeito karaokê vivo com pop elástico (`122% -> 108%`), palavras futuras em prata suave
        e alternância de cores vibrantes (Ouro `#FFE600`, Verde Neon `#00FF66` e Coral `#FF3355`).
      - Headline do Teaser em duas cores com contorno grosso (sem caixa cinza quadrada em cima do rosto).
    """

    def __init__(self, theme: Dict[str, Any]) -> None:
        self.theme = theme

    def generate_files(
        self,
        words: List[WordTimestamp],
        decision: DirectorDecision,
        ass_path: Path,
        srt_path: Path,
        layout_mode: str = "split_screen",
    ) -> Tuple[Path, Path]:
        mapped = remap_words_to_final_timeline(words, decision)
        cap_cfg = self.theme.get("captions", {})
        chunks = chunk_words(
            mapped,
            max_words=int(cap_cfg.get("max_words_per_chunk", 2)),
            max_chars=int(cap_cfg.get("max_chars_per_chunk", 16)),
            pause_threshold=float(cap_cfg.get("pause_split_threshold_sec", 0.24)),
            split_boundary_sec=decision.teaser_duration if decision.cold_open_enabled else 0.0,
        )

        ass_content = self._render_ass(chunks, decision, layout_mode=layout_mode)
        srt_content = self._render_srt(chunks)

        ass_path.parent.mkdir(parents=True, exist_ok=True)
        ass_path.write_text(ass_content, encoding="utf-8-sig")
        srt_path.write_text(srt_content, encoding="utf-8")
        return ass_path, srt_path

    def _render_ass(
        self,
        chunks: List[List[WordTimestamp]],
        decision: DirectorDecision,
        layout_mode: str = "split_screen",
    ) -> str:
        w_canvas = int(self.theme.get("canvas_width", 1080))
        h_canvas = int(self.theme.get("canvas_height", 1920))
        cap = self.theme.get("captions", {})
        hl = self.theme.get("teaser_headline", {})

        font_name = cap.get("font_name", "Arial Black")
        font_size = int(cap.get("font_size", 84))
        primary_ass = hex_to_ass_color(cap.get("primary_text_hex", "#FFFFFF"), 0)
        upcoming_ass = hex_to_ass_color(cap.get("upcoming_text_hex", "#C8C8C8"), 25)
        gold_ass = hex_to_ass_color(cap.get("active_word_hex", "#FFE600"), 0)
        green_ass = hex_to_ass_color(cap.get("alt_active_word_hex", "#00FF66"), 0)
        coral_ass = hex_to_ass_color(cap.get("hype_word_hex", "#FF3355"), 0)
        outline_ass = hex_to_ass_color(cap.get("outline_hex", "#000000"), 0)
        shadow_ass = hex_to_ass_color(cap.get("shadow_hex", "#000000"), 75)

        outline_w = int(cap.get("outline_width", 8))
        shadow_d = int(cap.get("shadow_depth", 5))
        uppercase = bool(cap.get("uppercase", True))
        active_scale = int(cap.get("active_word_scale_pct", 122))

        is_split = layout_mode == "split_screen"
        cap_y = int(cap.get("split_screen_pos_y", 685) if is_split else cap.get("full_screen_pos_y", 1340))
        hl_y = int(cap_y if is_split else hl.get("full_screen_pos_y", 340))

        hl_font = hl.get("font_name", font_name)
        hl_size = int(hl.get("font_size", 68))
        hl_text_ass = hex_to_ass_color(hl.get("text_hex", "#FFFFFF"), 0)
        hl_accent_ass = hex_to_ass_color(hl.get("accent_hex", "#FFE600"), 0)
        hl_outline_ass = hex_to_ass_color(hl.get("outline_hex", "#000000"), 0)
        hl_outline_w = int(hl.get("outline_width", 9))
        hl_shadow_d = int(hl.get("shadow_depth", 6))

        # Alignment=5 (center-middle) controlado via `\pos(540, Y)` para precisão cirúrgica
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w_canvas}
PlayResY: {h_canvas}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: OpusCap,{font_name},{font_size},{primary_ass},{gold_ass},{outline_ass},{shadow_ass},-1,0,0,0,100,100,1.5,0,1,{outline_w},{shadow_d},5,50,50,50,1
Style: TeaserHook,{hl_font},{hl_size},{hl_text_ass},{hl_accent_ass},{hl_outline_ass},{shadow_ass},-1,0,0,0,100,100,1.5,0,1,{hl_outline_w},{hl_shadow_d},5,60,60,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events: List[str] = []

        # 1. Headline de impacto em 2 cores EXCLUSIVA durante os 2-3s do Cold Open Teaser
        if (
            decision.cold_open_enabled
            and decision.teaser_duration > 0.3
            and hl.get("enabled", True)
            and decision.teaser_headline.strip()
        ):
            raw_hl = escape_ass_text(decision.teaser_headline.strip().upper())
            lines = textwrap.wrap(raw_hl, width=int(hl.get("max_chars_per_line", 18)))
            if len(lines) >= 2:
                # Primeira linha branca, última linha amarela ouro para contraste viral imediato
                colored_lines = lines[:-1] + [f"{{\\c{hl_accent_ass}&}}{lines[-1]}{{\\c{hl_text_ass}&}}"]
                wrapped = "\\N".join(colored_lines)
            else:
                words_hl = raw_hl.split()
                if len(words_hl) >= 2:
                    mid = max(1, len(words_hl) // 2)
                    p1 = " ".join(words_hl[:mid])
                    p2 = " ".join(words_hl[mid:])
                    wrapped = f"{p1}\\N{{\\c{hl_accent_ass}&}}{p2}{{\\c{hl_text_ass}&}}"
                else:
                    wrapped = f"{{\\c{hl_accent_ass}&}}{raw_hl}"

            end_str = format_ass_time(max(0.2, decision.teaser_duration - 0.06))
            anim_tags = f"{{\\pos({w_canvas // 2},{hl_y})\\fad(60,90)\\fscx120\\fscy120\\t(0,110,\\fscx100\\fscy100)}}"
            events.append(
                f"Dialogue: 3,0:00:00.00,{end_str},TeaserHook,,0,0,0,,{anim_tags}{wrapped}"
            )

        # 2. Legendas dinâmicas palavra a palavra (Submagic / OpusClip Pro)
        # Primeiro montamos todos os eventos em memória e depois travamos `end <= next_start - 0.02`
        # para GARANTIR matematicamente que duas legendas NUNCA se sobreponham na tela!
        raw_cap_events: List[Tuple[float, float, str]] = []
        for chunk_idx, chunk in enumerate(chunks):
            tokens = [
                escape_ass_text(w.word.upper() if uppercase else w.word)
                for w in chunk
            ]
            tilt_tag = ""
            if len(tokens) == 1:
                angle = 2 if (chunk_idx % 2 == 0) else -2
                tilt_tag = f"\\frz{angle}"

            for i, word_obj in enumerate(chunk):
                seg_start = word_obj.start
                if i < len(chunk) - 1:
                    seg_end = max(word_obj.end, chunk[i + 1].start)
                else:
                    seg_end = word_obj.end + 0.06

                active_color = self._pick_active_color(tokens[i], chunk_idx, gold_ass, green_ass, coral_ass)
                rendered_words: List[str] = []

                for j, tok in enumerate(tokens):
                    if j == i:
                        rendered_words.append(
                            f"{{\\c{active_color}&\\fscx{active_scale}\\fscy{active_scale}"
                            f"\\t(0,95,\\fscx108\\fscy108)}}{tok}"
                            f"{{\\c{primary_ass}&\\fscx100\\fscy100}}"
                        )
                    elif j < i:
                        rendered_words.append(f"{{\\c{primary_ass}&\\fscx100\\fscy100}}{tok}")
                    else:
                        rendered_words.append(f"{{\\c{upcoming_ass}&\\fscx96\\fscy96}}{tok}{{\\fscx100\\fscy100}}")

                pos_prefix = f"{{\\pos({w_canvas // 2},{cap_y}){tilt_tag}}}"
                line_text = pos_prefix + " ".join(rendered_words)
                raw_cap_events.append((seg_start, seg_end, line_text))

        # Clamping estrito anti-sobreposição
        for idx, (st_t, en_t, line_txt) in enumerate(raw_cap_events):
            if idx < len(raw_cap_events) - 1:
                next_start = raw_cap_events[idx + 1][0]
                en_t = min(en_t, next_start - 0.015)
            if en_t <= st_t + 0.02:
                en_t = st_t + 0.02
            events.append(
                f"Dialogue: 2,{format_ass_time(st_t)},{format_ass_time(en_t)},OpusCap,,0,0,0,,{line_txt}"
            )

        return header + "\n".join(events) + "\n"

    @staticmethod
    def _pick_active_color(
        token_upper: str,
        chunk_idx: int,
        gold_ass: str,
        green_ass: str,
        coral_ass: str,
    ) -> str:
        clean_w = re.sub(r"[^\wÀ-ÿ]", "", token_upper)
        if "!" in token_upper or "?" in token_upper or clean_w in HYPE_WORDS:
            return coral_ass if (chunk_idx % 2 == 0) else green_ass
        if any(ch.isdigit() for ch in clean_w):
            return green_ass
        if chunk_idx % 4 == 3:
            return green_ass
        return gold_ass

    @staticmethod
    def _render_srt(chunks: List[List[WordTimestamp]]) -> str:
        blocks: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            start_s = format_srt_time(chunk[0].start)
            end_s = format_srt_time(chunk[-1].end + 0.10)
            text = " ".join(w.word for w in chunk)
            blocks.append(f"{idx}\n{start_s} --> {end_s}\n{text}\n")
        return "\n".join(blocks)
