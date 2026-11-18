from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

from src.ai.trending_hashtags import fetch_brazil_realtime_trends, suggest_dynamic_hashtags
from src.models import DirectorDecision, MomentCluster, ProfileConfig, WordTimestamp

logger = logging.getLogger(__name__)


def build_transcript_lines(words: List[WordTimestamp], chunk_sec: float = 4.0) -> str:
    """Agrupa palavras em linhas com timestamps claros para o Gemini decidir o corte cirúrgico."""
    if not words:
        return "(Sem falas detectadas — momento puramente visual ou grito/reação)"

    lines: List[str] = []
    bucket: List[WordTimestamp] = [words[0]]
    for w in words[1:]:
        if (w.end - bucket[0].start) <= chunk_sec and (w.start - bucket[-1].end) <= 0.8:
            bucket.append(w)
        else:
            text = " ".join(x.word for x in bucket)
            lines.append(f"[{bucket[0].start:05.2f}s - {bucket[-1].end:05.2f}s] {text}")
            bucket = [w]
    if bucket:
        text = " ".join(x.word for x in bucket)
        lines.append(f"[{bucket[0].start:05.2f}s - {bucket[-1].end:05.2f}s] {text}")
    return "\n".join(lines)


class GeminiDirector:
    """
    Diretor de Edição via Gemini:
      - Escolhe `main_start` e `main_end` cirúrgicos removendo tempos mortos.
      - Seleciona `teaser_start` e `teaser_end` (2.0s–3.2s do clímax) para o Cold Open no segundo 00:00.
      - Cria a `teaser_headline` curta exibida só nos 2–3s iniciais.
      - Cria a `caption_text` minimalista estilo meme/reação (sem perguntas robóticas) + 3–4 hashtags.
      - Garante matematicamente que `teaser_duration + main_duration <= 58.0s` (anti-mute TikTok).
    """

    def __init__(
        self,
        profile: ProfileConfig,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ) -> None:
        self.profile = profile
        self.api_key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
        self.model_name = (model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
        self.groq_api_key = (groq_api_key if groq_api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
        self.groq_model = os.getenv("GROQ_DIRECTOR_MODEL", "openai/gpt-oss-120b").strip()

    def direct_cut(
        self,
        cluster: MomentCluster,
        words: List[WordTimestamp],
        raw_duration_sec: float,
        peak_hint_sec: float,
    ) -> DirectorDecision:
        prompt = self._build_prompt(cluster, words, raw_duration_sec, peak_hint_sec)
        # Clipes curtos da comunidade (Twitch/Kick <= 65s) e captura ao vivo da Kick (`<= 65s`) são preservados íntegros
        preserve_full_clip = cluster.mode in ("live_hot", "post_live", "kick_clip", "kick_live")
        prefer_clip_ending = cluster.mode == "kick_clip"

        # 1. Tenta Gemini quando chave AIza válida estiver configurada
        if self.api_key and not self.api_key.startswith("AQ."):
            try:
                raw_decision = self._call_gemini(prompt)
                return self.enforce_constraints(
                    raw_decision,
                    words,
                    raw_duration_sec,
                    peak_hint_sec,
                    preserve_full_clip=preserve_full_clip,
                    prefer_clip_ending=prefer_clip_ending,
                )
            except Exception as exc:
                logger.warning("Gemini falhou (%s). Acionando motor LLM Groq 120B...", exc)

        # 2. Motor LLM Groq (`openai/gpt-oss-120b`) — usa a mesma GROQ_API_KEY já configurada!
        if self.groq_api_key:
            try:
                raw_decision = self._call_groq_llm(prompt)
                return self.enforce_constraints(
                    raw_decision,
                    words,
                    raw_duration_sec,
                    peak_hint_sec,
                    preserve_full_clip=preserve_full_clip,
                    prefer_clip_ending=prefer_clip_ending,
                )
            except Exception as exc:
                logger.warning("Groq LLM falhou (%s). Tentando fallback...", exc)

        # 3. Se a chave Gemini começar com AQ., ainda tenta antes do fallback heurístico
        if self.api_key:
            try:
                raw_decision = self._call_gemini(prompt)
                return self.enforce_constraints(
                    raw_decision,
                    words,
                    raw_duration_sec,
                    peak_hint_sec,
                    preserve_full_clip=preserve_full_clip,
                    prefer_clip_ending=prefer_clip_ending,
                )
            except Exception as exc:
                logger.warning("Chave Gemini AQ. não aceita no endpoint público (%s).", exc)

        return self._fallback_decision(
            cluster,
            words,
            raw_duration_sec,
            peak_hint_sec,
            preserve_full_clip=preserve_full_clip,
            prefer_clip_ending=prefer_clip_ending,
        )

    def _build_prompt(
        self,
        cluster: MomentCluster,
        words: List[WordTimestamp],
        raw_duration_sec: float,
        peak_hint_sec: float,
    ) -> str:
        clip_titles = list({c.title for c in cluster.clips if c.title})[:8]
        transcript_block = build_transcript_lines(words)
        edt = self.profile.editing
        is_youtube = cluster.mode.startswith("youtube")
        is_kick = cluster.mode.startswith("kick")
        is_external_stream = is_youtube or cluster.mode in ("kick_vod", "kick_live")
        creator_name = (
            cluster.best_clip.broadcaster_name
            if (is_youtube or is_kick) and cluster.best_clip.broadcaster_name
            else self.profile.display_name
        )
        creator_handle = cluster.broadcaster_login or self.profile.twitch_login
        platform_label = "YouTube" if is_youtube else ("Kick" if is_kick else "Twitch")
        if is_youtube:
            default_tags = f'["#{creator_handle}", "#fy", "#fyp", "#viral", "#cortes"]'
        elif is_kick:
            default_tags = f'["#{creator_handle}", "#fy", "#fyp", "#viral", "#kick"]'
        else:
            default_tags = f'["#{creator_handle}", "#fy", "#fyp", "#viral", "#cortes"]'

        live_trends = fetch_brazil_realtime_trends(limit=15)
        trends_hint = ", ".join(live_trends) if live_trends else "(indisponível no momento)"

        return f"""
Você é um editor sênior de páginas virais brasileiras de cortes de lives no TikTok e Instagram Reels.

DADOS DO MOMENTO ({platform_label}):
- Criador: {creator_name} (@{creator_handle})
- Categoria/Jogo: {cluster.game_name or "Live"}
- Título da Live/Vídeo: {cluster.stream_title or "N/A"}
- Títulos/temas deste momento: {json.dumps(clip_titles, ensure_ascii=False)}
- Assuntos EM ALTA AGORA NO BRASIL (Google Trends tempo real): {trends_hint}
- Duração total do vídeo bruto extraído: {raw_duration_sec:.2f}s
- Ponto principal aproximado: ~{peak_hint_sec:.1f}s

TRANSCRIÇÃO COM TIMESTAMPS:
{transcript_block}

REGRAS OBRIGATÓRIAS DE EDIÇÃO:
1. CORTE PRINCIPAL (`main_start` e `main_end`):
   - {"Use o vídeo bruto acima como BUFFER DE CONTEXTO para encontrar o INÍCIO REAL da conversa/situação (antes do clímax em ~" + f"{peak_hint_sec:.0f}s" + ") e o DESFECHO COMPLETO após a reação." if is_external_stream else "Remova silêncios, enrolação ou fim de assuntos anteriores."}
   - Comece exatamente na frase de contexto/gancho que prepara o momento (`main_start`) e termine só após toda a reação/risada/desfecho final (`main_end` — deixe sempre +1.5s a +3s de respiro após a última fala).
   - {"NÃO comece o corte no meio de uma frase ou já com o assunto terminando; e se o assunto mudar para leitura de chat/doações ou caminhada sem relação, ENCERRE `main_end` antes dessa quebra (melhor 30–50s coesos do que 60s com assunto desconexo!)." if is_external_stream else ""}
   - {"ATENÇÃO (Clipe longo da Kick > 65s): o clipe original tem " + f"{raw_duration_sec:.1f}s" + " porque quem clipou incluiu a reação até o final! Prefira manter a janela final completa até ~" + f"{raw_duration_sec:.1f}s" + " (aproveitando de 45s a " + f"{edt.max_duration_seconds:.0f}s" + ") em vez de encerrar cedo aos 17s–30s." if cluster.mode == "kick_clip" and raw_duration_sec > max(65.0, edt.max_duration_seconds) else ""}
   - A soma final (`teaser` + `corte principal`) NUNCA pode passar de {edt.max_duration_seconds:.1f} segundos (se usar teaser de ~2.8s, o corte principal deve ter no máximo ~55.0s).
   - Mínimo desejado: {edt.min_duration_seconds:.1f}s (a menos que o vídeo bruto inteiro seja menor).
2. COLD OPEN / TEASER (`cold_open_enabled`, `teaser_start`, `teaser_end`):
   - Se o corte principal tiver pelo menos {edt.cold_open_min_main_duration:.0f}s e o clímax não estiver já nos primeiros 5s, ative `cold_open_enabled = true`.
   - Escolha os {edt.cold_open_min_seconds:.1f}s a {edt.cold_open_max_seconds:.1f}s mais explosivos/engraçados do clímax dentro da janela para aparecerem no segundo 00:00.
3. HEADLINE DO TEASER (`teaser_headline`):
   - Máximo 4 a 5 palavras curtas que aparecem na tela nos 2-3s iniciais.
   - Baseie-se LITERALMENTE no título/assunto do momento (`{cluster.dominant_title}`) ou na frase mais engraçada dita na transcrição.
   - PROIBIDO usar clichês genéricos de IA como "JOGADA DE OURO", "MOMENTO ÉPICO", "IMPERDÍVEL", "INACREDITÁVEL".
4. DESCRIÇÃO COMPACTA, DESCRITIVA E COM GANCHO (`caption_text` e `hashtags`):
   - `caption_text`: escreva **1 única frase curta e direta (ENTRE `40` E `85` CARACTERES no máximo — sem textão!)** explicando **o fato específico** da cena + **um gancho que dê vontade de assistir até o fim**.
   - Seja concreto com os detalhes reais da transcrição/título (ex.: o objeto, jogo, pessoa ou erro exato), **sem entregar a punchline inteira** e sem precisar de 2 frases longas. Pode usar no máximo 1 emoji se combinar.
   - ESTRITAMENTE PROIBIDO:
     * Enrolação genérica de IA (ex. RUIM: `"tenta X, mas acaba atrapalhando tudo"`, `"a situação vira risada total"`, `"transforma num caos completo"`, `"deixou todo mundo sem reação"`, `"galera surtando"`).
     * Textão acima de `85` caracteres (tem que caber antes do *"Ver mais"* no TikTok/Reels).
     * Fala avulsa sem explicar a cena (ex. RUIM: `"não conhece nada, galera"`, `"vai ser muito foda"`).
   - Exemplos de BOM tamanho e gancho (varie a estrutura conforme a cena):
     * `"Alanzoka parou o carro errado e se arrependeu na hora 👀"`
     * `"O momento em que o Bistecone quase perdeu o PS5 ao vivo 💀"`
     * `"Davy Jones foi bancar o encanador e olha no que deu 😂"`
   - `hashtags`: de 3 a {self.profile.copy.max_hashtags} hashtags focadas no que pode **ficar em alta neste momento** (ex: {default_tags}):
     1) criador/participante em evidência (`#{creator_handle}`),
     2) 1 ou 2 tags específicas do assunto/jogo/meme citado (e se algum tema em alta do Google Trends acima tiver ligação real com o clipe, inclua obrigatoriamente!),
     3) 1 ou 2 tags fortes de descoberta/alcance no algoritmo (`#fy`, `#fyp`, `#viral`, `#react`, `#clips`). Evite tags genéricas mortas ou que não combinem com a plataforma.
5. CORREÇÃO DE GRAFIA DO WHISPER (`word_corrections`):
   - Se notar na transcrição alguma palavra claramente ouvida errado pelo Whisper (ex: nome de jogo, gíria ou palavra trocada), inclua no dicionário `"word_corrections": {{"palavra_errada": "palavra_certa"}}` (caso contrário, `{{}}`).

Responda ESTRITAMENTE em JSON válido com as chaves:
{{
  "main_start": float,
  "main_end": float,
  "cold_open_enabled": bool,
  "teaser_start": float,
  "teaser_end": float,
  "teaser_headline": string,
  "caption_text": string,
  "hashtags": [string],
  "word_corrections": {{}},
  "virality_score": int,
  "reasoning": string
}}
""".strip()

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.35,
            ),
        )
        text = (response.text or "").strip()
        return json.loads(text)

    def _call_groq_llm(self, prompt: str) -> Dict[str, Any]:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.groq_model,
                "temperature": 0.35,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "Você é um diretor de cortes virais para TikTok e Reels. Responda apenas com um objeto JSON válido.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=45,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def enforce_constraints(
        self,
        raw: Dict[str, Any],
        words: List[WordTimestamp],
        raw_duration_sec: float,
        peak_hint_sec: float,
        preserve_full_clip: bool = True,
        prefer_clip_ending: bool = False,
    ) -> DirectorDecision:
        edt = self.profile.editing
        max_total = min(edt.max_duration_seconds, raw_duration_sec)

        main_start = max(0.0, min(raw_duration_sec - 2.0, float(raw.get("main_start", 0.0))))
        main_end = max(main_start + 3.0, min(raw_duration_sec, float(raw.get("main_end", raw_duration_sec))))

        # Aplica eventuais correções ortográficas identificadas pela IA nas palavras do Whisper
        corrections = raw.get("word_corrections")
        if isinstance(corrections, dict) and corrections and words:
            norm_map = {str(k).strip().lower(): str(v).strip() for k, v in corrections.items() if str(k).strip() and str(v).strip()}
            for w in words:
                clean_token = re.sub(r"[^\wÀ-ÿ]", "", w.word).lower()
                if clean_token in norm_map:
                    w.word = norm_map[clean_token]

        # Quando o vídeo de origem é um clipe curto da Twitch/Kick (<= 65s),
        # preservamos 100% do início (0.0s) ao fim (raw_duration_sec) SEM cortar nenhum segundo das pontas!
        if preserve_full_clip and raw_duration_sec <= max(65.0, edt.max_duration_seconds):
            main_start = 0.0
            main_end = raw_duration_sec
        else:
            if words:
                near_start = [w for w in words if abs(w.start - main_start) <= 0.8]
                if near_start:
                    w0 = min(near_start, key=lambda w: abs(w.start - main_start))
                    main_start = max(0.0, w0.start - 0.16)
                near_end = [w for w in words if abs(w.end - main_end) <= 0.8]
                if near_end:
                    w1 = min(near_end, key=lambda w: abs(w.end - main_end))
                    main_end = min(raw_duration_sec, w1.end + 1.35)

            # Evita cortar os últimos ~3.5s de risada/reação quando já está no fim do clipe
            if (raw_duration_sec - main_end) <= 3.5 and (raw_duration_sec - main_start) <= max_total:
                main_end = raw_duration_sec

            # Para clipes longos da Kick (> 65s): evita encerrar o clipe cedo aos 17s-30s amputando o final
            if prefer_clip_ending and raw_duration_sec > max_total:
                if raw_duration_sec <= 115.0 and (raw_duration_sec - main_end) <= 22.0:
                    main_end = raw_duration_sec
                    if (main_end - main_start) > max_total:
                        main_start = max(0.0, main_end - max_total)
                if (main_end - main_start) < min(50.0, max_total):
                    desired = min(max_total, raw_duration_sec - main_start)
                    main_end = min(raw_duration_sec, max(main_end, main_start + desired))
                    if (main_end - main_start) < min(50.0, max_total):
                        main_start = max(0.0, main_end - min(max_total, raw_duration_sec))

        cold_open = bool(raw.get("cold_open_enabled", True)) and edt.cold_open_enabled
        teaser_start = float(raw.get("teaser_start", 0.0))
        teaser_end = float(raw.get("teaser_end", 0.0))

        main_dur = main_end - main_start
        if main_dur < edt.cold_open_min_main_duration or raw_duration_sec < edt.cold_open_min_main_duration:
            cold_open = False

        if cold_open:
            # Garante que o teaser esteja numa região válida e tenha entre 2.0s e 3.2s
            teaser_start = max(main_start + 4.0, min(main_end - edt.cold_open_min_seconds, teaser_start))
            teaser_dur = max(edt.cold_open_min_seconds, min(edt.cold_open_max_seconds, teaser_end - teaser_start))
            teaser_end = min(raw_duration_sec, teaser_start + teaser_dur)
            if teaser_end - teaser_start < 1.6 or (teaser_start - main_start) < 3.5:
                cold_open = False
                teaser_start, teaser_end = 0.0, 0.0
        else:
            teaser_start, teaser_end = 0.0, 0.0

        teaser_dur = (teaser_end - teaser_start) if cold_open else 0.0

        # Trava matemática rígida <= 58.0s (incluindo o teaser!)
        allowed_main = max(5.0, max_total - teaser_dur)
        if (main_end - main_start) > allowed_main:
            # Preserva o pico dentro da janela cortada
            if main_start <= peak_hint_sec <= main_end:
                ideal_start = max(main_start, peak_hint_sec - allowed_main * 0.68)
                main_start = min(ideal_start, main_end - allowed_main)
                main_end = main_start + allowed_main
            else:
                main_end = main_start + allowed_main

        # Limpeza da descrição compacta (remove enchimento genérico de IA e mantém <= 115 chars)
        caption = str(raw.get("caption_text") or cluster_fallback_caption(self.profile)).strip()
        caption = re.sub(r"\s+", " ", caption)
        caption = re.sub(
            r"\s*(?:o que (?:vocês|vcs) acharam\??|comenta aí!?|compartilha com os amigos!?)\s*$",
            "",
            caption,
            flags=re.IGNORECASE,
        ).strip()
        if len(caption) > 115:
            # Tenta manter até a primeira oração completa antes de alongar
            cut_pos = caption.rfind(" ", 0, 112)
            caption = caption[: (cut_pos if cut_pos > 45 else 112)].rstrip(" ,;:") + "..."

        raw_tags = [str(t).strip() for t in (raw.get("hashtags") or []) if str(t).strip()]
        if not raw_tags:
            raw_tags = suggest_dynamic_hashtags(
                creator_handle=self.profile.twitch_login,
                max_tags=self.profile.copy.max_hashtags,
                base_discovery_tags=self.profile.copy.base_hashtags,
            )
        clean_tags: List[str] = []
        for t in raw_tags:
            tag = t if t.startswith("#") else f"#{t}"
            tag = re.sub(r"[^\w#À-ÿ]", "", tag)
            if tag and tag.lower() not in {x.lower() for x in clean_tags}:
                clean_tags.append(tag)
            if len(clean_tags) >= self.profile.copy.max_hashtags:
                break

        headline = str(raw.get("teaser_headline") or "").strip()
        if cold_open and not headline:
            headline = "OLHA O QUE ACONTECEU 💀"

        return DirectorDecision(
            main_start=round(main_start, 3),
            main_end=round(main_end, 3),
            cold_open_enabled=cold_open,
            teaser_start=round(teaser_start, 3),
            teaser_end=round(teaser_end, 3),
            teaser_headline=headline[:48],
            caption_text=caption,
            hashtags=clean_tags,
            virality_score=max(1, min(100, int(raw.get("virality_score", 75)))),
            reasoning=str(raw.get("reasoning", "")),
        )

    def _fallback_decision(
        self,
        cluster: MomentCluster,
        words: List[WordTimestamp],
        raw_duration_sec: float,
        peak_hint_sec: float,
        preserve_full_clip: bool = True,
        prefer_clip_ending: bool = False,
    ) -> DirectorDecision:
        edt = self.profile.editing
        cold_open = edt.cold_open_enabled and raw_duration_sec >= edt.cold_open_min_main_duration
        teaser_len = 2.5 if cold_open else 0.0
        target_main = min(raw_duration_sec, edt.max_duration_seconds - teaser_len)

        start = max(0.0, min(raw_duration_sec - target_main, peak_hint_sec - target_main * 0.62))
        end = min(raw_duration_sec, start + target_main)

        t_start = max(start + 5.0, min(end - 2.6, peak_hint_sec)) if cold_open else 0.0
        t_end = (t_start + 2.5) if cold_open else 0.0
        if cold_open and (t_end > end or (t_start - start) < 4.0):
            cold_open = False
            t_start, t_end = 0.0, 0.0

        title_clean = (cluster.dominant_title or "momento inesperado durante a live").strip()
        creator_label = cluster.best_clip.broadcaster_name or self.profile.display_name
        game_hint = f" em {cluster.game_name}" if cluster.game_name and cluster.game_name.lower() not in {"live", "youtube", "kick"} else ""
        fallback_desc = f"{creator_label} protagonizou uma cena inesperada{game_hint}: {title_clean}."
        transcript_hint = " ".join(w.word for w in words[:40])
        tags = suggest_dynamic_hashtags(
            creator_handle=cluster.broadcaster_login or self.profile.twitch_login,
            platform_mode=cluster.mode,
            game_name=cluster.game_name,
            title_text=f"{cluster.stream_title} {cluster.dominant_title}",
            transcript_snippet=transcript_hint,
            max_tags=self.profile.copy.max_hashtags,
            base_discovery_tags=self.profile.copy.base_hashtags,
        )

        return self.enforce_constraints(
            {
                "main_start": start,
                "main_end": end,
                "cold_open_enabled": cold_open,
                "teaser_start": t_start,
                "teaser_end": t_end,
                "teaser_headline": title_clean.upper()[:42] if cold_open else "",
                "caption_text": fallback_desc,
                "hashtags": tags[: self.profile.copy.max_hashtags],
                "virality_score": int(min(95, max(60, cluster.score))),
                "reasoning": "Fallback centrado no pico de clipes da comunidade.",
            },
            words=words,
            raw_duration_sec=raw_duration_sec,
            peak_hint_sec=peak_hint_sec,
            preserve_full_clip=preserve_full_clip,
            prefer_clip_ending=prefer_clip_ending,
        )


def cluster_fallback_caption(profile: ProfileConfig) -> str:
    return f"momento absurdo na live do {profile.display_name.lower()}"
