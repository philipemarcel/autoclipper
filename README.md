# 🎬 Twitch AutoClipper 24/7

Fábrica contínua e automatizada de cortes verticais (`9:16`) a partir dos momentos mais clipados de lives na **Twitch** (usando apenas conta comum de espectador), integrada aos seus **20 TB do Google Drive**, **Google Colab Ultra (`L4`/`A100`)**, **GitHub Actions 24/7**, fila de aprovação de 1 clique no **Discord** e publicação espaçada no **Instagram Reels** e **TikTok**.

## Principais Recursos

- **Modo Duplo 24/7 (`live_hot` + `post_live`)**: detecta explosões de clipes em tempo real durante a live e faz repescagem pós-live sem repetir momentos (`processed_moments.json`).
- **Extração Inteligente**: puxa contexto direto do VOD público (`yt-dlp --download-sections`), com *fallback* automático para o melhor clipe pronto se o VOD for fechado.
- **Diretor IA (Gemini)**: corta tempos mortos, trava a duração total em `<= 58s` (evita *mute* automático acima de 60s no TikTok), monta um *Cold Open* de `2–3s` do clímax logo no segundo `00:00` e escreve copy minimalista estilo meme/reação + 3–4 hashtags.
- **Auto-Layout Vertical 9:16**: alterna sozinho entre *Split-Screen* (webcam em cima + tela embaixo), *Face-Tracking* (câmera cheia) e *Fundo Blur* (sem rosto).
- **Legendas Dinâmicas OpusClip (`.ass`) + Headline de 3s**: destaque colorido palavra a palavra + headline curta exclusiva durante o *Cold Open*, salvando também `limpo_1080p60.mp4` e `legenda.srt` no Drive.
- **Aprovação de 1 Clique no Discord + Fila Espaçada (*Drip Queue*)**: envia prévia leve (`< 9 MB`) nativa no Discord + botões `Aprovar Fila`, `Postar Agora`, `Só Reels`, `Só TikTok`, `Limpar Música IA (Demucs)` e `Descartar`.

## Comandos Principais

```powershell
# 1. Monitorar Twitch e gerar cortes agora
python -m src.main watch

# 2. Disparar itens agendados vencidos na Fila Espaçada
python -m src.main drip-publish

# 3. Executar uma ação manual de corte (ex: limpar música de fundo com IA)
python -m src.main discord-action --action clean_audio --cut-id <ID_DO_CORTE>

# 4. Rodar bateria de testes automatizados
pytest -v
```

Consulte o passo a passo completo das chaves e deploy em [SETUP_GUIDE.md](SETUP_GUIDE.md) e o motor GPU em [colab/Twitch_AutoClipper_Ultra_GPU.ipynb](colab/Twitch_AutoClipper_Ultra_GPU.ipynb).
