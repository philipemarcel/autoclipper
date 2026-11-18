# Guia Completo de Configuração — Twitch AutoClipper 24/7

Este guia mostra passo a passo como conectar todas as peças gratuitas e sua conta **Google AI Ultra (20 TB)** para deixar o sistema operando 100% no piloto automático.

---

## Passo 1: Escolher o primeiro criador (`config/profiles.yaml`)

Abra `config/profiles.yaml` e troque `seu_criador_aqui` pelo login exato da URL do canal na Twitch (tudo em minúsculas):

```yaml
profiles:
  - id: "criador_01"
    enabled: true
    twitch_login: "alanzoka" # <-- exemplo: gaules, casimito, paulinholokobr, loud_coringa
    display_name: "Alanzoka"
```

> **Para expandir no futuro:** basta copiar e colar um novo item abaixo em `profiles:` com `id: "criador_02"`!

---

## Passo 2: Chaves da Twitch (2 minutos — conta comum de espectador)

1. Acesse https://dev.twitch.tv/console/apps logado na sua conta normal da Twitch.
2. Clique em **Register Your Application**:
   - **Name**: `MeuAutoClipper` (qualquer nome único)
   - **OAuth Redirect URLs**: `http://localhost`
   - **Category**: `Analytics Tool`
3. Clique em **Manage** no app criado:
   - Copie o **Client ID** → `TWITCH_CLIENT_ID`
   - Clique em **New Secret** e copie → `TWITCH_CLIENT_SECRET`

---

## Passo 3: Chaves de IA (Gemini + Groq Whisper gratuito)

1. **Gemini API** (Diretor de corte cirúrgico, Cold Open e Copy):
   - Acesse https://aistudio.google.com/apikey com sua conta Google e crie uma chave → `GEMINI_API_KEY`.
2. **Groq Cloud** (Transcrição *word-level* gratuita em 2 segundos para o GitHub Actions):
   - Acesse https://console.groq.com/keys, crie uma conta gratuita e gere uma API Key → `GROQ_API_KEY`.

---

## Passo 4: Conectar seus 20 TB do Google Drive

1. Acesse https://console.cloud.google.com/ na sua conta **Google AI Ultra**:
   - Crie um projeto novo → vá em **APIs & Services > Library** → ative a **Google Drive API**.
   - Em **OAuth consent screen**, escolha **External**, preencha nome/e-mail e adicione seu próprio Gmail em **Test users** (ou publique o app pessoal).
   - Em **Credentials > Create Credentials > OAuth client ID**, escolha **Desktop app** e baixe o JSON como `client_secret.json` na raiz deste projeto.
2. No seu terminal local, gere o token uma única vez:
   ```powershell
   pip install -r requirements.txt
   python scripts/auth_gdrive.py --client-secrets client_secret.json
   ```
3. Faça login no navegador e copie a linha JSON exibida no terminal → esse valor é o seu `GDRIVE_TOKEN_JSON`!

---

## Passo 5: Bot do Discord + Botões de 1 Clique (Cloudflare Worker gratuito)

### 5.1 Criar o Bot no Discord
1. Acesse https://discord.com/developers/applications → **New Application**.
2. Copie a **Public Key** na aba *General Information* (usaremos no Worker abaixo).
3. Na aba **Bot**:
   - Clique em **Reset Token** e copie → `DISCORD_BOT_TOKEN`.
4. Na aba **OAuth2 > URL Generator**:
   - Marque `bot` e as permissões `Send Messages`, `Embed Links`, `Attach Files`, `Read Message History`.
   - Abra a URL gerada e adicione o bot ao seu servidor privado de cortes.
5. No Discord, ative o *Modo Desenvolvedor*, clique com o botão direito no canal onde quer receber as prévias → **Copiar ID do canal** → `DISCORD_CHANNEL_ID`.

### 5.2 Subir o Receptor 24/7 Gratuito dos Botões (`cloudflare-discord-worker`)
1. Crie uma conta gratuita em https://dash.cloudflare.com e rode no terminal:
   ```powershell
   cd cloudflare-discord-worker
   npx wrangler login
   npx wrangler deploy
   ```
2. Cadastre os segredos no Worker:
   ```powershell
   npx wrangler secret put DISCORD_PUBLIC_KEY
   npx wrangler secret put GITHUB_PAT      # Token GitHub Fine-Grained com permissão Actions: Read & Write
   npx wrangler secret put GITHUB_OWNER    # Seu usuário do GitHub
   npx wrangler secret put GITHUB_REPO     # Nome do seu repositório (ex: twitch-autocliper)
   ```
3. Copie a URL do Worker gerada (`https://twitch-autocliper-discord.<sua-conta>.workers.dev`), volte no **Discord Developer Portal > General Information > Interactions Endpoint URL**, cole a URL e clique em **Save Changes** (o Worker responderá ao `PING` na hora!).

---

## Passo 6: Conectar Instagram Reels e TikTok

### Instagram Reels (Meta Graph API oficial)
1. Sua conta de cortes no Instagram deve estar no modo **Criador de Conteúdo** ou **Profissional** vinculada a uma Página do Facebook.
2. Em https://developers.facebook.com/apps, crie um app tipo *Business*, adicione o produto **Instagram Graph API** e gere um Token de Longa Duração com as permissões `instagram_basic` e `instagram_content_publish`:
   - `IG_ACCOUNT_ID`: ID numérico da conta Instagram Business/Creator
   - `IG_ACCESS_TOKEN`: Token gerado

### TikTok (Content Posting API v2)
1. Acesse https://developers.tiktok.com/, crie um app e adicione o produto **Content Posting API** (escopo `video.upload` e `video.publish`).
2. Com `mode: "inbox"` em `config/profiles.yaml` (padrão), **você não precisa esperar auditoria comercial**: ao aprovar no Discord, o vídeo já cai direto nas notificações/rascunhos do seu app TikTok pronto para publicar com 1 toque!
3. Quando seu app passar na revisão do TikTok para postagem direta, basta trocar para `mode: "direct"` no `config/profiles.yaml`!

---

## Passo 7: Ativar o 24/7 no GitHub Actions + Colab Ultra

1. Suba este projeto para um repositório no seu GitHub.
2. Em **Settings > Secrets and variables > Actions > New repository secret**, adicione:
   - `TWITCH_CLIENT_ID`
   - `TWITCH_CLIENT_SECRET`
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
   - `GDRIVE_TOKEN_JSON`
   - `DISCORD_BOT_TOKEN`
   - `DISCORD_CHANNEL_ID`
   - `IG_ACCOUNT_ID`
   - `IG_ACCESS_TOKEN`
   - `TIKTOK_ACCESS_TOKEN`
3. Pronto! O workflow **Watch Twitch & Generate Cuts (24/7)** já rodará sozinho a cada 15 minutos, e o **Smart Drip Queue Publisher** publicará sua fila aprovada respeitando o intervalo de 1h15!
4. Para usar suas GPUs `L4`/`A100` do **Google AI Ultra**, suba o arquivo `colab/Twitch_AutoClipper_Ultra_GPU.ipynb` no seu Google Colab, ative `Ambiente de execução > Alterar o tipo de ambiente de execução > L4 / A100 GPU` e rode com 1 clique!
