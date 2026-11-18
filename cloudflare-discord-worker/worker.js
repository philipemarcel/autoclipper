/**
 * Cloudflare Worker Gratuito (24/7) — Receptor do Discord + Site/Termos/Verificação TikTok Production
 *
 * Rotas:
 *  - POST /           : Valida assinatura Ed25519 do Discord e dispara GitHub Actions.
 *  - GET /            : Página oficial do AutoClipper Studio (Web/Desktop URL do TikTok).
 *  - GET /terms       : Terms of Service (TikTok Production).
 *  - GET /privacy     : Privacy Policy (TikTok Production).
 *  - GET /tiktok*.txt : Arquivo de verificação URL Prefix do TikTok Developers.
 */

function hexToUint8Array(hex) {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return bytes;
}

async function verifyDiscordSignature(request, bodyText, publicKeyHex) {
  const signatureHex = request.headers.get("X-Signature-Ed25519");
  const timestamp = request.headers.get("X-Signature-Timestamp");
  if (!signatureHex || !timestamp || !publicKeyHex) return false;

  try {
    const key = await crypto.subtle.importKey(
      "raw",
      hexToUint8Array(publicKeyHex),
      { name: "Ed25519", namedCurve: "Ed25519" },
      false,
      ["verify"]
    );
    const message = new TextEncoder().encode(timestamp + bodyText);
    return await crypto.subtle.verify("Ed25519", key, hexToUint8Array(signatureHex), message);
  } catch {
    return false;
  }
}

const ACTION_LABELS = {
  approve_both: "✅ Aprovado! Enviando para a Fila Espaçada (Instagram Reels + TikTok)...",
  now_both: "⚡ Fura-fila acionado! Publicando AGORA no Reels + TikTok...",
  approve_reels: "📸 Aprovado para a Fila Espaçada (Somente Instagram Reels)...",
  approve_tiktok: "🎵 Aprovado! Enviando para a Fila Espaçada do TikTok...",
  now_tiktok: "⚡ Fura-fila acionado! Enviando AGORA para o TikTok...",
  clean_audio: "🔇 Isolando voz e removendo música de fundo com IA (Demucs)... Um novo preview chegará em instantes!",
  reject: "🗑️ Corte descartado.",
};

function renderHtmlPage(title, heading, bodyHtml) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${title} — AutoClipper Studio</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 760px; margin: 48px auto; padding: 0 20px; line-height: 1.6; color: #1f2937; background: #f9fafb; }
    .card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 32px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    h1 { color: #111827; margin-top: 0; }
    a { color: #6366f1; }
    nav { margin-bottom: 24px; font-size: 14px; }
    nav a { margin-right: 16px; text-decoration: none; font-weight: 600; }
  </style>
</head>
<body>
  <div class="card">
    <nav>
      <a href="/">Home</a>
      <a href="/terms">Terms of Service</a>
      <a href="/privacy">Privacy Policy</a>
    </nav>
    <h1>${heading}</h1>
    ${bodyHtml}
  </div>
</body>
</html>`;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (request.method === "GET" || request.method === "HEAD") {
      // 1. Verificação de arquivo `.txt` exigida pelo TikTok Developers (Verify URL prefix)
      if (env.TIKTOK_VERIFY_FILENAME && env.TIKTOK_VERIFY_CONTENT && path === `/${env.TIKTOK_VERIFY_FILENAME}`) {
        return new Response(env.TIKTOK_VERIFY_CONTENT, {
          status: 200,
          headers: { "Content-Type": "text/plain; charset=utf-8" },
        });
      }

      if (path === "/terms" || path === "/terms/") {
        const html = renderHtmlPage(
          "Terms of Service",
          "Terms of Service",
          `<p><strong>Effective Date:</strong> November 18, 2026</p>
           <p>AutoClipper Studio is an automated video clipping, captioning, and publishing tool that connects authorized creator accounts to publish short-form 9:16 livestream highlights via the official TikTok Content Posting API.</p>
           <p>By authorizing your account through TikTok Login Kit, you confirm that you own or have permission to publish the selected video content and agree to comply with TikTok's Community Guidelines and Terms of Service.</p>
           <p>You may revoke access at any time in your TikTok account settings. Contact: <a href="https://github.com/philipemarcel/autoclipper">github.com/philipemarcel/autoclipper</a>.</p>`
        );
        return new Response(html, { status: 200, headers: { "Content-Type": "text/html; charset=utf-8" } });
      }

      if (path === "/privacy" || path === "/privacy/") {
        const html = renderHtmlPage(
          "Privacy Policy",
          "Privacy Policy",
          `<p><strong>Effective Date:</strong> November 18, 2026</p>
           <p>AutoClipper Studio respects your privacy. When you authorize your TikTok account via OAuth 2.0, we store only the access and refresh tokens strictly required to upload and publish videos you explicitly approve.</p>
           <p>We do not sell, rent, or share user data or tokens with third parties. Media files and metadata are stored privately in the creator's own Google Drive.</p>
           <p>To request deletion of stored tokens, revoke app permissions in TikTok or contact the repository owner at <a href="https://github.com/philipemarcel/autoclipper">github.com/philipemarcel/autoclipper</a>.</p>`
        );
        return new Response(html, { status: 200, headers: { "Content-Type": "text/html; charset=utf-8" } });
      }

      const homeHtml = renderHtmlPage(
        "Home",
        "AutoClipper Studio",
        `<p>Automated short-form 9:16 video clipping, dynamic captioning, and creator approval studio integrated with TikTok Login Kit and Content Posting API.</p>
         <ul>
           <li>Detects high-engagement livestream highlights</li>
           <li>Formats vertical 9:16 cuts with dynamic word-level captions</li>
           <li>Publishes creator-approved clips via TikTok Content Posting API</li>
         </ul>
         <p>Source &amp; Documentation: <a href="https://github.com/philipemarcel/autoclipper">github.com/philipemarcel/autoclipper</a></p>`
      );
      return new Response(homeHtml, { status: 200, headers: { "Content-Type": "text/html; charset=utf-8" } });
    }

    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    const bodyText = await request.text();
    const isValid = await verifyDiscordSignature(request, bodyText, env.DISCORD_PUBLIC_KEY);
    if (!isValid) {
      return new Response("Invalid request signature", { status: 401 });
    }

    const interaction = JSON.parse(bodyText);

    // 1. Handshake obrigatório do Discord Developer Portal
    if (interaction.type === 1) {
      return Response.json({ type: 1 });
    }

    // 2. Comando Slash `/cortar` enviado no chat do Discord (`APPLICATION_COMMAND`)
    if (interaction.type === 2 && interaction.data?.name === "cortar") {
      const opts = {};
      for (const opt of interaction.data.options || []) {
        opts[opt.name] = opt.value;
      }
      const ytUrl = String(opts.url || "").trim();
      const maxCuts = String(opts.quantidade || 3);
      const timeRange = String(opts.trecho || "").trim();
      const username = interaction.member?.user?.username || interaction.user?.username || "admin";

      ctx.waitUntil(
        fetch(
          `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/actions/workflows/discord_action.yml/dispatches`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${env.GITHUB_PAT}`,
              Accept: "application/vnd.github+json",
              "User-Agent": "TwitchAutoClipper-DiscordWorker",
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              ref: env.GITHUB_BRANCH || "main",
              inputs: {
                action: "youtube_cut",
                cut_id: "youtube",
                youtube_url: ytUrl,
                max_cuts: maxCuts,
                time_range: timeRange,
                actor: username,
              },
            }),
          }
        )
      );

      const rangeLabel = timeRange ? `Trecho \`${timeRange}\`` : "Automático (Picos Mais Assistidos / IA)";
      return Response.json({
        type: 4,
        data: {
          content:
            `🎬 **Pedido de cortes recebido por \`${username}\`!**\n` +
            `• **Vídeo/Live:** <${ytUrl}>\n` +
            `• **Quantidade:** até \`${maxCuts}\` corte(s) \`9:16\`\n` +
            `• **Seleção:** ${rangeLabel}\n` +
            `⏳ *Analisando, legendando e renderizando na nuvem... Os cortes prontos chegarão aqui em ~1–3 minutos!*`,
        },
      });
    }

    // 3. Clique em botão interativo (`MESSAGE_COMPONENT`)
    if (interaction.type === 3 && interaction.data?.custom_id) {
      const [action, cutId] = interaction.data.custom_id.split(":");
      const username = interaction.member?.user?.username || interaction.user?.username || "admin";
      const messageId = interaction.message?.id || "";

      // Dispara o GitHub Actions em background usando `ctx.waitUntil` (responde ao Discord em <100ms)
      ctx.waitUntil(
        fetch(
          `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/actions/workflows/discord_action.yml/dispatches`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${env.GITHUB_PAT}`,
              Accept: "application/vnd.github+json",
              "User-Agent": "TwitchAutoClipper-DiscordWorker",
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              ref: env.GITHUB_BRANCH || "main",
              inputs: {
                action: action,
                cut_id: cutId,
                discord_message_id: messageId,
                actor: username,
              },
            }),
          }
        )
      );

      const statusLine = ACTION_LABELS[action] || `Processando ${action}...`;
      const prevContent = interaction.message?.content || "";

      return Response.json({
        type: 7,
        data: {
          content: `${prevContent}\n\n> **Status (${username}):** ${statusLine}`,
          embeds: interaction.message?.embeds || [],
          components: [],
        },
      });
    }

    return Response.json({ type: 4, data: { content: "Interação recebida.", flags: 64 } });
  },
};
