/**
 * Cloudflare Worker Gratuito (24/7) — Receptor dos Botões de 1 Clique do Discord
 *
 * Responsabilidades:
 *  1. Valida a assinatura criptográfica Ed25519 exigida pelo Discord (`X-Signature-Ed25519`).
 *  2. Responde ao `PING` de verificação do Portal Discord Developer (`type === 1`).
 *  3. Ao clicar em qualquer botão (`Aprovar Fila`, `Postar Agora`, `Só Reels`, `Só TikTok`, `Limpar Música IA`, `Descartar`),
 *     dispara imediatamente o workflow `discord_action.yml` no GitHub Actions e atualiza o status no chat do Discord!
 *
 * Variáveis Secretas no Cloudflare Worker (`npx wrangler secret put ...`):
 *  - DISCORD_PUBLIC_KEY  : Public Key do seu aplicativo no Discord Developer Portal
 *  - GITHUB_PAT          : Fine-grained Personal Access Token com permissão `Actions: Read and write`
 *  - GITHUB_OWNER        : Seu usuário/organização no GitHub
 *  - GITHUB_REPO         : Nome do repositório (ex: `twitch-autocliper`)
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
  approve_tiktok: "🎵 Aprovado para a Fila Espaçada (Somente TikTok)...",
  clean_audio: "🔇 Isolando voz e removendo música de fundo com IA (Demucs)... Um novo preview chegará em instantes!",
  reject: "🗑️ Corte descartado.",
};

export default {
  async fetch(request, env, ctx) {
    if (request.method !== "POST") {
      return new Response("Twitch AutoClipper Discord Worker Online.", { status: 200 });
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

    // 2. Clique em botão interativo (`MESSAGE_COMPONENT`)
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

      // `type: 7` atualiza a mensagem original desativando os botões já clicados (evita clique duplo)
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
