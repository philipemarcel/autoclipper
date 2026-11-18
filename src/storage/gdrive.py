from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


class DriveStore:
    """
    Gerenciador unificado do Google Drive (20 TB Google AI Ultra).
    Funciona em 3 modos transparentes:
      1. API Oficial v3 (`GDRIVE_TOKEN_JSON`) -> ideal para GitHub Actions 24/7 na nuvem.
      2. Drive Montado (`GDRIVE_MOUNT_PATH`, ex: `/content/drive/MyDrive/TwitchAutoClipper`) -> ideal no Google Colab Ultra.
      3. Espelho Local (`local_drive_mirror/`) -> fallback automático para testes locais sem credenciais.
    """

    def __init__(
        self,
        root_folder_name: Optional[str] = None,
        mount_path: Optional[str] = None,
        token_json_str: Optional[str] = None,
        fallback_local_dir: Optional[Path] = None,
    ) -> None:
        self.root_folder_name = root_folder_name or os.getenv("GDRIVE_ROOT_FOLDER_NAME", "TwitchAutoClipper")
        env_mount = (mount_path if mount_path is not None else os.getenv("GDRIVE_MOUNT_PATH", "")).strip()
        env_token = (token_json_str if token_json_str is not None else os.getenv("GDRIVE_TOKEN_JSON", "")).strip()

        self.service = None
        self.mode = "local"
        self.local_root: Optional[Path] = None
        self._folder_cache: Dict[str, str] = {}

        if env_token:
            self.service = self._build_drive_service(env_token)
            self.mode = "api"
            logger.info("DriveStore inicializado em modo Google Drive API v3 (%s).", self.root_folder_name)
        elif env_mount:
            self.local_root = Path(env_mount)
            self.local_root.mkdir(parents=True, exist_ok=True)
            self.mode = "mount"
            logger.info("DriveStore inicializado em modo Drive Montado: %s", self.local_root)
        else:
            base = fallback_local_dir or (Path(__file__).resolve().parent.parent.parent / "local_drive_mirror")
            self.local_root = base / self.root_folder_name
            self.local_root.mkdir(parents=True, exist_ok=True)
            self.mode = "local"
            logger.info("DriveStore em modo espelho local: %s", self.local_root)

    @staticmethod
    def _build_drive_service(token_json_str: str):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        info = json.loads(token_json_str)
        creds = Credentials.from_authorized_user_info(info, scopes=DRIVE_SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # ------------------------------------------------------------------
    # Resolução de pastas
    # ------------------------------------------------------------------
    def ensure_folder_path(self, parts: List[str]) -> Tuple[str, str]:
        """
        Garante a árvore `[self.root_folder_name, *parts]` no Google Drive.
        Retorna `(folder_id_ou_path, folder_web_url)`.
        """
        full_parts = [self.root_folder_name] + [p.strip("/") for p in parts if p.strip("/")]

        if self.mode != "api":
            assert self.local_root is not None
            rel = Path(*parts) if parts else Path(".")
            target = self.local_root / rel
            target.mkdir(parents=True, exist_ok=True)
            return str(target), target.as_uri()

        parent_id = "root"
        accum_key = ""
        for part in full_parts:
            accum_key = f"{accum_key}/{part}"
            if accum_key in self._folder_cache:
                parent_id = self._folder_cache[accum_key]
                continue

            escaped = part.replace("'", "\\'")
            query = (
                f"name = '{escaped}' and '{parent_id}' in parents "
                f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            )
            resp = self.service.files().list(q=query, fields="files(id, name, webViewLink)", pageSize=5).execute()
            files = resp.get("files", [])
            if files:
                folder_id = files[0]["id"]
            else:
                meta = {
                    "name": part,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [parent_id],
                }
                created = self.service.files().create(body=meta, fields="id, webViewLink").execute()
                folder_id = created["id"]

            self._folder_cache[accum_key] = folder_id
            parent_id = folder_id

        folder_url = f"https://drive.google.com/drive/folders/{parent_id}"
        return parent_id, folder_url

    # ------------------------------------------------------------------
    # Upload e Download de Arquivos
    # ------------------------------------------------------------------
    def upload_file(
        self,
        local_path: Path,
        folder_parts: List[str],
        remote_filename: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Envia ou atualiza um arquivo em `folder_parts/remote_filename`.
        Retorna `(file_id_ou_path, web_view_url)`.
        """
        filename = remote_filename or local_path.name
        folder_ref, _ = self.ensure_folder_path(folder_parts)

        if self.mode != "api":
            dest = Path(folder_ref) / filename
            if local_path.resolve() != dest.resolve():
                shutil.copy2(local_path, dest)
            return str(dest), dest.as_uri()

        from googleapiclient.http import MediaFileUpload

        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or "application/octet-stream"
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True, chunksize=8 * 1024 * 1024)

        existing_id = self._find_file_in_folder(folder_ref, filename)
        if existing_id:
            updated = (
                self.service.files()
                .update(fileId=existing_id, media_body=media, fields="id, webViewLink")
                .execute()
            )
            file_id = updated["id"]
            url = updated.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"
            return file_id, url

        meta = {"name": filename, "parents": [folder_ref]}
        created = (
            self.service.files()
            .create(body=meta, media_body=media, fields="id, webViewLink")
            .execute()
        )
        file_id = created["id"]
        url = created.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"
        return file_id, url

    def download_file_by_ref(self, file_ref: str, dest_path: Path) -> Path:
        """Baixa um arquivo a partir do ID do Google Drive ou copia se for caminho montado."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode != "api" or Path(file_ref).exists():
            shutil.copy2(Path(file_ref), dest_path)
            return dest_path

        from googleapiclient.http import MediaIoBaseDownload

        request = self.service.files().get_media(fileId=file_ref)
        with open(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return dest_path

    def make_public_download_url(self, file_id: str) -> str:
        """
        Habilita leitura pública por link no arquivo do Drive e retorna a URL HTTPS direta
        (necessária para a Instagram Reels Graph API puxar o vídeo diretamente).
        """
        if self.mode != "api":
            raise RuntimeError(
                "Para publicar no Instagram Reels via API, configure GDRIVE_TOKEN_JSON para gerar URL HTTPS pública."
            )
        self.service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            fields="id",
        ).execute()
        # Endpoint direto compatível com download binário sem página de confirmação
        return f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={self._get_or_omit_api_key()}" if os.getenv("GOOGLE_API_KEY") else f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"

    @staticmethod
    def _get_or_omit_api_key() -> str:
        return os.getenv("GOOGLE_API_KEY", "").strip()

    def _find_file_in_folder(self, folder_id: str, filename: str) -> Optional[str]:
        escaped = filename.replace("'", "\\'")
        query = f"name = '{escaped}' and '{folder_id}' in parents and trashed = false"
        resp = self.service.files().list(q=query, fields="files(id)", pageSize=1).execute()
        files = resp.get("files", [])
        return files[0]["id"] if files else None

    # ------------------------------------------------------------------
    # Estado JSON Atômico (`_state/processed_moments.json`, `_state/drip_queue.json`)
    # ------------------------------------------------------------------
    def load_state_json(self, state_filename: str, default: Any) -> Any:
        folder_ref, _ = self.ensure_folder_path(["_state"])

        if self.mode != "api":
            p = Path(folder_ref) / state_filename
            if not p.exists():
                return default
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        file_id = self._find_file_in_folder(folder_ref, state_filename)
        if not file_id:
            return default

        from googleapiclient.http import MediaIoBaseDownload

        buf = io.BytesIO()
        req = self.service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        raw = buf.getvalue().decode("utf-8").strip()
        return json.loads(raw) if raw else default

    def save_state_json(self, state_filename: str, data: Any) -> None:
        folder_ref, _ = self.ensure_folder_path(["_state"])
        serialized = json.dumps(data, ensure_ascii=False, indent=2)

        if self.mode != "api":
            p = Path(folder_ref) / state_filename
            tmp = p.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(serialized)
            tmp.replace(p)
            return

        from googleapiclient.http import MediaIoBaseUpload

        media = MediaIoBaseUpload(
            io.BytesIO(serialized.encode("utf-8")),
            mimetype="application/json",
            resumable=False,
        )
        existing_id = self._find_file_in_folder(folder_ref, state_filename)
        if existing_id:
            self.service.files().update(fileId=existing_id, media_body=media).execute()
        else:
            meta = {"name": state_filename, "parents": [folder_ref]}
            self.service.files().create(body=meta, media_body=media).execute()
