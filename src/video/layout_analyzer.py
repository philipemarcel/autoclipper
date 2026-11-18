from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.models import EditingConfig

logger = logging.getLogger(__name__)


@dataclass
class CropRect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class LayoutSpec:
    mode: str  # "split_screen" | "face_track" | "blur_center"
    src_width: int
    src_height: int
    webcam_crop: Optional[CropRect] = None
    gameplay_crop: Optional[CropRect] = None
    face_track_crop: Optional[CropRect] = None


class LayoutAnalyzer:
    """
    Analisa frames do corte para escolher automaticamente o melhor layout vertical 9:16 (1080x1920):
      1. `split_screen`: quando detecta webcam consistente num canto/borda (rosto em cima + tela embaixo).
      2. `face_track`: quando o criador está em câmera cheia / Just Chatting (corte 9:16 centralizado no rosto).
      3. `blur_center`: quando não há rosto (jogo tela cheia com fundo desfocado dinâmico).
    """

    def __init__(self, editing_cfg: EditingConfig) -> None:
        self.cfg = editing_cfg

    def analyze(
        self,
        video_path: Path,
        start_sec: float,
        end_sec: float,
        num_samples: int = 18,
    ) -> LayoutSpec:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo para análise visual: {video_path}")

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        # Se o vídeo de entrada já for vertical (9:16), mantém tela cheia
        if src_h > src_w:
            cap.release()
            return LayoutSpec(
                mode="face_track",
                src_width=src_w,
                src_height=src_h,
                face_track_crop=CropRect(0, 0, src_w, src_h),
            )

        # Override manual de layout no `profiles.yaml`
        forced_mode = (self.cfg.layout_mode or "auto").lower().strip()
        if forced_mode == "blur_center":
            cap.release()
            return LayoutSpec(mode="blur_center", src_width=src_w, src_height=src_h)

        if forced_mode == "split_screen" and self.cfg.fixed_webcam_box and len(self.cfg.fixed_webcam_box) == 4:
            cap.release()
            nx, ny, nw, nh = self.cfg.fixed_webcam_box
            cam_rect = self._clamp_rect(int(nx * src_w), int(ny * src_h), int(nw * src_w), int(nh * src_h), src_w, src_h)
            gp_rect = self._compute_gameplay_crop(src_w, src_h, cam_rect)
            return LayoutSpec(
                mode="split_screen",
                src_width=src_w,
                src_height=src_h,
                webcam_crop=cam_rect,
                gameplay_crop=gp_rect,
            )

        # Amostragem uniforme de frames no trecho selecionado
        span = max(1.0, end_sec - start_sec)
        timestamps = [start_sec + (span * (i + 0.5) / num_samples) for i in range(num_samples)]
        all_faces: List[Tuple[float, float, float, float]] = []  # (cx_norm, cy_norm, w_norm, h_norm)

        detector_fn, cleanup_fn = self._init_face_detector()
        try:
            for ts in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                faces = detector_fn(frame)
                for f in faces:
                    all_faces.append(f)
        finally:
            cleanup_fn()
            cap.release()

        min_hits = max(3, int(num_samples * 0.22))
        if len(all_faces) < min_hits:
            logger.info("Auto-Layout: poucos rostos (%d) -> usando `blur_center`.", len(all_faces))
            return LayoutSpec(mode="blur_center", src_width=src_w, src_height=src_h)

        # 1. Procura primeiro por Webcam de Canto/Lateral (rosto pequeno/médio h <= 0.27 nas laterais cx < 0.38 ou cx > 0.62)
        # Isola das caras de personagens de jogo que aparecem no meio da tela!
        edge_cam_faces = [
            f for f in all_faces if (0.03 <= f[3] <= 0.27) and (f[0] < 0.38 or f[0] > 0.62)
        ]
        if len(edge_cam_faces) >= min_hits:
            edge_arr = np.array(edge_cam_faces, dtype=np.float64)
            # Separa lado esquerdo vs lado direito para achar o canto dominante da webcam
            left_faces = edge_arr[edge_arr[:, 0] < 0.5]
            right_faces = edge_arr[edge_arr[:, 0] >= 0.5]
            dom_arr = left_faces if len(left_faces) >= len(right_faces) else right_faces

            if len(dom_arr) >= min_hits:
                med_cx, med_cy, med_w, med_h = np.median(dom_arr, axis=0)
                # Filtra outliers dentro do quadrante da webcam
                inliers = dom_arr[
                    (np.abs(dom_arr[:, 0] - med_cx) < 0.12) & (np.abs(dom_arr[:, 1] - med_cy) < 0.14)
                ]
                if len(inliers) >= min_hits:
                    med_cx, med_cy, med_w, med_h = np.median(inliers, axis=0)
                    cam_rect = self._compute_webcam_crop_from_face(med_cx, med_cy, med_w, med_h, src_w, src_h)
                    gp_rect = self._compute_gameplay_crop(src_w, src_h, cam_rect)
                    logger.info(
                        "Auto-Layout: webcam lateral detectada em (cx=%.2f, cy=%.2f, h=%.2f) -> `split_screen`.",
                        med_cx,
                        med_cy,
                        med_h,
                    )
                    return LayoutSpec(
                        mode="split_screen",
                        src_width=src_w,
                        src_height=src_h,
                        webcam_crop=cam_rect,
                        gameplay_crop=gp_rect,
                    )

        # 2. Verifica Câmera Cheia (Just Chatting / IRL): exige rosto GRANDE (h > 0.27)
        large_faces = [f for f in all_faces if f[3] > 0.27]
        if len(large_faces) >= min_hits:
            lg_arr = np.array(large_faces, dtype=np.float64)
            med_cx, med_cy, med_w, med_h = np.median(lg_arr, axis=0)
            crop_h = src_h
            crop_w = int(round(src_h * (9.0 / 16.0)))
            crop_w = min(src_w, crop_w - (crop_w % 2))
            center_x = int(round(med_cx * src_w))
            crop_x = max(0, min(src_w - crop_w, center_x - crop_w // 2))
            crop_x -= crop_x % 2

            logger.info("Auto-Layout: câmera cheia detectada (h=%.2f) -> `face_track` (x=%d).", med_h, crop_x)
            return LayoutSpec(
                mode="face_track",
                src_width=src_w,
                src_height=src_h,
                face_track_crop=CropRect(x=crop_x, y=0, w=crop_w, h=crop_h),
            )

        # 3. Caso contrário (ex: apenas rostos pequenos de NPCs no centro do jogo sem webcam fixa), usa `blur_center`
        logger.info("Auto-Layout: gameplay sem webcam fixa ou câmera cheia -> `blur_center`.")
        return LayoutSpec(mode="blur_center", src_width=src_w, src_height=src_h)

    def _compute_webcam_crop_from_face(
        self,
        cx_norm: float,
        cy_norm: float,
        fw_norm: float,
        fh_norm: float,
        src_w: int,
        src_h: int,
    ) -> CropRect:
        """
        Enquadra o close-up da webcam na proporção 1080:680 (~1.588) do painel superior do Split-Screen,
        garantindo que o recorte nunca vaze para fora da borda lateral/inferior da webcam.
        """
        target_aspect = 1080.0 / 680.0
        cx_px = int(round(cx_norm * src_w))
        cy_px = int(round((cy_norm + fh_norm * 0.04) * src_h))

        box_h = int(round(min(0.22, max(0.145, fh_norm * 1.52)) * src_h))
        box_w = int(round(box_h * target_aspect))

        # Como o streamer fica no centro da própria webcam, se a webcam está na esquerda (cx < 0.32),
        # encostamos mais à esquerda; se está na direita (cx > 0.68), ancoramos mais à direita para não pegar jogo na borda interna!
        if cx_norm < 0.32:
            max_safe_w = max(180, int(round(cx_px * 1.78)))
            if box_w > max_safe_w:
                box_w = max_safe_w
                box_h = int(round(box_w / target_aspect))
            x = max(0, min(src_w - box_w, int(round(cx_px - box_w * 0.56))))
        elif cx_norm > 0.68:
            dist_right = src_w - cx_px
            max_safe_w = max(180, int(round(dist_right * 1.78)))
            if box_w > max_safe_w:
                box_w = max_safe_w
                box_h = int(round(box_w / target_aspect))
            x = max(0, min(src_w - box_w, int(round(cx_px - box_w * 0.44))))
        else:
            x = max(0, min(src_w - box_w, cx_px - box_w // 2))

        y = max(0, min(src_h - box_h, cy_px - box_h // 2))
        return self._clamp_rect(x, y, box_w, box_h, src_w, src_h)

    def _compute_gameplay_crop(self, src_w: int, src_h: int, cam_rect: CropRect) -> CropRect:
        """
        Recorta o centro do jogo preservando 100% da altura (`0..src_h`) e ~65% da largura central (`aspect ~1.16`)
        para manter a legenda de tradução inferior do jogo inteira e legível acima dos botões do TikTok.
        """
        target_aspect = 1080.0 / 930.0
        gp_h = src_h
        gp_w = int(round(gp_h * target_aspect))
        gp_w = min(src_w, gp_w - (gp_w % 2))

        # Centraliza exatamente no meio horizontal (onde ficam as legendas de tradução do jogo)
        ideal_center_x = src_w // 2
        gp_x = max(0, min(src_w - gp_w, ideal_center_x - gp_w // 2))
        return self._clamp_rect(gp_x, 0, gp_w, gp_h, src_w, src_h)

    @staticmethod
    def _clamp_rect(x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> CropRect:
        w = max(64, min(max_w, w - (w % 2)))
        h = max(64, min(max_h, h - (h % 2)))
        x = max(0, min(max_w - w, x - (x % 2)))
        y = max(0, min(max_h - h, y - (y % 2)))
        return CropRect(x=x, y=y, w=w, h=h)

    @staticmethod
    def _init_face_detector():
        """
        Hierarquia de detectores faciais compatível com Python 3.10–3.14 e OpenCV 4.x/5.x:
          1. MediaPipe FaceDetection (se instalado)
          2. OpenCV `FaceDetectorYN` (YuNet DNN leve de ~230 KB — altíssima precisão em webcams gamer)
          3. OpenCV `CascadeClassifier` (legado OpenCV 4.x) ou fallback seguro `blur_center` offline.
        """
        try:
            import mediapipe as mp  # type: ignore

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.52)

                def _detect_mp(bgr_frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
                    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    res = mp_fd.process(rgb)
                    out: List[Tuple[float, float, float, float]] = []
                    if res.detections:
                        for det in res.detections:
                            bb = det.location_data.relative_bounding_box
                            w_n = max(0.01, float(bb.width))
                            h_n = max(0.01, float(bb.height))
                            cx_n = float(bb.xmin) + w_n / 2.0
                            cy_n = float(bb.ymin) + h_n / 2.0
                            if 0.0 <= cx_n <= 1.0 and 0.0 <= cy_n <= 1.0:
                                out.append((cx_n, cy_n, w_n, h_n))
                    return out

                return _detect_mp, mp_fd.close
        except Exception:
            pass

        # 2. OpenCV YuNet (`FaceDetectorYN` — suportado no OpenCV 4.6+ e OpenCV 5.x)
        if hasattr(cv2, "FaceDetectorYN"):
            model_dir = Path(__file__).resolve().parent.parent.parent / "config" / "models"
            model_path = model_dir / "face_detection_yunet_2023mar.onnx"
            if not model_path.exists():
                try:
                    import urllib.request

                    model_dir.mkdir(parents=True, exist_ok=True)
                    yunet_url = (
                        "https://github.com/opencv/opencv_zoo/raw/main/models/"
                        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
                    )
                    urllib.request.urlretrieve(yunet_url, str(model_path))
                except Exception as exc:
                    logger.warning("Download YuNet ONNX indisponível (%s).", exc)

            if model_path.exists() and model_path.stat().st_size > 50_000:
                detector = cv2.FaceDetectorYN.create(
                    str(model_path),
                    "",
                    (640, 360),
                    0.55,
                    0.3,
                    5000,
                )

                def _detect_yunet(bgr_frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
                    h_img, w_img = bgr_frame.shape[:2]
                    resized = cv2.resize(bgr_frame, (640, 360))
                    detector.setInputSize((640, 360))
                    _, faces = detector.detect(resized)
                    out: List[Tuple[float, float, float, float]] = []
                    if faces is not None:
                        for row in faces:
                            rx, ry, rw, rh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                            cx_n = (rx + rw / 2.0) / 640.0
                            cy_n = (ry + rh / 2.0) / 360.0
                            w_n = rw / 640.0
                            h_n = rh / 360.0
                            if 0.0 <= cx_n <= 1.0 and 0.0 <= cy_n <= 1.0 and w_n > 0.01 and h_n > 0.01:
                                out.append((cx_n, cy_n, w_n, h_n))
                    return out

                return _detect_yunet, lambda: None

        # 3. Fallback legado OpenCV 4.x ou lista vazia segura
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(cascade_path)

            def _detect_cv(bgr_frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
                h_img, w_img = bgr_frame.shape[:2]
                gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                rects = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(36, 36))
                return [
                    ((rx + rw / 2.0) / w_img, (ry + rh / 2.0) / h_img, rw / w_img, rh / h_img)
                    for rx, ry, rw, rh in rects
                ]

            return _detect_cv, lambda: None

        return (lambda _frame: []), (lambda: None)
