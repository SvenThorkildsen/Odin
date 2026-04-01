import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class FeatureExtractor(nn.Module):
    """
    ResNet18 feature extractor.
    Returns:
      - global embedding: [B, C]
      - spatial feature map: [B, C, Hf, Wf]
    """

    def __init__(self) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap = self.stem(x)
        emb = self.pool(fmap).flatten(1)
        emb = F.normalize(emb, dim=1)
        fmap = F.normalize(fmap, dim=1)
        return emb, fmap


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class LiveAnomalyDetector:
    def __init__(
        self,
        model_dir: Path,
        device: str = "cpu",
        threshold_override: Optional[float] = None,
        score_smoothing: int = 3,
    ) -> None:
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.model = FeatureExtractor().to(self.device).eval()

        config = load_json(model_dir / "config.json")
        self.image_size = int(config["image_size"])
        self.threshold = float(
            threshold_override if threshold_override is not None else config["threshold"]
        )

        self.memory_global = np.load(model_dir / "memory_global.npy").astype(np.float32)
        self.memory_patch = np.load(model_dir / "memory_patch.npy").astype(np.float32)

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.score_history = deque(maxlen=max(1, score_smoothing))

    @staticmethod
    def pairwise_l2_min(query: np.ndarray, memory: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """
        For each row in query, compute minimum L2 distance to memory rows.
        """
        q = query.astype(np.float32)
        m = memory.astype(np.float32)

        best = np.full((q.shape[0],), np.inf, dtype=np.float32)

        for start in range(0, m.shape[0], batch_size):
            chunk = m[start : start + batch_size]
            dists = np.sqrt(
                np.maximum(
                    1e-12,
                    ((q[:, None, :] - chunk[None, :, :]) ** 2).sum(axis=2),
                )
            )
            best = np.minimum(best, dists.min(axis=1))

        return best

    def preprocess_bgr_frame(
        self,
        frame_bgr: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Returns:
          - input tensor for model
          - ROI frame in BGR
        """
        if roi is not None:
            x, y, w, h = roi
            frame_bgr = frame_bgr[y : y + h, x : x + w]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        x_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return x_tensor, frame_bgr

    @torch.no_grad()
    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[float, float, np.ndarray, str, np.ndarray]:
        """
        Returns:
          raw_score,
          smoothed_score,
          heatmap_small,
          decision,
          roi_frame_bgr
        """
        x_tensor, roi_frame = self.preprocess_bgr_frame(frame_bgr, roi=roi)
        emb, fmap = self.model(x_tensor)

        global_vec = emb.squeeze(0).cpu().numpy()[None, :]
        patches = (
            fmap.squeeze(0)
            .permute(1, 2, 0)
            .reshape(-1, fmap.shape[1])
            .cpu()
            .numpy()
        )

        raw_score = float(self.pairwise_l2_min(global_vec, self.memory_global)[0])
        patch_scores = self.pairwise_l2_min(patches, self.memory_patch)

        h_feat = fmap.shape[2]
        w_feat = fmap.shape[3]
        heatmap_small = patch_scores.reshape(h_feat, w_feat)

        self.score_history.append(raw_score)
        smoothed_score = float(np.mean(self.score_history))

        decision = "FAIL" if smoothed_score >= self.threshold else "PASS"
        return raw_score, smoothed_score, heatmap_small, decision, roi_frame

    @staticmethod
    def heatmap_to_color(
        heatmap_small: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> np.ndarray:
        """
        Converts anomaly heatmap to a colored BGR heatmap.
        """
        heatmap = heatmap_small - heatmap_small.min()
        if heatmap.max() > 1e-8:
            heatmap = heatmap / heatmap.max()

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(
            heatmap_uint8,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        return heatmap_color

    @staticmethod
    def overlay_heatmap(
        image_bgr: np.ndarray,
        heatmap_color_bgr: np.ndarray,
        alpha: float = 0.35,
    ) -> np.ndarray:
        return cv2.addWeighted(heatmap_color_bgr, alpha, image_bgr, 1.0 - alpha, 0.0)

    @staticmethod
    def draw_label(
        image_bgr: np.ndarray,
        decision: str,
        raw_score: float,
        smoothed_score: float,
        threshold: float,
        fps: float,
    ) -> np.ndarray:
        out = image_bgr.copy()

        label = f"{decision} | score={smoothed_score:.4f} | raw={raw_score:.4f} | thr={threshold:.4f} | fps={fps:.1f}"
        color = (0, 0, 255) if decision == "FAIL" else (0, 200, 0)

        cv2.rectangle(out, (10, 10), (min(out.shape[1] - 10, 900), 60), (0, 0, 0), -1)
        cv2.putText(
            out,
            label,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return out


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    # Best-effort settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def parse_roi(roi_text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """
    ROI format: x,y,w,h
    """
    if roi_text is None:
        return None

    parts = [int(v.strip()) for v in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be in format x,y,w,h")
    return tuple(parts)  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time anomaly detection")
    parser.add_argument("--model_dir", type=str, required=True, help="Folder containing saved model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold from config.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--roi", type=str, default=None, help="Optional ROI as x,y,w,h")
    parser.add_argument("--overlay_alpha", type=float, default=0.35, help="Heatmap overlay alpha")
    parser.add_argument("--score_smoothing", type=int, default=3, help="Number of recent scores to average")
    parser.add_argument(
        "--save_failures_dir",
        type=str,
        default=None,
        help="Optional folder to save frames classified as FAIL",
    )
    parser.add_argument(
        "--save_interval_sec",
        type=float,
        default=1.0,
        help="Minimum seconds between saved FAIL images",
    )
    parser.add_argument(
        "--show_original",
        action="store_true",
        help="Show original ROI instead of heatmap overlay",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    roi = parse_roi(args.roi)

    if args.save_failures_dir is not None:
        save_failures_dir = Path(args.save_failures_dir)
        ensure_dir(save_failures_dir)
    else:
        save_failures_dir = None

    detector = LiveAnomalyDetector(
        model_dir=model_dir,
        device=args.device,
        threshold_override=args.threshold,
        score_smoothing=args.score_smoothing,
    )

    print(f"Loaded model from: {model_dir}")
    print(f"Image size: {detector.image_size}")
    print(f"Threshold: {detector.threshold:.6f}")
    print(f"Device: {args.device}")
    if roi is not None:
        print(f"ROI: {roi}")

    cap = open_camera(args.camera, args.width, args.height)

    prev_time = time.time()
    last_save_time = 0.0

    print("Controls:")
    print("  q      Quit")
    print("  +/=    Increase threshold")
    print("  -/_    Decrease threshold")
    print("  s      Save current frame")
    print("  h      Toggle heatmap/original view")

    show_overlay = not args.show_original

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame")
                continue

            now = time.time()
            dt = max(1e-6, now - prev_time)
            fps = 1.0 / dt
            prev_time = now

            raw_score, smoothed_score, heatmap_small, decision, roi_frame = detector.infer_frame(
                frame,
                roi=roi,
            )

            heatmap_color = detector.heatmap_to_color(
                heatmap_small,
                target_width=roi_frame.shape[1],
                target_height=roi_frame.shape[0],
            )

            if show_overlay:
                display = detector.overlay_heatmap(
                    roi_frame,
                    heatmap_color,
                    alpha=args.overlay_alpha,
                )
            else:
                display = roi_frame.copy()

            display = detector.draw_label(
                display,
                decision=decision,
                raw_score=raw_score,
                smoothed_score=smoothed_score,
                threshold=detector.threshold,
                fps=fps,
            )

            if decision == "FAIL" and save_failures_dir is not None:
                if now - last_save_time >= args.save_interval_sec:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    millis = int((now % 1.0) * 1000)
                    out_path = save_failures_dir / f"fail_{timestamp}_{millis:03d}.jpg"
                    cv2.imwrite(str(out_path), display)
                    last_save_time = now

            if roi is not None:
                x, y, w, h = roi
                frame_with_roi = frame.copy()
                cv2.rectangle(frame_with_roi, (x, y), (x + w, y + h), (255, 255, 0), 2)
                preview_small = cv2.resize(frame_with_roi, (640, 360))
                cv2.imshow("Camera Preview", preview_small)

            cv2.imshow("Live Anomaly Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                detector.threshold += 0.01
                print(f"Threshold increased to {detector.threshold:.6f}")
            elif key in (ord("-"), ord("_")):
                detector.threshold = max(0.0, detector.threshold - 0.01)
                print(f"Threshold decreased to {detector.threshold:.6f}")
            elif key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = Path(f"snapshot_{timestamp}.jpg")
                cv2.imwrite(str(out_path), display)
                print(f"Saved snapshot to {out_path}")
            elif key == ord("h"):
                show_overlay = not show_overlay
                print(f"Heatmap overlay: {'ON' if show_overlay else 'OFF'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()