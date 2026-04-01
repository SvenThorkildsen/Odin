import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


class AnomalyInspector:
    def __init__(self, device: str = "cpu", image_size: int = 256) -> None:
        self.device = torch.device(device)
        self.image_size = image_size
        self.model = FeatureExtractor().to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.memory_global = None
        self.memory_patch = None
        self.threshold = None

    def load_image(self, path: Path) -> Tuple[torch.Tensor, Image.Image]:
        img = Image.open(path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        return x, img

    @torch.no_grad()
    def extract_features(self, image_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        global_features = []
        patch_features = []

        for path in image_paths:
            x, _ = self.load_image(path)
            emb, fmap = self.model(x)

            global_features.append(emb.squeeze(0).cpu().numpy())

            # [1, C, Hf, Wf] -> [Hf*Wf, C]
            patches = fmap.squeeze(0).permute(1, 2, 0).reshape(-1, fmap.shape[1])
            patch_features.append(patches.cpu().numpy())

        global_features_np = np.stack(global_features, axis=0)
        patch_features_np = np.concatenate(patch_features, axis=0)
        return global_features_np, patch_features_np

    @staticmethod
    def pairwise_l2_min(query: np.ndarray, memory: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """
        For each row in query, compute the minimum L2 distance to memory rows.
        Memory batching avoids large RAM spikes.
        """
        q = query.astype(np.float32)
        m = memory.astype(np.float32)

        best = np.full((q.shape[0],), np.inf, dtype=np.float32)

        for start in range(0, m.shape[0], batch_size):
            chunk = m[start : start + batch_size]  # [M, D]
            # cdist result shape: [Q, M]
            dists = np.sqrt(
                np.maximum(
                    1e-12,
                    ((q[:, None, :] - chunk[None, :, :]) ** 2).sum(axis=2),
                )
            )
            best = np.minimum(best, dists.min(axis=1))

        return best

    def train(self, train_folder: Path, model_dir: Path, percentile_threshold: float = 95.0) -> None:
        image_paths = list_images(train_folder)
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {train_folder}")

        print(f"Found {len(image_paths)} training images.")
        if len(image_paths) < 10:
            print("Warning: fewer than 10 training images may give unstable results.")
        if len(image_paths) > 20:
            print("Note: more than 20 images is fine; this method benefits from more good images.")

        memory_global, memory_patch = self.extract_features(image_paths)

        # Training scores on good images only.
        # These are not valid test scores, but can be used to set an initial threshold.
        train_scores = []
        for i in range(memory_global.shape[0]):
            q = memory_global[i : i + 1]
            others = np.delete(memory_global, i, axis=0)
            if others.shape[0] == 0:
                score = 0.0
            else:
                score = float(self.pairwise_l2_min(q, others)[0])
            train_scores.append(score)

        threshold = float(np.percentile(train_scores, percentile_threshold))

        ensure_dir(model_dir)
        np.save(model_dir / "memory_global.npy", memory_global)
        np.save(model_dir / "memory_patch.npy", memory_patch)
        save_json(
            model_dir / "config.json",
            {
                "image_size": self.image_size,
                "threshold": threshold,
                "percentile_threshold": percentile_threshold,
                "num_train_images": len(image_paths),
            },
        )

        print(f"Model saved to: {model_dir}")
        print(f"Initial threshold: {threshold:.6f}")
        print("You can adjust this threshold later in config.json.")

    def load_model(self, model_dir: Path) -> None:
        config = load_json(model_dir / "config.json")
        self.threshold = float(config["threshold"])

        self.memory_global = np.load(model_dir / "memory_global.npy")
        self.memory_patch = np.load(model_dir / "memory_patch.npy")

        print(f"Loaded model from {model_dir}")
        print(f"Threshold: {self.threshold:.6f}")

    @torch.no_grad()
    def score_image(self, image_path: Path) -> Tuple[float, np.ndarray]:
        if self.memory_global is None or self.memory_patch is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        x, _ = self.load_image(image_path)
        emb, fmap = self.model(x)

        global_vec = emb.squeeze(0).cpu().numpy()[None, :]
        patches = fmap.squeeze(0).permute(1, 2, 0).reshape(-1, fmap.shape[1]).cpu().numpy()

        image_score = float(self.pairwise_l2_min(global_vec, self.memory_global)[0])
        patch_scores = self.pairwise_l2_min(patches, self.memory_patch)

        h_feat = fmap.shape[2]
        w_feat = fmap.shape[3]
        heatmap = patch_scores.reshape(h_feat, w_feat)

        return image_score, heatmap

    def save_heatmap_overlay(
        self,
        image_path: Path,
        heatmap: np.ndarray,
        output_path: Path,
        alpha: float = 0.45,
    ) -> None:
        original = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        base = np.array(original).astype(np.float32)

        heatmap_norm = heatmap - heatmap.min()
        if heatmap_norm.max() > 1e-8:
            heatmap_norm = heatmap_norm / heatmap_norm.max()

        heat_img = Image.fromarray((heatmap_norm * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        heat_np = np.array(heat_img).astype(np.float32) / 255.0

        # Simple red overlay without additional libraries
        overlay = base.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] * (1 - alpha) + 255 * heat_np * alpha, 0, 255)
        overlay[..., 1] = np.clip(overlay[..., 1] * (1 - alpha), 0, 255)
        overlay[..., 2] = np.clip(overlay[..., 2] * (1 - alpha), 0, 255)

        Image.fromarray(overlay.astype(np.uint8)).save(output_path)

    def test(
        self,
        test_folder: Path,
        model_dir: Path,
        output_dir: Path,
        save_heatmaps: bool = True,
        threshold_override: float = None,
    ) -> None:
        self.load_model(model_dir)
        if threshold_override is not None:
            self.threshold = float(threshold_override)
            print(f"Using overridden threshold: {self.threshold:.6f}")

        image_paths = list_images(test_folder)
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {test_folder}")

        ensure_dir(output_dir)
        if save_heatmaps:
            ensure_dir(output_dir / "heatmaps")

        results = []
        for path in image_paths:
            score, heatmap = self.score_image(path)
            decision = "FAIL" if score >= self.threshold else "PASS"

            record = {
                "image": str(path),
                "score": score,
                "threshold": self.threshold,
                "decision": decision,
            }
            results.append(record)

            print(f"{path.name}: score={score:.6f} -> {decision}")

            if save_heatmaps:
                heatmap_path = output_dir / "heatmaps" / f"{path.stem}_heatmap.png"
                self.save_heatmap_overlay(path, heatmap, heatmap_path)

        save_json(output_dir / "results.json", {"results": results})
        print(f"Results saved to: {output_dir / 'results.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Good-only visual anomaly inspection")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train on good images only")
    train_parser.add_argument("--train_dir", type=str, required=True, help="Folder of good training images")
    train_parser.add_argument("--model_dir", type=str, required=True, help="Where to save the model")
    train_parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    train_parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=95.0,
        help="Initial threshold percentile based on training-image leave-one-out scores",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )

    test_parser = subparsers.add_parser("test", help="Test images against a trained model")
    test_parser.add_argument("--test_dir", type=str, required=True, help="Folder of images to test")
    test_parser.add_argument("--model_dir", type=str, required=True, help="Folder containing saved model")
    test_parser.add_argument("--output_dir", type=str, required=True, help="Where to save test results")
    test_parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    test_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override. If omitted, uses config.json threshold.",
    )
    test_parser.add_argument(
        "--no_heatmaps",
        action="store_true",
        help="Disable saving heatmap overlays",
    )
    test_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inspector = AnomalyInspector(device=args.device, image_size=args.image_size)

    if args.command == "train":
        inspector.train(
            train_folder=Path(args.train_dir),
            model_dir=Path(args.model_dir),
            percentile_threshold=args.threshold_percentile,
        )

    elif args.command == "test":
        inspector.test(
            test_folder=Path(args.test_dir),
            model_dir=Path(args.model_dir),
            output_dir=Path(args.output_dir),
            save_heatmaps=not args.no_heatmaps,
            threshold_override=args.threshold,
        )


if __name__ == "__main__":
    main()