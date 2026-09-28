"""Shared inference infrastructure for the AeroTwin model wrappers.

Provides:
* Weight-file management with descriptive errors that the API layer maps to
  HTTP 503 when ``backend/app/models/weights/`` entries are missing.
* Device resolution (``AEROTWIN_DEVICE``: auto/cuda/cpu, default auto).
* ``MiniUNet`` — the compact encoder/decoder backbone used by all dense
  predictors (replaceable per task when the final trained architectures land).
* Sliding-window tiling with cosine-tapered overlap blending so high-resolution
  UAV imagery never blows GPU/CPU memory (anti-OOM rule).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import settings

logger = logging.getLogger("aerotwin.models")


class ModelWeightsMissingError(Exception):
    """A required ``.pt`` weight file is absent → API maps this to HTTP 503."""


class ModelLoadError(Exception):
    """Weights exist but cannot be loaded/matched → API maps this to HTTP 503."""


def require_weights(filename: str, *, weights_dir: Path | None = None) -> Path:
    """Return the weight path or raise a descriptive 503-mapped error."""
    directory = weights_dir if weights_dir is not None else settings.weights_dir
    path = directory / filename
    if not path.is_file():
        raise ModelWeightsMissingError(
            f"Model weights '{filename}' not found in {directory}. Place the trained "
            f".pt file there and retry; the API returns HTTP 503 until weights are present."
        )
    return path


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve the inference device (env ``AEROTWIN_DEVICE``: auto|cpu|cuda)."""
    choice = (requested or os.environ.get("AEROTWIN_DEVICE", "auto")).lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(choice)


class ConvBlock(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers (the MiniUNet workhorse)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MiniUNet(nn.Module):
    """Compact U-Net backbone producing a per-pixel logit map (B, C, H, W)."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.bottleneck = ConvBlock(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Conv2d(b * 4, b * 2, kernel_size=1)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.Conv2d(b * 2, b, kernel_size=1)
        self.dec1 = ConvBlock(b * 2, b)
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bn = self.bottleneck(self.pool(e2))
        d2 = F.interpolate(self.up2(bn), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(self.up1(d2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def load_checkpoint(module: nn.Module, weights_path: Path, device: torch.device) -> None:
    """Load a state dict into *module*, moving it to *device* in eval mode.

    Raises:
        ModelLoadError: Unreadable file or architecture/weight mismatch.
    """
    try:
        checkpoint = torch.load(str(weights_path), map_location=device, weights_only=True)
    except Exception as exc:
        raise ModelLoadError(f"Failed to read model weights {weights_path}: {exc}") from exc
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ModelLoadError(
            f"Unsupported checkpoint format in {weights_path}: expected a state dict."
        )
    if not state_dict:
        raise ModelLoadError(f"Empty state dict in {weights_path}.")
    try:
        module.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(
            f"Weights/architecture mismatch for {weights_path}: {exc}"
        ) from exc
    module.to(device)
    module.eval()


# ---------------------------------------------------------------------------
# Sliding-window tiling with overlap blending (memory-safe inference)
# ---------------------------------------------------------------------------
def axis_origins(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering *length* with *tile*-sized windows.

    The final window is flush with the axis end; axes shorter than the tile
    yield a single origin of 0 (the slice then clamps naturally).
    """
    if length <= tile:
        return [0]
    step = max(tile - overlap, 1)
    origins = list(range(0, length - tile + 1, step))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


def generate_tile_origins(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """All ``(y, x)`` origins tiling a ``height x width`` image."""
    return [
        (y, x)
        for y in axis_origins(height, tile_size, overlap)
        for x in axis_origins(width, tile_size, overlap)
    ]


class TileBlender:
    """Accumulates overlapping tile predictions with cosine-tapered blending.

    Interior seams get a raised-cosine ramp on each contributing tile so the
    stitched result fades smoothly across overlaps (image borders stay at full
    weight), and :meth:`finalize` divides by the accumulated weight map.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        *,
        ramp: int,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.channels = channels
        self.height = height
        self.width = width
        self.ramp = max(int(ramp), 1)
        self.prediction = np.zeros((channels, height, width), dtype=dtype)
        self.weight = np.zeros((height, width), dtype=dtype)

    @staticmethod
    def _profile(ramp: int) -> np.ndarray:
        """Descending cross-fade profile over *ramp* samples (~1 → ~0, never 0).

        Its mirror sums to exactly 1, so overlapping tiles cross-fade smoothly
        instead of stepping at the seam.
        """
        return (
            0.5 + 0.5 * np.cos(np.pi * (np.arange(ramp, dtype=np.float64) + 0.5) / ramp)
        ).astype(np.float32)

    def _axis_weight(self, start: int, size: int, total: int) -> np.ndarray:
        """Per-axis weights: taper only at seams that touch another tile."""
        weights = np.ones(size, dtype=np.float32)
        ramp = min(self.ramp, size // 2)
        if ramp < 1:
            return weights
        profile = self._profile(ramp)
        if start > 0:  # left seam: ascending into this tile's interior
            weights[:ramp] = profile[::-1]
        if start + size < total:  # right seam: descending toward this tile's end
            weights[-ramp:] = profile
        return weights

    def add(self, prediction: np.ndarray, y: int, x: int) -> None:
        """Accumulate a ``(C, th, tw)`` tile prediction at offset ``(y, x)``."""
        _, tile_h, tile_w = prediction.shape
        weight_map = self._axis_weight(y, tile_h, self.height)[
            :, None
        ] * self._axis_weight(x, tile_w, self.width)[None, :]
        self.prediction[:, y : y + tile_h, x : x + tile_w] += prediction * weight_map[None, :, :]
        self.weight[y : y + tile_h, x : x + tile_w] += weight_map

    def finalize(self) -> np.ndarray:
        """Return the blended ``(C, H, W)`` prediction map."""
        return self.prediction / np.maximum(self.weight, 1e-8)[None, :, :]


def model_predict_proba(
    model: nn.Module,
    image_bgr: np.ndarray,
    *,
    num_classes: int,
    tile_size: int,
    tile_overlap: int,
    device: torch.device,
) -> np.ndarray:
    """Run softmax inference on a BGR frame, tiling when it exceeds *tile_size*.

    Returns a ``(num_classes, H, W)`` float32 probability map. Images that fit
    in one tile take a single forward pass; larger images are processed window
    by window (batch of 1 per tile) so peak memory stays bounded regardless of
    input resolution — the anti-OOM requirement for UAV imagery.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) BGR image, got shape {image_bgr.shape}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]

    def forward(tile_rgb: np.ndarray) -> np.ndarray:
        tensor = (
            torch.from_numpy(np.ascontiguousarray(tile_rgb.transpose(2, 0, 1)))
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probabilities.astype(np.float32, copy=False)

    if height <= tile_size and width <= tile_size:
        return forward(rgb)

    blender = TileBlender(num_classes, height, width, ramp=tile_overlap)
    for y, x in generate_tile_origins(height, width, tile_size, tile_overlap):
        tile = rgb[y : y + tile_size, x : x + tile_size]
        blender.add(forward(tile), y, x)
    return blender.finalize()
