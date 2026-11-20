"""Shared inference infrastructure for the AeroTwin model wrappers.

Provides:
* Weight-file management with descriptive errors that the API layer maps to
  HTTP 503 when ``backend/app/models/weights/`` entries are missing.
* Device resolution (``AEROTWIN_DEVICE``: auto/cuda/cpu, default auto).
* ``MiniUNet`` — the compact encoder/decoder backbone used by all dense
  predictors (replaceable per task when the final trained architectures land).
* Unified ``ModelInterface`` supporting three checkpoint formats:
  - plain ``MiniUNet`` state dicts (backward compatible),
  - TorchScript archives (``torch.jit.load``),
  - Ultralytics YOLO segmentation checkpoints (``ultralytics.YOLO``).
* Sliding-window tiling with cosine-tapered overlap blending so high-resolution
  UAV imagery never blows GPU/CPU memory (anti-OOM rule).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

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


# ---------------------------------------------------------------------------
# Unified model interface
# ---------------------------------------------------------------------------


class ModelInterface(Protocol):
    """Common inference contract for every supported checkpoint format.

    ``predict_proba`` accepts an ``(H, W, 3)`` RGB float image in ``[0, 1]``
    and returns ``(C, H, W)`` float32 class probabilities.
    """

    num_classes: int
    device: torch.device

    def predict_proba(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return ``(C, H, W)`` class probabilities for an RGB float image."""
        ...

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        ...

    def to(self, device: torch.device) -> None:
        """Move the model to *device*."""
        ...


class MiniUNetModel:
    """MiniUNet backbone loaded from a plain state_dict checkpoint (backward compatible)."""

    def __init__(self, num_classes: int, base_channels: int = 16) -> None:
        self.model = MiniUNet(in_channels=3, num_classes=num_classes, base_channels=base_channels)
        self.num_classes = num_classes
        self.device = torch.device("cpu")

    def load_checkpoint(self, weights_path: Path, device: torch.device) -> None:
        """Load a state dict into the MiniUNet backbone."""
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
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ModelLoadError(
                f"Weights/architecture mismatch for {weights_path}: {exc}"
            ) from exc
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict_proba(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run softmax inference on an RGB float [0,1] image tile."""
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1)))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probabilities.astype(np.float32, copy=False)

    def eval(self) -> None:
        self.model.eval()

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)


class TorchScriptModel:
    """TorchScript archive (``torch.jit.script`` / ``torch.jit.trace`` export).

    The wrapped module is expected to return class-index maps (e.g. the output
    of ``torch.argmax``).  These are one-hot encoded into ``(C, H, W)``
    probabilities so the rest of the pipeline stays format-agnostic.
    """

    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.device = device
        self.module: torch.jit.ScriptModule | None = None

    def load_checkpoint(self, weights_path: Path, device: torch.device) -> None:
        self.module = torch.jit.load(str(weights_path), map_location=device)
        self.device = device
        self.module.eval()
        self.module.to(device)

    def predict_proba(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run the TorchScript module and one-hot encode class indices.

        Indices outside ``[0, num_classes)`` (e.g. when the exported model was
        trained with more classes than the local :data:`MATERIAL_CLASSES`)
        are treated as background on a per-pixel basis.
        """
        if self.module is None:
            raise ModelLoadError("TorchScript module has not been loaded.")
        # TorchScript segmentation models are typically exported from backbones
        # trained with ImageNet normalisation; apply it here so the input
        # distribution matches what the model expects.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (image_rgb - mean) / std
        tensor = (
            torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            output = self.module(tensor)
        if isinstance(output, tuple):
            output = output[0]
        indices = output.squeeze(0).squeeze(0).long().cpu().numpy()  # (H, W)
        h, w = indices.shape
        probs = np.zeros((self.num_classes, h, w), dtype=np.float32)
        for c in range(self.num_classes):
            probs[c] = (indices == c).astype(np.float32)
        # Per-pixel fallback: pixels whose predicted index is outside
        # [0, num_classes) receive background probability 1.0.
        pixel_sum = probs.sum(axis=0)  # (H, W): 1.0 where a local class matched
        unmatched = pixel_sum == 0.0
        probs[0] = np.where(unmatched, 1.0, probs[0])
        return probs

    def eval(self) -> None:
        if self.module is not None:
            self.module.eval()

    def to(self, device: torch.device) -> None:
        self.device = device
        if self.module is not None:
            self.module.to(device)


class UltralyticsModel:
    """Ultralytics YOLO segmentation checkpoint.

    Runs ``ultralytics.YOLO`` inference on each tile and converts the
    per-instance segmentation masks into ``(C, H, W)`` class probability maps
    using ``class_map`` (Ultralytics class index → local class index).
    Instances not present in ``class_map`` are treated as background.
    """

    def __init__(
        self,
        num_classes: int,
        class_map: dict[int, int] | None = None,
        device: torch.device | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError as exc:
            raise ModelLoadError(
                "Ultralytics checkpoints require the ``ultralytics`` package; "
                "install it with ``pip install ultralytics``."
            ) from exc
        self.num_classes = num_classes
        self.class_map = class_map or {}
        self.device = device or torch.device("cpu")
        self.model = None
        self.weights_path: Path | None = None

    def load_checkpoint(self, weights_path: Path, device: torch.device) -> None:
        from ultralytics import YOLO

        self.weights_path = weights_path
        self.device = device
        self.model = YOLO(str(weights_path))
        self.model.to(device)

    def predict_proba(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run Ultralytics segmentation inference and build class probability maps."""
        if self.model is None:
            raise ModelLoadError("Ultralytics model has not been loaded.")
        import cv2

        h, w = image_rgb.shape[:2]
        # Ultralytics expects uint8 BGR input.
        image_bgr_uint8 = (cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)

        results = self.model.predict(image_bgr_uint8, verbose=False)
        result = results[0]

        probs = np.zeros((self.num_classes, h, w), dtype=np.float32)

        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()  # (num_instances, H_inf, W_inf)
            classes = result.boxes.cls.cpu().numpy().astype(int)  # (num_instances,)
            confs = result.boxes.conf.cpu().numpy()  # (num_instances,)

            # Resize masks back to the original tile resolution if needed.
            if masks.shape[-2:] != (h, w):
                masks_resized = np.stack(
                    [
                        cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                        for mask in masks
                    ]
                )
            else:
                masks_resized = masks

            for mask, cls_idx, conf in zip(masks_resized, classes, confs):
                local_idx = self.class_map.get(int(cls_idx), 0)
                if local_idx > 0:  # skip background
                    probs[local_idx] = np.maximum(probs[local_idx], mask * float(conf))

        # Background = 1 - max foreground probability (per pixel).
        if self.num_classes > 1:
            foreground_max = probs[1:].max(axis=0)
            probs[0] = np.clip(1.0 - foreground_max, 0.0, 1.0)

        return np.clip(probs, 0.0, 1.0).astype(np.float32, copy=False)

    def eval(self) -> None:
        if self.model is not None:
            self.model.model.eval()

    def to(self, device: torch.device) -> None:
        self.device = device
        if self.model is not None:
            self.model.to(device)


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


def load_model(
    weights_path: Path,
    num_classes: int,
    device: torch.device,
    *,
    class_map: dict[int, int] | None = None,
    base_channels: int = 16,
) -> ModelInterface:
    """Auto-detect checkpoint format and return a ready-to-use model wrapper.

    Detection order:
    1. TorchScript archive (``torch.jit.load``)
    2. Plain state dict (``torch.load`` with ``weights_only=True``)
    3. Ultralytics checkpoint (``torch.load`` with ``weights_only=False`` +
       a dict containing an Ultralytics ``model`` object)

    Raises:
        ModelLoadError: The file is unreadable or its format is unrecognised.
    """
    weights_path = Path(weights_path)

    # 1) TorchScript archive
    try:
        model = TorchScriptModel(num_classes, device)
        model.load_checkpoint(weights_path, device)
        logger.info("Loaded TorchScript model %s (num_classes=%d)", weights_path.name, num_classes)
        return model
    except Exception as exc:
        logger.debug("Not a TorchScript archive: %s: %s", weights_path.name, exc)

    # 2) Plain state dict (MiniUNet)
    try:
        checkpoint = torch.load(str(weights_path), map_location=device, weights_only=True)
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
        model = MiniUNetModel(num_classes, base_channels=base_channels)
        model.model.load_state_dict(state_dict, strict=True)
        model.device = device
        model.model.to(device)
        model.model.eval()
        logger.info(
            "Loaded MiniUNet state_dict %s (num_classes=%d)", weights_path.name, num_classes
        )
        return model
    except ModelLoadError:
        raise
    except Exception as exc:
        logger.debug("Not a plain state dict: %s: %s", weights_path.name, exc)

    # 3) Ultralytics segmentation checkpoint
    try:
        checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model_obj = checkpoint["model"]
            if hasattr(model_obj, "names") and hasattr(model_obj, "yaml"):
                model = UltralyticsModel(num_classes, class_map=class_map, device=device)
                model.load_checkpoint(weights_path, device)
                logger.info(
                    "Loaded Ultralytics model %s (num_classes=%d, class_map=%s)",
                    weights_path.name,
                    num_classes,
                    class_map,
                )
                return model
    except Exception as exc:
        logger.debug("Not an Ultralytics checkpoint: %s: %s", weights_path.name, exc)

    raise ModelLoadError(
        f"Unsupported checkpoint format in {weights_path}: "
        f"expected a MiniUNet state dict, a TorchScript archive, or an "
        f"Ultralytics segmentation checkpoint."
    )


# Keep the original ``load_checkpoint`` signature for backward compatibility.
# It now delegates to ``MiniUNetModel.load_checkpoint``.
def load_checkpoint(module: nn.Module, weights_path: Path, device: torch.device) -> None:
    """Load a state dict into *module*, moving it to *device* in eval mode.

    Thin wrapper kept for backward compatibility; new code should use
    :func:`load_model` which auto-detects the checkpoint format.

    Raises:
        ModelLoadError: Unreadable file or architecture/weight mismatch.
    """
    if isinstance(module, MiniUNetModel):
        module.load_checkpoint(weights_path, device)
        return
    wrapper = MiniUNetModel(module.num_classes if hasattr(module, "num_classes") else 2)
    wrapper.load_checkpoint(weights_path, device)
    module.load_state_dict(wrapper.model.state_dict(), strict=True)
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
    model: ModelInterface,
    image_bgr: np.ndarray,
    *,
    num_classes: int,
    tile_size: int,
    tile_overlap: int,
    device: torch.device,
) -> np.ndarray:
    """Run inference on a BGR frame, tiling when it exceeds *tile_size*.

    Delegates to ``model.predict_proba`` which returns ``(C, H, W)`` float32
    class probabilities. Images that fit in one tile take a single forward
    pass; larger images are processed window by window so peak memory stays
    bounded regardless of input resolution — the anti-OOM requirement for
    UAV imagery.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) BGR image, got shape {image_bgr.shape}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]

    def forward(tile_rgb: np.ndarray) -> np.ndarray:
        return model.predict_proba(tile_rgb)

    if height <= tile_size and width <= tile_size:
        return forward(rgb)

    blender = TileBlender(num_classes, height, width, ramp=tile_overlap)
    for y, x in generate_tile_origins(height, width, tile_size, tile_overlap):
        tile = rgb[y : y + tile_size, x : x + tile_size]
        blender.add(forward(tile), y, x)
    return blender.finalize()
