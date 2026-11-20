"""Material segmentation model wrapper (Phase 3: AI-Based Analysis).

Semantic segmentation of UAV frames into Apple's DMS46 material classes using
the TorchScript DMS46 model.  Weights load from
``backend/app/models/weights/material_model.pt``; when absent, a descriptive
error propagates to the API which answers HTTP 503.

The DMS46 model is a TorchScript archive whose forward pass already performs
``torch.argmax`` internally.  It returns a tuple ``(mask,)`` where ``mask`` is
an ``(1, 1, H, W)`` int64 tensor of categorical class IDs in range 0-45.
No softmax, no additional argmax, and no probability mapping should be
applied — doing so would collapse every non-zero class to background (0).
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..core.config import settings
from ..core.database import update_job_record
from ..schemas.processing import AnalysisRunManifest, JobStatus, MaterialSegmentationRequest
from ..services.uav_preprocessor import list_run_frames
from .base import (
    ModelWeightsMissingError,
    require_weights,
    resolve_device,
)

logger = logging.getLogger("aerotwin.models.material")

MATERIAL_WEIGHTS_FILENAME: str = "material_model.pt"
MATERIAL_CLASSES: tuple[str, ...] = (
    "background",
    "concrete",
    "metal",
    "glass",
    "brick",
    "roofing",
)

DEFAULT_TILE_SIZE: int = 512
DEFAULT_TILE_OVERLAP: int = 32
JOB_TYPE_MATERIAL_SEGMENTATION: str = "material_segmentation"

# DMS46 semantic label IDs (from Apple's taxonomy.json semantic_labels,
# filtered to the 46 classes used by the trained model). Model output class ID k
# (0-45) corresponds to semantic label DMS46_TAXONOMY_IDS[k].
DMS46_TAXONOMY_IDS: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
    24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48,
    49, 50, 51, 52, 53, 56,
]

# DMS46 shortnames indexed by model class ID (0-45), from taxonomy.json.
DMS46_CLASS_NAMES: tuple[str, ...] = (
    "none", "hide", "bone", "brick", "cardboard", "carpet", "ceilingtile",
    "ceramic", "chalkboard", "clutter", "concrete", "cork", "engineeredstone",
    "fabric", "fiberglass", "fire", "foliage", "food", "fur", "gemstone",
    "glass", "hair", "icannottell", "ice", "leather", "liquid", "metal",
    "mirror", "notonlist", "paint", "paper", "pearl", "photograph",
    "clearplastic", "plastic", "rubber", "sand", "sponge", "stone",
    "polishedstone", "styrofoam", "tile", "wallpaper", "water", "wax",
    "whiteboard",
)

# DMS46 sRGB colours indexed by model class ID (0-45), from taxonomy.json.
DMS46_CLASS_COLOURS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0), (188, 188, 137), (0, 188, 0), (188, 188, 0), (0, 0, 188),
    (188, 0, 188), (0, 188, 188), (241, 241, 241), (0, 137, 137), (225, 0, 0),
    (137, 188, 0), (225, 188, 0), (137, 0, 188), (137, 188, 188), (0, 225, 137),
    (225, 188, 188), (0, 137, 0), (188, 137, 0), (137, 225, 188),
    (188, 137, 188), (0, 137, 188), (188, 225, 0), (0, 225, 188),
    (188, 225, 188), (188, 137, 137), (188, 0, 0), (0, 188, 137),
    (188, 188, 225), (137, 0, 137), (225, 0, 137), (137, 188, 137),
    (225, 225, 188), (137, 0, 225), (225, 0, 225), (137, 188, 225),
    (225, 188, 225), (0, 137, 225), (188, 137, 137), (188, 188, 188),
    (188, 225, 137), (137, 0, 0), (188, 137, 225), (0, 225, 225),
    (188, 225, 225), (137, 137, 137),
)

# Explicit aggregation: DMS46 model class ID → local application class index.
# Class ID 0 ("none") is background. Unmapped classes default to background but
# are logged so they are not silently ignored.
DMS46_TO_LOCAL: dict[int, int] = {
    # background (local 0): none, icannottell, notonlist
    0: 0,   # none → background
    22: 0,  # icannottell → background
    28: 0,  # notonlist → background
    # concrete (local 1): concrete, engineeredstone, stone, polishedstone
    10: 1,  # concrete → concrete
    12: 1,  # engineeredstone → concrete
    43: 1,  # stone → concrete
    44: 1,  # polishedstone → concrete
    # metal (local 2): metal, mirror
    26: 2,  # metal → metal
    27: 2,  # mirror → metal
    # glass (local 3): glass
    20: 3,  # glass → glass
    # brick (local 4): brick
    3: 4,   # brick → brick
    # roofing (local 5): tile
    41: 5,  # tile → roofing
    6: 5,   # ceilingtile → roofing
}

# 46-class DMS46 palette from the working Python script.
# These are the exact names and hex colors requested by the user.
DMS46_CLASS_DISPLAY_NAMES: tuple[str, ...] = (
    "Animal skin", "Bone/teeth/horn", "Brickwork", "Cardboard", "Carpet/rug",
    "Ceiling tile", "Ceramic", "Chalkboard/blackboard", "Clutter", "Concrete",
    "Cork/corkboard", "Engineered stone", "Fabric/cloth", "Fire", "Foliage",
    "Food", "Fur", "Gemstone/quartz", "Glass", "Hair", "Ice", "Leather",
    "Metal", "Mirror", "Paint/plaster/enamel", "Paper", "Photograph/painting",
    "Plastic, clear", "Plastic, non-clear", "Rubber/latex", "Sand",
    "Skin/lips", "Sky", "Snow", "Soil/mud", "Stone, natural",
    "Stone, polished", "Tile", "Wallpaper", "Water", "Wax", "Whiteboard",
    "Wicker", "Wood", "Wood, tree", "Asphalt",
)

# Exact HEX colors matching the user's requested palette (0-45).
# Format: (R, G, B) as uint8 values.
DMS46_CLASS_COLOURS: tuple[tuple[int, int, int], ...] = (
    (188, 188, 137),  # 0 Animal skin
    (0, 188, 0),      # 1 Bone/teeth/horn
    (188, 188, 0),    # 2 Brickwork
    (0, 0, 188),      # 3 Cardboard
    (188, 0, 188),    # 4 Carpet/rug
    (0, 188, 188),    # 5 Ceiling tile
    (241, 241, 241),  # 6 Ceramic
    (0, 137, 137),    # 7 Chalkboard/blackboard
    (225, 0, 0),      # 8 Clutter
    (137, 188, 0),    # 9 Concrete
    (225, 188, 0),    # 10 Cork/corkboard
    (137, 0, 188),    # 11 Engineered stone
    (137, 188, 188),  # 12 Fabric/cloth
    (225, 188, 188),  # 13 Fire
    (0, 137, 0),      # 14 Foliage
    (188, 137, 0),    # 15 Food
    (137, 225, 188),  # 16 Fur
    (188, 137, 188),  # 17 Gemstone/quartz
    (0, 137, 188),    # 18 Glass
    (188, 225, 0),    # 19 Hair
    (188, 225, 188),  # 20 Ice
    (137, 137, 0),    # 21 Leather
    (137, 225, 0),    # 22 Metal
    (225, 225, 0),    # 23 Mirror
    (225, 137, 188),  # 24 Paint/plaster/enamel
    (0, 225, 0),      # 25 Paper
    (0, 0, 137),      # 26 Photograph/painting
    (188, 0, 0),      # 27 Plastic, clear
    (0, 188, 137),    # 28 Plastic, non-clear
    (188, 0, 137),    # 29 Rubber/latex
    (0, 0, 225),      # 30 Sand
    (225, 188, 137),  # 31 Skin/lips
    (0, 188, 225),    # 32 Sky
    (188, 188, 225),  # 33 Snow
    (225, 0, 137),    # 34 Soil/mud
    (225, 225, 188),  # 35 Stone, natural
    (137, 0, 225),    # 36 Stone, polished
    (137, 188, 225),  # 37 Tile
    (225, 188, 225),  # 38 Wallpaper
    (0, 137, 225),    # 39 Water
    (188, 137, 137),  # 40 Wax
    (188, 188, 188),  # 41 Whiteboard
    (188, 225, 137),  # 42 Wicker
    (137, 0, 0),      # 43 Wood
    (188, 137, 225),  # 44 Wood, tree
    (137, 137, 137),  # 45 Asphalt
)

_MEAN_255 = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD_255 = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def ensure_model_available(*, weights_dir: Path | None = None) -> Path:
    """Raise :class:`ModelWeightsMissingError` (→ HTTP 503) unless weights exist."""
    return require_weights(MATERIAL_WEIGHTS_FILENAME, weights_dir=weights_dir)


class MaterialSegmenter:
    """Material segmentation over BGR frames using Apple's DMS46 TorchScript model."""

    def __init__(
        self,
        weights_path: Path | None = None,
        *,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        device: str | None = None,
        base_channels: int = 16,
    ) -> None:
        if weights_path is None:
            weights_path = ensure_model_available()
        elif not weights_path.is_file():
            raise ModelWeightsMissingError(
                f"Model weights '{weights_path.name}' not found at {weights_path}. "
                "The API returns HTTP 503 until weights are present."
            )
        self.weights_path = weights_path
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.device = resolve_device(device)
        self.num_classes = 46
        self.model: torch.jit.ScriptModule | None = None
        self._load_torchscript()
        logger.info(
            "MaterialSegmenter ready | weights=%s device=%s num_classes=%d",
            weights_path.name,
            self.device,
            self.num_classes,
        )

    def _load_torchscript(self) -> None:
        """Load the DMS46 TorchScript archive."""
        logger.info("Loading TorchScript model from %s", self.weights_path)
        self.model = torch.jit.load(str(self.weights_path), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        logger.info(
            "TorchScript model loaded | weights=%s device=%s",
            self.weights_path.name,
            self.device,
        )

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess a BGR frame exactly as Apple's DMS46 reference script."""
        orig_h, orig_w = image_bgr.shape[:2]
        logger.debug(
            "Preprocessing | orig_size=(%d, %d) device=%s", orig_w, orig_h, self.device
        )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        new_dim = self.tile_size
        h, w = image_rgb.shape[:2]
        scale_x = float(new_dim) / float(h)
        scale_y = float(new_dim) / float(w)
        scale = min(scale_x, scale_y)
        new_h = math.ceil(scale * h)
        new_w = math.ceil(scale * w)
        pil_img = Image.fromarray(image_rgb).resize((new_w, new_h), Image.LANCZOS)
        image_resized = np.array(pil_img)

        logger.debug(
            "Resized | size=(%d, %d) scale=%.4f interpolation=LANCZOS", new_w, new_h, scale
        )

        image_tensor = (
            torch.from_numpy(image_resized.transpose((2, 0, 1))).float().unsqueeze(0)
        )

        mean = torch.tensor(_MEAN_255, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(_STD_255, dtype=torch.float32).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        logger.debug(
            "Tensor | shape=%s dtype=%s min=%.3f max=%.3f",
            tuple(image_tensor.shape),
            image_tensor.dtype,
            float(image_tensor.min()),
            float(image_tensor.max()),
        )

        return image_tensor.to(self.device), (orig_h, orig_w)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an ``(H, W)`` int64 map of DMS46 class IDs (0-45).

        The output is resized back to the original frame dimensions using
        nearest-neighbour interpolation.
        """
        if self.model is None:
            raise ModelWeightsMissingError(
                f"TorchScript model not loaded from {self.weights_path}."
            )

        input_tensor, (orig_h, orig_w) = self._preprocess(image_bgr)

        with torch.inference_mode():
            output = self.model(input_tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]
        logger.debug(
            "Raw output | type=%s shape=%s dtype=%s",
            type(output).__name__,
            tuple(output.shape),
            output.dtype,
        )

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Expected model output to be a Tensor, got {type(output)}"
            )
        if output.dtype != torch.int64:
            logger.warning(
                "Output dtype is %s, expected int64; casting.", output.dtype
            )
            output = output.long()

        mask_resized = output.squeeze(0).squeeze(0).cpu().numpy().astype(np.int64)

        min_id = int(mask_resized.min())
        max_id = int(mask_resized.max())
        unique_ids = np.unique(mask_resized)
        logger.debug(
            "Mask before resize | shape=%s min=%d max=%d unique_ids=%s",
            mask_resized.shape, min_id, max_id, unique_ids.tolist()[:20],
        )
        if min_id < 0 or max_id > 45:
            logger.error(
                "Invalid DMS46 class IDs: min=%d max=%d (expected 0-45)",
                min_id, max_id,
            )

        mask = cv2.resize(
            mask_resized,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

        logger.debug(
            "Mask after resize | shape=%s min=%d max=%d unique=%s",
            mask.shape, int(mask.min()), int(mask.max()),
            np.unique(mask).tolist()[:20],
        )

        return mask

    def predict_local(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an ``(H, W)`` int64 map of *local* class indices (0-5).

        DMS46 class IDs (0-45) are aggregated to the six application categories
        via :data:`DMS46_TO_LOCAL`.  Classes not in the mapping default to
        background (0) but are logged so they are not silently ignored.
        """
        dms46_mask = self.predict(image_bgr)
        local_mask = np.zeros_like(dms46_mask, dtype=np.int64)

        lut = np.zeros(46, dtype=np.int64)
        for dms46_id, local_id in DMS46_TO_LOCAL.items():
            lut[dms46_id] = local_id
        unmapped = set(range(46)) - set(DMS46_TO_LOCAL.keys())
        if unmapped:
            logger.warning(
                "DMS46 class IDs not in aggregation mapping (→ background): %s",
                sorted(unmapped),
            )
        local_mask = lut[dms46_mask]

        unique, counts = np.unique(local_mask, return_counts=True)
        total = local_mask.size
        dist = {
            MATERIAL_CLASSES[int(uid)]: f"{int(c) / total * 100:.2f}%"
            for uid, c in zip(unique, counts)
        }
        logger.info("Local class distribution | %s", dist)

        unique_vals = np.unique(local_mask)
        logger.info(
            "Saved mask | shape=%s dtype=%s unique=%s min=%d max=%d",
            local_mask.shape, local_mask.dtype, unique_vals.tolist(),
            int(local_mask.min()), int(local_mask.max()),
        )

        return local_mask

    def predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Sequential per-frame inference keeps peak memory at one tile-batch."""
        return [self.predict(image) for image in images]


def run_material_segmentation_job(
    job_id: str,
    frames_run_dir: Path,
    params: MaterialSegmentationRequest,
    *,
    output_root: Path | None = None,
    db_path: Path | None = None,
) -> None:
    """Background task: segment every frame of a run into material masks.

    Writes ``mask_<frame_id>.png`` (uint8 class indices) plus a reproducibility
    ``manifest.json`` under ``data/processed/material_masks/<stem>/job_<id8>/``
    and records status/metrics in ``aerotwin.db``. Never raises — failures land
    on the job row; partial output directories are removed.
    """
    update_job_record(job_id, {"status": JobStatus.RUNNING.value}, db_path)
    started_at = datetime.now(timezone.utc)
    timer = time.perf_counter()
    root = output_root if output_root is not None else settings.material_masks_dir
    out_dir = root / frames_run_dir.parent.name / f"job_{job_id[:8]}"
    try:
        frames = list_run_frames(frames_run_dir)
        segmenter = MaterialSegmenter(tile_size=params.tile_size or DEFAULT_TILE_SIZE)
        out_dir.mkdir(parents=True, exist_ok=False)
        frame_times: list[float] = []
        for frame_path in frames:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Unreadable frame image: {frame_path.name}")
            frame_timer = time.perf_counter()
            mask = segmenter.predict(image)   # Raw DMS46 class IDs 0-45
            frame_times.append(time.perf_counter() - frame_timer)
            mask_path = out_dir / f"mask_{frame_path.stem.removeprefix('frame_')}.png"
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8)):
                raise RuntimeError(f"Failed to write mask: {mask_path}")
        processing_time = time.perf_counter() - timer
        completed_at = datetime.now(timezone.utc)
        manifest = AnalysisRunManifest(
            job_id=job_id,
            job_type=JOB_TYPE_MATERIAL_SEGMENTATION,
            frames_run=frames_run_dir.name,
            frames_run_path=str(frames_run_dir),
            frames_processed=len(frames),
            params=params.model_dump(mode="json"),
            class_names=list(DMS46_CLASS_DISPLAY_NAMES),
            weights_file=segmenter.weights_path.name,
            device=str(segmenter.device),
            tile_size=segmenter.tile_size,
            torch_version=torch.__version__,
            opencv_version=cv2.__version__,
            processing_time_seconds=processing_time,
            started_at=started_at,
            completed_at=completed_at,
        )
        (out_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        metrics = {
            "frames_processed": len(frames),
            "mean_frame_inference_seconds": float(np.mean(frame_times)) if frame_times else 0.0,
            "total_inference_seconds": float(np.sum(frame_times)) if frame_times else 0.0,
            "device": str(segmenter.device),
            "class_names": list(DMS46_CLASS_DISPLAY_NAMES),
        }
        update_job_record(
            job_id,
            {
                "status": JobStatus.COMPLETED.value,
                "output_dir": str(out_dir),
                "frames_written": len(frames),
                "total_video_frames": len(frames),
                "processing_time_seconds": processing_time,
                "metrics_json": json.dumps(metrics),
                "completed_at": completed_at.isoformat(),
            },
            db_path,
        )
        logger.info(
            "Material segmentation job %s finished: %d frames in %.3fs -> %s",
            job_id,
            len(frames),
            processing_time,
            out_dir,
        )
    except Exception as exc:  # noqa: BLE001 — job boundary: everything lands on the row
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        update_job_record(
            job_id,
            {
                "status": JobStatus.FAILED.value,
                "error": str(exc),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            db_path,
        )
        logger.exception("Material segmentation job %s failed", job_id)
