"""Central configuration and pathlib-based path management for AeroTwin AI.

Every filesystem location the backend touches (raw UAV imagery, intermediate
outputs, model weights, and the SQLite database) is resolved here so that no
other module hard-codes absolute paths. ``AEROTWIN_*`` environment variables
override the defaults to keep the app testable and portable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Frozen (PyInstaller) vs. development path resolution
# ---------------------------------------------------------------------------
#: True inside a PyInstaller bundle (``sys.frozen`` is set by the bootloader).
IS_FROZEN: bool = bool(getattr(sys, "frozen", False))
#: Root of the PyInstaller bundle — ``sys._MEIPASS`` points at the ``_internal``
#: directory of an onedir build (or the temp extraction dir of a onefile build).
#: ``None`` during development.
BUNDLE_DIR: Path | None = Path(sys._MEIPASS).resolve() if IS_FROZEN else None


def _default_backend_dir() -> Path:
    """Repository ``backend/`` in dev; the bundle root when frozen.

    PyInstaller keeps the ``app`` package layout, so model weights bundled by
    ``aerotwin_backend.spec`` live at ``<bundle>/app/models/weights`` — the same
    relative location as in the repository. Resolving through ``sys._MEIPASS``
    instead of ``__file__`` keeps production builds free of path errors even
    when ``__file__`` is misleading (onefile temp extraction, zip imports).
    """
    if IS_FROZEN and BUNDLE_DIR is not None:
        return BUNDLE_DIR
    # config.py lives at backend/app/core/config.py:
    #   parents[0] = core/, [1] = app/, [2] = backend/, [3] = repository root
    return Path(__file__).resolve().parents[2]


def _default_project_root() -> Path:
    """Repository root in dev; the bundle's parent when frozen (unused for data)."""
    if IS_FROZEN and BUNDLE_DIR is not None:
        return BUNDLE_DIR.parent
    return Path(__file__).resolve().parents[3]


def _user_data_dir() -> Path:
    """Writable per-user data directory for frozen builds.

    An installed app lives under ``Program Files`` / ``/Applications`` which are
    read-only, so raw videos, frames, and ``aerotwin.db`` go to the per-user
    roaming/app-data location instead (``AEROTWIN_DATA_DIR`` still overrides).
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "AeroTwinAI" / "data"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AeroTwinAI" / "data"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "aerotwin-ai" / "data"


def _default_data_dir() -> Path:
    """``data/`` next to the repository in dev; per-user directory when frozen."""
    if IS_FROZEN:
        return _user_data_dir()
    return _default_project_root() / "data"


def _default_weights_dir() -> Path:
    """Bundled weights inside ``sys._MEIPASS`` when frozen; repo copy in dev."""
    return _default_backend_dir() / "app" / "models" / "weights"


BACKEND_DIR: Path = _default_backend_dir()
PROJECT_ROOT: Path = _default_project_root()

DEFAULT_DATA_DIR: Path = _default_data_dir()
#: Vite dev-server origins plus the ``null`` origin that Chromium sends for the
#: packaged Electron page loaded from ``file://`` — without it the desktop build
#: could not call the sidecar (or read image pixels for the canvas viewers).
DEFAULT_CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,null"


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved from environment variables."""

    app_name: str
    app_version: str
    project_root: Path
    backend_dir: Path
    data_dir: Path
    database_path: Path
    inputs_dir: Path
    raw_images_dir: Path
    orthomosaics_dir: Path
    raw_videos_dir: Path
    processed_dir: Path
    frames_dir: Path
    material_masks_dir: Path
    structural_elements_dir: Path
    crack_maps_dir: Path
    aggregated_dir: Path
    outputs_dir: Path
    weights_dir: Path
    host: str
    port: int
    cors_origins: tuple[str, ...]
    random_seed: int

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings, letting ``AEROTWIN_*`` environment variables win."""
        data_dir = Path(os.environ.get("AEROTWIN_DATA_DIR", DEFAULT_DATA_DIR)).resolve()
        cors_origins = tuple(
            origin.strip()
            for origin in os.environ.get("AEROTWIN_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
            if origin.strip()
        )
        return cls(
            app_name="AeroTwin AI Backend",
            app_version="0.1.0",
            project_root=PROJECT_ROOT,
            backend_dir=BACKEND_DIR,
            data_dir=data_dir,
            database_path=data_dir / "aerotwin.db",
            inputs_dir=data_dir / "inputs",
            raw_images_dir=data_dir / "inputs" / "raw_images",
            orthomosaics_dir=data_dir / "inputs" / "orthomosaics",
            raw_videos_dir=data_dir / "inputs" / "raw_videos",
            processed_dir=data_dir / "processed",
            frames_dir=data_dir / "processed" / "frames",
            material_masks_dir=data_dir / "processed" / "material_masks",
            structural_elements_dir=data_dir / "processed" / "structural_elements",
            crack_maps_dir=data_dir / "processed" / "crack_maps",
            aggregated_dir=data_dir / "processed" / "aggregated",
            outputs_dir=data_dir / "outputs",
            weights_dir=Path(
                os.environ.get("AEROTWIN_WEIGHTS_DIR", _default_weights_dir())
            ).resolve(),
            host=os.environ.get("AEROTWIN_HOST", "127.0.0.1"),
            port=int(os.environ.get("AEROTWIN_PORT", "8000")),
            cors_origins=cors_origins,
            random_seed=int(os.environ.get("AEROTWIN_RANDOM_SEED", "42")),
        )

    def managed_directories(self) -> tuple[Path, ...]:
        """All directories the backend owns (mirrors the project structure reference)."""
        return (
            self.raw_images_dir,
            self.orthomosaics_dir,
            self.raw_videos_dir,
            self.processed_dir / "preprocessed",
            self.frames_dir,
            self.material_masks_dir,
            self.structural_elements_dir,
            self.crack_maps_dir,
            self.aggregated_dir,
            self.outputs_dir / "assessments",
            self.outputs_dir / "reports",
            self.outputs_dir / "visualizations",
            self.weights_dir,
        )

    def ensure_directories(self) -> None:
        """Idempotently create managed directories; existing contents are never touched."""
        for directory in self.managed_directories():
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_frames_run(self, frames_path: str) -> Path:
        """Resolve a frames-run reference (``<stem>/<run_id>``) under data/processed/frames/.

        Raises:
            ValueError: The reference is empty or escapes the frames directory.
            FileNotFoundError: The referenced run directory does not exist.
        """
        relative = frames_path.strip().replace("\\", "/")
        if not relative:
            raise ValueError("frames_path must not be empty")
        root = self.frames_dir.resolve()
        candidate = (root / relative).resolve()
        if candidate == root or not candidate.is_relative_to(root):
            raise ValueError(f"frames_path escapes the frames directory: {frames_path!r}")
        if not candidate.is_dir():
            raise FileNotFoundError(f"Frames run not found: {candidate}")
        return candidate

    def resolve_managed_path(self, root: Path, relative: str = "") -> Path:
        """Resolve *relative* inside a managed *root*, rejecting traversal.

        Used by the artifact API that streams masks/maps/reports to the
        frontend. Accepts ``""`` for the root itself and both slash styles.

        Raises:
            ValueError: The reference escapes *root* (or is absolute).
            FileNotFoundError: The resolved path does not exist.
        """
        cleaned = (relative or "").strip().replace("\\", "/").strip("/")
        for segment in cleaned.split("/"):
            if segment in {"..", "."}:
                raise ValueError(f"Path escapes the managed directory: {relative!r}")
        base = root.resolve()
        candidate = (base / cleaned).resolve() if cleaned else base
        if candidate != base and not candidate.is_relative_to(base):
            raise ValueError(f"Path escapes the managed directory: {relative!r}")
        if not candidate.exists():
            raise FileNotFoundError(f"Path not found: {candidate}")
        return candidate


settings: Settings = Settings.from_env()
