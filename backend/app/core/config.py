"""Central configuration and pathlib-based path management for AeroTwin AI.

Every filesystem location the backend touches (raw UAV imagery, intermediate
outputs, model weights, and the SQLite database) is resolved here so that no
other module hard-codes absolute paths. ``AEROTWIN_*`` environment variables
override the defaults to keep the app testable and portable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# config.py lives at backend/app/core/config.py:
#   parents[0] = core/, [1] = app/, [2] = backend/, [3] = repository root
BACKEND_DIR: Path = Path(__file__).resolve().parents[2]
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"
DEFAULT_CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"


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
    processed_dir: Path
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
            processed_dir=data_dir / "processed",
            outputs_dir=data_dir / "outputs",
            weights_dir=BACKEND_DIR / "app" / "models" / "weights",
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
            self.processed_dir / "preprocessed",
            self.processed_dir / "material_masks",
            self.processed_dir / "structural_elements",
            self.processed_dir / "crack_maps",
            self.outputs_dir / "assessments",
            self.outputs_dir / "reports",
            self.outputs_dir / "visualizations",
            self.weights_dir,
        )

    def ensure_directories(self) -> None:
        """Idempotently create managed directories; existing contents are never touched."""
        for directory in self.managed_directories():
            directory.mkdir(parents=True, exist_ok=True)


settings: Settings = Settings.from_env()
