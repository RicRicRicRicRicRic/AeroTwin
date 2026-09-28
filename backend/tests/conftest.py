"""Shared test configuration.

Forces ``AEROTWIN_DATA_DIR`` to a throw-away directory BEFORE any app module
is imported, so the test suite never reads or writes the real
``data/aerotwin.db`` or pollutes the raw-input stores.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_DATA_DIR: Path = Path(tempfile.mkdtemp(prefix="aerotwin_test_data_"))
os.environ["AEROTWIN_DATA_DIR"] = str(_TEST_DATA_DIR)

# Model weights are isolated too: tests never read the real
# backend/app/models/weights/ contents (fake fixtures are written here).
_TEST_WEIGHTS_DIR: Path = Path(tempfile.mkdtemp(prefix="aerotwin_test_weights_"))
os.environ["AEROTWIN_WEIGHTS_DIR"] = str(_TEST_WEIGHTS_DIR)
