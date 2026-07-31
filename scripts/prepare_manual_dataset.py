#!/usr/bin/env python3
"""Prepare the Lagos 30 W field collection for model training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smart_mppt.manual_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
