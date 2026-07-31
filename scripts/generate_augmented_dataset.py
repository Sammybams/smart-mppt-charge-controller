#!/usr/bin/env python3
"""Generate physics-guided 30 W samples using lux as the input."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smart_mppt.augmentation import main


if __name__ == "__main__":
    raise SystemExit(main())
