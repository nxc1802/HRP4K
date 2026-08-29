#!/usr/bin/env python3
"""Entry-point script for training and evaluating Raw-4K Shallow Scout (MobileNetV3-Small Stem + Stage 1).

Usage:
  # Smoke test (2-4 samples, 1 epoch)
  python train_raw4k_scout.py --smoke --output-dir outputs/scout_raw4k_smoke

  # Full training on raw 4K
  python train_raw4k_scout.py --epochs 30 --batch-size 4 --device auto
"""

import sys
from pathlib import Path

# Add src to sys.path
repo_root = Path(__file__).resolve().parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hrp4k.training.raw4k_shallow_scout import main

if __name__ == "__main__":
    main()
