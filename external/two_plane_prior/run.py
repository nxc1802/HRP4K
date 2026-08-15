"""External execution template for Two-Plane Prior on HRP4K."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Two-Plane Prior HRP4K Runner")
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument("--split", default="test", help="Evaluation split")
    parser.add_argument("--weights", required=True, help="Path to checkpoint")
    parser.add_argument("--output", required=True, help="Output canonical prediction JSON")
    args = parser.parse_args()

    print(f"[TPP Adapter] Launching Two-Plane Prior inference on {args.split}...")
    print(f"[TPP Adapter] Ensure the isolated Python 3.8 / PyTorch 1.6 / MMDetection environment is active.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
