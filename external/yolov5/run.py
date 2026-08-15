"""External runner template for official YOLOv5 repository."""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Official YOLOv5 HRP4K Runner")
    parser.add_argument("--data", required=True, help="Dataset directory or YAML")
    parser.add_argument("--split", default="test", help="Split to evaluate")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--output", required=True, help="Output canonical predictions JSON")
    args = parser.parse_args()

    print(f"[YOLOv5 Official] Running inference on {args.split} with {args.weights}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
