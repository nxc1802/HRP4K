"""External runner template for RT-DETR v1/v2."""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Official RT-DETR HRP4K Runner")
    parser.add_argument("--config", help="Model configuration YAML")
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument("--split", default="test", help="Split to evaluate")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--output", required=True, help="Output canonical predictions JSON")
    args = parser.parse_args()

    print(f"[RT-DETR] Running inference on {args.split} with {args.weights}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
