"""External execution template for AutoFocus on HRP4K."""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="AutoFocus HRP4K Runner")
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument("--split", default="test", help="Evaluation split")
    parser.add_argument("--weights", required=True, help="Path to checkpoint")
    parser.add_argument("--output", required=True, help="Output canonical prediction JSON")
    args = parser.parse_args()

    print(f"[AutoFocus Adapter] Running AutoFocus on {args.split} with {args.weights}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
