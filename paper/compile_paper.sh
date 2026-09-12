#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "🚀 Compiling HRP4K Research Paper with Tectonic..."
echo "=========================================================="

/opt/homebrew/bin/tectonic -o "$SCRIPT_DIR" paper.tex

echo "✅ Compilation successful!"
ls -lh "$SCRIPT_DIR/paper.pdf"
