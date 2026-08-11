# Script to download and extract HRP4K dataset from HuggingFace
import os
import sys
import zipfile
import urllib.request
import argparse
from pathlib import Path

HF_DATASET_URL = "https://huggingface.co/datasets/Cuong2004/HRP4K/resolve/main/HRP4K.zip"

def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[DOWNLOADING] {mb_downloaded:.1f} / {mb_total:.1f} MB ({percent:.1f}%)")
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Download HRP4K dataset from HuggingFace")
    parser.add_argument("--dest", type=str, default="./HRP4K", help="Destination folder for dataset")
    parser.add_argument("--url", type=str, default=HF_DATASET_URL, help="HuggingFace dataset zip URL")
    args = parser.parse_args()

    dest_dir = Path(args.dest).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "HRP4K.zip"

    train_img_dir = dest_dir / "HRP4K" / "train" / "images"
    test_img_dir = dest_dir / "HRP4K" / "test" / "images"

    if train_img_dir.exists() and len(list(train_img_dir.glob("*.*"))) >= 2286:
        print(f"[EXISTS] Dataset already extracted at {dest_dir / 'HRP4K'}")
        print(f" - Train images: {len(list(train_img_dir.glob('*.*')))}")
        print(f" - Test images: {len(list(test_img_dir.glob('*.*')))}")
        return

    if not zip_path.exists() or zip_path.stat().st_size < 1e6:
        print(f"[INFO] Downloading HRP4K zip from {args.url} to {zip_path}...")
        urllib.request.urlretrieve(args.url, zip_path, download_progress_hook)
        print("\n[DOWNLOAD COMPLETE]")

    print(f"[INFO] Extracting {zip_path} to {dest_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    print("[SUCCESS] Dataset extraction complete!")
    if train_img_dir.exists():
        print(f" - Train images: {len(list(train_img_dir.glob('*.*')))}")
    if test_img_dir.exists():
        print(f" - Test images: {len(list(test_img_dir.glob('*.*')))}")

if __name__ == "__main__":
    main()
