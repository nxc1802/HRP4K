"""All-in-one script for synchronous training of Raw-4K Shallow Scout on Marimo GPU.

Workflow (Single Step):
1. Setup environment & site-packages
2. Clone/Pull HRP4K repo & verify/unpack dataset
3. Run train_raw4k_scout synchronously on GPU Blackwell (batch_size=16, ram_cache=True)
4. Upload all checkpoints, metrics, and reports to Hugging Face Hub Cuong2004/HRP4K
"""

import os
import sys
import time
import json
import zipfile
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------------------------------
# 1. Environment & Site-Packages Setup
# ------------------------------------------------------------------------------
print("=" * 80)
print("🚀 [1/4] Setting up Environment, Site-Packages & Repository...")
print("=" * 80)

for sp in [
    "/usr/local/lib/python3.13/site-packages",
    "/marimo/HRP4K/src",
    "/marimo/HRP4K",
]:
    if sp not in sys.path and os.path.exists(sp):
        sys.path.insert(0, sp)

# Load environment variables
dotenv_path = Path("/marimo/HRP4K/.env")
if not dotenv_path.is_file() and Path(".env").is_file():
    dotenv_path = Path(".env")

if dotenv_path.is_file():
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            os.environ.setdefault(k, v)

# Clone or pull repo
repo_dir = Path("/marimo/HRP4K")
if not repo_dir.exists():
    print("Cloning repository https://github.com/nxc1802/HRP4K.git ...")
    subprocess.run(["git", "clone", "https://github.com/nxc1802/HRP4K.git", str(repo_dir)], check=True)
else:
    print("Repository /marimo/HRP4K already exists. Pulling latest updates...")
    subprocess.run(["git", "-C", str(repo_dir), "pull"], check=False)

# Force reload hrp4k modules in case they were cached
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("hrp4k"):
        del sys.modules[mod_name]

# Send initial toast in Marimo if marimo is imported
try:
    import marimo as mo
    mo.status.toast("🚀 Bắt đầu quy trình 1-bước: Huấn luyện Raw-4K Scout trực tiếp trên Blackwell GPU!")
except Exception:
    pass

# ------------------------------------------------------------------------------
# 2. Data Verification & Fast Parallel Unpack
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("📦 [2/4] Verifying & Unpacking Dataset (HRP4K.zip)...")
print("=" * 80)

data_dir = repo_dir
train_json = data_dir / "train.json"
train_img_dir = data_dir / "train" / "images"
zip_path = data_dir / "HRP4K.zip"

if not train_json.is_file() or not train_img_dir.is_dir() or len(list(train_img_dir.glob("*.jpg"))) < 4000:
    if zip_path.is_file():
        print(f"Extracting {zip_path} with 16 parallel worker threads...")
        t0 = time.time()
        with zipfile.ZipFile(zip_path, "r") as zf:
            infolist = zf.infolist()
            total = len(infolist)
            
            # Extract JSON annotations first
            for j_name in ["HRP4K/train.json", "HRP4K/valid.json", "HRP4K/test.json"]:
                try:
                    zf.extract(j_name, "/marimo")
                except Exception:
                    pass

            def worker_chunk(chunk_infos):
                with zipfile.ZipFile(zip_path, "r") as worker_zf:
                    for info in chunk_infos:
                        dest = os.path.join("/marimo", info.filename)
                        if os.path.exists(dest) and not info.is_dir():
                            if os.path.getsize(dest) == info.file_size:
                                continue
                        worker_zf.extract(info, "/marimo")

            num_threads = 16
            chunk_size = (total + num_threads - 1) // num_threads
            chunks = [infolist[i:i+chunk_size] for i in range(0, total, chunk_size)]
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                list(executor.map(worker_chunk, chunks))
        print(f"✅ Parallel extraction completed in {time.time() - t0:.2f}s")
    else:
        print("Dataset zip not found locally, will auto-download via ensure_dataset.")
else:
    print(f"✅ Dataset already fully extracted ({len(list(train_img_dir.glob('*.jpg')))} train images ready).")

# ------------------------------------------------------------------------------
# 3. Synchronous Training on GPU Blackwell
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("⚡ [3/4] Launching Synchronous Training on NVIDIA RTX PRO 6000 Blackwell...")
print("=" * 80)

import torch
from hrp4k.training.raw4k_shallow_scout import train_raw4k_scout
from hrp4k.infra.upload import upload_to_hf

print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print(f"Config: Epochs=30 | Batch Size=16 | RAM Cache=True | Workers=6")

output_dir = data_dir / "outputs" / "raw4k_scout_blackwell"
output_dir.mkdir(parents=True, exist_ok=True)

hf_token = os.environ.get("HF_TOKEN")
hf_repo = os.environ.get("HF_REPO", "Cuong2004/HRP4K")

t_train_start = time.time()
results = train_raw4k_scout(
    data_dir=data_dir,
    output_dir=output_dir,
    epochs=30,
    batch_size=16,
    lr=1e-3,
    lambda_cov=2.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    smoke=False,
    resume=True,
    hf_repo=hf_repo,
    hf_token=hf_token,
    hf_sync=True,
    num_workers=6,
    ram_cache=True,
)
t_train_total = time.time() - t_train_start

print("\n" + "=" * 80)
print(f"🎉 Training Finished in {t_train_total/60:.2f} minutes!")
print(f"Best Region Recall @ 0.75: {results.get('best_region_recall_75', 0.0)*100:.2f}%")
print("=" * 80)

# ------------------------------------------------------------------------------
# 4. Final Hugging Face Upload & Report Display
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"☁️ [4/4] Uploading Final Checkpoints & Artifacts to Hugging Face ({hf_repo})...")
print("=" * 80)

if hf_token:
    try:
        upload_to_hf(
            repo_id=hf_repo,
            local_path=output_dir,
            token=hf_token,
            repo_type="dataset",
            path_in_repo=f"checkpoints/{output_dir.name}",
            commit_message=f"Sync Raw-4K Shallow Scout trained on Blackwell GPU (Recall@0.75: {results.get('best_region_recall_75', 0.0)*100:.2f}%)",
        )
        print(f"✅ Successfully uploaded all artifacts to Hugging Face: {hf_repo}/checkpoints/{output_dir.name}")
    except Exception as exc:
        print(f"⚠️ Hugging Face upload warning: {exc}")
else:
    print("⚠️ Skipping Hugging Face upload: HF_TOKEN not set.")

# Display final toast
try:
    import marimo as mo
    r75 = results.get("best_region_recall_75", 0.0) * 100
    mo.status.toast(f"🎉 Huấn luyện hoàn tất! Recall@0.75: {r75:.2f}% | Đã sync HF Cuong2004/HRP4K")
except Exception:
    pass

report_file = output_dir / "raw4k_scout_report.md"
if report_file.is_file():
    print("\n--- Summary Report ---")
    print(report_file.read_text(encoding="utf-8"))
