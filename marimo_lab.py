import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", app_title="HRP4K Benchmark Studio — Marimo Lab")


@app.cell
def _():
    import html
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import time
    import marimo as mo
    import torch

    DEFAULT_HF_REPO = os.environ.get("HF_REPO", "Cuong2004/HRP4K")
    DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")
    env_path = Path(".env")
    if not DEFAULT_HF_TOKEN and env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                DEFAULT_HF_TOKEN = line.split("=", 1)[1].strip()
            elif line.startswith("HF_REPO=") and not os.environ.get("HF_REPO"):
                DEFAULT_HF_REPO = line.split("=", 1)[1].strip()

    # Pre-configure environment variables
    if DEFAULT_HF_TOKEN and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = DEFAULT_HF_TOKEN
    if DEFAULT_HF_REPO and not os.environ.get("HF_REPO"):
        os.environ["HF_REPO"] = DEFAULT_HF_REPO

    def run_command_live(cmd: list[str] | str, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> tuple[int, str]:
        """Runs a command, captures output, and returns (exit_code, combined_output)."""
        current_env = os.environ.copy()
        if env:
            current_env.update(env)
        
        # Inject HF token and repo if missing
        if "HF_TOKEN" not in current_env and DEFAULT_HF_TOKEN:
            current_env["HF_TOKEN"] = DEFAULT_HF_TOKEN
        if "HF_REPO" not in current_env and DEFAULT_HF_REPO:
            current_env["HF_REPO"] = DEFAULT_HF_REPO

        # Ensure python unbuffered and src in PYTHONPATH
        current_env["PYTHONUNBUFFERED"] = "1"
        src_path = str(Path.cwd() / "src")
        if "PYTHONPATH" in current_env:
            current_env["PYTHONPATH"] = f"{src_path}:{current_env['PYTHONPATH']}"
        else:
            current_env["PYTHONPATH"] = src_path

        shell = isinstance(cmd, str)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=current_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=shell,
                bufsize=1,
                universal_newlines=True,
            )
            output_lines = []
            if proc.stdout:
                for line in proc.stdout:
                    output_lines.append(line)
            proc.wait()
            return proc.returncode, "".join(output_lines)
        except Exception as e:
            return 1, f"Execution error: {e}"

    return (
        DEFAULT_HF_REPO,
        DEFAULT_HF_TOKEN,
        Path,
        env_path,
        html,
        json,
        mo,
        os,
        run_command_live,
        shutil,
        subprocess,
        sys,
        time,
        torch,
    )


@app.cell
def _(DEFAULT_HF_REPO, Path, mo, subprocess, sys, torch):
    # System Diagnostics & Hardware Discovery
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A (CPU Only)"
    vram_gb = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if cuda_available else "N/A"
    
    # Git commit hash
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        git_branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        git_info = f"Branch: `{git_branch_name}` | Commit: `{git_hash}`"
    except Exception:
        git_info = "Not a git repository yet"

    header_md = mo.md(
        f"""
        # 🔬 HRP4K Benchmark Studio — Marimo Lab
        Notebook tương tác thực thi toàn diện quy trình nghiên cứu, huấn luyện baseline và đánh giá độ phân giải cao cho tập dữ liệu HRP4K (**Phase 0 → Phase 3**).
        
        ---
        """
    )

    stats_grid = mo.hstack(
        [
            mo.stat(value="CUDA GPU" if cuda_available else "CPU Only", label="Accelerator", caption=gpu_name),
            mo.stat(value=vram_gb, label="Total VRAM", caption="GPU Memory" if cuda_available else "No GPU Detected"),
            mo.stat(value=torch.__version__, label="PyTorch Version", caption=f"Python {sys.version.split()[0]}"),
            mo.stat(value=DEFAULT_HF_REPO, label="HF Cloud Sync", caption="Connected & Pre-configured"),
        ],
        justify="space-between",
    )

    sys_info_banner = mo.vstack([header_md, stats_grid])
    sys_info_banner
    return (
        cuda_available,
        git_branch_name,
        git_hash,
        git_info,
        gpu_name,
        header_md,
        stats_grid,
        sys_info_banner,
        vram_gb,
    )


@app.cell
def _(mo):
    git_repo_url = mo.ui.text(
        value="https://github.com/nxc1802/HRP4K.git",
        label="Git Repository URL:",
        full_width=True,
    )
    git_branch = mo.ui.text(
        value="main",
        label="Git Branch:",
    )
    git_btn = mo.ui.run_button(
        label="🔄 Đồng Bộ Git (Clone hoặc Pull)",
        kind="neutral",
    )

    git_ui = mo.vstack([
        mo.md("## 🐙 0. Quản Lý Git Repository (Clone / Pull)"),
        mo.md("Tự động kiểm tra: Nếu thư mục repo đã tồn tại thì thực hiện `git pull`, ngược lại sẽ tự động `git clone` mã nguồn mới nhất."),
        mo.hstack([git_repo_url, git_branch, git_btn], justify="start", align="end", gap=1),
    ])
    git_ui
    return git_branch, git_btn, git_repo_url, git_ui


@app.cell
def _(Path, git_branch, git_btn, git_repo_url, mo, run_command_live):
    if not git_btn.value:
        git_result = mo.md("_Nhấn nút **Đồng Bộ Git** để cập nhật hoặc tải mã nguồn mới nhất._")
    else:
        repo_dir = Path("HRP4K")
        cwd_git = Path(".git")

        if cwd_git.exists():
            cmd = f"git pull origin {git_branch.value}"
            code, out = run_command_live(cmd)
            target = "Thư mục hiện tại"
        elif (repo_dir / ".git").exists():
            cmd = f"git -C HRP4K pull origin {git_branch.value}"
            code, out = run_command_live(cmd)
            target = "Thư mục HRP4K"
        else:
            cmd = f"git clone -b {git_branch.value} {git_repo_url.value} HRP4K"
            code, out = run_command_live(cmd)
            target = "HRP4K (Clone mới)"

        status_kind = "success" if code == 0 else "danger"
        status_title = "✅ Đồng bộ Git thành công!" if code == 0 else "❌ Lỗi đồng bộ Git!"

        git_result = mo.callout(
            mo.vstack([
                mo.md(f"### {status_title}"),
                mo.md(f"- **Thao tác:** `{cmd}`"),
                mo.md(f"- **Target:** {target}"),
                mo.accordion({"Chi tiết nhật ký Git (stdout/stderr)": mo.md(f"```text\n{out}\n```")}),
            ]),
            kind=status_kind,
        )

    git_result
    return (
        cmd,
        code,
        cwd_git,
        git_result,
        out,
        repo_dir,
        status_kind,
        status_title,
        target,
    )


@app.cell
def _(mo):
    dep_pip_cb = mo.ui.checkbox(value=True, label="Nâng cấp pip (--upgrade pip)")
    dep_pkg_cb = mo.ui.checkbox(value=True, label="Cài đặt hrp4k editable (pip install -e .)")
    dep_btn = mo.ui.run_button(
        label="⚡ Cài Đặt Dependencies & Framework",
        kind="warn",
    )

    dep_ui = mo.vstack([
        mo.md("## 📦 1. Cài Đặt Dependencies & Framework CLI"),
        mo.md("Cài đặt các gói phụ thuộc (`ultralytics`, `pycocotools`, `sahi`, `opencv-python`, `huggingface_hub`) và package `hrp4k` ở chế độ editable."),
        mo.hstack([dep_pip_cb, dep_pkg_cb, dep_btn], justify="start", align="center", gap=1),
    ])
    dep_ui
    return dep_btn, dep_pip_cb, dep_pkg_cb, dep_ui


@app.cell
def _(dep_btn, dep_pip_cb, dep_pkg_cb, mo, run_command_live, sys):
    if not dep_btn.value:
        dep_result = mo.md("_Nhấn nút **Cài Đặt Dependencies** nếu cần cài đặt hoặc cập nhật môi trường._")
    else:
        logs = []
        py_exe = sys.executable

        if dep_pip_cb.value:
            c1, o1 = run_command_live(f"{py_exe} -m pip install -q --upgrade pip")
            logs.append(f"=== Pip Upgrade ===\n{o1}")

        c2, o2 = run_command_live(
            f'{py_exe} -m pip install -q "ultralytics>=8.3.0" "pycocotools>=2.0.7" "sahi>=0.12.5" "opencv-python>=4.8" "pyyaml>=6.0" "huggingface_hub"'
        )
        logs.append(f"=== Core Dependencies ===\n{o2}")

        if dep_pkg_cb.value:
            c3, o3 = run_command_live(f"{py_exe} -m pip install -q -e .")
            logs.append(f"=== Package hrp4k install -e . ===\n{o3}")

        # Check version
        c4, o4 = run_command_live(f"{py_exe} -m hrp4k --version")
        logs.append(f"=== Version Check ===\n{o4}")

        success = (c2 == 0)
        dep_result = mo.callout(
            mo.vstack([
                mo.md(f"### {'✅ Cài đặt Dependencies thành công!' if success else '⚠️ Cảnh báo trong quá trình cài đặt'}") ,
                mo.md(f"**Phiên bản CLI:** `{o4.strip() if o4.strip() else 'hrp4k installed'}`"),
                mo.accordion({"Xem toàn bộ Output Log": mo.md(f"```text\n{''.join(logs)}\n```")}),
            ]),
            kind="success" if success else "warn",
        )

    dep_result
    return c1, c2, c3, c4, dep_result, logs, o1, o2, o3, o4, py_exe, success


@app.cell
def _(DEFAULT_HF_REPO, DEFAULT_HF_TOKEN, mo):
    hf_token_input = mo.ui.text(
        value=DEFAULT_HF_TOKEN,
        placeholder="hf_your_write_token_here",
        kind="password",
        label="Hugging Face Token (Write Access):",
        full_width=True,
    )
    hf_repo_input = mo.ui.text(
        value=DEFAULT_HF_REPO,
        placeholder="Cuong2004/HRP4K",
        label="HF Repository ID:",
        full_width=True,
    )
    hf_save_btn = mo.ui.run_button(
        label="💾 Cập Nhật .env & Xác Thực",
        kind="neutral",
    )

    hf_ui = mo.vstack([
        mo.md("## ☁️ 1.1 Cấu Hình Hugging Face Cloud Synchronization (`.env`)"),
        mo.md("Token và Repository đã được **tích hợp sẵn mặc định** (`Cuong2004/HRP4K`). Checkpoints (`best.pt`, `last.pt`, `results.csv`) sẽ tự động sao lưu ngầm lên Cloud sau mỗi Epoch."),
        mo.hstack([hf_token_input, hf_repo_input], justify="space-between", gap=1),
        hf_save_btn,
    ])
    hf_ui
    return hf_repo_input, hf_save_btn, hf_token_input, hf_ui


@app.cell
def _(DEFAULT_HF_REPO, DEFAULT_HF_TOKEN, Path, hf_repo_input, hf_save_btn, hf_token_input, mo):
    token = hf_token_input.value.strip() or DEFAULT_HF_TOKEN
    repo = hf_repo_input.value.strip() or DEFAULT_HF_REPO

    env_lines = [f"HF_TOKEN={token}", f"HF_REPO={repo}"]
    env_file = Path(".env")
    env_file.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    user_info = "Đang kiểm tra token..."
    auth_success = False
    if token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            who = api.whoami()
            user_info = f"Đăng nhập thành công với tài khoản: **{who.get('name', who.get('fullname', 'User'))}**"
            auth_success = True
        except Exception as ex:
            user_info = f"Lỗi xác thực Hugging Face: {ex}"
            auth_success = False

    hf_result = mo.callout(
        mo.vstack([
            mo.md(f"### {'✅ Hugging Face Cloud Sẵn Sàng!' if auth_success else '⚠️ Hugging Face Token'}") ,
            mo.md(f"- **Target HF Repository:** `{repo}`"),
            mo.md(f"- **Tài khoản xác thực:** {user_info}"),
        ]),
        kind="success" if auth_success else "warn",
    )

    hf_result
    return (
        api,
        auth_success,
        env_file,
        env_lines,
        hf_result,
        repo,
        token,
        user_info,
        who,
    )


@app.cell
def _(mo):
    data_dir_input = mo.ui.text(
        value="HRP4K",
        label="Thư mục Dataset đích:",
    )
    data_setup_btn = mo.ui.run_button(
        label="📂 Khởi Tạo Dataset (hrp4k setup-data)",
        kind="neutral",
    )

    data_ui = mo.vstack([
        mo.md("## 📁 2. Thiết Lập Dataset (Setup Data)"),
        mo.md("Tự động nhận diện dữ liệu Kaggle Input (`/kaggle/input/...`), tạo symlink chuẩn hoặc tự động tải từ Hugging Face Hub (`Cuong2004/HRP4K`)."),
        mo.hstack([data_dir_input, data_setup_btn], justify="start", align="end", gap=1),
    ])
    data_ui
    return data_dir_input, data_setup_btn, data_ui


@app.cell
def _(
    DEFAULT_HF_REPO,
    DEFAULT_HF_TOKEN,
    Path,
    data_dir_input,
    data_setup_btn,
    mo,
    run_command_live,
    sys,
):
    if not data_setup_btn.value:
        data_result = mo.md("_Nhấn nút **Khởi Tạo Dataset** để kiểm tra, tải từ HF hoặc liên kết dữ liệu Kaggle._")
    else:
        py_exe_data = sys.executable
        c_data, o_data = run_command_live(
            f"{py_exe_data} -m hrp4k setup-data --data {data_dir_input.value}",
            env={"HF_TOKEN": DEFAULT_HF_TOKEN, "HF_REPO": DEFAULT_HF_REPO}
        )
        
        target_path = Path(data_dir_input.value)
        train_json = target_path / "train.json"
        val_json = target_path / "valid.json"
        test_json = target_path / "test.json"

        files_status = [
            {"Tệp / Thư mục": "train.json", "Tồn tại": "✅" if train_json.exists() else "❌", "Đường dẫn": str(train_json)},
            {"Tệp / Thư mục": "valid.json", "Tồn tại": "✅" if val_json.exists() else "❌", "Đường dẫn": str(val_json)},
            {"Tệp / Thư mục": "test.json", "Tồn tại": "✅" if test_json.exists() else "❌", "Đường dẫn": str(test_json)},
        ]

        data_result = mo.callout(
            mo.vstack([
                mo.md(f"### {'✅ Dataset đã sẵn sàng!' if c_data == 0 else '❌ Lỗi thiết lập dataset'}") ,
                mo.ui.table(files_status),
                mo.accordion({"Xem Chi Tiết Setup Data Output": mo.md(f"```text\n{o_data}\n```")}),
            ]),
            kind="success" if c_data == 0 else "danger",
        )

    data_result
    return (
        c_data,
        data_result,
        files_status,
        o_data,
        py_exe_data,
        target_path,
        test_json,
        train_json,
        val_json,
    )


@app.cell
def _(mo):
    p0_samples_slider = mo.ui.slider(
        start=4,
        stop=32,
        step=4,
        value=12,
        label="Số mẫu kiểm tra chất lượng:",
    )
    p0_out_input = mo.ui.text(
        value="outputs/phase0",
        label="Thư mục xuất báo cáo:",
    )
    p0_btn = mo.ui.run_button(
        label="📊 Chạy Phase 0 (Audit & Scale Bins)",
        kind="neutral",
    )

    p0_ui = mo.vstack([
        mo.md("## 📊 3. Phase 0: Kiểm Định Dataset & Phân Bố Scale Bins"),
        mo.md("Thực hiện kiểm tra tính toàn vẹn dataset, nhãn COCO và phân tích phân bố kích thước đối tượng (*Tiny*, *Small*, *Medium*, *Large*)."),
        mo.hstack([p0_samples_slider, p0_out_input, p0_btn], justify="start", align="end", gap=1),
    ])
    p0_ui
    return p0_btn, p0_out_input, p0_samples_slider, p0_ui


@app.cell
def _(
    Path,
    data_dir_input,
    json,
    mo,
    p0_btn,
    p0_out_input,
    p0_samples_slider,
    run_command_live,
    sys,
):
    if not p0_btn.value:
        p0_result = mo.md("_Nhấn nút **Chạy Phase 0** để bắt đầu kiểm định dataset._")
    else:
        py_exe_p0 = sys.executable
        cmd_p0 = f"{py_exe_p0} -m hrp4k phase0 --data {data_dir_input.value} --output {p0_out_input.value} --quality-samples {p0_samples_slider.value}"
        c_p0, o_p0 = run_command_live(cmd_p0)

        # Parse output json if available
        summary_path = Path(p0_out_input.value) / "dataset_report.json"
        metrics_table = None
        if summary_path.exists():
            try:
                report_data = json.loads(summary_path.read_text(encoding="utf-8"))
                metrics_table = report_data
            except Exception:
                pass

        p0_result = mo.callout(
            mo.vstack([
                mo.md(f"### {'✅ Phase 0 Hoàn Thành!' if c_p0 == 0 else '❌ Lỗi thực thi Phase 0'}") ,
                mo.md(f"**Lệnh đã chạy:** `{cmd_p0}`"),
                mo.accordion({"Báo cáo chi tiết Phase 0 (JSON/Output)": mo.md(f"```json\n{json.dumps(metrics_table, indent=2) if metrics_table else o_p0}\n```")}),
            ]),
            kind="success" if c_p0 == 0 else "danger",
        )

    p0_result
    return (
        c_p0,
        cmd_p0,
        metrics_table,
        o_p0,
        p0_result,
        py_exe_p0,
        report_data,
        summary_path,
    )


@app.cell
def _(mo):
    p1_model_dropdown = mo.ui.dropdown(
        options={
            "yolo11m": "YOLO11m (Khuyên dùng - 1280x1280)",
            "yolov8m": "YOLOv8m",
            "yolov5m-compat": "YOLOv5m (Ultralytics compat)",
            "yolov5m-official": "YOLOv5m (Official)",
            "rt-detr-v1": "RT-DETR-L (v1 Transformer)",
            "rt-detr-v2": "RT-DETR-X (v2 Transformer SOTA)",
            "all": "TẤT CẢ 6 Models (all)",
        },
        value="yolo11m",
        label="Chọn Model:",
    )
    p1_imgsz_dropdown = mo.ui.dropdown(
        options={
            "1280": "1280 (Khuyên dùng: Nhanh, mAP cao, ổn định VRAM)",
            "original": "original (Kích thước gốc 4K 3840x2176 - batch 1)",
            "640": "640 (Tiêu chuẩn nhanh)",
        },
        value="1280",
        label="Độ phân giải (imgsz):",
    )
    p1_batch_slider = mo.ui.slider(
        start=1,
        stop=32,
        step=1,
        value=16,
        label="Batch Size:",
    )
    p1_epochs_slider = mo.ui.slider(
        start=1,
        stop=300,
        step=10,
        value=150,
        label="Số Epochs:",
    )
    p1_conf_number = mo.ui.number(
        start=0.0001,
        stop=0.5,
        step=0.001,
        value=0.001,
        label="Val Confidence (Paper COCO = 0.001):",
    )
    p1_weights_input = mo.ui.text(
        value="",
        placeholder="outputs/runs/yolo11m_1280/weights/last.pt (để trống nếu dùng preset)",
        label="Tùy chọn Weights/Checkpoint (.pt):",
        full_width=True,
    )
    p1_output_input = mo.ui.text(
        value="outputs/runs/yolo11m_1280",
        label="Thư mục lưu outputs:",
        full_width=True,
    )
    p1_resume_cb = mo.ui.checkbox(value=False, label="Resume (--resume từ checkpoint)")
    p1_allow_full_cb = mo.ui.checkbox(value=True, label="Authorize Full Run (--allow-full)")
    p1_hf_sync_cb = mo.ui.checkbox(value=True, label="Background HF Cloud Sync")

    p1_btn = mo.ui.run_button(
        label="🚀 Bắt Đầu Huấn Luyện (Phase 1)",
        kind="warn",
    )

    p1_ui = mo.vstack([
        mo.md("## 🏋️ 4. Phase 1: Huấn Luyện Baseline Detectors"),
        mo.md("Cấu hình và khởi chạy huấn luyện các mô hình Baseline. Checkpoints (`best.pt`, `last.pt`, `results.csv`) sẽ được tự động đồng bộ ngầm lên Hugging Face Cloud (`Cuong2004/HRP4K`) sau mỗi Epoch."),
        mo.hstack([p1_model_dropdown, p1_imgsz_dropdown], justify="space-between", gap=1),
        mo.hstack([p1_batch_slider, p1_epochs_slider, p1_conf_number], justify="space-between", gap=1),
        mo.hstack([p1_weights_input, p1_output_input], justify="space-between", gap=1),
        mo.hstack([p1_resume_cb, p1_allow_full_cb, p1_hf_sync_cb], justify="start", gap=1),
        p1_btn,
    ])
    p1_ui
    return (
        p1_allow_full_cb,
        p1_batch_slider,
        p1_btn,
        p1_conf_number,
        p1_epochs_slider,
        p1_hf_sync_cb,
        p1_imgsz_dropdown,
        p1_model_dropdown,
        p1_output_input,
        p1_resume_cb,
        p1_ui,
        p1_weights_input,
    )


@app.cell
def _(
    DEFAULT_HF_REPO,
    DEFAULT_HF_TOKEN,
    Path,
    mo,
    p1_allow_full_cb,
    p1_batch_slider,
    p1_btn,
    p1_conf_number,
    p1_epochs_slider,
    p1_hf_sync_cb,
    p1_imgsz_dropdown,
    p1_model_dropdown,
    p1_output_input,
    p1_resume_cb,
    p1_weights_input,
    run_command_live,
    sys,
):
    if not p1_btn.value:
        p1_result = mo.md("_Nhấn nút **Bắt Đầu Huấn Luyện** để khởi chạy Phase 1._")
    else:
        py_exe_p1 = sys.executable
        args_p1 = [
            py_exe_p1, "-m", "hrp4k", "phase1",
            "--model", p1_model_dropdown.value,
            "--imgsz", str(p1_imgsz_dropdown.value),
            "--batch", str(p1_batch_slider.value),
            "--epochs", str(p1_epochs_slider.value),
            "--confidence", str(p1_conf_number.value),
            "--output", p1_output_input.value,
            "--hf-repo", DEFAULT_HF_REPO,
            "--hf-token", DEFAULT_HF_TOKEN,
        ]

        if p1_allow_full_cb.value:
            args_p1.append("--allow-full")
        if p1_resume_cb.value:
            args_p1.append("--resume")
        if not p1_hf_sync_cb.value:
            args_p1.append("--no-hf-sync")
        if p1_weights_input.value.strip():
            args_p1.extend(["--weights", p1_weights_input.value.strip()])

        cmd_p1_str = " ".join(args_p1)
        c_p1, o_p1 = run_command_live(
            args_p1,
            env={"HF_TOKEN": DEFAULT_HF_TOKEN, "HF_REPO": DEFAULT_HF_REPO}
        )

        best_pt = Path(p1_output_input.value) / "weights" / "best.pt"
        last_pt = Path(p1_output_input.value) / "weights" / "last.pt"
        results_csv = Path(p1_output_input.value) / "results.csv"

        p1_result = mo.callout(
            mo.vstack([
                mo.md(f"### {'✅ Huấn Luyện Phase 1 Thành Công!' if c_p1 == 0 else '❌ Lỗi trong quá trình huấn luyện'}"),
                mo.md(f"- **Lệnh thực thi:** `{cmd_p1_str}`"),
                mo.md(f"- **Best Checkpoint:** `{best_pt}` ({'Đã tạo' if best_pt.exists() else 'Chưa thấy'})"),
                mo.md(f"- **Last Checkpoint:** `{last_pt}` ({'Đã tạo' if last_pt.exists() else 'Chưa thấy'})"),
                mo.accordion({"Nhật ký chi tiết quá trình huấn luyện (Training Logs)": mo.md(f"```text\n{o_p1}\n```")}),
            ]),
            kind="success" if c_p1 == 0 else "danger",
        )

    p1_result
    return (
        args_p1,
        best_pt,
        c_p1,
        cmd_p1_str,
        last_pt,
        o_p1,
        p1_result,
        py_exe_p1,
        results_csv,
    )


@app.cell
def _(mo):
    p2_method_dropdown = mo.ui.dropdown(
        options={
            "sliced-nms": "Sliced-NMS (Tile 960x960, overlap 20% - Khuyên dùng)",
            "perspective-grid": "Perspective-Grid (Bám sát dải mặt đường ở xa)",
            "sahi": "SAHI (Sliced Aided Hyper Inference)",
            "resize": "Resize (Standard)",
            "all": "TẤT CẢ Phương Pháp (all)",
        },
        value="sliced-nms",
        label="Phương pháp Inference:",
    )
    p2_split_dropdown = mo.ui.dropdown(
        options=["test", "valid", "train"],
        value="test",
        label="Tập dữ liệu:",
    )
    p2_weights_input = mo.ui.text(
        value="outputs/runs/yolo11m_1280/weights/best.pt",
        label="Checkpoint weights (.pt):",
        full_width=True,
    )
    p2_tile_size_slider = mo.ui.slider(
        start=320,
        stop=1920,
        step=64,
        value=960,
        label="Tile Size (px):",
    )
    p2_overlap_slider = mo.ui.slider(
        start=0.0,
        stop=0.5,
        step=0.05,
        value=0.2,
        label="Overlap Ratio:",
    )
    p2_limit_slider = mo.ui.slider(
        start=0,
        stop=500,
        step=10,
        value=0,
        label="Limit số ảnh (0 = Toàn bộ):",
    )
    p2_out_input = mo.ui.text(
        value="outputs/predictions/yolo11m_sliced_nms.json",
        label="File JSON dự đoán đầu ra:",
        full_width=True,
    )
    p2_btn = mo.ui.run_button(
        label="⚡ Chạy Phase 2 Inference",
        kind="neutral",
    )

    p2_ui = mo.vstack([
        mo.md("## 🔍 5. Phase 2: High-Resolution Inference & Slicing"),
        mo.md("Thực thi các phương pháp phân giải độ phân giải cao (`sliced-nms`, `perspective-grid`, `sahi`, `all`) và tự động tính toán COCO Benchmark Metrics."),
        mo.hstack([p2_method_dropdown, p2_split_dropdown], justify="space-between", gap=1),
        p2_weights_input,
        mo.hstack([p2_tile_size_slider, p2_overlap_slider, p2_limit_slider], justify="space-between", gap=1),
        p2_out_input,
        p2_btn,
    ])
    p2_ui
    return (
        p2_btn,
        p2_limit_slider,
        p2_method_dropdown,
        p2_out_input,
        p2_overlap_slider,
        p2_split_dropdown,
        p2_tile_size_slider,
        p2_ui,
        p2_weights_input,
    )


@app.cell
def _(
    data_dir_input,
    mo,
    p2_btn,
    p2_limit_slider,
    p2_method_dropdown,
    p2_out_input,
    p2_overlap_slider,
    p2_split_dropdown,
    p2_tile_size_slider,
    p2_weights_input,
    run_command_live,
    sys,
):
    if not p2_btn.value:
        p2_result = mo.md("_Nhấn nút **Chạy Phase 2 Inference** để bắt đầu dự đoán._")
    else:
        py_exe_p2 = sys.executable
        args_p2 = [
            py_exe_p2, "-m", "hrp4k", "phase2",
            "--data", data_dir_input.value,
            "--split", p2_split_dropdown.value,
            "--method", p2_method_dropdown.value,
            "--weights", p2_weights_input.value.strip(),
            "--tile-size", str(p2_tile_size_slider.value),
            "--overlap", str(p2_overlap_slider.value),
            "--output", p2_out_input.value.strip(),
        ]
        if p2_limit_slider.value > 0:
            args_p2.extend(["--limit", str(p2_limit_slider.value)])

        cmd_p2_str = " ".join(args_p2)
        c_p2, o_p2 = run_command_live(args_p2)

        p2_result = mo.callout(
            mo.vstack([
                mo.md(f"### {'✅ Phase 2 Inference Hoàn Thành!' if c_p2 == 0 else '❌ Lỗi trong quá trình inference'}") ,
                mo.md(f"- **Lệnh đã chạy:** `{cmd_p2_str}`"),
                mo.md(f"- **Predictions lưu tại:** `{p2_out_input.value}`"),
                mo.accordion({"Kết quả chi tiết Phase 2": mo.md(f"```text\n{o_p2}\n```")}),
            ]),
            kind="success" if c_p2 == 0 else "danger",
        )

    p2_result
    return args_p2, c_p2, cmd_p2_str, o_p2, p2_result, py_exe_p2


@app.cell
def _(mo):
    p3_gt_input = mo.ui.text(
        value="HRP4K/test.json",
        label="Ground Truth COCO JSON:",
        full_width=True,
    )
    p3_pred_input = mo.ui.text(
        value="outputs/predictions/yolo11m_sliced_nms.json",
        label="Predictions JSON:",
        full_width=True,
    )
    p3_conf_number = mo.ui.number(
        start=0.001,
        stop=0.9,
        step=0.05,
        value=0.25,
        label="Confidence Threshold:",
    )
    p3_out_input = mo.ui.text(
        value="outputs/metrics/yolo11m_sliced_nms_metrics.json",
        label="Output Metrics JSON:",
        full_width=True,
    )

    p3_eval_btn = mo.ui.run_button(
        label="📊 Đánh Giá Chi Tiết (Phase 3 COCO)",
        kind="neutral",
    )
    p3_diag_btn = mo.ui.run_button(
        label="🔬 Chẩn Đoán Lỗi (Diagnose)",
        kind="warn",
    )

    p3_ui = mo.vstack([
        mo.md("## 📈 6. Phase 3: Đánh Giá Chi Tiết & Chẩn Đoán Lỗi"),
        mo.md("Đánh giá độ chính xác theo quy chuẩn COCO/FPPI và phân tích các dạng lỗi phát hiện đối tượng (*Localization*, *Background FP*, *False Negatives*)."),
        mo.hstack([p3_gt_input, p3_pred_input], justify="space-between", gap=1),
        mo.hstack([p3_conf_number, p3_out_input], justify="space-between", gap=1),
        mo.hstack([p3_eval_btn, p3_diag_btn], justify="start", gap=1),
    ])
    p3_ui
    return (
        p3_conf_number,
        p3_diag_btn,
        p3_eval_btn,
        p3_gt_input,
        p3_out_input,
        p3_pred_input,
        p3_ui,
    )


@app.cell
def _(
    Path,
    json,
    mo,
    p3_conf_number,
    p3_eval_btn,
    p3_gt_input,
    p3_out_input,
    p3_pred_input,
    run_command_live,
    sys,
):
    if not p3_eval_btn.value:
        p3_eval_result = mo.md("_Nhấn nút **Đánh Giá Chi Tiết (Phase 3)** để tính toán các chỉ số COCO mAP._")
    else:
        py_exe_p3 = sys.executable
        cmd_p3 = f"{py_exe_p3} -m hrp4k phase3 --ground-truth {p3_gt_input.value} --predictions {p3_pred_input.value} --output {p3_out_input.value} --confidence {p3_conf_number.value}"
        c_p3, o_p3 = run_command_live(cmd_p3)

        metric_json = Path(p3_out_input.value)
        table_items = []
        if metric_json.exists():
            try:
                m_data = json.loads(metric_json.read_text(encoding="utf-8"))
                for k, v in m_data.items():
                    if isinstance(v, (int, float)):
                        table_items.append({"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)})
            except Exception:
                pass

        p3_eval_result = mo.callout(
            mo.vstack([
                mo.md(f"### {'✅ Phase 3 Evaluation Thành Công!' if c_p3 == 0 else '❌ Lỗi đánh giá Phase 3'}") ,
                mo.md(f"- **Command:** `{cmd_p3}`"),
                mo.ui.table(table_items) if table_items else mo.md("_Không tìm thấy file JSON kết quả._"),
                mo.accordion({"Xem Chi Tiết Output": mo.md(f"```text\n{o_p3}\n```")}),
            ]),
            kind="success" if c_p3 == 0 else "danger",
        )

    p3_eval_result
    return (
        c_p3,
        cmd_p3,
        k,
        m_data,
        metric_json,
        o_p3,
        p3_eval_result,
        py_exe_p3,
        table_items,
        v,
    )


@app.cell
def _(
    mo,
    p3_diag_btn,
    p3_gt_input,
    p3_pred_input,
    run_command_live,
    sys,
):
    if not p3_diag_btn.value:
        p3_diag_result = mo.md("_Nhấn nút **Chẩn Đoán Lỗi (Diagnose)** để phân tích chi tiết các loại sai số phát hiện._")
    else:
        py_exe_diag = sys.executable
        diag_out = "outputs/diagnostics"
        cmd_diag = f"{py_exe_diag} -m hrp4k diagnose --ground-truth {p3_gt_input.value} --predictions {p3_pred_input.value} --output {diag_out}"
        c_diag, o_diag = run_command_live(cmd_diag)

        p3_diag_result = mo.callout(
            mo.vstack([
                mo.md(f"### {'✅ Hoàn Thành Chẩn Đoán Lỗi!' if c_diag == 0 else '❌ Lỗi trong quá trình chẩn đoán'}") ,
                mo.md(f"- **Command:** `{cmd_diag}`"),
                mo.md(f"- **Thư mục báo cáo chẩn đoán:** `{diag_out}`"),
                mo.accordion({"Chi tiết phân tích lỗi (Diagnostics Log)": mo.md(f"```text\n{o_diag}\n```")}),
            ]),
            kind="success" if c_diag == 0 else "danger",
        )

    p3_diag_result
    return c_diag, cmd_diag, diag_out, o_diag, p3_diag_result, py_exe_diag


@app.cell
def _(DEFAULT_HF_REPO, DEFAULT_HF_TOKEN, mo):
    hf_push_repo = mo.ui.text(
        value=DEFAULT_HF_REPO,
        label="Target HF Repo ID:",
    )
    hf_push_path = mo.ui.text(
        value="outputs",
        label="Thư mục/File cần đẩy:",
    )
    hf_push_type = mo.ui.dropdown(
        options=["dataset", "model"],
        value="dataset",
        label="Loại Repo:",
    )
    hf_push_token = mo.ui.text(
        value=DEFAULT_HF_TOKEN,
        placeholder="HF Write Token",
        kind="password",
        label="HF Token (Write Access):",
        full_width=True,
    )
    hf_push_btn = mo.ui.run_button(
        label="☁️ Đẩy Lên Hugging Face (push-hf)",
        kind="warn",
    )

    hf_push_ui = mo.vstack([
        mo.md("## ☁️ 7. Đẩy Checkpoints & Kết Quả Lên Hugging Face Thủ Công"),
        mo.md("Đóng gói và đẩy thư mục `outputs/` hoặc các checkpoint quan trọng lên Hugging Face Hub (`Cuong2004/HRP4K`)."),
        mo.hstack([hf_push_repo, hf_push_type, hf_push_path], justify="space-between", gap=1),
        hf_push_token,
        hf_push_btn,
    ])
    hf_push_ui
    return (
        hf_push_btn,
        hf_push_path,
        hf_push_repo,
        hf_push_token,
        hf_push_type,
        hf_push_ui,
    )


@app.cell
def _(
    DEFAULT_HF_TOKEN,
    hf_push_btn,
    hf_push_path,
    hf_push_repo,
    hf_push_token,
    hf_push_type,
    mo,
    os,
    run_command_live,
    sys,
):
    if not hf_push_btn.value:
        hf_push_result = mo.md("_Nhấn nút **Đẩy Lên Hugging Face** để tải toàn bộ kết quả lên Cloud._")
    else:
        token_to_use = hf_push_token.value.strip() or os.environ.get("HF_TOKEN") or DEFAULT_HF_TOKEN
        py_exe_push = sys.executable

        if not token_to_use:
            hf_push_result = mo.callout(
                mo.md("❌ Không tìm thấy HF Token. Vui lòng nhập token hoặc cấu hình trong mục 1.1!"),
                kind="danger",
            )
        else:
            args_push = [
                py_exe_push, "-m", "hrp4k", "push-hf",
                "--repo", hf_push_repo.value.strip(),
                "--path", hf_push_path.value.strip(),
                "--token", token_to_use,
                "--repo-type", hf_push_type.value,
            ]
            c_push, o_push = run_command_live(
                args_push,
                env={"HF_TOKEN": token_to_use, "HF_REPO": hf_push_repo.value.strip()}
            )

            hf_push_result = mo.callout(
                mo.vstack([
                    mo.md(f"### {'✅ Đẩy dữ liệu lên Hugging Face thành công!' if c_push == 0 else '❌ Lỗi tải lên Hugging Face'}") ,
                    mo.md(f"- **Repository URL:** [https://huggingface.co/datasets/{hf_push_repo.value.strip()}](https://huggingface.co/datasets/{hf_push_repo.value.strip()})"),
                    mo.accordion({"Chi tiết kết quả tải lên": mo.md(f"```text\n{o_push}\n```")}),
                ]),
                kind="success" if c_push == 0 else "danger",
            )

    hf_push_result
    return args_push, c_push, hf_push_result, o_push, py_exe_push, token_to_use


if __name__ == "__main__":
    app.run()
