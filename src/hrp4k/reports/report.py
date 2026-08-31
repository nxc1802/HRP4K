"""Experiment Final report auto-updater.

Automatically updates docs/Experiment_Final.md and its benchmark tables
after each experiment completion.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_FINAL_PATH = Path(__file__).resolve().parents[3] / "docs" / "Experiment_Final.md"
HF_BASE_URL = "https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main"


def _format_metric(value: Any, precision: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


def _format_pct(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def _extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract metrics from experiment result into display format."""
    m = result.get("test_metrics") or result.get("val_metrics") or result.get("metrics") or {}
    scale = m.get("scale", {})

    return {
        "AP50": _format_pct(m.get("AP50") or m.get("metrics/mAP50(B)")),
        "AP75": _format_pct(m.get("AP75") or m.get("metrics/mAP75(B)")),
        "AP50_95": _format_pct(m.get("AP50_95") or m.get("metrics/mAP50-95(B)")),
        "AP_ultra_fine": _format_pct(scale.get("ultra_fine", {}).get("AP50")),
        "AP_small": _format_pct(scale.get("fine", {}).get("AP50")),
        "AP_medium": _format_pct(scale.get("medium", {}).get("AP50")),
        "AP_large": _format_pct(scale.get("large", {}).get("AP50")),
        "Precision": _format_pct(m.get("precision") or m.get("metrics/precision(B)")),
        "Recall": _format_pct(m.get("recall") or m.get("metrics/recall(B)")),
        "F1": _format_pct(m.get("f1")),
        "FPPI": _format_metric(m.get("FPPI_official") or m.get("FPPI_all_images"), 3),
        "Latency": _format_metric(result.get("mean_latency_ms"), 1) + " ms" if result.get("mean_latency_ms") else "—",
    }


def _update_table_row(content: str, config, metrics: dict[str, Any]) -> str:
    """Update corresponding table row in markdown content."""
    exp_id = getattr(config, "experiment_id", "")
    hf_link = f"[{exp_id[:12]}]({HF_BASE_URL}/experiments/{exp_id})"

    # Determine row label
    if config.phase == "resolution":
        res_label = f"**{config.resolution.upper()}**"
        # Match table row for this resolution
        pattern = rf"(\| \*\*{re.escape(config.resolution.upper())}.*?\|).*?\n"
    elif config.phase == "slicing":
        method_map = {
            "resize": "Full Image (Baseline 640)",
            "sliced-nms": "Sliced-NMS (25 crops)",
            "sahi": "SAHI (32 crops)",
            "perspective-grid": "Perspective Grid (9 crops)",
        }
        method_name = method_map.get(config.method, config.method)
        pattern = rf"(\| \*\*{re.escape(method_name)}.*?\n)"
    else:
        return content

    new_row = (
        f"| **{config.resolution.upper() if config.phase == 'resolution' else method_name}** | "
        f"{metrics['AP50']} | {metrics['AP75']} | {metrics['AP50_95']} | "
        f"{metrics['AP_ultra_fine']} | {metrics['AP_small']} | {metrics['AP_medium']} | {metrics['AP_large']} | "
        f"{metrics['Precision']} | {metrics['Recall']} | {metrics['F1']} | {metrics['FPPI']} | {metrics['Latency']} | "
        f"{hf_link} |\n"
    )

    return content


def _build_experiment_entry(config, result: dict[str, Any]) -> str:
    """Build a markdown entry for a single experiment."""
    metrics = _extract_metrics(result)
    exp_id = getattr(config, "experiment_id", "unknown")
    hf_link = f"{HF_BASE_URL}/experiments/{exp_id}"

    # Scale metrics
    scale_section = ""
    raw_metrics = result.get("test_metrics") or result.get("metrics") or {}
    if "scale" in raw_metrics and isinstance(raw_metrics["scale"], dict):
        scale_section = "\n**Scale-wise Metrics:**\n\n"
        scale_section += "| Scale | AP50 | AP75 | AP50:95 | Positives |\n"
        scale_section += "| :--- | ---: | ---: | ---: | ---: |\n"
        for scale_name, scale_data in raw_metrics["scale"].items():
            if isinstance(scale_data, dict):
                scale_section += f"| {scale_name} | {_format_pct(scale_data.get('AP50'))} | {_format_pct(scale_data.get('AP75'))} | {_format_pct(scale_data.get('AP50_95'))} | {scale_data.get('positives', '—')} |\n"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    entry = f"""
### `{config.name}`
- **Experiment ID**: `{exp_id}`
- **Detector**: {config.detector}
- **Phase**: {config.phase}
- **Resolution**: {config.resolution} ({config.imgsz}px)
- **Method**: {getattr(config, 'method', 'N/A')}
- **Epochs**: {config.epochs}
- **Optimizer**: {config.optimizer} (lr0={config.lr0})
- **Effective Batch**: {config.batch * getattr(config, 'accumulation', 1)}
- **Status**: ✅ Completed
- **Timestamp**: {now}

| AP50 | AP75 | AP50:95 | Precision | Recall | F1 | FPPI | Latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {metrics['AP50']} | {metrics['AP75']} | {metrics['AP50_95']} | {metrics['Precision']} | {metrics['Recall']} | {metrics['F1']} | {metrics['FPPI']} | {metrics['Latency']} |
{scale_section}
**Hugging Face**: [🔗 Open Experiment Artifacts]({hf_link})
"""
    return entry


def update_experiment_final(config, result: dict[str, Any]) -> None:
    """Update docs/Experiment_Final.md with results from a completed experiment."""
    path = EXPERIMENT_FINAL_PATH

    if not path.exists():
        return

    content = path.read_text(encoding="utf-8")
    entry = _build_experiment_entry(config, result)

    marker = f"### `{config.name}`"
    if marker in content:
        # Update existing entry block
        start_idx = content.index(marker)
        next_entry = content.find("### `", start_idx + len(marker))
        if next_entry != -1:
            content = content[:start_idx] + entry.strip() + "\n\n" + content[next_entry:]
        else:
            content = content[:start_idx] + entry.strip() + "\n"
    else:
        content += "\n" + entry.strip() + "\n"

    path.write_text(content, encoding="utf-8")
    print(f"[Report] Successfully recorded results for {config.name} in {path.name}")
