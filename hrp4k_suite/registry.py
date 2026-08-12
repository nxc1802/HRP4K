from __future__ import annotations


METHOD_REGISTRY = {
    "resize": {"type": "inference", "requires_training": False, "implementation": "native", "status": "ready"},
    "uniform-2": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
    "uniform-3": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
    "sliced-nms": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
    "sahi": {"type": "crop", "requires_training": False, "implementation": "official-library", "status": "optional-ready"},
    "perspective-grid": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
    "autofocus": {"type": "coarse-to-fine", "requires_training": True, "implementation": "paper-reproduction", "status": "external-required"},
    "adazoom": {"type": "adaptive-crop", "requires_training": True, "implementation": "paper-reproduction", "status": "external-required"},
    "fovea": {"type": "nonlinear-warp", "requires_training": True, "implementation": "paper-reproduction", "status": "external-required"},
    "two-plane-prior": {"type": "nonlinear-warp", "requires_training": True, "implementation": "paper-reproduction", "status": "external-required"},
    "zoomdet": {"type": "nonlinear-warp", "requires_training": True, "implementation": "official-code-adaptation", "status": "external-required"},
}

METHOD_STATUS = {
    name: f"{entry['status']} ({entry['implementation']})"
    for name, entry in METHOD_REGISTRY.items()
}
