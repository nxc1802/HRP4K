from __future__ import annotations

import time


def cuda_synchronize_if_needed() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


class Timer:
    def __enter__(self):
        cuda_synchronize_if_needed()
        self.started = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *_args):
        cuda_synchronize_if_needed()
        self.elapsed_ms = (time.perf_counter() - self.started) * 1000.0
