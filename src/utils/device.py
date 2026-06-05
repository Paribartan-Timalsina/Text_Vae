"""Device detection and AMP context helpers."""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device, enabled: bool = True):
    """Return an autocast context manager for the given device."""
    if not enabled:
        return torch.amp.autocast(device_type="cpu", enabled=False)
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return torch.amp.autocast(device_type="cpu", enabled=False)
