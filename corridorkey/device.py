"""Device-neutral helpers for the CorridorKey v2 runtime (CUDA, Apple MPS, CPU).

The runtime was written against CUDA. These helpers keep the call sites
unchanged in spirit (autocast, cache release, synchronize, memory budget)
while dispatching on the selected device type.
"""
from __future__ import annotations

import contextlib

import torch

_DTYPES = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
}

# Active device and autocast dtype, set once per process by `configure`.
_STATE = {"device": torch.device("cuda"), "dtype": torch.bfloat16}


def resolve_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available.")
    return torch.device(name)


def resolve_dtype(name: str | None, device: torch.device) -> torch.dtype:
    """Autocast dtype. Default: bfloat16 everywhere (the network's training dtype)."""
    if name is None or name == "auto":
        return torch.bfloat16
    try:
        return _DTYPES[name.lower()]
    except KeyError:
        raise ValueError(f"unsupported --dtype {name!r} (bfloat16, float16, float32)")


def configure(device, dtype=None) -> None:
    dev = device if isinstance(device, torch.device) else resolve_device(str(device))
    _STATE["device"] = dev
    _STATE["dtype"] = dtype if isinstance(dtype, torch.dtype) else resolve_dtype(dtype, dev)


def device() -> torch.device:
    return _STATE["device"]


def amp_dtype() -> torch.dtype:
    return _STATE["dtype"]


def autocast():
    """torch.autocast for the active device; float32 means autocast off."""
    dtype = _STATE["dtype"]
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(_STATE["device"].type, dtype=dtype)


def empty_cache() -> None:
    kind = _STATE["device"].type
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def synchronize() -> None:
    kind = _STATE["device"].type
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


# Unified memory is shared with the OS, ffmpeg and the frame readers.
MPS_RESERVE_GB = 8.0
# The window table is CUDA-measured. On MPS the driver keeps about 2-3.5x the
# tensor peak allocated (measured on M1 Ultra), and beyond that the run slows
# down sharply, so the budget handed to the table is divided by this factor.
MPS_OVERHEAD = 3.0
CPU_RESERVE_GB = 8.0


def memory_budget_gb() -> float:
    """Memory available to this process for the Hann-window estimate, in GiB."""
    kind = _STATE["device"].type
    if kind == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        # Memory this process already holds (the loaded model) is part of the budget.
        return (free_bytes + torch.cuda.memory_reserved()) / 2**30
    if kind == "mps":
        held = torch.mps.driver_allocated_memory()
        limit = torch.mps.recommended_max_memory()
        budget = limit
        try:
            import psutil
            budget = min(limit, psutil.virtual_memory().available + held)
        except ImportError:
            pass
        return max(0.0, budget / 2**30 - MPS_RESERVE_GB) / MPS_OVERHEAD
    try:
        import psutil
        return max(0.0, psutil.virtual_memory().available / 2**30 - CPU_RESERVE_GB)
    except ImportError:
        return 16.0


def non_blocking() -> bool:
    """Async host-to-device copies only on CUDA (pinned-memory semantics).

    On MPS a non_blocking copy from pageable CPU memory may still be in flight
    when the source tensor is freed; copy synchronously there.
    """
    import os
    override = os.environ.get("CORRIDORKEY_NON_BLOCKING")
    if override is not None:
        return override == "1"
    return _STATE["device"].type == "cuda"
