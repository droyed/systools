"""
occupy_vram.py
--------------
Utility to allocate and hold a precise target amount of GPU VRAM using PyTorch.

Usage:
    handles = occupy_vram(target_gb=4.0)   # occupy 4 GB
    # ... do your work ...
    release_vram(handles)                  # free it all
"""

import math
import time
from contextlib import contextmanager
from typing import Optional

import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _bytes_allocated(device: torch.device) -> int:
    """Currently allocated (not just reserved) bytes on the device."""
    return torch.cuda.memory_allocated(device)


def _bytes_free(device: torch.device) -> int:
    """Free bytes still available for allocation on the device."""
    total  = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    # memory that is reserved but not yet used can be reused without asking the OS
    free_in_pool = reserved - allocated
    free_outside_pool = total - reserved
    return free_in_pool + free_outside_pool


def _fmt(n_bytes: int) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} TB"


# ── core function ─────────────────────────────────────────────────────────────

def occupy_vram(
    target_bytes: Optional[int] = None,
    *,
    target_gb: Optional[float] = None,
    target_mb: Optional[float] = None,
    device: Optional[torch.device | str | int] = None,
    dtype: torch.dtype = torch.float16,   # float16 → 2 B/elem (saves address space)
    tolerance_mb: float = 1.0,            # acceptable error in MB
    chunk_mb: float = 256.0,              # coarse allocation chunk
    fine_chunk_mb: float = 1.0,           # fine-tuning chunk
    verbose: bool = True,
) -> list[torch.Tensor]:
    """
    Allocate GPU VRAM until `target_bytes` (or target_gb / target_mb) of
    **additional** memory is occupied, on top of whatever is already in use.

    Parameters
    ----------
    target_bytes   : exact number of bytes to occupy (mutually exclusive with
                     target_gb / target_mb)
    target_gb      : gigabytes shorthand
    target_mb      : megabytes shorthand
    device         : CUDA device; defaults to cuda:0
    dtype          : element dtype for the backing tensors (float16 recommended)
    tolerance_mb   : stop when remaining delta < this value (default 1 MB)
    chunk_mb       : large allocation step in MB
    fine_chunk_mb  : fine-tuning step in MB (used near the target)
    verbose        : print progress

    Returns
    -------
    list[torch.Tensor]
        Live tensors that are holding the VRAM.  Keep a reference to the list;
        pass it to `release_vram()` when done.

    Raises
    ------
    ValueError  : bad arguments
    RuntimeError: not enough free VRAM to satisfy the request
    """

    # ── argument validation ──────────────────────────────────────────────────
    given = sum(x is not None for x in (target_bytes, target_gb, target_mb))
    if given != 1:
        raise ValueError("Provide exactly one of: target_bytes, target_gb, target_mb")

    if target_gb is not None:
        target_bytes = int(target_gb * 1024 ** 3)
    elif target_mb is not None:
        target_bytes = int(target_mb * 1024 ** 2)

    if target_bytes <= 0:
        raise ValueError(f"target_bytes must be positive, got {target_bytes}")

    # ── device setup ─────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-capable GPU found.")

    if device is None:
        device = torch.device("cuda:0")
    else:
        device = torch.device(device)

    if device.type != "cuda":
        raise ValueError(f"device must be a CUDA device, got {device}")

    torch.cuda.synchronize(device)

    # ── pre-flight check ─────────────────────────────────────────────────────
    baseline      = _bytes_allocated(device)
    target_total  = baseline + target_bytes
    free_now      = _bytes_free(device)

    if target_bytes > free_now:
        raise RuntimeError(
            f"Requested {_fmt(target_bytes)} but only {_fmt(free_now)} is free "
            f"on {device}."
        )

    if verbose:
        props = torch.cuda.get_device_properties(device)
        print(f"[occupy_vram] Device      : {device} ({props.name})")
        print(f"[occupy_vram] Total VRAM  : {_fmt(props.total_memory)}")
        print(f"[occupy_vram] Baseline    : {_fmt(baseline)}")
        print(f"[occupy_vram] Target extra: {_fmt(target_bytes)}")

    # ── dtype bookkeeping ─────────────────────────────────────────────────────
    bytes_per_elem = torch.finfo(dtype).bits // 8 if dtype.is_floating_point \
                     else torch.iinfo(dtype).bits // 8

    def _elems_for_mb(mb: float) -> int:
        return max(1, int(mb * 1024 ** 2 / bytes_per_elem))

    tol_bytes   = int(tolerance_mb  * 1024 ** 2)
    chunk_elems = _elems_for_mb(chunk_mb)
    fine_elems  = _elems_for_mb(fine_chunk_mb)

    # ── allocation loop ───────────────────────────────────────────────────────
    handles: list[torch.Tensor] = []

    def _remaining() -> int:
        return target_total - _bytes_allocated(device)

    try:
        # Phase 1 – coarse fill
        while _remaining() > tol_bytes:
            delta      = _remaining()
            need_elems = max(1, delta // bytes_per_elem)

            # Don't over-shoot: cap at coarse chunk size unless we're close
            if need_elems > chunk_elems and delta > int(fine_chunk_mb * 1024 ** 2 * 4):
                alloc_elems = min(chunk_elems, need_elems)
            else:
                alloc_elems = min(fine_elems, need_elems)

            t = torch.empty(alloc_elems, dtype=dtype, device=device)
            handles.append(t)
            torch.cuda.synchronize(device)

        # Phase 2 – fine trim (1-element at a time if needed)
        while _remaining() > tol_bytes:
            handles.append(torch.empty(1, dtype=dtype, device=device))

    except torch.cuda.OutOfMemoryError:
        # Partial allocation – release what we grabbed and re-raise
        _do_release(handles, device, verbose=False)
        raise RuntimeError(
            "Ran out of VRAM during allocation.  Try a smaller target or "
            "increase tolerance_mb."
        )

    torch.cuda.synchronize(device)
    occupied = _bytes_allocated(device) - baseline

    if verbose:
        print(f"[occupy_vram] Allocated   : {_fmt(occupied)}  "
              f"(error: {_fmt(abs(occupied - target_bytes))})")
        print(f"[occupy_vram] Tensors held: {len(handles)}")

    return handles


# ── release helper ────────────────────────────────────────────────────────────

def _do_release(handles: list[torch.Tensor],
                device: torch.device,
                verbose: bool) -> None:
    freed_before = torch.cuda.memory_allocated(device)
    handles.clear()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    freed_after = torch.cuda.memory_allocated(device)
    if verbose:
        print(f"[occupy_vram] Released    : {_fmt(freed_before - freed_after)}")


def release_vram(handles: list[torch.Tensor],
                 device: Optional[torch.device | str | int] = None,
                 verbose: bool = True) -> None:
    """
    Free all tensors returned by `occupy_vram`.

    Parameters
    ----------
    handles : list returned by occupy_vram
    device  : same device passed to occupy_vram (defaults to cuda:0)
    verbose : print how much was freed
    """
    if device is None:
        device = torch.device("cuda:0")
    else:
        device = torch.device(device)
    _do_release(handles, device, verbose)


# ── context-manager interface ─────────────────────────────────────────────────

@contextmanager
def vram_occupied(
    target_bytes: Optional[int] = None,
    *,
    target_gb: Optional[float] = None,
    target_mb: Optional[float] = None,
    device: Optional[torch.device | str | int] = None,
    dtype: torch.dtype = torch.float16,
    tolerance_mb: float = 1.0,
    verbose: bool = True,
):
    """
    Context-manager wrapper around occupy_vram / release_vram.

    Example
    -------
    with vram_occupied(target_gb=2.0):
        train_one_epoch(model, loader)
    # VRAM automatically freed on exit
    """
    handles = occupy_vram(
        target_bytes=target_bytes,
        target_gb=target_gb,
        target_mb=target_mb,
        device=device,
        dtype=dtype,
        tolerance_mb=tolerance_mb,
        verbose=verbose,
    )
    try:
        yield handles
    finally:
        release_vram(handles, device=device, verbose=verbose)
