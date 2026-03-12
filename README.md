# systools — System utilities for Python

## Overview

`systools` is a growing collection of system-level utilities for Python.

The current release includes a GPU VRAM allocation tool for PyTorch, useful for:

- **Testing memory limits** — verify your workload fits within a budget
- **Reserving VRAM** — pre-occupy memory to simulate a loaded GPU
- **Benchmarking** — run experiments under controlled memory pressure

Allocation is accurate to within 1 MB by default, using a two-phase strategy (coarse chunks, then fine-tuning).

## Installation

Requires Python ≥ 3.9 and a PyTorch build with CUDA support.

```bash
pip install .
```

For an editable install during development:

```bash
pip install -e .
```

## Quick Start

### Functional API

```python
from systools.occupy_vram import occupy_vram, release_vram

# Occupy 4 GB of VRAM on cuda:0
handles = occupy_vram(target_gb=4.0)

# ... do your work ...

release_vram(handles)
```

### Context Manager API

```python
from systools.occupy_vram import vram_occupied

with vram_occupied(target_mb=512, verbose=True):
    # VRAM is held inside the block
    train_one_epoch(model, loader)
# VRAM is automatically released on exit
```

See [`demos/demo_occupy_vram.py`](demos/demo_occupy_vram.py) for a runnable example.

## API Reference

### `occupy_vram(...) -> list[torch.Tensor]`

Allocates GPU VRAM until the requested amount of **additional** memory is occupied on top of whatever is already in use.

```python
occupy_vram(
    target_bytes: int | None = None,
    *,
    target_gb: float | None = None,
    target_mb: float | None = None,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype = torch.float16,
    tolerance_mb: float = 1.0,
    chunk_mb: float = 256.0,
    fine_chunk_mb: float = 1.0,
    verbose: bool = True,
) -> list[torch.Tensor]
```

| Parameter | Description |
|---|---|
| `target_bytes` | Exact bytes to occupy. Mutually exclusive with `target_gb` / `target_mb`. |
| `target_gb` | Gigabytes shorthand. |
| `target_mb` | Megabytes shorthand. |
| `device` | CUDA device (e.g. `"cuda:0"`, `1`). Defaults to `cuda:0`. |
| `dtype` | Element dtype for backing tensors. `float16` (2 B/elem) recommended. |
| `tolerance_mb` | Stop when remaining delta is below this value (default 1 MB). |
| `chunk_mb` | Coarse allocation step in MB (default 256 MB). |
| `fine_chunk_mb` | Fine-tuning step in MB (default 1 MB). |
| `verbose` | Print progress to stdout. |

**Returns:** A list of live tensors holding the VRAM. Keep a reference and pass it to `release_vram()` when done.

**Raises:**
- `ValueError` — bad arguments (wrong number of targets, non-positive size, non-CUDA device)
- `RuntimeError` — CUDA unavailable, or insufficient free VRAM

---

### `release_vram(handles, device=None, verbose=True) -> None`

Frees all tensors returned by `occupy_vram` and calls `torch.cuda.empty_cache()`.

```python
release_vram(
    handles: list[torch.Tensor],
    device: torch.device | str | int | None = None,
    verbose: bool = True,
) -> None
```

| Parameter | Description |
|---|---|
| `handles` | List returned by `occupy_vram`. |
| `device` | Same device passed to `occupy_vram`. Defaults to `cuda:0`. |
| `verbose` | Print how much VRAM was freed. |

---

### `vram_occupied(...)` — context manager

Context-manager wrapper around `occupy_vram` / `release_vram`. Accepts the same parameters as `occupy_vram` (except `chunk_mb` and `fine_chunk_mb`). Yields the handles list and always releases on exit, even if an exception occurs.

```python
@contextmanager
def vram_occupied(
    target_bytes: int | None = None,
    *,
    target_gb: float | None = None,
    target_mb: float | None = None,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype = torch.float16,
    tolerance_mb: float = 1.0,
    verbose: bool = True,
)
```

## Development

Install with dev dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Clean build artifacts:

```bash
make clean
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
