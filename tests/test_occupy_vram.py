import pytest
from unittest.mock import patch, MagicMock
import torch
from systools.occupy_vram import (
    _fmt, _bytes_free, _bytes_allocated,
    occupy_vram, release_vram, _do_release, vram_occupied,
)

try:
    _OOM = torch.cuda.OutOfMemoryError  # PyTorch >= 2.0
except AttributeError:
    _OOM = Exception


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cuda_env():
    """Patches shared across most tests. Tests add memory_allocated/torch.empty inline."""
    props = MagicMock()
    props.total_memory = 8 * 1024 ** 3   # 8 GB
    props.name = "Test GPU"

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.synchronize"), \
         patch("torch.cuda.memory_reserved", return_value=0), \
         patch("torch.cuda.get_device_properties", return_value=props), \
         patch("torch.cuda.empty_cache") as mock_empty_cache:
        yield {
            "props": props,
            "empty_cache": mock_empty_cache,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _ma_side_effect(target_bytes):
    """7-call memory_allocated sequence for 1 Phase-1 loop iteration, then exit."""
    return [0, 0, 0, 0, target_bytes, target_bytes, target_bytes]


# ── Section 1: _fmt (pure, no mocks) ─────────────────────────────────────────

def test_fmt_bytes():
    assert _fmt(0) == "0.00 B"
    assert _fmt(1) == "1.00 B"
    assert _fmt(512) == "512.00 B"
    assert _fmt(1023) == "1023.00 B"


def test_fmt_kilobytes():
    assert _fmt(1024) == "1.00 KB"
    assert _fmt(2048) == "2.00 KB"
    assert _fmt(1024 * 1023) == "1023.00 KB"


def test_fmt_megabytes():
    assert _fmt(1024 ** 2) == "1.00 MB"
    assert _fmt(int(1.5 * 1024 ** 2)) == "1.50 MB"


def test_fmt_gigabytes():
    assert _fmt(1024 ** 3) == "1.00 GB"
    assert _fmt(4 * 1024 ** 3) == "4.00 GB"


def test_fmt_terabytes():
    assert _fmt(1024 ** 4) == "1.00 TB"


def test_fmt_boundary_exact_1024():
    assert _fmt(1023) == "1023.00 B"
    assert _fmt(1024) == "1.00 KB"


# ── Section 2: _bytes_free ────────────────────────────────────────────────────

def test_bytes_free_arithmetic():
    device = torch.device("cuda:0")
    total = 8 * 1024 ** 3
    reserved = 2 * 1024 ** 3
    allocated = 1 * 1024 ** 3

    props = MagicMock()
    props.total_memory = total

    with patch("torch.cuda.get_device_properties", return_value=props), \
         patch("torch.cuda.memory_reserved", return_value=reserved), \
         patch("torch.cuda.memory_allocated", return_value=allocated):
        result = _bytes_free(device)

    # free_in_pool + free_outside_pool = (reserved-allocated) + (total-reserved) = total-allocated
    assert result == total - allocated


def test_bytes_free_fully_reserved():
    device = torch.device("cuda:0")
    total = 4 * 1024 ** 3
    reserved = total          # all memory reserved
    allocated = 1 * 1024 ** 3

    props = MagicMock()
    props.total_memory = total

    with patch("torch.cuda.get_device_properties", return_value=props), \
         patch("torch.cuda.memory_reserved", return_value=reserved), \
         patch("torch.cuda.memory_allocated", return_value=allocated):
        result = _bytes_free(device)

    assert result == total - allocated


def test_bytes_free_nothing_allocated():
    device = torch.device("cuda:0")
    total = 6 * 1024 ** 3

    props = MagicMock()
    props.total_memory = total

    with patch("torch.cuda.get_device_properties", return_value=props), \
         patch("torch.cuda.memory_reserved", return_value=0), \
         patch("torch.cuda.memory_allocated", return_value=0):
        result = _bytes_free(device)

    assert result == total


# ── Section 3: _bytes_allocated ───────────────────────────────────────────────

def test_bytes_allocated_wraps_cuda():
    device = torch.device("cuda:0")
    with patch("torch.cuda.memory_allocated", return_value=42) as mock_ma:
        result = _bytes_allocated(device)
    mock_ma.assert_called_once_with(device)
    assert result == 42


# ── Section 4: Argument validation ───────────────────────────────────────────

def test_occupy_no_target_raises():
    with pytest.raises(ValueError, match="exactly one"):
        occupy_vram()


def test_occupy_two_targets_raises():
    with pytest.raises(ValueError, match="exactly one"):
        occupy_vram(target_gb=1.0, target_mb=512.0)


def test_occupy_all_three_targets_raises():
    with pytest.raises(ValueError, match="exactly one"):
        occupy_vram(target_bytes=1024, target_gb=1.0, target_mb=512.0)


def test_occupy_zero_bytes_raises():
    with pytest.raises(ValueError, match="must be positive"):
        occupy_vram(target_bytes=0)


def test_occupy_negative_bytes_raises():
    with pytest.raises(ValueError, match="must be positive"):
        occupy_vram(target_bytes=-1)


def test_occupy_negative_mb_raises():
    with pytest.raises(ValueError, match="must be positive"):
        occupy_vram(target_mb=-1.0)


# ── Section 5: CUDA / device checks ──────────────────────────────────────────

def test_occupy_no_cuda_raises():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="No CUDA"):
            occupy_vram(target_bytes=1024)


def test_occupy_cpu_device_raises():
    with patch("torch.cuda.is_available", return_value=True):
        with pytest.raises(ValueError, match="must be a CUDA device"):
            occupy_vram(target_bytes=1024, device="cpu")


# ── Section 6: Pre-flight check ───────────────────────────────────────────────

def test_occupy_insufficient_vram_raises(cuda_env):
    # memory_allocated returns total → free = total - total = 0 < target
    total = cuda_env["props"].total_memory
    with patch("torch.cuda.memory_allocated", return_value=total):
        with pytest.raises(RuntimeError, match="only"):
            occupy_vram(target_bytes=2 * 1024 ** 3)


# ── Section 7: Success paths ──────────────────────────────────────────────────

def test_occupy_success_target_mb(cuda_env):
    target_mb = 512.0
    target_bytes = int(target_mb * 1024 ** 2)
    fake_tensor = MagicMock()

    with patch("torch.cuda.memory_allocated", side_effect=_ma_side_effect(target_bytes)), \
         patch("torch.empty", return_value=fake_tensor) as mock_empty:
        handles = occupy_vram(target_mb=target_mb, verbose=False)

    assert isinstance(handles, list)
    assert len(handles) == 1
    mock_empty.assert_called_once()
    kwargs = mock_empty.call_args.kwargs
    assert kwargs["dtype"] == torch.float16
    assert kwargs["device"] == torch.device("cuda:0")


def test_occupy_success_target_gb(cuda_env):
    target_gb = 1.0
    target_bytes = int(target_gb * 1024 ** 3)
    fake_tensor = MagicMock()

    with patch("torch.cuda.memory_allocated", side_effect=_ma_side_effect(target_bytes)), \
         patch("torch.empty", return_value=fake_tensor) as mock_empty:
        handles = occupy_vram(target_gb=target_gb, verbose=False)

    assert isinstance(handles, list)
    assert len(handles) == 1
    mock_empty.assert_called_once()


def test_occupy_success_target_bytes(cuda_env):
    target_bytes = 256 * 1024 ** 2  # 256 MB
    fake_tensor = MagicMock()

    with patch("torch.cuda.memory_allocated", side_effect=_ma_side_effect(target_bytes)), \
         patch("torch.empty", return_value=fake_tensor):
        handles = occupy_vram(target_bytes=target_bytes, verbose=False)

    assert isinstance(handles, list)
    assert len(handles) == 1


def test_occupy_verbose_true_prints(cuda_env, capsys):
    target_mb = 128.0
    target_bytes = int(target_mb * 1024 ** 2)
    fake_tensor = MagicMock()

    with patch("torch.cuda.memory_allocated", side_effect=_ma_side_effect(target_bytes)), \
         patch("torch.empty", return_value=fake_tensor):
        occupy_vram(target_mb=target_mb, verbose=True)

    out = capsys.readouterr().out
    assert "Test GPU" in out
    assert "Device" in out
    assert "Total VRAM" in out
    assert "Allocated" in out


def test_occupy_verbose_false_silent(cuda_env, capsys):
    target_mb = 128.0
    target_bytes = int(target_mb * 1024 ** 2)
    fake_tensor = MagicMock()

    with patch("torch.cuda.memory_allocated", side_effect=_ma_side_effect(target_bytes)), \
         patch("torch.empty", return_value=fake_tensor):
        occupy_vram(target_mb=target_mb, verbose=False)

    assert capsys.readouterr().out == ""


# ── Section 8: OOM ────────────────────────────────────────────────────────────

# memory_allocated call sequence when torch.empty raises OOM:
#   1 baseline, 2 _bytes_free, 3 phase1-cond, 4 delta, 5+6 _do_release(before/after)
_OOM_MA_SIDE_EFFECT = [0, 0, 0, 0, 0, 0]


def test_occupy_oom_raises_runtime_error(cuda_env):
    with patch("torch.cuda.memory_allocated", side_effect=_OOM_MA_SIDE_EFFECT), \
         patch("torch.empty", side_effect=_OOM("OOM")):
        with pytest.raises(RuntimeError, match="Ran out of VRAM"):
            occupy_vram(target_mb=512.0, verbose=False)


def test_occupy_oom_clears_handles(cuda_env):
    # Verify RuntimeError is raised (which requires _do_release to have run)
    with patch("torch.cuda.memory_allocated", side_effect=_OOM_MA_SIDE_EFFECT), \
         patch("torch.empty", side_effect=_OOM("OOM")):
        with pytest.raises(RuntimeError, match="Ran out of VRAM"):
            occupy_vram(target_mb=512.0, verbose=False)


# ── Section 9: _do_release / release_vram ────────────────────────────────────

def test_do_release_clears_handles_and_calls_cache(cuda_env):
    device = torch.device("cuda:0")
    handles = [MagicMock(), MagicMock()]

    with patch("torch.cuda.memory_allocated", return_value=0):
        _do_release(handles, device, verbose=False)

    assert handles == []
    cuda_env["empty_cache"].assert_called_once()


def test_do_release_verbose_prints(cuda_env, capsys):
    device = torch.device("cuda:0")
    handles = [MagicMock()]

    with patch("torch.cuda.memory_allocated", return_value=0):
        _do_release(handles, device, verbose=True)

    assert "Released" in capsys.readouterr().out


def test_release_vram_delegates_to_do_release():
    handles = [MagicMock()]
    device = torch.device("cuda:1")

    with patch("systools.occupy_vram._do_release") as mock_do_release:
        release_vram(handles, device=device, verbose=False)

    mock_do_release.assert_called_once_with(handles, torch.device("cuda:1"), False)


def test_release_vram_default_device():
    handles = []
    with patch("systools.occupy_vram._do_release") as mock_do_release:
        release_vram(handles)

    assert mock_do_release.call_args[0][1] == torch.device("cuda:0")


# ── Section 10: vram_occupied context manager ─────────────────────────────────

def test_vram_occupied_calls_occupy_and_release():
    handles = [MagicMock()]
    with patch("systools.occupy_vram.occupy_vram", return_value=handles) as mock_occ, \
         patch("systools.occupy_vram.release_vram") as mock_rel:
        with vram_occupied(target_mb=256.0, verbose=False):
            mock_rel.assert_not_called()
        mock_occ.assert_called_once()
        mock_rel.assert_called_once()


def test_vram_occupied_releases_on_exception():
    handles = [MagicMock()]
    with patch("systools.occupy_vram.occupy_vram", return_value=handles), \
         patch("systools.occupy_vram.release_vram") as mock_rel:
        with pytest.raises(ValueError):
            with vram_occupied(target_mb=256.0, verbose=False):
                raise ValueError("test error")
        mock_rel.assert_called_once()


def test_vram_occupied_yields_handles():
    handles = ["fake_handle_1", "fake_handle_2"]
    with patch("systools.occupy_vram.occupy_vram", return_value=handles), \
         patch("systools.occupy_vram.release_vram"):
        with vram_occupied(target_mb=256.0) as h:
            assert h is handles
