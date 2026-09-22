# SPDX-License-Identifier: AGPL-3.0-only
"""Display-reserve KV backing for GLM-5.3 on GB10.

Derived from coolbho3k's display_kv.py (AGPL-3.0-only).  This adaptation
keeps vLLM's native KV descriptors/layout unchanged and replaces only the one
shared byte backing with a contiguous UVA span:

    ordinary registered RAM prefix + 1.75 GiB DRM display-reserve suffix

The requested vLLM KV size MUST include the display suffix. For the production
GLM experiment the 10,240,000,000-byte operational 11 GB pin becomes
12,119,048,192 bytes while ordinary RAM consumption remains capped at the
production pin.
"""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DISPLAY_BYTES = 1792 * 2**20
QUANTUM = 65536
DEFAULT_EXPECTED_TOTAL_BYTES = 11 * 1024**3 + DISPLAY_BYTES

NATIVE_PINS = {
    "vllm.v1.worker.utils": (
        "5ed6caaf5797c214398be2a28fe7b6b24a4e9eb7b9b72fa6f17afec4a495875c"
    ),
    "vllm.v1.worker.gpu.attn_utils": (
        "e91729ae69c9d53691c76aa3617bd6a5e2712811c442678a5330c48d3afc6d7e"
    ),
    "vllm.v1.worker.gpu_model_runner": (
        "5c246e2c6c2178d98df69bc3b6a63b44cd815eb129547051816782bcd3cd00b4"
    ),
    "vllm.v1.worker.gpu_worker": (
        "a2ab95f9562025295d6de4f5246a50d372c4deb13382d5e9a8bac73041f744ce"
    ),
}

_ALLOC_LINE = "buf = torch.zeros(buf_size, dtype=torch.int8, device=device)"
_REPLACEMENT_LINE = (
    "buf = _display_kv_backing(buf_size, dtype=torch.int8, device=device)"
)

_owner: Owner | None = None
_expected_allocation_size: int | None = None
_installed = False


@dataclass(frozen=True)
class PoolLayout:
    ordinary_bytes: int
    display_bytes: int
    total_bytes: int       # exact byte count requested by vLLM
    backing_bytes: int     # owned span, including an invisible alignment tail
    padding_bytes: int


def pool_layout(total_bytes: int) -> PoolLayout:
    """Compute an alignment-safe ordinary/display split for requested KV bytes."""
    if type(total_bytes) is not int or total_bytes <= DISPLAY_BYTES:
        raise ValueError("total KV bytes must be larger than display reserve")
    unaligned_ordinary = total_bytes - DISPLAY_BYTES
    ordinary = ((unaligned_ordinary + QUANTUM - 1) // QUANTUM) * QUANTUM
    backing = ordinary + DISPLAY_BYTES
    padding = backing - total_bytes
    if not 0 <= padding < QUANTUM:
        raise RuntimeError(f"invalid display-KV alignment padding: {padding}")
    return PoolLayout(ordinary, DISPLAY_BYTES, total_bytes, backing, padding)


def _validate_layout(layout: PoolLayout) -> None:
    exact = os.environ.get("DISPLAY_KV_EXPECTED_TOTAL_BYTES", "").strip()
    if exact and layout.total_bytes != int(exact):
        raise RuntimeError(
            f"display-KV exact-size guard: requested {layout.total_bytes}, expected {exact}"
        )

    minimum = os.environ.get("DISPLAY_KV_MIN_TOTAL_BYTES", "").strip()
    if minimum and layout.total_bytes < int(minimum):
        raise RuntimeError(
            f"display-KV minimum-size guard: requested {layout.total_bytes}, minimum {minimum}"
        )

    max_ordinary = os.environ.get("DISPLAY_KV_MAX_ORDINARY_BYTES", "").strip()
    if max_ordinary and layout.ordinary_bytes > int(max_ordinary):
        raise RuntimeError(
            "display-KV ordinary-RAM guard: "
            f"requested {layout.ordinary_bytes}, maximum {max_ordinary}"
        )


def expected_allocation_size() -> Optional[int]:
    """Exposed for tests and diagnostics."""
    return _expected_allocation_size


class Owner:
    """Own the registered ordinary prefix and DRM display suffix."""

    def __init__(self, layout: PoolLayout, library_path: Path):
        import torch

        if not torch.cuda.is_initialized() or torch.cuda.current_device() != 0:
            raise RuntimeError("existing CUDA0 context must be initialized first")

        self.layout = layout
        self.lib = ctypes.CDLL(str(library_path))
        self.lib.ds41_display_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self.lib.ds41_display_create.restype = ctypes.c_void_p
        self.lib.ds41_display_pointer.argtypes = [ctypes.c_void_p]
        self.lib.ds41_display_pointer.restype = ctypes.c_uint64
        self.lib.ds41_display_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ds41_display_destroy.restype = None
        self.lib.ds41_display_error.restype = ctypes.c_char_p

        self.handle = self.lib.ds41_display_create(
            layout.ordinary_bytes, layout.display_bytes
        )
        if not self.handle:
            raise RuntimeError(self.lib.ds41_display_error().decode())
        self.pointer = self.lib.ds41_display_pointer(self.handle)
        self.__cuda_array_interface__ = {
            "shape": (layout.backing_bytes,),
            "strides": None,
            "typestr": "|i1",
            "data": (self.pointer, False),
            "version": 3,
        }

    def tensor(self):
        import torch

        tensor = torch.as_tensor(self, device="cuda:0")
        if (
            tensor.data_ptr() != self.pointer
            or tensor.dtype != torch.int8
            or tensor.numel() != self.layout.backing_bytes
        ):
            raise RuntimeError("CUDA array interface copied or misread external storage")
        return tensor

    def close_for_test(self) -> None:
        if self.handle:
            self.lib.ds41_display_destroy(self.handle)
            self.handle = None


def _emit(receipt: dict[str, Any]) -> None:
    """Emit diagnostics without ever contaminating machine-readable stdout."""
    print(json.dumps(receipt), file=sys.stderr, flush=True)


def _memory() -> dict[str, int]:
    return {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split(":")[0] in ("MemFree", "MemAvailable")
    }


def _library_path() -> Path:
    path = Path(
        os.environ.get(
            "DISPLAY_KV_LIBRARY", "/opt/display-kv/libdisplay_kv_glm.so"
        )
    )
    if not path.is_file():
        raise RuntimeError(f"display-KV library not found: {path}")
    return path


def backing(size: int, dtype: Any, device: Any):
    """Replacement for the final KV ``torch.zeros`` allocation."""
    import torch

    global _owner

    if _expected_allocation_size is None:
        # A profiling allocation outside Worker.initialize_from_config remains
        # native and must not consume the one display carveout.
        return torch.zeros(size, dtype=dtype, device=device)

    if size != _expected_allocation_size:
        raise ValueError(
            f"final KV allocation {size} differs from admitted descriptor "
            f"{_expected_allocation_size}"
        )
    if dtype != torch.int8 or torch.device(device) != torch.device("cuda:0"):
        raise ValueError(f"unexpected KV backing request: dtype={dtype}, device={device}")
    if _owner is not None:
        raise RuntimeError("display-KV backing may be allocated exactly once per worker")

    # Exact equality is optional because vLLM's hybrid-cache planner may round
    # the CLI budget down to a block-aligned backing size. _validate_layout()
    # enforces the configured minimum and ordinary-RAM ceiling instead.

    layout = pool_layout(size)
    _validate_layout(layout)
    before = _memory()
    owner = Owner(layout, _library_path())
    _owner = owner  # process-lifetime ownership, including graph lifetimes
    full_tensor = owner.tensor()
    # vLLM sees exactly the requested bytes. Any <64 KiB alignment tail remains
    # owned/mapped after the visible tensor and is never handed to vLLM.
    tensor = full_tensor[:size]
    if tensor.numel() != size or tensor.data_ptr() != owner.pointer:
        raise RuntimeError("failed to expose exact requested KV view")
    full_tensor.zero_()
    torch.cuda.synchronize()
    after = _memory()

    # Only the ordinary prefix should reduce normal available RAM.  Leave a
    # tolerance for allocator/OS bookkeeping, but fail loudly on a second full
    # ordinary allocation of the display suffix.
    delta = max(before["MemAvailable"] - after["MemAvailable"], 0)
    if delta > layout.ordinary_bytes + 256 * 2**20:
        raise RuntimeError(
            "display-KV pool consumed unexpected ordinary RAM: "
            f"delta={delta}, ordinary={layout.ordinary_bytes}"
        )

    receipt = {
        "stage": "glm_display_kv_allocated",
        "pid": os.getpid(),
        "ordinary_bytes": layout.ordinary_bytes,
        "display_bytes": layout.display_bytes,
        "requested_bytes": layout.total_bytes,
        "backing_bytes": layout.backing_bytes,
        "alignment_padding_bytes": layout.padding_bytes,
        "storage_pointer": owner.pointer,
        "memavailable_before": before["MemAvailable"],
        "memavailable_after": after["MemAvailable"],
        "torch_allocator_tracks_external_pool": False,
    }
    _emit(receipt)
    receipt_dir = Path(os.environ.get("DISPLAY_KV_RECEIPT_DIR", "/cache/display-kv"))
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / f"{os.getpid()}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return tensor


def compile_allocator(
    original: Callable[..., Any], replacement: Callable[..., Any]
) -> Callable[..., Any]:
    """Compile the pinned allocator after replacing its one backing line."""
    source = inspect.getsource(original)
    matches = source.count(_ALLOC_LINE)
    if matches != 1:
        raise RuntimeError(
            f"allocator patch target must occur exactly once; found {matches}"
        )
    patched_source = source.replace(_ALLOC_LINE, _REPLACEMENT_LINE, 1)
    namespace = dict(original.__globals__)
    namespace["_display_kv_backing"] = replacement
    exec(compile(patched_source, "<display_kv_glm>", "exec"), namespace)
    patched = namespace.get(original.__name__)
    if not isinstance(patched, types.FunctionType):
        raise RuntimeError("patched allocator did not compile to a function")
    return patched


@contextmanager
def real_allocation(kv_cache_config):
    """Mark exactly the final KV initialization lifecycle."""
    global _expected_allocation_size

    if _expected_allocation_size is not None or _owner is not None:
        raise RuntimeError("final display-KV initialization must occur exactly once")
    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("expected one shared final KV backing size")
    size = sizes.pop()
    pool_layout(size)  # validate before entering the lifecycle
    _expected_allocation_size = size
    try:
        yield
        if _owner is None:
            raise RuntimeError("final KV initialization did not use display backing")
    finally:
        _expected_allocation_size = None


def wrap_initialize_from_config(original):
    """Wrap Worker.initialize_from_config with the final-allocation context."""
    if getattr(original, "_display_kv_wrapped", False):
        return original

    def wrapped(self, kv_cache_config):
        with real_allocation(kv_cache_config):
            return original(self, kv_cache_config)

    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    wrapped._display_kv_wrapped = True
    return wrapped


def _import_and_pin(name: str):
    module = __import__(name, fromlist=[""])
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    expected = NATIVE_PINS[name]
    if actual != expected:
        raise RuntimeError(
            f"unreviewed display-KV target {name}: expected {expected}, "
            f"got {actual} ({module.__file__})"
        )
    return module


def install() -> None:
    """Install the allocator and worker lifecycle hooks; fail closed on drift."""
    global _installed
    if _installed:
        return

    utils = _import_and_pin("vllm.v1.worker.utils")
    attn = _import_and_pin("vllm.v1.worker.gpu.attn_utils")
    legacy_runner = _import_and_pin("vllm.v1.worker.gpu_model_runner")
    worker_module = _import_and_pin("vllm.v1.worker.gpu_worker")

    original = utils.allocate_kv_cache
    if not (
        attn.allocate_kv_cache is original
        and legacy_runner.allocate_kv_cache is original
    ):
        raise RuntimeError("native allocator namespace binding already changed")

    patched = compile_allocator(original, backing)
    utils.allocate_kv_cache = patched
    attn.allocate_kv_cache = patched
    legacy_runner.allocate_kv_cache = patched

    worker_cls = worker_module.Worker
    worker_cls.initialize_from_config = wrap_initialize_from_config(
        worker_cls.initialize_from_config
    )
    _installed = True
    _emit(
        {
            "stage": "glm_display_kv_hooks_installed",
            "exact_total_bytes": os.environ.get(
                "DISPLAY_KV_EXPECTED_TOTAL_BYTES", ""
            ),
            "min_total_bytes": os.environ.get(
                "DISPLAY_KV_MIN_TOTAL_BYTES", ""
            ),
            "max_ordinary_bytes": os.environ.get(
                "DISPLAY_KV_MAX_ORDINARY_BYTES", ""
            ),
            "display_bytes": DISPLAY_BYTES,
        }
    )


def _reset_for_tests() -> None:
    """Reset pure module state. Never call in a serving worker."""
    global _owner, _expected_allocation_size, _installed
    if _owner is not None and hasattr(_owner, "close_for_test"):
        _owner.close_for_test()
    _owner = None
    _expected_allocation_size = None
    _installed = False
