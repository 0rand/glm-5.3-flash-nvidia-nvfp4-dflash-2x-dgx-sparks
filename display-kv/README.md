# Display-KV runtime shim — internals

AGPL-3.0 adaptation of coolbho3k's display-reserve allocator
(https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark) for the
GLM-5.3-Flash NVFP4 stack. The technique: on headless GB10 nodes, the ~2 GiB
firmware-reserved display memory can be reached through a DRM scanout buffer +
`mmap` + `cuMemHostRegister(DEVICEMAP | IOMEMORY)` and used as device memory.

## Files

- `display_kv_glm.py` — Python shim. Replaces vLLM's final KV `torch.zeros`
  backing with a contiguous UVA span: aligned ordinary-RAM prefix + the full
  1.75 GiB display suffix. vLLM sees exactly its requested byte count; any
  <64 KiB alignment tail is hidden. SHA-pins the four hooked vLLM files and
  fails closed (exit 78) on drift. All diagnostics go to **stderr** — stdout
  must stay machine-clean (cpuinfo.py parses it).
- `sitecustomize.py` — auto-activates the shim when the launcher sets
  `PYTHONPATH=/opt/display-kv` + `DISPLAY_KV_ENABLED=1`.
- `libdisplay_kv_glm.so` — C allocator (built from `toolkit/repo/.../display_kv.c`).
- `toolkit/` — DRM module control, probes, tests, build script, AGPL source.
- `README.md` here documents design + guards; the launcher is
  `../start-display-kv.sh`.

## Invariants enforced at runtime

- Final KV total must exceed the production pin (`DISPLAY_KV_MIN_TOTAL_BYTES`).
- Ordinary registered RAM must never exceed the production pin
  (`DISPLAY_KV_MAX_ORDINARY_BYTES` = 10,240,000,000).
- Display suffix is exactly 1.75 GiB (1,879,048,192 bytes).
- Exactly one backing allocation per worker; receipts written to
  `/tmp/display-kv-receipts/<pid>.json` inside each container.

## DRM handling (runtime-only, never persisted)

`toolkit/display-drm-mode.sh on` reloads `nvidia_drm` with `modeset=1 fbdev=0`
on the local node (one interactive sudo prompt on the head).
`toolkit/prepare-display-drm-both.sh on` additionally reloads it on the worker
over SSH using **passwordless sudo**. No modprobe.d or initramfs changes; a
reboot restores boot defaults. The scripts refuse to reload while a display-KV
pool is mapped (they check `/proc/*/maps` as root).

## Measured properties (our two-node cluster, driver 580.x, 2026-09-22)

- Integrity: full 1.75 GiB sweep, 0 mismatches, both nodes.
- SM-side read of the display segment: 163–168 GB/s (ordinary cudaMalloc:
  235–269 GB/s). Write ~112 GB/s. Copy-engine (DMA) access is catastrophic
  (0.9–2.3 GB/s) — attention kernels use SM loads, which is why KV works fine.
- Registered ordinary RAM inside the contiguous span reads at the same ~166 GB/s
  as the display segment — the pin+DEVICEMAP registration path itself, not the
  carveout, carries the bandwidth cost. A/B on this stack showed **no decode
  tax** within fluctuation (fill run ~10% faster, within run-to-run noise).
- Driver-dependent: verified on 580.x; failure reported upstream on 595.84
  (`CUDA_ERROR_INVALID_VALUE` at the IOMEMORY register). Re-probe after any
  driver change: `toolkit/verify-display-kv.sh`.

## Rebuilding the .so

```bash
cd toolkit && ./build.sh     # needs CUDA toolkit headers + driver on the host
```

Then copy the rebuilt `libdisplay_kv_glm.so` up one level, replacing the
shipped binary, and rerun `./start-display-kv.sh preflight`.
