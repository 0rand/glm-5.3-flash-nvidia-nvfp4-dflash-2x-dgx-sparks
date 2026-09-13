# GLM-5.3-Flash · NVIDIA official NVFP4 · DFlash2 · 2× DGX Spark

Serving **nvidia/GLM-5.3-Flash-NVFP4** (uniform NVFP4) with the pilcothink 0.28 runtime
on two GB10 nodes, TP=2 over RoCE, with a DFlash2 speculative draft, CUDA graphs and
async scheduling — tuned for **long context**.

## Verified result (2026-09-13)

```
context      : 700,160 tokens (2,735 x 256)
KV pool      : 892,139 tokens  (1.27x at full context)
hardmode     : 94/100   (166/176 points, Hard Mode 38/38 = 100%)
e2e          : 996 s   (88 scenarios, parallel 4, high effort)
boot         : 980 s to serving
config       : GMU 0.88 · seqs 4 · batch 1024 · cudagraphs <=16 · async ON
               DFlash2 k=5 · VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
```

Context scale-up against the earlier runs on the same iron:

| config | ctx | KV pool | hardmode | e2e |
|---|---|---|---|---|
| eager, k=7 (baseline) | 262,144 | 642,370 | 91/100 | 1331 s |
| graphs ≤64, batch 2048, 10 seqs | 262,144 | 544,019 | 90/100 | 973 s |
| **graphs ≤16, batch 1024, 4 seqs, GME off** | **700,160** | **892,139** | **94/100** | **996 s** |

2.7× the context, +4 quality points, same wall time, and a *larger* pool than either
earlier run. `docs/FINDINGS.md` explains why — the CUDA-graph memory estimator was
silently reserving ~2.1 GiB of KV, and batch/seqs were taking another ~2 GiB.

## Quickstart

```bash
cp .env.example .env      # then edit the [EDIT] lines for your cluster
./download.sh             # image (once, then shipped over the fabric) + weights + draft
./start.sh                # boot, wait for health, print the KV pool
./run-tests.sh            # spec-bench x2 -> throughput -> hardmode
./status.sh               # containers, health, served model, pool
./stop.sh                 # stop both nodes
```

`./run-tests.sh --boot` does the boot and the whole measurement set in one go.

## Requirements

- 2× DGX Spark (GB10, 128 GB unified memory each) on a RoCE fabric
- Docker on both nodes, passwordless SSH from head to worker
- `tool-eval-bench` on the head (for `run-tests.sh`)
- The pilcothink runtime image — it supplies the `b12x` MoE/linear backends and the
  sparse-MLA attention default this recipe relies on

The official NVFP4 checkpoint ships **no MTP heads** (0 `mtp`/`nextn` tensors in
147,661), so a separate DFlash2 draft model is mandatory here. If you have the
local-inference-lab weights instead, see the sibling stack below — those carry native MTP.

## Layout

```
.env / .env.example   all configuration; scripts contain no site values
start.sh              boot both ranks, wait for health, report the pool
stop.sh               stop both ranks
status.sh             containers, health, served model, boot markers
tail-log.sh           follow the boot log or the container log
download.sh           image + weights + draft, synced to the worker over RoCE
run-tests.sh          the measurement set (spec-bench x2, throughput, hardmode)
replicate-hardmode.sh byte-matched e2e hardmode run
vendor/               run_cluster_dual.sh — upstream cluster launcher (see below)
docs/FINDINGS.md      memory model, failure catalogue, measurement protocol
logs/                 boot + test logs (gitignored)
```

## Provenance

- Runtime recipe: **pilcothink** `DGX_Spark_vllm_Dockerfile` `0.28` (upstream project).
  `vendor/run_cluster_dual.sh` is upstream, vendored unmodified — it is the proven
  multi-node launcher and is not ours to rewrite.
- Weights: NVIDIA official `nvidia/GLM-5.3-Flash-NVFP4`, revision-pinned in `.env`.
- Draft: incoai `GLM-5.3-Flash-DFlash2`, mounted as `/workspace/models/<DRAFT_NAME>`.
- Related stack in this cluster: the local-inference-lab NVFP4 build (native MTP,
  eager, 900K ctx) — see the lab repo for that lineage.

## Tuning notes

Read `docs/FINDINGS.md` before changing memory knobs. Short version:

- **GMU is a fraction of *whole-system* RAM on GB10** and this runtime enforces it as a
  **fatal** pre-load check. Measure the ceiling on your build; ~0.88 here.
- **CUDA-graph memory profiling** (vLLM ≥ v0.21) taxes graph mode ~2.1 GiB of KV.
  `ESTIMATE_CUDAGRAPHS=0` returns it — with `CAPTURE_SIZE` small the estimate is waste.
- **`CAPTURE_SIZE` and `BATCH`/`SEQS` are KV-budget knobs**, not just perf knobs.
- Long-context changes are memory tests first: the engine's own `GPU KV cache size`
  line, not arithmetic, decides whether your context fits.
