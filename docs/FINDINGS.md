# FINDINGS — GLM-5.3-Flash NVIDIA NVFP4 on 2× DGX Spark

Everything here was measured on a 2-node GB10 cluster (head + worker over RoCE),
2026-09-12/13, with the pilcothink `0.28` runtime.
Numbers are the engine's own; where something is extrapolated it says so.

---

## 1. Quality-verified 700K baseline

```
image     pilcothink/vllm_spark_glm53:0.28
weights   nvidia/GLM-5.3-Flash-NVFP4 @ 09b04e5e74bca08ca8549fc736d4cdd8624bfde3
draft     glm53-dflash2-orig   (DFlash2, k=5)
GMU 0.88 · ctx 700,160 · seqs 4 · batch 1024 · capture <=16 · async ON
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
```

Result:

| metric | value |
|---|---|
| KV pool | **892,139 tokens**, 1.27x at 700,160 |
| hardmode | **94/100** (166/176), Hard Mode **38/38** |
| e2e | **996 s** |
| boot | 980 s (weights 744 s, engine init 118 s) |
| spec-bench (warmed) | structured 49.7 t/s α=88% · code 43.0 α=71% · filler 33.5 α=52% |
| throughput c1 | decode 32.5/33.1/33.1 t/s at depth 0/2K/8K · TTFT 1.02/3.19/10.15 s |
| failures | TC-43 (empty `web_search` query), TC-68 (invalid JSON); safety flag TC-47 (duplicate event during a correction) |

TC-47 reproduces the earlier 262K run — pre-existing behaviour, not a regression.

### 1.1 Launch-verified 900K profile

The persisted long-context profile keeps the same DFlash2/graph schedule and adds the
upstream 9 GiB KV allocation:

```
GMU 0.88 · ctx 900,096 · KV 9 GiB · seqs 4 · batch 1024 · capture <=16 · async ON
```

Measured result:

| metric | value |
|---|---|
| boot | **PASS**, ~980 s; `logs/boot-20260913-163451.log` |
| KV pool | **1,080,115 tokens**, 1.20x at 900,096 |
| graph capture | 12 s, 0.59 GiB |
| API smoke | HTTP 200; exact content `DRAGON ONLINE` |
| warmed pp1024/tg1024 c1 | **32.0 t/s** decode, 1,245 t/s prefill |
| pp1024/tg512 c1, depths 0/2K/8K | **40.3 / 28.8 / 33.7 t/s** decode |
| hardmode | **not rerun**; do not attribute the 700K profile's 94/100 here |

The first cold pp1024/tg1024 pass was 27.9 t/s; the identical warmed pass was 32.0.
DFlash counters on the warmed run proved k=5 was live: 1,530 draft tokens over 306
draft steps, with 725 accepted (2.37 accepted tokens/step; 47.4% token acceptance).

`GMU=0.885` was tested first and rejected by the engine itself: 107.28 GiB free versus
107.69 GiB demanded. The working profile therefore uses 0.88 while the explicit KV pin
preserves 900K context without stopping Forgejo, SearXNG, or Hermes services.

---

### 1.2 The CURRENT production profile — display-KV + k=4 (Primo's preferred, 2026-09-26/27)

The live `.env` has moved past both profiles above. This is the recipe actually serving
:8100 since the 2026-09-26/27 boot (`logs/boot-20260927-003712.log`), and it is
**Primo's preferred trade**: k was cut 5→4, batch raised 1024→2048, GMU trimmed to 0.87,
and the display-KV unlock (+1.75 GiB/rank) is active. Reasoning (prose) dominates the
output mix and still runs ~35 t/s effective; code/structured give up a little decode vs
k=5 but hold higher acceptance (78-83% at depth 0, up to 92% on structured bursts).

```
image     pilcothink/vllm_spark_glm53:0.28
weights   nvidia/GLM-5.3-Flash-NVFP4 @ 09b04e5e74bca08ca8549fc736d4cdd8624bfde3
draft     glm53-dflash2-orig (DFlash2, k=4, probabilistic draft / standard rejection)
GMU 0.87 · KV pin 10,240,000,000 B (+1,879,048,192 B display suffix via display-KV shim)
ctx 1,048,576 (native 4096x256, no YaRN) · seqs 4 · batch 2048 · capture <=8 · async ON
graphs ON (FULL_AND_PIECEWISE), ESTIMATE_CUDAGRAPHS=0, kv_cache_dtype fp8
launch    ./start-display-kv.sh   (worker addr = bare WORKER_IP; SSH target separate)
```

Measured on this boot:

| metric | value |
|---|---|
| KV pool | **1,399,445 tokens** (1.33x at 1,048,576) — display-KV +18.4% over the 10 GiB pin |
| DFlash acceptance | mean acceptance length **median 3.35** over 1,202 windows (~64% @ k=4) |
| decode (real traffic, 1,089 windows >20 t/s) | median 37.8 t/s, max 73.2 |
| spec-bench d0 (2026-09-27T07-51, temp 0.9) | filler 35.7 (α 65%) · code 41.4 (α 79%) · structured 42.0 (α 83%) |
| throughput d0/4K/8K (2026-09-27T07-56) | 35.3 / 34.7 / 33.8 t/s tg128, TTFT 2.3/6.0/8.7 s at pp2048 |
| hardmode (v2.7.1, 184-pt set) | **89-95 spread across 11 runs 2026-09-26/27**, median 91, best 95 (174/184) |

Quality under the new 184-point bench matches the old 176-point record (91-95 band
identical). The k=4 + batch-2048 + display-KV combination is the keeper. The k=6 probe
(boot-20260926-224348) and k=5 probe were both tested the same night and rejected in
favor of k=4.

## 2. Memory model on GB10 (unified memory)

Three facts stack up, and missing any one of them produces a wrong boot decision.

**2.1 `gpu_memory_utilization` is a fraction of *system* RAM.**
`request_memory()` (`vllm/v1/worker/utils.py`) demands
`ceil(init_snapshot.total_memory × GMU)` be **free at worker init, before weights load**.
On a discrete card that is a slice of VRAM; on GB10 `total_memory` = **121.69 GiB =
the whole machine**, OS and containers included. GMU 0.895 therefore requires 108.91 GiB
*free*, which a box also running host services cannot supply.

**2.2 Whether that check is fatal depends on the build.**

| build | behaviour at 0.895 |
|---|---|
| `pilcothink/vllm_spark_glm53:0.28` | **raises ValueError** — boot dead in ~45 s |
| `glm53-flash-lab:local` (LIL stack) | logs the same numbers as **INFO** and proceeds |

Measured free-at-check on the head: **106.88 / 107.39 GiB** (pilcothink, cache-warm and
post-cache-drop) vs **109.07–109.74 GiB** (lab build). This is why "0.895 ran all the time"
on the lab stack and cannot run here — the gate, not the hardware, was the difference.
**Never port a GMU value between images.** Practical ceiling on this box: **0.88**.

**2.3 Page cache is not the main term.** A full cache drop moved free-at-check by only
**+0.5 GiB**. The bulk is the engine's own pre-check footprint (~6–7 GiB: APIServer +
EngineCore + WorkerProc + NCCL + warmups) plus OS/container overhead.

---

## 3. The CUDA-graph memory estimator tax (vLLM ≥ v0.21)

**The single most expensive surprise of this work.** In graph mode the engine reserves
part of the KV budget for a CUDA-graph memory estimate, and says so in one INFO line:

```
[gpu_worker.py:696] CUDA graph memory profiling is enabled (default since v0.21.0).
The current --gpu-memory-utilization=0.8800 is equivalent to --gpu-memory-utilization=0.8642
without CUDA graph memory profiling. ... To disable, set
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
```

Immediately above it, the two numbers that matter:

```
[gpu_worker.py:681] Available KV cache memory: 7.28 GiB     <- before the reservation
... available KV cache memory (5.17 GiB)                    <- after
```

**It cost 2.11 GiB of pool.** With `CAPTURE_SIZE=16` the estimate was pure waste — the
capture itself finished in 2 s and **freed 0.74 GiB** (negative!). Disabling it took the
pool from 5.17 → 7.28 GiB. Same effect at `CAPTURE_SIZE=64`: GMU 0.87 was "equivalent to
0.8516", i.e. −2.24 GiB.

**Rule:** in graph mode, check that line. If the pool is the binding constraint and the
capture set is small, disable the estimator — then watch the capture step for OOM.

---

## 4. KV budget levers (all measured, same GMU 0.87 unless noted)

**4.1 Eager vs graphs.** capture ≤64 cost **~4.0 GiB** of pool:

| mode | available KV | tokens |
|---|---|---|
| eager | 11.41 GiB | 921,732 |
| graphs ≤64 | 7.41 GiB | 544,019 |

**4.2 Capture size.** `--max-cudagraph-capture-size 64` yields capture sizes
`[1,2,4,8,16,24,32,40,48,56,64]` = 295 batch-slots. Capping at **16** keeps 31 slots and
returns **~3.6 GiB** to KV. On a single-sequence workload the high slots are pure waste.

**4.3 Batch and sequence count.** With GME off, at ctx 700,160:

| batch | seqs | available KV |
|---|---|---|
| 2048 | 10 | 7.28 GiB |
| **1024** | **4** | **9.28 GiB** (+2.00) |

Larger batches take RAM away from the KV cache hard, and they are not needed for quality.

**4.4 The 700K pool grows as +4.11 GiB total:** 5.17 → 9.28 GiB = GME off
(+2.11) + batch/seqs trim (+2.00). That is the measured difference between "700K does
not fit" and 94/100 at 700K.

Do **not** extrapolate that 892,139-token pool to a larger `max_model_len`. The runtime's
non-KV allocations change with the configured context: at `CTX=900096`, GMU 0.88 and no
pin, the engine measured only 5.85 GiB available and estimated a 668,160-token ceiling.
The working 900K profile uses the upstream explicit 9 GiB KV allocation and measures a
1,080,115-token pool. Engine output, not per-token arithmetic, is authoritative.

---

## 5. Failure catalogue (all three real ones)

**5.1 GMU gate — fatal, dies before weights load (~45 s in).**
```
ValueError: Free memory on device cuda:0 (106.88/121.69 GiB) on startup is less than
desired GPU memory utilization (0.89, 108.3 GiB).
```
Try: measure free-at-check on your build, set GMU ≤ ceiling. On this box 0.88 passes,
0.89/0.895 cannot.

**5.2 KV-too-small for the requested context — dies after weights load (~15 min in).**
```
ValueError: To serve at least one request with the model's max seq len (700160),
(6.11 GiB KV cache is needed, which is larger than the available KV cache memory
(5.17 GiB). Based on the available memory, the estimated maximum model length is 566784.
```
The engine states the largest context that *does* fit — use it instead of guessing.
This is what the GME tax and the batch/seqs trim were actually fighting.

**5.3 `${VAR:+--flag}` fires when `VAR=0`.** `${ENFORCE_EAGER:+--enforce-eager}` expands
for any set-and-non-empty value, so `ENFORCE_EAGER=0` passed `--enforce-eager` and vLLM
cancelled the graphs ("Enforce eager set, disabling torch.compile and CUDAGraphs") while
the launcher still printed "graphs ON". A whole night of "graph" runs was actually eager.
**Fix:** explicit `[ "$VAR" = "1" ]`, and verify the engine's resolved config
(`cudagraph_mode`) rather than trusting your own flags.

**5.4 SSH target leaked into the cluster bind address — rank 1 dies, rank 0 reports Gloo.**

The launcher accepts one worker list for both SSH and node addressing. Passing
`WORKER_SSH_TARGET=xraan@192.168.0.88` produced the worker-side root cause:

```
zmq.error.ZMQError: No such device (addr='tcp://xraan@192.168.0.88:0')
```

Rank 0 then emitted only the misleading secondary error:

```
gloo/transport/tcp/pair.cc:553] Connection closed by peer [192.168.0.88]:...
```

Fix at the source: pass the bare `WORKER_IP` to upstream `--workers`; use
`WORKER_SSH_TARGET` only for explicit SSH cleanup/sync operations. The successful 900K
boot resolves `master_addr=192.168.0.8`, `mq_connect_ip=192.168.0.8`, and both ranks
remain alive. `vendor/run_cluster_dual.sh` is unchanged from upstream (SHA-256
`ad3a7bfc8a47989fa4c74e4f8c6d77acf44a60c36b523ef84dbd6d211267be83`).

---

## 6. Measurement protocol

- **Two spec-bench passes.** The first is JIT/warm-up evidence and is slower; the *second*
  is the warmed result. Both are kept, only the second is quoted.
- **Hardmode is the deployment decision**: 88 scenarios, parallel 4, temp 0, seed 42,
  max-turns 32, timeout 360, thinking on, `reasoning_effort=high`, wall clock measured
  around the invocation (`replicate-hardmode.sh`). Also run it with a generous budget —
  a clipped run is not a result.
- **Always quote the bench build.** The evaluator moved 40 commits in one day and scores
  moved with it.
- **Parallel-4 wall time is not a clean speed ranking** when the two arms use different
  speculative schedulers (MTP scales with multiple sequences; DFlash2 suffers when new
  prefills join decoding). Quality and per-workload spec rates are the comparable numbers.
- **Prefill and decode separately.** `pp1024/tg512` at `c1`, depths 0/2048/8192.
  TTFT is where depth hurts; decode holds flat at 33 t/s.

---

## 7. Environment facts worth keeping

- **drop caches without root** (page cache is reclaimable; free *is* what the gate reads):
  ```
  docker run --rm --privileged -v /proc:/hostproc alpine \
    sh -c 'sync; echo 3 > /hostproc/sys/vm/drop_caches'
  ```
  Bought 116.37 GiB free on the head, which is what let GMU 0.88 pass the gate.
- **Image distribution: pull once, ship over the fabric** (`docker save | ssh docker load`,
  RoCE fabric). Never `docker pull` the same image per node.
- **`--kv-cache-memory` does not bypass the GMU gate** — the worker still runs
  `request_memory()` first. The pin only changes KV sizing, not the admission check.
- **A failed boot tears down its own container.** Poll health *and* container existence
  and abort early; otherwise the runner sleeps its full 40-minute timeout against nothing.

---

## 8. Open items

- Run the full hardmode suite on the 900K profile; the 700K score must not be transferred.
- The GME reservation vs actual capture cost should be re-checked on future image bumps —
  if the estimator gets smarter, `ESTIMATE_CUDAGRAPHS=0` may stop being the right default.
- TC-47 (duplicate event during correction) and TC-68 (invalid JSON) are reproducible
  model behaviours worth tracking across builds; they are not config-dependent so far.
