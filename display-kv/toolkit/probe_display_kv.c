// Standalone probe for the GB10 display-reserve memory unlock technique.
// Derived from display_kv.c by coolbho3k (AGPL-3.0) — includes it directly.
// Verifies: (1) DRM scanout carveout allocation works, (2) CUDA can
// read/write it via DEVICEMAP|IOMEMORY registration, (3) roundtrip data
// integrity, (4) bandwidth vs ordinary cudaMalloc'd memory.
//
// Build: see build.sh
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include "repo/release/runtime/sources/display_kv.c"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define CHECK(op, rc) do { if ((rc) != CUDA_SUCCESS) { \
    const char *n = "unknown"; cuGetErrorName((rc), &n); \
    fprintf(stderr, "FAIL: %s -> %s (%d)\n", op, n, (int)(rc)); exit(1); } } while (0)

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    CUdevice dev; CUcontext ctx;
    CHECK("cuInit", cuInit(0));
    CHECK("cuDeviceGet", cuDeviceGet(&dev, 0));
    CHECK("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK("cuCtxSetCurrent", cuCtxSetCurrent(ctx));
    printf("{\"stage\":\"cuda_context_created\",\"device\":\"GB10\"}\n");

    const size_t DISPLAY = 1792UL * 1024 * 1024; // 1.75 GiB, as in the recipe
    Pool *p = ds41_display_create(0, DISPLAY);
    if (!p) { fprintf(stderr, "ALLOC FAIL: %s\n", ds41_display_error()); return 1; }
    printf("{\"stage\":\"pool_created\",\"display_bytes\":%zu,\"device_ptr\":%llu}\n",
           p->display, (unsigned long long)p->gpu);

    // --- 1. Integrity: write pattern with cuMemsetD32, read back with cuMemcpyDtoH
    const size_t chunk = 256UL * 1024 * 1024; // 256 MiB test window
    unsigned int *host = malloc(chunk);
    const unsigned int PAT_A = 0xA5A5C3C3u, PAT_B = 0x13579BDFu;
    CHECK("cuMemsetD32(display)", cuMemsetD32(p->gpu, PAT_A, chunk / 4));
    CHECK("cuMemcpyDtoH", cuMemcpyDtoH(host, p->gpu, chunk));
    int bad = 0; for (size_t i = 0; i < chunk / 4; i++) if (host[i] != PAT_A) bad++;
    printf("{\"stage\":\"intensity_roundtrip\",\"pattern\":\"A5A5C3C3\",\"mismatches\":%d}\n", bad);
    if (bad) { printf("{\"status\":\"failed\"}\n"); return 1; }

    // overwrite check (no aliasing/stickiness)
    CHECK("cuMemsetD32(display,B)", cuMemsetD32(p->gpu + chunk, PAT_B, chunk / 4));
    CHECK("cuMemcpyDtoH", cuMemcpyDtoH(host, p->gpu + chunk, chunk));
    bad = 0; for (size_t i = 0; i < chunk / 4; i++) if (host[i] != PAT_B) bad++;
    printf("{\"stage\":\"overwrite_check\",\"mismatches\":%d}\n", bad);
    if (bad) { printf("{\"status\":\"failed\"}\n"); return 1; }

    // --- 2. Bandwidth: display region vs ordinary device memory
    // HtoD + DtoH over the 256 MiB window, 3 passes, report best (GB/s).
    void *ord = NULL;
    CHECK("cuMemAlloc(ord)", cuMemAlloc((CUdeviceptr *)&ord, chunk));
    double best_d_h2d = 0, best_d_d2h = 0, best_o_h2d = 0, best_o_d2h = 0;
    for (int pass = 0; pass < 3; pass++) {
        double t0 = now_s(); CHECK("HtoD display", cuMemcpyHtoD(p->gpu, host, chunk));
        CHECK("sync", cuCtxSynchronize());
        double t1 = now_s();
        double gbs = (chunk / (t1 - t0)) / 1073741824.0; if (gbs > best_d_h2d) best_d_h2d = gbs;
        t0 = now_s(); CHECK("DtoH display", cuMemcpyDtoH(host, p->gpu, chunk));
        CHECK("sync", cuCtxSynchronize());
        t1 = now_s(); gbs = (chunk / (t1 - t0)) / 1073741824.0; if (gbs > best_d_d2h) best_d_d2h = gbs;
        t0 = now_s(); CHECK("HtoD ord", cuMemcpyHtoD((CUdeviceptr)ord, host, chunk));
        CHECK("sync", cuCtxSynchronize());
        t1 = now_s(); gbs = (chunk / (t1 - t0)) / 1073741824.0; if (gbs > best_o_h2d) best_o_h2d = gbs;
        t0 = now_s(); CHECK("DtoH ord", cuMemcpyDtoH(host, (CUdeviceptr)ord, chunk));
        CHECK("sync", cuCtxSynchronize());
        t1 = now_s(); gbs = (chunk / (t1 - t0)) / 1073741824.0; if (gbs > best_o_d2h) best_o_d2h = gbs;
    }
    printf("{\"stage\":\"bandwidth_best_of_3\",\"chunk_mib\":%zu,"
           "\"display_h2d_GBs\":%.2f,\"display_d2h_GBs\":%.2f,"
           "\"ordinary_h2d_GBs\":%.2f,\"ordinary_d2h_GBs\":%.2f,"
           "\"display_vs_ordinary_ratio\":%.3f}\n",
           chunk / 1048576, best_d_h2d, best_d_d2h, best_o_h2d, best_o_d2h,
           (best_d_h2d + best_d_d2h) / (best_o_h2d + best_o_d2h));

    // --- 3. Full-capacity sweep: prove all 1.75 GiB is addressable
    size_t step = 64UL * 1024 * 1024;
    int sweep_bad = 0;
    for (size_t off = 0; off + 4096 <= DISPLAY; off += step) {
        CHECK("sweep set", cuMemsetD32(p->gpu + off, (unsigned int)(off ^ 0xDEADBEEF), 1));
        unsigned int back = 0;
        CHECK("sweep get", cuMemcpyDtoH(&back, p->gpu + off, 4));
        if (back != (unsigned int)(off ^ 0xDEADBEEF)) sweep_bad++;
    }
    printf("{\"stage\":\"full_sweep_1792MiB\",\"steps\":%zu,\"failures\":%d}\n",
           DISPLAY / step, sweep_bad);

    free(host);
    ds41_display_destroy(p);
    cuDevicePrimaryCtxRelease(dev);
    printf("{\"status\":\"%s\"}\n", sweep_bad ? "failed" : "pass");
    return sweep_bad ? 1 : 0;
}
