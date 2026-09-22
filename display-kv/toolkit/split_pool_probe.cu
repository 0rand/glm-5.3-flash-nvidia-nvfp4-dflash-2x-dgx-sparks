// Split-pool probe: contiguous UVA span = ordinary-RAM prefix + display suffix.
// This is the GLM-stack shape: KV backed by (ordinary + 1.75 GiB display) as ONE
// device pointer. Tests up to 8 GiB ordinary prefix + integrity + SM bandwidth
// on both segments. Same-process SM kernels (registration dies with the process).
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

extern "C" {
void* ds41_display_create(size_t ordinary, size_t display);
unsigned long long ds41_display_pointer(void* pool);
void ds41_display_destroy(void* pool);
const char* ds41_display_error(void);
}

__global__ void read_kernel(const float4* __restrict__ src, float* __restrict__ sink, size_t n4) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float acc = 0.f;
    for (; i < n4; i += gridDim.x * (size_t)blockDim.x) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) sink[0] = acc;
}

static void chk(CUresult rc, const char* op) {
    if (rc != CUDA_SUCCESS) {
        const char* n = "unknown"; cuGetErrorName(rc, &n);
        fprintf(stderr, "FAIL: %s -> %s\n", op, n); exit(1);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    size_t ordinary = (argc > 1) ? strtoull(argv[1], nullptr, 0) * (size_t)1024 * 1024 : (size_t)7 * 1024 * 1024 * 1024;
    CUdevice dev; CUcontext ctx;
    cuInit(0); cuDeviceGet(&dev, 0); cuDevicePrimaryCtxRetain(&ctx, dev); cuCtxSetCurrent(ctx);

    void* pool = ds41_display_create(ordinary, 1792UL * 1024 * 1024);
    if (!pool) { fprintf(stderr, "ALLOC FAIL: %s\n", ds41_display_error()); return 1; }
    unsigned long long dptr = ds41_display_pointer(pool);
    printf("{\"stage\":\"pool\",\"ordinary_mib\":%zu,\"display_mib\":1792,\"ptr\":%llu}\n",
           ordinary >> 20, dptr);

    // Integrity on both segments
    const size_t win = 64UL << 20;
    unsigned int* host = (unsigned int*)malloc(win);
    chk(cuMemsetD32((CUdeviceptr)dptr, 0x11111111u, win / 4), "memset ord");
    chk(cuMemcpyDtoH(host, (CUdeviceptr)dptr, win), "dtoh ord");
    int bad = 0; for (size_t i = 0; i < win / 4; i++) if (host[i] != 0x11111111u) bad++;
    printf("{\"stage\":\"ordinary_integrity\",\"mismatches\":%d}\n", bad);

    unsigned long long disp = dptr + ordinary;
    chk(cuMemsetD32((CUdeviceptr)disp, 0x22222222u, win / 4), "memset disp");
    chk(cuMemcpyDtoH(host, (CUdeviceptr)disp, win), "dtoh disp");
    bad = 0; for (size_t i = 0; i < win / 4; i++) if (host[i] != 0x22222222u) bad++;
    printf("{\"stage\":\"display_integrity\",\"mismatches\":%d}\n", bad);

    // SM read bandwidth on both segments (same kernels, same process)
    const size_t bytes = 256UL << 20;
    float* sink; cudaMalloc(&sink, 4);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto bench = [&](const void* p) {
        size_t n4 = bytes / 16;
        read_kernel<<<4096, 256>>>((const float4*)p, sink, n4); cudaDeviceSynchronize();
        float best = 1e30f;
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(a);
            read_kernel<<<4096, 256>>>((const float4*)p, sink, n4);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            if (ms < best) best = ms;
        }
        return (float)((double)bytes / (best / 1000.0)) / 1e9f;
    };
    float ord_gbs = bench((const void*)dptr);
    float disp_gbs = bench((const void*)disp);
    printf("{\"stage\":\"sm_read\",\"ordinary_GBs\":%.2f,\"display_GBs\":%.2f,\"ratio\":%.3f}\n",
           ord_gbs, disp_gbs, disp_gbs / ord_gbs);

    free(host);
    ds41_display_destroy(pool);
    cuDevicePrimaryCtxRelease(dev);
    printf("{\"status\":\"done\"}\n");
    return 0;
}
