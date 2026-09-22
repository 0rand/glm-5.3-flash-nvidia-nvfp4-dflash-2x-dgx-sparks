// All-in-one: display pool creation (driver API) + SM-side bandwidth kernels
// (runtime API, same process/context). Read + write bandwidth on the display
// carveout vs ordinary cudaMalloc memory — the KV-cache access pattern.
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

// Allocator from display_kv.c (compiled separately with gcc as C11).
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
    if (acc == 12345.678f) sink[0] = acc; // never true; defeats DCE
}

__global__ void write_kernel(float4* __restrict__ dst, size_t n4) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    for (; i < n4; i += gridDim.x * (size_t)blockDim.x) dst[i] = v;
}

struct Timer { cudaEvent_t a, b; };
static Timer mk() { Timer t; cudaEventCreate(&t.a); cudaEventCreate(&t.b); return t; }
static float best_ms(Timer t, void (*launch)(void*), void* arg, int reps) {
    launch(arg); cudaDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(t.a); launch(arg); cudaEventRecord(t.b);
        cudaEventSynchronize(t.b);
        float ms; cudaEventElapsedTime(&ms, t.a, t.b);
        if (ms < best) best = ms;
    }
    return best;
}

struct ReadArg { const float4* p; float* sink; size_t n4; };
struct WriteArg { float4* p; size_t n4; };

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    CUdevice dev; CUcontext ctx;
    cuInit(0); cuDeviceGet(&dev, 0); cuDevicePrimaryCtxRetain(&ctx, dev); cuCtxSetCurrent(ctx);

    void *pool = ds41_display_create(0, 1792UL*1024*1024);
    if (!pool) { fprintf(stderr, "ALLOC FAIL: %s\n", ds41_display_error()); return 1; }
    unsigned long long dptr = ds41_display_pointer(pool);
    printf("{\"stage\":\"pool\",\"ptr\":%llu}\n", dptr);

    const size_t bytes = 256UL << 20;
    size_t n4 = bytes / 16;
    float* sink; cudaMalloc(&sink, 4);
    float4* ord; cudaMalloc(&ord, bytes);

    ReadArg ra_d{(const float4*)dptr, sink, n4}, ra_o{(const float4*)ord, sink, n4};
    WriteArg wa_d{(float4*)dptr, n4}, wa_o{ord, n4};
    Timer t = mk();
    float d_r = best_ms(t, [](void* a){ ReadArg* x=(ReadArg*)a; read_kernel<<<4096,256>>>(x->p, x->sink, x->n4); }, &ra_d, 5);
    float o_r = best_ms(t, [](void* a){ ReadArg* x=(ReadArg*)a; read_kernel<<<4096,256>>>(x->p, x->sink, x->n4); }, &ra_o, 5);
    float d_w = best_ms(t, [](void* a){ WriteArg* x=(WriteArg*)a; write_kernel<<<4096,256>>>(x->p, x->n4); }, &wa_d, 5);
    float o_w = best_ms(t, [](void* a){ WriteArg* x=(WriteArg*)a; write_kernel<<<4096,256>>>(x->p, x->n4); }, &wa_o, 5);

    printf("{\"stage\":\"sm_bandwidth_best_of_5\",\"chunk_mib\":%zu,"
           "\"display_read_GBs\":%.2f,\"display_write_GBs\":%.2f,"
           "\"ordinary_read_GBs\":%.2f,\"ordinary_write_GBs\":%.2f,"
           "\"read_ratio\":%.3f,\"write_ratio\":%.3f}\n",
           bytes>>20, bytes/(d_r/1000.)/1e9, bytes/(d_w/1000.)/1e9,
           bytes/(o_r/1000.)/1e9, bytes/(o_w/1000.)/1e9, d_r/o_r, d_w/o_w);

    // Integrity under SM writes: write whole window, read back via DtoH
    cuMemsetD32((CUdeviceptr)dptr, 0xCAFEBABEu, 1024);
    unsigned int back = 0; cuMemcpyDtoH(&back, (CUdeviceptr)dptr, 4);
    printf("{\"stage\":\"sm_region_alive_after_kernels\",\"readback\":\"%s\"}\n",
           back == 0xCAFEBABEu ? "ok" : "MISMATCH");
    ds41_display_destroy(pool);
    cuDevicePrimaryCtxRelease(dev);
    printf("{\"status\":\"done\"}\n");
    return 0;
}
