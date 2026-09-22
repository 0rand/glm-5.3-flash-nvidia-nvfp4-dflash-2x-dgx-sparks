// SPDX-License-Identifier: AGPL-3.0-only
// Owned, contiguous UVA span: ordinary registered RAM + DRM scanout carveout.
// No physical address access, modesetting, persistent changes or CUDA context creation.
#define _GNU_SOURCE
#include <cuda.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct {
    void *base;
    size_t ordinary, display, total;
    CUdeviceptr gpu;
    CUcontext context;
    int fd, ordinary_registered, display_registered;
    unsigned handle;
} Pool;
static _Thread_local char error_text[512];
const char *ds41_display_error(void) { return error_text; }
static int check_cuda(const char *op,CUresult rc) {
    if(rc==CUDA_SUCCESS)return 1;
    const char *name="unknown";cuGetErrorName(rc,&name);
    snprintf(error_text,sizeof(error_text),"%s: %s (%d)",op,name,rc);return 0;
}
static int check_sys(const char *op,int ok) {
    if(ok)return 1;
    snprintf(error_text,sizeof(error_text),"%s: %s",op,strerror(errno));return 0;
}
void ds41_display_destroy(Pool *p) {
    if(!p)return;
    // Caller owns tensor lifetime; never release while views/graphs are live.
    CUcontext previous=NULL;cuCtxGetCurrent(&previous);
    if(p->context)cuCtxSetCurrent(p->context);
    if(p->ordinary_registered||p->display_registered)cuCtxSynchronize();
    if(p->ordinary_registered)cuMemHostUnregister(p->base);
    if(p->display_registered)cuMemHostUnregister((char *)p->base+p->ordinary);
    if(p->base!=MAP_FAILED)munmap(p->base,p->total);
    if(p->handle){struct drm_gem_close c={.handle=p->handle};ioctl(p->fd,DRM_IOCTL_GEM_CLOSE,&c);}
    if(p->fd>=0)close(p->fd);
    if(previous!=p->context)cuCtxSetCurrent(previous);
    free(p);
}
Pool *ds41_display_create(size_t ordinary,size_t display) {
    error_text[0]=0;
    const size_t quantum=65536;
    // Canglong: raise ordinary cap for split-pool use (GLM 9 GiB KV backing).
    // Mechanics are size-agnostic; cap is configurable via env, default upstream.
    {
        const char *cap_env = getenv("DS41_ORDINARY_CAP_GIB");
        size_t cap = cap_env ? (size_t)atoi(cap_env) * 1024UL * 1024 * 1024 : 1024UL * 1024 * 1024;
        if(ordinary>cap||display!=1792UL*1024*1024||ordinary%quantum) {
            snprintf(error_text,sizeof(error_text),"ordinary exceeds cap (%zu MiB) in 64KiB units, or display != 1.75GiB",cap>>20);return NULL;
        }
    }
    Pool *p=calloc(1,sizeof(*p));if(!p)return NULL;
    p->base=MAP_FAILED;p->fd=-1;p->ordinary=ordinary;p->display=display;p->total=ordinary+display;
    if(!check_cuda("cuCtxGetCurrent",cuCtxGetCurrent(&p->context)))goto fail;
    if(!p->context){snprintf(error_text,sizeof(error_text),"No current CUDA context on caller thread");goto fail;}
    // Canglong patch: select DRM card via env (our efifb owns card0; NVIDIA GPU is card1).
    const char *card_path = getenv("DS41_DRM_CARD");
    if (!card_path || !*card_path) card_path = "/dev/dri/card0";
    p->fd=open(card_path,O_RDWR|O_CLOEXEC);
    if(!check_sys("open DRM card",p->fd>=0))goto fail;
    struct drm_mode_create_dumb c={.width=4096,.height=display/16384,.bpp=32};
    if(!check_sys("DRM create scanout",ioctl(p->fd,DRM_IOCTL_MODE_CREATE_DUMB,&c)==0))goto fail;
    p->handle=c.handle;
    if(c.size!=display){snprintf(error_text,sizeof(error_text),"Unexpected scanout size");goto fail;}
    struct drm_mode_map_dumb m={.handle=p->handle};
    if(!check_sys("DRM map offset",ioctl(p->fd,DRM_IOCTL_MODE_MAP_DUMB,&m)==0))goto fail;
    // MAP_FIXED is restricted to this freshly reserved private virtual span.
    p->base=mmap(NULL,p->total,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(!check_sys("reserve owned host VA",p->base!=MAP_FAILED))goto fail;
    if(ordinary && !check_sys("map ordinary prefix",mmap(p->base,ordinary,PROT_READ|PROT_WRITE,
        MAP_FIXED|MAP_PRIVATE|MAP_ANONYMOUS,-1,0)==p->base))goto fail;
    void *display_base=(char *)p->base+ordinary;
    if(!check_sys("map scanout suffix",mmap(display_base,display,PROT_READ|PROT_WRITE,
        MAP_FIXED|MAP_SHARED,p->fd,m.offset)==display_base))goto fail;
    CUdeviceptr ordinary_gpu=0,display_gpu=0;
    if(ordinary) {
        if(!check_cuda("register ordinary",cuMemHostRegister(p->base,ordinary,CU_MEMHOSTREGISTER_DEVICEMAP)))goto fail;
        p->ordinary_registered=1;
        if(!check_cuda("ordinary device pointer",cuMemHostGetDevicePointer(&ordinary_gpu,p->base,0)))goto fail;
    }
    if(!check_cuda("register display IO",cuMemHostRegister(display_base,display,
        CU_MEMHOSTREGISTER_DEVICEMAP|CU_MEMHOSTREGISTER_IOMEMORY)))goto fail;
    p->display_registered=1;
    if(!check_cuda("display device pointer",cuMemHostGetDevicePointer(&display_gpu,display_base,0)))goto fail;
    if((ordinary && ordinary_gpu!=(CUdeviceptr)(uintptr_t)p->base) ||
        display_gpu!=(CUdeviceptr)(uintptr_t)display_base) {
        snprintf(error_text,sizeof(error_text),"Driver did not preserve contiguous UVA: host=%p ordinary=%"PRIx64" display=%"PRIx64,
            p->base,(uint64_t)ordinary_gpu,(uint64_t)display_gpu);goto fail;
    }
    p->gpu=(CUdeviceptr)(uintptr_t)p->base;
    return p;
fail:
    ds41_display_destroy(p);return NULL;
}
uint64_t ds41_display_pointer(Pool *p){return p ? p->gpu : 0;}
size_t ds41_display_size(Pool *p){return p ? p->total : 0;}

#ifdef DS41_DISPLAY_PROBE_MAIN
int display_gpu_probe(CUdeviceptr,size_t);
int main(void) {
    setvbuf(stdout,NULL,_IONBF,0);
    CUdevice d;CUcontext ctx;
    if(cuInit(0)||cuDeviceGet(&d,0)||cuDevicePrimaryCtxRetain(&ctx,d)||cuCtxSetCurrent(ctx))return 2;
    Pool *p=ds41_display_create(1024UL*1024*1024,1792UL*1024*1024);
    if(!p){fprintf(stderr,"%s\n",error_text);return 1;}
    printf("{\"stage\":\"combined_contiguous_uva\",\"ordinary_bytes\":%zu,\"display_bytes\":%zu,\"total_bytes\":%zu,\"pointer\":%"PRIu64"}\n",p->ordinary,p->display,p->total,(uint64_t)p->gpu);
    int ok=display_gpu_probe(p->gpu,p->total);
    ds41_display_destroy(p);cuDevicePrimaryCtxRelease(d);
    printf("{\"status\":\"%s\"}\n",ok?"pass":"failed");return ok?0:1;
}
#endif
