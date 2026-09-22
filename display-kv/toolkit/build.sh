#!/bin/bash
# Build the display-KV probe (host-side, AGPL source from coolbho3k repo).
# Usage: ./build.sh          -> builds libdisplay_kv.so + probe binary
set -e
cd "$(dirname "$0")"
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=$(dirname $(ls /usr/lib/*/libcuda.so.1 2>/dev/null | head -1) 2>/dev/null || echo /usr/local/cuda/lib64)
echo "CUDA_INC=$CUDA_INC CUDA_LIB=$CUDA_LIB"
gcc -O2 -fPIC -shared -I"$CUDA_INC" \
    -o libdisplay_kv.so repo/release/runtime/sources/display_kv.c -L"$CUDA_LIB" -lcuda
gcc -O2 -I"$CUDA_INC" \
    -o probe_display_kv probe_display_kv.c -L"$CUDA_LIB" -lcuda -lpthread
echo "BUILD OK:"
ls -la libdisplay_kv.so probe_display_kv
