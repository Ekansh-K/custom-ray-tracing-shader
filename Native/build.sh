#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$root/UnityRayTracer/Assets/Plugins/x86_64/libCudaRayTracer.so}"
embree="${EMBREE_ROOT:-$root/Native/ThirdParty/embree}"
[[ -f "$embree/include/embree3/rtcore.h" ]] || { echo "Run Tools/install_embree.sh first" >&2; exit 1; }
[[ -f "$root/Native/ThirdParty/optix/include/optix_stubs.h" ]] || { echo "Run Tools/install_optix.sh first" >&2; exit 1; }
mkdir -p "$(dirname -- "$output")"
trap 'rm -f -- "$output.tmp.so"' EXIT
# Replace the file atomically so an existing loader never sees a half-written library.
nvcc -O3 --fmad=false -std=c++17 -arch=sm_75 --shared -Xcompiler=-fPIC,-fopenmp --cudart shared \
  "$root/Native/renderer.cu" -I"$embree/include" -I"$root/Native/ThirdParty/optix/include" -ldl -lcurand -o "$output.tmp.so" \
  -Xlinker -rpath -Xlinker '$ORIGIN' \
  -Xlinker -rpath -Xlinker /usr/local/cuda/lib64 \
  -Xlinker -rpath -Xlinker "$embree/lib"
mv -- "$output.tmp.so" "$output"
