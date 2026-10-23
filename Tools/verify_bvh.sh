#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
embree="${EMBREE_ROOT:-$root/Native/ThirdParty/embree}"
mkdir -p "$root/scratch/bvh"
nvcc -O3 --fmad=false -std=c++17 -arch=sm_75 -Xcompiler=-fopenmp \
  "$root/Tools/verify_bvh.cu" -I"$embree/include" -I"$root/Native/ThirdParty/optix/include" -ldl -lcurand \
  -Xlinker -rpath -Xlinker "$embree/lib" -o "$root/scratch/bvh/verify-bvh"
if [[ "${1:-}" == "--sanitize" ]]; then
  exec compute-sanitizer --tool memcheck --error-exitcode 1 "$root/scratch/bvh/verify-bvh"
fi
exec "$root/scratch/bvh/verify-bvh"
