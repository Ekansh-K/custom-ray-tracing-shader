#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
exec "$HOME/Unity/Hub/Editor/6000.3.26f1/Editor/Unity" \
  -projectPath "$project_root/UnityRayTracer" -force-vulkan \
  -acceptSoftwareTermsForThisRunOnly "$@"
