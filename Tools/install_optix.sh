#!/usr/bin/env bash
# OptiX 9.0 matches the installed R580 driver. Its runtime ships with NVIDIA's driver.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
commit=fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd
archive="$root/scratch/quality/optix-9.0.0.tar.gz"
prefix="$root/Native/ThirdParty/optix"
mkdir -p "$(dirname -- "$archive")" "$prefix"
if [[ ! -f "$archive" ]]; then
  curl -fL --retry 2 "https://codeload.github.com/NVIDIA/optix-dev/tar.gz/$commit" -o "$archive"
fi
echo "d5239da7ad098ae24afc6651d0cd3020ab0346a575ff4935a60e5494fd0bbe28  $archive" | sha256sum -c -
if [[ ! -f "$prefix/include/optix_stubs.h" ]]; then
  tar -xzf "$archive" -C "$prefix" --strip-components=1
fi
plugins="$root/UnityRayTracer/Assets/Plugins/x86_64"
cp "$prefix/LICENSE.txt" "$plugins/OptiX-LICENSE.txt"
cp "$prefix/license_info.txt" "$plugins/OptiX-license-info.txt"
echo 'OptiX 9.0 headers ready. NVIDIA R570 or later is required for the AI denoiser.'
