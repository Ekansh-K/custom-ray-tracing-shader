#!/usr/bin/env bash
# Install the official optimized Linux x86-64 release locally, without sudo.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version=3.13.5
archive="$root/scratch/embree/embree-$version.x86_64.linux.tar.gz"
prefix="$root/Native/ThirdParty/embree"
mkdir -p "$(dirname -- "$archive")" "$prefix"
if [[ ! -f "$archive" ]]; then
  curl -fL --retry 2 "https://github.com/RenderKit/embree/releases/download/v$version/embree-$version.x86_64.linux.tar.gz" -o "$archive"
fi
echo "6454c9793e50213f681a34be3b0128f263caad0d663a1409f0bc2f98644bfa98  $archive" | sha256sum -c -
if [[ ! -f "$prefix/include/embree3/rtcore.h" || ! -f "$prefix/lib/libembree3.so.3" ]]; then
  tar -xzf "$archive" -C "$prefix" --strip-components=1 \
    "embree-$version.x86_64.linux/include" "embree-$version.x86_64.linux/lib" "embree-$version.x86_64.linux/doc"
fi
# Bundle native dependencies and notices for Unity Linux player builds. Hard-link
# where possible to avoid storing duplicate runtime libraries on this filesystem.
plugins="$root/UnityRayTracer/Assets/Plugins/x86_64"
mkdir -p "$plugins"
for name in libembree3.so.3 libtbb.so.12; do
  source="$(readlink -f "$prefix/lib/$name")"
  if [[ ! "$source" -ef "$plugins/$name" ]]; then
    ln -f "$source" "$plugins/$name" 2>/dev/null || cp "$source" "$plugins/$name"
  fi
done
cp "$prefix/doc/LICENSE.txt" "$plugins/Embree-LICENSE.txt"
cp "$prefix/doc/third-party-programs.txt" "$plugins/Embree-third-party.txt"
cp "$prefix/doc/third-party-programs-TBB.txt" "$plugins/TBB-third-party.txt"
echo "Official Embree $version and TBB runtime ready."
