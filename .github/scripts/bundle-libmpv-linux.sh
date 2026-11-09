#!/usr/bin/env bash
# Copies a libmpv.so.2 and the libraries it needs into a folder that can sit beside flex, with
# each one finding the others there. What belongs to the machine (the C library, graphics,
# sound, the display) is left out, along with any library that has to match its drivers.
#
#   bundle-libmpv-linux.sh <libmpv.so.2> <output dir>
#
# Needs: bash, ldd, patchelf, dpkg (for the copyright files).
set -euo pipefail

LIBMPV=$1
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
HOST='^(ld-linux|libc|libm|libdl|libpthread|librt|libresolv|libgcc_s|libstdc\+\+|libGL|libEGL|libGLX|libGLdispatch|libOpenGL|libgbm|libdrm|libva|libvdpau|libX|libxcb|libxkbcommon|libwayland|libasound|libpulse|libpipewire|libdbus|libsystemd|libudev|libcap|libelf)'

mkdir -p "$OUT/libmpv-deps" "$OUT/licenses/libmpv-deps"
cp -L "$LIBMPV" "$OUT/libmpv.so.2"
patchelf --set-rpath '$ORIGIN/libmpv-deps' "$OUT/libmpv.so.2"
ldd "$LIBMPV" | awk '/=> \// { print $3 }' | while read -r lib; do
    name=$(basename "$lib")
    [[ $name =~ $HOST ]] && continue
    cp -L "$lib" "$OUT/libmpv-deps/$name"
    chmod u+w "$OUT/libmpv-deps/$name"
    patchelf --set-rpath '$ORIGIN' "$OUT/libmpv-deps/$name"
    real=$(readlink -f "$lib")
    package=$(dpkg -S "$real" 2>/dev/null | head -1 | cut -d: -f1 || dpkg -S "${real#/usr}" 2>/dev/null | head -1 | cut -d: -f1 || true)
    if [[ -n $package && -f /usr/share/doc/$package/copyright ]]; then
        cp "/usr/share/doc/$package/copyright" "$OUT/licenses/libmpv-deps/$package.txt"
    fi
done
echo "Bundled $(ls "$OUT/libmpv-deps" | wc -l) libraries, $(du -sh "$OUT" | cut -f1)"
