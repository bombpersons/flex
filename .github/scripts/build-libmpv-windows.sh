#!/usr/bin/env bash
# Builds a single-file libmpv-2.dll with Flex's patch, using the same toolchain as the libmpv
# builds Flex used to ship. Runs in its container, and takes hours the first time.
#
#   build-libmpv-windows.sh <output dir>
set -euo pipefail

OUT=$(mkdir -p "$1" && cd "$1" && pwd)
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TOOLCHAIN=bda10a2816ce9908d419b3f4b564227330457e76
MPV_COMMIT=$(cat "$ROOT/packaging/mpv/windows/COMMIT")

git config --global --add safe.directory '*'
git config --global user.name flex-build
git config --global user.email build@localhost
WORK=$(mktemp -d)
git clone https://github.com/shinchiro/mpv-winbuild-cmake "$WORK/toolchain"
cd "$WORK/toolchain"
git checkout -q "$TOOLCHAIN"
PATCH=$(ls "$ROOT"/packaging/mpv/windows/*.patch)
sed -i "s|^    UPDATE_COMMAND \"\"|    GIT_TAG $MPV_COMMIT\n    UPDATE_COMMAND \"\"\n    PATCH_COMMAND git -C <SOURCE_DIR> apply $PATCH|" packages/mpv.cmake
grep -q "GIT_TAG $MPV_COMMIT" packages/mpv.cmake

BUILD=build_x86_64
mkdir -p clang_root
cmake -DTARGET_ARCH=x86_64-w64-mingw32 -DSINGLE_SOURCE_LOCATION="$PWD/src_packages" -DRUSTUP_LOCATION="$PWD/clang_root/install_rustup" -G Ninja -B "$BUILD" -S "$PWD"
ninja -C "$BUILD" download || true
ninja -C "$BUILD" gcc
rm -rf "$BUILD/toolchain"
ninja -C "$BUILD" mpv
DLL=$(find "$BUILD" -name libmpv-2.dll -size +1M | head -1)
[[ -n $DLL ]] || { echo "error: no libmpv-2.dll was built" >&2; exit 1; }
grep -q mpv_flex_set_audio_tap "$DLL" || { echo "error: the DLL isn't patched" >&2; exit 1; }
cp "$DLL" "$OUT/libmpv-2.dll"
