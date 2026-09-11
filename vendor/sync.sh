#!/usr/bin/env bash
# One-way sync of the vendored FlashRT kernel sources.
#
#   vendor/sync.sh <FlashRT checkout> [commit]
#
# Copies every path listed in FLASHRT_VENDOR.txt from the FlashRT checkout
# (at the given commit, default: the commit recorded in the manifest),
# rewrites the manifest with the new commit and hashes, and refuses if the
# checkout is dirty. Kernels are never edited here: change them in
# FlashRT, then run this.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
manifest="$here/FLASHRT_VENDOR.txt"
src="${1:?usage: sync.sh <FlashRT checkout> [commit]}"
commit="${2:-$(sed -n 's/^flashrt_commit=//p' "$manifest")}"
[ -z "$(git -C "$src" status --porcelain)" ] || { echo "refusing: $src has uncommitted changes" >&2; exit 1; }
full="$(git -C "$src" rev-parse --verify "$commit^{commit}")"
paths="$(sed -n 's/^[0-9a-f]\{64\}  //p' "$manifest")"
for p in $paths; do
    mkdir -p "$here/$(dirname "$p")"
    git -C "$src" show "$full:$p" > "$here/$p"
done
{
    echo "# FlashRT kernels vendored into FlashRT-llama.cpp. One-way: edit in FlashRT, then run vendor/sync.sh."
    echo "flashrt_repo=https://github.com/flashrt-project/FlashRT"
    echo "flashrt_commit=$full"
    echo "# sha256  path (relative to the FlashRT checkout and to vendor/)"
    (cd "$here" && echo "$paths" | tr ' ' '\n' | sort | xargs sha256sum)
} > "$manifest"
echo "vendored $(echo "$paths" | wc -w) files from FlashRT $full"
