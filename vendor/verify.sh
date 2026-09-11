#!/usr/bin/env bash
# Check that the vendored files still match the manifest (no local edits).
set -euo pipefail
cd "$(dirname "$0")" && sed -n '/^[0-9a-f]\{64\}  /p' FLASHRT_VENDOR.txt | sha256sum -c --quiet && echo "vendor/ matches FLASHRT_VENDOR.txt"
