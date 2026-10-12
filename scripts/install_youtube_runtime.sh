#!/usr/bin/env bash
# Optional YouTube JavaScript runtime; system packages remain administrator-managed.
set -euo pipefail
export DENO_INSTALL="${DENO_INSTALL:-$HOME/.deno}"
export PATH="$DENO_INSTALL/bin:$PATH"
if command -v deno >/dev/null 2>&1; then
    deno --version
    exit 0
fi
command -v curl >/dev/null || { echo 'Install curl first' >&2; exit 2; }
command -v unzip >/dev/null || command -v 7z >/dev/null || { echo 'Install unzip or 7z first' >&2; exit 2; }
installer=$(mktemp)
trap 'rm -f "$installer"' EXIT
curl --fail --location --retry 3 https://deno.land/install.sh -o "$installer"
CI=1 sh "$installer"
"$DENO_INSTALL/bin/deno" --version
