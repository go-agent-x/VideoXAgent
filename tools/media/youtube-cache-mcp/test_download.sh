#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
YT_DLP="${YT_DLP:-$PROJECT_ROOT/.venvs/base/bin/yt-dlp}"

"$YT_DLP" \
  --remote-components ejs:github \
  --js-runtimes deno:$(which deno) \
  --cookies "$SCRIPT_DIR/www.youtube.com_cookies.txt" \
  -N 1 \
  --limit-rate 1M \
  -o "%(id)s.%(ext)s" \
  "https://www.youtube.com/watch?v=ECMMct_jnEM"
