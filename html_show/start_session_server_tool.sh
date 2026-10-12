#!/usr/bin/env bash
# Also support the documented entry point being invoked with `sh` (e.g. dash).
if [ -z "${BASH_VERSION:-}" ] || shopt -oq posix; then
  exec bash +o posix "$0" "$@"
fi

# Usage:
#   sh html_show/start_session_server_tool.sh                            # default: port 8923, localhost, auto timestamp tag
#   bash html_show/start_session_server_tool.sh --tag exp1               # custom tag: tool_workflow_exp1.html
#   bash html_show/start_session_server_tool.sh --port 9000              # custom port
#   VIDEOX_WEB_TOKEN=... bash html_show/start_session_server_tool.sh --ip 0.0.0.0
#   bash html_show/start_session_server_tool.sh --data /path/to/data.json  # custom data JSON
#   bash html_show/start_session_server_tool.sh --data /downloads/run/results.json --data-root /downloads/run
#   bash html_show/start_session_server_tool.sh --path-map /old/workspace=/downloads/run  # repeatable
#   bash html_show/start_session_server_tool.sh --enrich                 # explicitly update answer/metadata fields
#   bash html_show/start_session_server_tool.sh --no-tag                 # no tag, overwrite default tool_workflow.html
#   bash html_show/start_session_server_tool.sh --stop                   # stop the background server
#   bash html_show/start_session_server_tool.sh --fg                     # run server in foreground (old behavior)
#
# Each run generates a uniquely tagged HTML file (default: timestamp) so you can
# compare different experiments side by side.  All generated files are served by
# the same HTTP server from html_show/.
# After generating the new page, each run stops older servers from this checkout
# (including servers on other ports) before starting a fresh one.
#
# The server runs in the BACKGROUND by default (survives terminal disconnect).
# Use --stop to shut it down, or --fg to run in foreground.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT=8923
HOST_IP="${VIDEOX_WEB_HOST:-127.0.0.1}"
TAG=""
TAG_SET=false
NO_TAG=false
STOP_SERVER=false
FOREGROUND=false
DATA_JSON="${REPO_ROOT}/conversations/all_data_path.json"
DATA_ROOT=""
PATH_MAPS=()
ENRICH_DATA=false

while [ $# -gt 0 ]; do
  case "$1" in
    --port|--ip|--tag|--data|--data-root|--path-map)
      OPTION="$1"
      if [ $# -lt 2 ] || [ -z "$2" ]; then
        echo "ERROR: $OPTION requires a value." >&2
        exit 1
      fi
      shift
      case "$OPTION" in
        --port) PORT="$1" ;;
        --ip) HOST_IP="$1" ;;
        --tag) TAG="$1"; TAG_SET=true ;;
        --data) DATA_JSON="$1" ;;
        --data-root) DATA_ROOT="$1" ;;
        --path-map) PATH_MAPS+=("$1") ;;
      esac
      ;;
    --no-tag) NO_TAG=true ;;
    --stop) STOP_SERVER=true ;;
    --fg) FOREGROUND=true ;;
    --enrich) ENRICH_DATA=true ;;
    *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

case "$PORT" in
  ''|*[!0-9]*) echo "ERROR: Invalid port: $PORT" >&2; exit 1 ;;
esac
if [ "${#PORT}" -gt 5 ] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "ERROR: Port must be between 1 and 65535." >&2
  exit 1
fi
PORT=$((10#$PORT))

# Use the selected environment instead of a machine-specific Anaconda path.
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for CANDIDATE in "${REPO_ROOT}/.venv/bin/python3" "${REPO_ROOT}/venv/bin/python3" python3 python; do
    if command -v "$CANDIDATE" >/dev/null 2>&1; then
      PYTHON_BIN="$CANDIDATE"
      break
    fi
  done
fi
if [ -z "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info.major != 3)' >/dev/null 2>&1; then
  echo "ERROR: Python 3 is required. Set PYTHON_BIN to its executable path." >&2
  exit 1
fi

PID_FILE="${REPO_ROOT}/html_show/server_${PORT}.pid"
LOG_FILE="${REPO_ROOT}/html_show/server_${PORT}.log"

# ps is available on both macOS and Linux. Recognize only direct Python servers
# with this checkout's absolute script/root paths; never trust a saved PID alone.
server_port_for_command() {
  local command="$1" interpreter args server_port
  case "$command" in *" ${SCRIPT_DIR}/server.py "*) ;; *) return 1 ;; esac
  interpreter="${command%%" ${SCRIPT_DIR}/server.py "*}"
  # Accept the configured executable (also if its path contains spaces), or a
  # direct Python executable from a previous environment, never a shell wrapper.
  if [ "$interpreter" != "$PYTHON_BIN" ] && \
     ! [[ "$interpreter" =~ ^([^[:space:]]*/)?[Pp]ython([0-9]+(\.[0-9]+)*)?$ ]]; then
    return 1
  fi
  args="${command#*" ${SCRIPT_DIR}/server.py "}"
  server_port="${args%% *}"
  case "$server_port" in ''|*[!0-9]*) return 1 ;; esac
  [ "$args" = "$server_port $REPO_ROOT" ] || return 1
  echo "$server_port"
}

find_server_pids() {
  local selected_port="${1:-}" processes pid command server_port
  if ! processes=$(ps -axww -o pid= -o command=); then
    echo "ERROR: Cannot inspect running servers; refusing to start another instance." >&2
    return 1
  fi
  while read -r pid command; do
    case "$command" in *" ${SCRIPT_DIR}/server.py "*) ;; *) continue ;; esac
    server_port=$(server_port_for_command "$command") || continue
    [ -z "$selected_port" ] || [ "$server_port" = "$selected_port" ] || continue
    echo "$pid $server_port"
  done <<< "$processes"
}

is_server_pid() {
  local command server_port
  command=$(ps -ww -p "$1" -o command= 2>/dev/null) || return 1
  read -r command <<< "$command"
  server_port=$(server_port_for_command "$command") || return 1
  [ "$server_port" = "$2" ]
}

stop_server_pid() {
  local pid="$1" server_port="$2" i
  if is_server_pid "$pid" "$server_port"; then
    echo "Stopping previous server (PID $pid) on port $server_port..."
    kill -TERM "$pid" 2>/dev/null || true
    for ((i = 0; i < 30; i++)); do
      is_server_pid "$pid" "$server_port" || break
      sleep 0.1
    done
    # Recheck identity before escalation: the PID may already have exited.
    if is_server_pid "$pid" "$server_port"; then
      echo "Server did not exit after 3s; forcing PID $pid to stop."
      kill -KILL "$pid" 2>/dev/null || true
      for ((i = 0; i < 20; i++)); do
        is_server_pid "$pid" "$server_port" || break
        sleep 0.1
      done
    fi
    if is_server_pid "$pid" "$server_port"; then
      echo "ERROR: Could not stop server PID $pid; refusing to start another instance." >&2
      return 1
    fi
  fi
  rm -f "${SCRIPT_DIR}/server_${server_port}.pid"
}

stop_previous_servers() {
  local servers pid server_port
  servers=$(find_server_pids "${1:-}") || return 1
  while read -r pid server_port; do
    [ -n "$pid" ] || continue
    stop_server_pid "$pid" "$server_port" || return 1
  done <<< "$servers"
}

port_in_use() {
  "$PYTHON_BIN" - "$PORT" "$HOST_IP" <<'PY'
import socket
import sys
host = sys.argv[2]
if host in ('0.0.0.0', '::'):
    host = '127.0.0.1' if host == '0.0.0.0' else '::1'
try:
    with socket.create_connection((host, int(sys.argv[1])), timeout=0.5):
        sys.exit(0)
except OSError:
    sys.exit(1)
PY
}

html_ready() {
  "$PYTHON_BIN" - "$PORT" "$HTML_NAME" "$HOST_IP" <<'PY'
import os
import base64
import sys
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener
try:
    host = sys.argv[3]
    if host in ('0.0.0.0', '::'):
        host = '127.0.0.1' if host == '0.0.0.0' else '::1'
    if ':' in host:
        host = '[' + host.strip('[]') + ']'
    origin = 'http://{}:{}'.format(host, sys.argv[1])
    headers = {}
    token = os.environ.get('VIDEOX_WEB_TOKEN', '')
    if token:
        credentials = base64.b64encode(('videoxagent:' + token).encode('utf-8')).decode('ascii')
        headers['Authorization'] = 'Basic ' + credentials
    opener = build_opener(ProxyHandler({}))
    for path in ('/healthz', '/html_show/' + quote(sys.argv[2])):
        with opener.open(Request(origin + path, headers=headers), timeout=0.5) as response:
            if response.status != 200:
                sys.exit(1)
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
}

# --stop: kill background server and exit
if [ "$STOP_SERVER" = true ]; then
  stop_previous_servers "$PORT"
  rm -f "$PID_FILE"
  echo "No server from this checkout remains on port $PORT."
  exit 0
fi

# Validate exposure before generating files or replacing an existing server.
# The token stays in the environment, never in process arguments or URLs.
"$PYTHON_BIN" - "$REPO_ROOT" "$HOST_IP" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from utils.web_security import WebAccessPolicy
try:
    WebAccessPolicy(host=sys.argv[2])
except ValueError as error:
    raise SystemExit(str(error))
PY
export VIDEOX_WEB_HOST="$HOST_IP"

# Auto-generate timestamp tag if none provided and --no-tag not set
if [ "$NO_TAG" = false ] && [ "$TAG_SET" = false ]; then
  TAG="$(date +%Y%m%d_%H%M%S)"
fi
if [ "$NO_TAG" = true ]; then
  TAG=""
fi

# --- Step 1: Optionally enrich data JSON; viewing a downloaded run is read-only. ---
DATA_JSON=$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$DATA_JSON")
if [ -n "$DATA_ROOT" ]; then
  DATA_ROOT=$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$DATA_ROOT")
fi
if [ "$ENRICH_DATA" = true ] && [ -f "$DATA_JSON" ]; then
  echo "Enriching ${DATA_JSON} with gt_answer / model_answer / metadata..."
  "$PYTHON_BIN" "${REPO_ROOT}/html_show/add_gt_answer.py" "$DATA_JSON"
  echo ""
fi

# --- Step 2: Generate HTML dashboard ---
GEN_ARGS=("--data" "$DATA_JSON")
if [ -n "$DATA_ROOT" ]; then
  GEN_ARGS+=("--data-root" "$DATA_ROOT")
fi
# macOS Bash 3 treats an empty array as unset with `set -u`.
if [ -n "${PATH_MAPS[*]-}" ]; then
  for PATH_MAPPING in "${PATH_MAPS[@]}"; do
    GEN_ARGS+=("--path-map" "$PATH_MAPPING")
  done
fi
if [ -n "$TAG" ]; then
  GEN_ARGS+=("--tag" "$TAG")
fi

echo "Generating tool_workflow HTML (tag=${TAG:-<none>})..."
"$PYTHON_BIN" "${REPO_ROOT}/html_show/gen_dashboard.py" "${GEN_ARGS[@]}"

# Preserve the server's positional command line for safe lifecycle recognition.
# Both generator and server receive the same root and prefix mapping configuration.
export VIDEOX_DATA_FILE="$DATA_JSON"
export VIDEOX_DATA_ROOT="$DATA_ROOT"
export VIDEOX_PATH_MAPS
if [ -n "${PATH_MAPS[*]-}" ]; then
  VIDEOX_PATH_MAPS=$("$PYTHON_BIN" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${PATH_MAPS[@]}")
else
  VIDEOX_PATH_MAPS='[]'
fi

# Determine the generated filename for display
if [ -n "$TAG" ]; then
  HTML_NAME="tool_workflow_${TAG}.html"
else
  HTML_NAME="tool_workflow.html"
fi
HTML_URL_NAME=$("$PYTHON_BIN" -c 'import sys; from urllib.parse import quote; print(quote(sys.argv[1]))' "$HTML_NAME")

# --- Step 3: Replace previous HTTP servers after generation succeeds ---
stop_previous_servers
rm -f "$PID_FILE"
if port_in_use; then
  echo "ERROR: Port $PORT is occupied by another process."
  echo "It was not stopped. Use --port to specify a different port."
  exit 1
fi

if [ "$FOREGROUND" = true ]; then
  echo ""
  echo "Starting server in FOREGROUND on port $PORT..."
  echo "Press Ctrl+C to stop"
  echo ""
  echo "Open in browser:"
  echo "  - Tool Workflow:        http://$HOST_IP:$PORT/html_show/${HTML_URL_NAME}"
  echo ""
  exec "$PYTHON_BIN" "${REPO_ROOT}/html_show/server.py" "$PORT" "${REPO_ROOT}"
fi

# If startup fails or is interrupted, do not leave an unhealthy server behind.
SERVER_PID=""
cleanup_failed_start() {
  if [ -n "$SERVER_PID" ]; then
    stop_server_pid "$SERVER_PID" "$PORT" || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_failed_start EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Background mode (default) - survives terminal disconnect
echo "Starting server in background on port $PORT..."
nohup "$PYTHON_BIN" "${REPO_ROOT}/html_show/server.py" "$PORT" "${REPO_ROOT}" \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# --- Step 4: Health check ---
HTML_READY=false
for ((i = 0; i < 10; i++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  if html_ready; then
    HTML_READY=true
    break
  fi
  sleep 0.5
done
if [ "$HTML_READY" = false ]; then
  echo "ERROR: Dashboard is not accessible. Check log: $LOG_FILE" >&2
  tail -10 "$LOG_FILE" 2>/dev/null || true
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    rm -f "$PID_FILE"
  fi
  exit 1
fi
echo "[OK] HTML accessible (HTTP 200), server PID $SERVER_PID, log: $LOG_FILE"
# A healthy background server intentionally outlives this launcher.
trap - EXIT INT TERM

# --- Step 5: Display URLs ---
echo ""
echo "=========================================="
echo " Open in browser:"
echo "=========================================="
echo "  Tool Workflow:        http://$HOST_IP:$PORT/html_show/${HTML_URL_NAME}"
echo ""
echo "All generated dashboards in html_show/:"
"$PYTHON_BIN" - "$SCRIPT_DIR" "$HOST_IP" "$PORT" <<'PY'
import sys
from pathlib import Path
from urllib.parse import quote
files = sorted(Path(sys.argv[1]).glob('tool_workflow*.html'), key=lambda p: p.stat().st_mtime, reverse=True)
for path in files[:10]:
    print('  - http://{}:{}/html_show/{}'.format(sys.argv[2], sys.argv[3], quote(path.name)))
PY
echo ""
echo "To stop: sh html_show/start_session_server_tool.sh --port $PORT --stop"
echo ""
