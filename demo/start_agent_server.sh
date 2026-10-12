#!/bin/bash
# Start the Agent Runner web server
# Usage: bash demo/start_agent_server.sh [port] [--config PATH] [--host HOST] [--media-root PATH]
# Default port: 8080

set -eu

cd "$(dirname "$0")/.."

PORT=8080
WEB_HOST="${VIDEOX_WEB_HOST:-127.0.0.1}"
MEDIA_ARGS=()
AGENT_EXP_CONFIG="${AGENT_EXP_CONFIG:-exp_configs/validation/videomme_v2_gemini.yaml}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            if [ "$#" -lt 2 ]; then
                echo "--config requires a YAML path" >&2
                exit 2
            fi
            AGENT_EXP_CONFIG="$2"
            shift 2 ;;
        --host)
            if [ "$#" -lt 2 ]; then
                echo "--host requires an address" >&2
                exit 2
            fi
            WEB_HOST="$2"
            shift 2 ;;
        --media-root)
            if [ "$#" -lt 2 ]; then
                echo "--media-root requires a directory" >&2
                exit 2
            fi
            MEDIA_ARGS+=(--media-root "$2")
            shift 2 ;;
        -h|--help)
            echo "Usage: bash demo/start_agent_server.sh [port] [--config PATH] [--host HOST] [--media-root PATH]"
            echo "Config paths are relative to the repository root, or absolute."
            echo "Default host: 127.0.0.1. Non-loopback binding requires VIDEOX_WEB_TOKEN."
            echo "Additional media directories: repeat --media-root or set VIDEOX_MEDIA_ROOTS (colon-separated)."
            echo "For remote access, use an SSH tunnel or an HTTPS reverse proxy."
            exit 0 ;;
        *[!0-9]*|'') echo "Unknown argument: $1" >&2; exit 2 ;;
        *) PORT="$1"; shift ;;
    esac
done
if [ ! -f "$AGENT_EXP_CONFIG" ]; then
    echo "Experiment config not found: $AGENT_EXP_CONFIG" >&2
    exit 2
fi
export AGENT_EXP_CONFIG

AGENT_PYTHON="${PYTHON_BIN:-./.venvs/base/bin/python}"
if [ ! -x "$AGENT_PYTHON" ]; then
    echo "Agent Python not found: $AGENT_PYTHON" >&2
    echo "Install first: bash scripts/install_profiles.sh base" >&2
    exit 1
fi

echo "Starting Agent Runner Server on $WEB_HOST:$PORT..."
echo "Default experiment config: $AGENT_EXP_CONFIG"
echo ""

# Check if aiohttp is installed
if ! "$AGENT_PYTHON" -c "import aiohttp" 2>/dev/null; then
    echo "aiohttp is missing. Run: bash scripts/install_profiles.sh base" >&2
    exit 1
fi

# Run the server
exec "$AGENT_PYTHON" demo/agent_server.py "$PORT" --host "$WEB_HOST" ${MEDIA_ARGS[@]+"${MEDIA_ARGS[@]}"}
