#!/bin/bash
# Run batch inference with the base profile environment.
# Usage: bash auto_deepagents.sh EXP_CONFIG [--log]
# EXP_CONFIG_ENV may supply the configuration when no positional path is given.

if [ -z "${BASH_VERSION:-}" ] || shopt -oq posix; then
    exec bash +o posix "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# HuggingFace offline mode (Avoid network timeout)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ===== Parse arguments =====
ENABLE_LOG=false
EXP_CONFIG=""

for arg in "$@"; do
    case $arg in
        --log)
            ENABLE_LOG=true
            ;;
        -h|--help)
            echo "Usage: bash auto_deepagents.sh EXP_CONFIG [--log]"
            echo "Environment: PYTHON_BIN (default: .venvs/base/bin/python), EXP_CONFIG_ENV"
            exit 0
            ;;
        --*)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
        *)
            # Use the first non-option argument as the configuration file
            if [ -z "$EXP_CONFIG" ]; then
                EXP_CONFIG="$arg"
            else
                echo "Only one experiment configuration may be selected." >&2
                exit 2
            fi
            ;;
    esac
done

# Require a real batch configuration rather than an obsolete default filename.
EXP_CONFIG="${EXP_CONFIG:-${EXP_CONFIG_ENV:-}}"
if [ -z "$EXP_CONFIG" ]; then
    echo "Usage: bash auto_deepagents.sh EXP_CONFIG [--log]" >&2
    exit 2
fi
if [ ! -f "$EXP_CONFIG" ]; then
    echo "Experiment configuration not found: $EXP_CONFIG" >&2
    exit 2
fi

AGENT_PYTHON="${PYTHON_BIN:-$SCRIPT_DIR/.venvs/base/bin/python}"
if ! AGENT_PYTHON=$(command -v "$AGENT_PYTHON") || [ ! -x "$AGENT_PYTHON" ]; then
    echo "Base Python not found: ${PYTHON_BIN:-$SCRIPT_DIR/.venvs/base/bin/python}" >&2
    echo "Install it with: bash scripts/install_profiles.sh base" >&2
    exit 1
fi
export PATH="$(dirname "$AGENT_PYTHON"):$PATH"

# ===== Set up logging =====
if [ "$ENABLE_LOG" = true ]; then
    mkdir -p logs
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="logs/deepagents_${TIMESTAMP}.log"
    echo "📝 Logging enabled: $LOG_FILE"
    # tee also works on platforms that do not provide GNU stdbuf.
    exec > >(tee -a "$LOG_FILE") 2>&1
    # Enable unbuffered Python output
    export PYTHONUNBUFFERED=1
fi

# Print configuration details before starting
echo ""
echo "🚀 Checking configuration before starting..."
"$AGENT_PYTHON" utils/print_config.py --config "$EXP_CONFIG"
echo ""

# ===== Read parameters from the configuration file =====
echo ""
echo "📄 Experiment configuration file: $EXP_CONFIG"


# Read exp_name from the configuration file if present
EXP_NAME=$("$AGENT_PYTHON" - "$EXP_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
name = config.get("exp_name") or ""
if not isinstance(name, str):
    raise SystemExit("exp_name must be a string")
print(name)
PY
)

# Set the DATA_FILE path based on exp_name
if [ -n "$EXP_NAME" ]; then
    DATA_FILE="conversations/all_data_path_${EXP_NAME}.json"
    echo "📁 Experiment name: $EXP_NAME"
else
    DATA_FILE="conversations/all_data_path.json"
fi

if [ -f "$DATA_FILE" ]; then
    echo ""
    echo "⚠️  Existing file detected: $DATA_FILE"
    echo "Choose an action:"
    echo "  1) Continue appending to this file"
    echo "  2) Rename the file and restart"
    echo "  3) Exit the script"
    read -p "Enter an option [1/2/3]: " choice
    case $choice in
        1)
            echo "✅ Continuing to append to $DATA_FILE"
            ;;
        2)
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            read -p "Enter a filename suffix (press Enter to use timestamp ${TIMESTAMP}): " SUFFIX
            if [ -z "$SUFFIX" ]; then
                SUFFIX="$TIMESTAMP"
            fi
            NEW_NAME="conversations/all_data_path_${SUFFIX}.json"
            mv "$DATA_FILE" "$NEW_NAME"
            echo "✅ Renamed to: $NEW_NAME"
            echo "🔄 Restarting the script..."
            exec bash "$SCRIPT_DIR/auto_deepagents.sh" "$@"
            ;;
        3)
            echo "❌ Script exited"
            exit 0
            ;;
        *)
            echo "❌ Invalid option; exiting the script"
            exit 1
            ;;
    esac
fi




# ===== Write experiment configuration to all_data_path.json =====
"$AGENT_PYTHON" utils/write_experiment_config.py --config "$EXP_CONFIG"

# Use Python to parse YAML and extract data_files (supports both new and legacy formats)
# New format: [{path: "xxx.jsonl", qid_file: "xxx.txt"}, ...]
# Legacy format: ["xxx.jsonl", ...]
# Output: one "data_file_path|qid_file_path" or "data_file_path|" per line
DATA_ENTRIES=$("$AGENT_PYTHON" - "$EXP_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    data = yaml.safe_load(handle).get("data_files", [])
for item in data:
    if isinstance(item, dict):
        print(item.get('path', '') + '|' + item.get('qid_file', ''))
    else:
        print(str(item) + '|')
PY
)


# ===== Process all data files =====
while IFS='|' read -r data_file qid_file; do
    if [ -n "$data_file" ]; then
        echo ""
        echo "===== Processing $(basename "$data_file") ====="
        if [ -n "$qid_file" ]; then
            echo "   qid_file: $qid_file"
            "$AGENT_PYTHON" cmd_deepagents.py -f "$data_file" --qid-file "$qid_file" --config "$EXP_CONFIG"
        else
            "$AGENT_PYTHON" cmd_deepagents.py -f "$data_file" --config "$EXP_CONFIG"
        fi
    fi
done <<< "$DATA_ENTRIES"
