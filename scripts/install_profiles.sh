#!/usr/bin/env bash
# Install MCP dependencies once into isolated profile environments.
# No package installation is performed when an MCP server is later launched.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
VENV_ROOT="$REPO_ROOT/.venvs"
TORCH_BACKEND="cpu"
PADDLE_BACKEND="cpu"
CONFIG_SOURCE=""
CONFIG_OUTPUT="$REPO_ROOT/.mcp.profiles.json"
DRY_RUN=0
SKIP_CONFIG=0
SKIP_SAM2=0
INSTALL_DEV=0
INSTALL_EVALUATION=0
DISABLED_SERVERS=()
TARGETS=()

usage() {
    cat <<'EOF'
Usage: scripts/install_profiles.sh [options] PROFILE [PROFILE ...]

Profiles: base, vision, audio, ocr, music, all

Options:
  --python PATH          Python 3.11 interpreter (default: auto-detect)
  --venv-root PATH       Profile environment directory (default: .venvs)
  --torch-backend NAME   cpu or cu126 (default: cpu)
  --paddle-backend NAME  cpu or gpu (default: cpu)
  --config-source PATH   MCP source (default: existing output, then repository .mcp.profiles.json)
  --config-output PATH   Generated config (default: .mcp.profiles.json)
  --disable-server NAME  Omit an MCP service from the generated config (repeatable)
  --skip-config          Do not generate an MCP config
  --skip-sam2            Do not install pinned SAM2 in the vision profile
  --dev                 Include development/test tools in base (requires base/all)
  --evaluation          Include evaluation/plotting tools in base (requires base/all)
  --dry-run              Print commands without creating or changing anything
  -h, --help             Show this help

Examples:
  scripts/install_profiles.sh base vision --torch-backend cpu
  scripts/install_profiles.sh base --dev
  scripts/install_profiles.sh all --torch-backend cu126 --paddle-backend gpu
EOF
}

die() {
    echo "install_profiles: $*" >&2
    exit 2
}

run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            [[ $# -ge 2 ]] || die "--python requires a value"
            PYTHON_BIN="$2"; shift 2 ;;
        --venv-root)
            [[ $# -ge 2 ]] || die "--venv-root requires a value"
            VENV_ROOT="$2"; shift 2 ;;
        --torch-backend)
            [[ $# -ge 2 ]] || die "--torch-backend requires a value"
            TORCH_BACKEND="$2"; shift 2 ;;
        --paddle-backend)
            [[ $# -ge 2 ]] || die "--paddle-backend requires a value"
            PADDLE_BACKEND="$2"; shift 2 ;;
        --config-source)
            [[ $# -ge 2 ]] || die "--config-source requires a value"
            CONFIG_SOURCE="$2"; shift 2 ;;
        --config-output)
            [[ $# -ge 2 ]] || die "--config-output requires a value"
            CONFIG_OUTPUT="$2"; shift 2 ;;
        --disable-server)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "--disable-server requires a service name"
            DISABLED_SERVERS+=("$2"); shift 2 ;;
        --skip-config) SKIP_CONFIG=1; shift ;;
        --skip-sam2) SKIP_SAM2=1; shift ;;
        --dev) INSTALL_DEV=1; shift ;;
        --evaluation) INSTALL_EVALUATION=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) die "unknown option: $1" ;;
        *) TARGETS+=("$1"); shift ;;
    esac
done

[[ ${#TARGETS[@]} -gt 0 ]] || { usage; exit 2; }
[[ "$TORCH_BACKEND" == "cpu" || "$TORCH_BACKEND" == "cu126" ]] \
    || die "--torch-backend must be cpu or cu126"
[[ "$PADDLE_BACKEND" == "cpu" || "$PADDLE_BACKEND" == "gpu" ]] \
    || die "--paddle-backend must be cpu or gpu"

if [[ " ${TARGETS[*]} " == *" all "* ]]; then
    [[ ${#TARGETS[@]} -eq 1 ]] || die "all cannot be combined with another profile"
    TARGETS=(base vision audio ocr music)
fi

for profile in "${TARGETS[@]}"; do
    case "$profile" in
        base|vision|audio|ocr|music) ;;
        *) die "unknown profile: $profile" ;;
    esac
done

if [[ "$INSTALL_DEV" == 1 || "$INSTALL_EVALUATION" == 1 ]]; then
    [[ " ${TARGETS[*]} " == *" base "* ]] \
        || die "--dev and --evaluation require the base profile (or all)"
fi
if [[ "$SKIP_CONFIG" == 1 && ${#DISABLED_SERVERS[@]} -gt 0 ]]; then
    die "--disable-server requires configuration generation; remove --skip-config"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3.11 "$REPO_ROOT/venv/bin/python" python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            candidate_version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            if [[ "$candidate_version" == "3.11" ]]; then
                PYTHON_BIN="$candidate"
                break
            fi
        fi
    done
fi
if [[ -z "$PYTHON_BIN" && "$DRY_RUN" == 1 ]]; then
    PYTHON_BIN="python3.11"
fi

if [[ "$DRY_RUN" == 0 ]]; then
    [[ -n "$PYTHON_BIN" ]] || die "Python 3.11 was not found; install it or pass --python PATH"
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
    python_version="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "$python_version" == "3.11" ]] \
        || die "dependency profiles currently support Python 3.11; found $python_version"
fi

if [[ " ${TARGETS[*]} " == *" vision "* && "$SKIP_SAM2" == 0 ]]; then
    command -v git >/dev/null 2>&1 || die "git is required to install the pinned SAM2 revision"
fi
if [[ " ${TARGETS[*]} " == *" base "* || " ${TARGETS[*]} " == *" audio "* || " ${TARGETS[*]} " == *" music "* ]]; then
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "install_profiles: warning: ffmpeg is not installed; media tools will not run" >&2
    fi
fi

install_torch() {
    local profile_python="$1"
    local index_url="https://download.pytorch.org/whl/$TORCH_BACKEND"
    # The local version suffix forces pip to switch an existing CPU/CUDA build.
    local packages=("torch==2.11.0+$TORCH_BACKEND" "torchvision==0.26.0+$TORCH_BACKEND")
    if [[ "$2" == "audio" ]]; then
        packages+=("torchaudio==2.11.0+$TORCH_BACKEND")
    fi
    run "$profile_python" -m pip install "${packages[@]}" --index-url "$index_url"
}

install_profile() {
    local profile="$1"
    local venv_path="$VENV_ROOT/$profile"
    local profile_python="$venv_path/bin/python"
    local requirements="$REPO_ROOT/requirements/profiles/$profile.txt"

    echo "==> Installing $profile profile in $venv_path"
    if [[ ! -x "$profile_python" ]]; then
        run "$PYTHON_BIN" -m venv "$venv_path"
    fi
    # Upgrade pip in a separate process. Upgrading pip and build tools together
    # can leave old pip resolving metadata that the same command just replaced.
    run "$profile_python" -m pip install --upgrade pip
    run "$profile_python" -m pip install --upgrade setuptools wheel

    if [[ "$profile" == "vision" || "$profile" == "audio" ]]; then
        install_torch "$profile_python" "$profile"
    fi
    if [[ "$profile" == "ocr" ]]; then
        if [[ "$PADDLE_BACKEND" == "gpu" ]]; then
            run "$profile_python" -m pip install paddlepaddle-gpu==2.6.2
        else
            run "$profile_python" -m pip install paddlepaddle==2.6.2
        fi
    fi

    if [[ "$profile" == "base" ]]; then
        local project_requirement="$REPO_ROOT"
        local extras=""
        if [[ "$INSTALL_DEV" == 1 ]]; then
            extras="dev"
        fi
        if [[ "$INSTALL_EVALUATION" == 1 ]]; then
            extras="${extras:+$extras,}evaluation"
        fi
        if [[ -n "$extras" ]]; then
            project_requirement="${REPO_ROOT}[$extras]"
        fi
        # Resolve project dependencies and profile pins together. A second,
        # unconstrained editable install could otherwise replace pinned packages.
        run "$profile_python" -m pip install -r "$requirements" -e "$project_requirement"
    else
        run "$profile_python" -m pip install -r "$requirements"
    fi

    if [[ "$profile" == "vision" && "$SKIP_SAM2" == 0 ]]; then
        run "$profile_python" -m pip install --no-build-isolation \
            "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
    fi

    run "$profile_python" -m pip check
    local verify_args=("$profile")
    if [[ "$profile" == "vision" || "$profile" == "audio" ]]; then
        verify_args+=(--torch-backend "$TORCH_BACKEND")
    fi
    if [[ "$profile" == "vision" && "$SKIP_SAM2" == 1 ]]; then
        verify_args+=(--skip-sam2)
    fi
    # Base verification also exercises frame-diff's SSIM implementation.
    # A clean pip check alone cannot detect undeclared runtime dependencies.
    run "$profile_python" "$REPO_ROOT/scripts/verify_profile.py" "${verify_args[@]}"
}

cd "$REPO_ROOT"
if [[ "$SKIP_CONFIG" == 0 ]]; then
    if [[ -z "$CONFIG_SOURCE" ]]; then
        if [[ -e "$CONFIG_OUTPUT" || -L "$CONFIG_OUTPUT" ]]; then
            CONFIG_SOURCE="$CONFIG_OUTPUT"
        else
            CONFIG_SOURCE="$REPO_ROOT/.mcp.profiles.json"
        fi
    fi
    [[ -f "$CONFIG_SOURCE" && -r "$CONFIG_SOURCE" ]] \
        || die "MCP source is not a readable file: $CONFIG_SOURCE. Use --config-source PATH or --skip-config."
fi
for profile in "${TARGETS[@]}"; do
    install_profile "$profile"
done

if [[ "$SKIP_CONFIG" == 0 ]]; then
    config_args=(--source "$CONFIG_SOURCE" --output "$CONFIG_OUTPUT" --venv-root "$VENV_ROOT")
    device_profiles=()
    for profile in "${TARGETS[@]}"; do
        if [[ "$profile" == "vision" || "$profile" == "audio" ]]; then
            device_profiles+=(--device-profile "$profile")
        fi
    done
    if [[ ${#device_profiles[@]} -gt 0 ]]; then
        runtime_device="cpu"
        [[ "$TORCH_BACKEND" == "cpu" ]] || runtime_device="cuda"
        config_args+=(--device "$runtime_device" "${device_profiles[@]}")
    fi
    if [[ " ${TARGETS[*]} " == *" ocr "* ]]; then
        config_args+=(--ocr-device "$PADDLE_BACKEND")
    fi
    if [[ ${#DISABLED_SERVERS[@]} -gt 0 ]]; then
        for server in "${DISABLED_SERVERS[@]}"; do
            config_args+=(--disable-server "$server")
        done
    fi
    run "$PYTHON_BIN" "$REPO_ROOT/scripts/configure_mcp_profiles.py" "${config_args[@]}"
fi

if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run complete. No packages were installed and no checks were executed."
else
    echo "Selected profiles passed dependency/smoke checks. Model inference is not verified."
fi
if [[ "$SKIP_CONFIG" == 0 && "$DRY_RUN" == 0 ]]; then
    echo "Generated MCP configuration: $CONFIG_OUTPUT"
fi
