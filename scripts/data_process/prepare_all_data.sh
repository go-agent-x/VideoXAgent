#!/usr/bin/env bash
# Prepare local benchmark archives and annotations; this script downloads no data.
set -euo pipefail

CALLER_DIR="$PWD"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_SRC="${DATA_SRC:-}"
FORCE=0
DATASETS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/data_process/prepare_all_data.sh [options] [DATASET ...]

Datasets: longvideobench, lvbench, minerva, videomme, videomme_v2
If omitted, all five datasets are selected. Existing exports are skipped.

Options:
  --data-src DIR   Directory of prepared archives/annotations (or set DATA_SRC)
  --force          Re-extract/copy and regenerate existing exports
  -h, --help       Show this help

Environment:
  DATA_SRC         Source directory; --data-src takes precedence
  PYTHON_BIN       Python interpreter (default: <repo>/.venvs/base/bin/python)

Relative source/interpreter paths are resolved from the calling directory.
See scripts/data_process/README.md for the required source layout.
EOF
}

die() {
    echo "prepare_all_data: $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-src)
            [[ $# -ge 2 && -n "$2" ]] || die "--data-src requires a directory"
            DATA_SRC="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) die "unknown option: $1" ;;
        *) DATASETS+=("$1"); shift ;;
    esac
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(longvideobench lvbench minerva videomme videomme_v2)
fi
for dataset in "${DATASETS[@]}"; do
    case "$dataset" in
        longvideobench|lvbench|minerva|videomme|videomme_v2) ;;
        *) die "unknown dataset: $dataset (see --help)" ;;
    esac
done

cd "$PROJECT_ROOT"

has_files() {
    local path
    for path in "$@"; do
        [[ -f "$path" ]] || return 1
    done
}

is_prepared() {
    # A failed/interrupted attempt can leave all expected filenames behind.
    [[ ! -e "data/.prepare_state/$1.incomplete" ]] || return 1
    case "$1" in
        longvideobench)
            has_files data/LongVideoBench/test/{short,medium,long}_duration_data.jsonl ;;
        lvbench)
            has_files data/LVBench/test/all_data.jsonl ;;
        minerva)
            has_files data/minerva/test/minerva_all_data.jsonl ;;
        videomme)
            has_files data/videomme/test/{short,medium,long}_duration_data.jsonl ;;
        videomme_v2)
            has_files data/Video-MME-v2/test/test.jsonl ;;
    esac
}

PENDING=()
for dataset in "${DATASETS[@]}"; do
    if [[ "$FORCE" == 0 ]] && is_prepared "$dataset"; then
        echo "Already exported: $dataset (skip; use --force to rebuild)"
    else
        PENDING+=("$dataset")
    fi
done
if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "All selected exports already exist; no files changed."
    exit 0
fi

[[ -n "$DATA_SRC" ]] || die "set DATA_SRC or pass --data-src DIR"
[[ "$DATA_SRC" = /* ]] || DATA_SRC="$CALLER_DIR/$DATA_SRC"
[[ -d "$DATA_SRC" ]] || die "source directory not found: $DATA_SRC"

AGENT_PYTHON="${PYTHON_BIN:-$PROJECT_ROOT/.venvs/base/bin/python}"
if [[ "$AGENT_PYTHON" == */* && "$AGENT_PYTHON" != /* ]]; then
    AGENT_PYTHON="$CALLER_DIR/$AGENT_PYTHON"
fi
if ! AGENT_PYTHON=$(command -v "$AGENT_PYTHON") || [[ ! -x "$AGENT_PYTHON" ]]; then
    die "base Python not found; run bash scripts/install_profiles.sh base --evaluation, or set PYTHON_BIN"
fi

require_file() {
    [[ -f "$1" && -r "$1" ]] || die "required input file not found or unreadable: $1"
}

# Check inputs for every pending dataset before extracting or copying any files.
for dataset in "${PENDING[@]}"; do
    case "$dataset" in
        longvideobench|lvbench) exporter="export_$dataset.py" ;;
        *) exporter="export_${dataset}_data.py" ;;
    esac
    require_file "$PROJECT_ROOT/scripts/data_process/$exporter"
    case "$dataset" in
        longvideobench)
            require_file "$DATA_SRC/longvideobench/videos.tar"
            require_file "$DATA_SRC/longvideobench/subtitles.tar"
            require_file "$DATA_SRC/longvideobench/lvb_val.json"
            ;;
        lvbench) require_file "$DATA_SRC/longvideo/LVBench.tar" ;;
        minerva)
            require_file "$DATA_SRC/minerva/minerva.json"
            [[ -d "$DATA_SRC/minerva/youtube_videos/youtube_cache" ]] \
                || die "Minerva video directory not found: $DATA_SRC/minerva/youtube_videos/youtube_cache"
            ;;
        videomme) require_file "$DATA_SRC/videomme/videomme.tar" ;;
        videomme_v2) require_file "$DATA_SRC/videomme/Video-MME-v2.tar" ;;
    esac
done

mkdir -p data/.prepare_state
for dataset in "${PENDING[@]}"; do
    echo "Preparing dataset: $dataset"
    incomplete_marker="data/.prepare_state/$dataset.incomplete"
    : > "$incomplete_marker"
    case "$dataset" in
        longvideobench)
            mkdir -p data/LongVideoBench
            tar -xf "$DATA_SRC/longvideobench/videos.tar" -C data/LongVideoBench
            tar -xf "$DATA_SRC/longvideobench/subtitles.tar" -C data/LongVideoBench
            cp "$DATA_SRC/longvideobench/lvb_val.json" data/LongVideoBench/
            for category in short medium long; do
                "$AGENT_PYTHON" scripts/data_process/export_longvideobench.py "$category"
            done
            ;;
        lvbench)
            tar -xf "$DATA_SRC/longvideo/LVBench.tar" -C data/
            "$AGENT_PYTHON" scripts/data_process/export_lvbench.py all
            ;;
        minerva)
            mkdir -p data/minerva data/youtube_cache
            for video in "$DATA_SRC/minerva/youtube_videos/youtube_cache/"*.mp4; do
                [[ -f "$video" ]] || continue
                if [[ ! "$video" -ef "data/youtube_cache/$(basename "$video")" ]]; then
                    cp "$video" data/youtube_cache/
                fi
            done
            if [[ ! "$DATA_SRC/minerva/minerva.json" -ef data/minerva/minerva.json ]]; then
                cp "$DATA_SRC/minerva/minerva.json" data/minerva/
            fi
            "$AGENT_PYTHON" scripts/data_process/export_minerva_data.py
            ;;
        videomme)
            tar -xf "$DATA_SRC/videomme/videomme.tar" -C data/
            for category in short medium long; do
                "$AGENT_PYTHON" scripts/data_process/export_videomme_data.py "$category"
            done
            ;;
        videomme_v2)
            tar -xf "$DATA_SRC/videomme/Video-MME-v2.tar" -C data/
            "$AGENT_PYTHON" scripts/data_process/export_videomme_v2_data.py
            ;;
    esac
    rm -f "$incomplete_marker"
done

echo "Selected datasets prepared successfully."
