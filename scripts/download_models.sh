#!/usr/bin/env bash
# Download MCP model weights from their public upstream sources.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
TORCH_CHECKPOINT_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints"
DRY_RUN=0
VENV_ROOT=""

usage() {
    cat <<'EOF'
Usage: bash scripts/download_models.sh [--dry-run] [--venv-root DIR] [COMPONENT ...]

Options:
  --dry-run        Print commands without executing them or creating directories
  --venv-root DIR  Select hf/python from DIR/<profile>/bin for each component
                   Relative paths are resolved from the repository root.
                   If omitted, hf/python are resolved through PATH as before.

Components:
  hf-core          Sentence-BERT, Depth Anything, VideoMAE and CLIP
  temporal-ground  Sentence-BERT model only
  depth            Depth Anything model only
  action           VideoMAE model only
  moment           CLIP model used by the public moment-retrieval fallback
  grounding-dino   BERT text encoder and GroundingDINO-B checkpoint
  sam              SAM ViT-H checkpoint
  sam2             SAM 2.1 Hiera-L checkpoint
  audio            faster-whisper and torchaudio alignment weights
  optical-flow     torchvision RAFT weights
  diarization      pyannote 3.1 diarization models (gated; need HF_TOKEN)
  emotion          FER and SpeechBrain SER models
  insightface      InsightFace buffalo_l model pack
  panns            PANNs Cnn14 AudioSet checkpoint
  ocr              Preload PaddleOCR's official models
  gaze             L2CS-Net (requires L2CS_WEIGHTS_URL; see MODEL_SETUP.md)
  all              All components above; gated/manual components are skipped if unconfigured

Environment:
  MODEL_DIR          Destination for repository-local checkpoints (default: ./models)
  HF_HOME            Hugging Face cache root
  TORCH_HOME         PyTorch cache root
  HF_TOKEN           Token for gated Hugging Face models
  WHISPER_MODEL_SIZE faster-whisper variant (default: large-v3)
  L2CS_WEIGHTS_URL   URL copied from the official L2CS-Net model zoo

Examples:
  bash scripts/download_models.sh --dry-run all
  bash scripts/download_models.sh --venv-root .venvs all
  bash scripts/download_models.sh grounding-dino sam2
  HF_TOKEN=hf_xxx bash scripts/download_models.sh diarization
EOF
}

quote_command() {
    local argument redact_next=0
    printf '  '
    for argument in "$@"; do
        if [[ "$redact_next" -eq 1 ]]; then
            argument='[REDACTED]'
            redact_next=0
        else
            case "$argument" in
                --token) redact_next=1 ;;
                --token=*) argument='--token=[REDACTED]' ;;
            esac
        fi
        printf '%q ' "$argument"
    done
    printf '\n'
}

run() {
    quote_command "$@"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

require_command() {
    if [[ "$DRY_RUN" -eq 0 ]] && ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command not found: $1" >&2
        exit 1
    fi
}

make_dir() {
    run mkdir -p "$1"
}

profile_command() {
    local profile="$1" executable="$2"
    if [[ -n "$VENV_ROOT" ]]; then
        printf '%s\n' "$VENV_ROOT/$profile/bin/$executable"
    else
        printf '%s\n' "$executable"
    fi
}

hf_download() {
    local profile="$1" hf_command
    shift
    hf_command="$(profile_command "$profile" hf)"
    require_command "$hf_command"
    run "$hf_command" download "$@"
}

download_file() {
    local url="$1"
    local destination="$2"
    require_command curl
    make_dir "$(dirname "$destination")"
    run curl --fail --location --retry 3 --continue-at - --output "$destination" "$url"
}

download_hf_core() {
    echo "[hf-core] Hugging Face model repositories"
    download_temporal_ground
    download_depth
    download_action
    download_moment
}

download_temporal_ground() {
    echo "[temporal-ground] Sentence-BERT"
    hf_download vision sentence-transformers/all-MiniLM-L6-v2
}

download_depth() {
    echo "[depth] Depth Anything V2 Small"
    hf_download vision depth-anything/Depth-Anything-V2-Small-hf
}

download_action() {
    echo "[action] VideoMAE Kinetics"
    hf_download vision MCG-NJU/videomae-base-finetuned-kinetics
}

download_moment() {
    echo "[moment] CLIP ViT-B/32"
    hf_download vision openai/clip-vit-base-patch32
}

download_grounding_dino() {
    echo "[grounding-dino] BERT and GroundingDINO-B"
    hf_download vision google-bert/bert-base-uncased --local-dir "$MODEL_DIR/bert-base-uncased"
    download_file \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swinb_cogcoor.pth" \
        "$MODEL_DIR/GroundingDINO/groundingdino_swinb_cogcoor.pth"
}

download_sam() {
    echo "[sam] SAM ViT-H"
    download_file \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
        "$MODEL_DIR/sam_vit_h_4b8939.pth"
}

download_sam2() {
    echo "[sam2] SAM 2.1 Hiera-L"
    download_file \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        "$MODEL_DIR/sam2.1_hiera_large.pt"
}

download_audio() {
    local whisper_size="${WHISPER_MODEL_SIZE:-large-v3}"
    local python_command
    python_command="$(profile_command audio python)"
    echo "[audio] faster-whisper ($whisper_size) and wav2vec2 alignment"
    hf_download audio "Systran/faster-whisper-$whisper_size"
    require_command "$python_command"
    run "$python_command" -c "import torchaudio; torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()"
}

download_optical_flow() {
    echo "[optical-flow] RAFT Large"
    download_file \
        "https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
        "$TORCH_CHECKPOINT_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
}

download_diarization() {
    echo "[diarization] pyannote speaker-diarization-3.1"
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "  skipped: accept the model conditions and set HF_TOKEN first" >&2
        return 0
    fi
    hf_download audio pyannote/speaker-diarization-3.1 --token "$HF_TOKEN"
    hf_download audio pyannote/segmentation-3.0 --token "$HF_TOKEN"
}

download_emotion() {
    echo "[emotion] facial and speech emotion models"
    hf_download audio trpakov/vit-face-expression
    hf_download audio speechbrain/emotion-recognition-wav2vec2-IEMOCAP \
        --local-dir "$HOME/.cache/speechbrain/speechbrain--emotion-recognition-wav2vec2-IEMOCAP"
    hf_download audio facebook/wav2vec2-base
}

download_insightface() {
    local archive="$MODEL_DIR/downloads/insightface-buffalo_l.zip"
    echo "[insightface] buffalo_l"
    download_file \
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" \
        "$archive"
    require_command unzip
    make_dir "${INSIGHTFACE_HOME:-$HOME/.insightface}/models/buffalo_l"
    run unzip -o "$archive" -d "${INSIGHTFACE_HOME:-$HOME/.insightface}/models/buffalo_l"
}

download_panns() {
    echo "[panns] Cnn14 AudioSet checkpoint"
    # This is the artifact linked by the official qiuqiangkong/audioset_tagging_cnn repository.
    download_file \
        "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1" \
        "$MODEL_DIR/Cnn14_mAP=0.431.pth"
}

download_ocr() {
    local python_command
    python_command="$(profile_command ocr python)"
    echo "[ocr] PaddleOCR official pretrained models"
    require_command "$python_command"
    run env PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True "$python_command" -c \
        "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)"
}

download_gaze() {
    echo "[gaze] L2CS-Net Gaze360 checkpoint"
    if [[ -z "${L2CS_WEIGHTS_URL:-}" ]]; then
        echo "  skipped: set L2CS_WEIGHTS_URL to the Gaze360 URL from the official L2CS-Net model zoo" >&2
        echo "  https://github.com/Ahmednull/L2CS-Net#models" >&2
        return 0
    fi
    download_file "$L2CS_WEIGHTS_URL" "$MODEL_DIR/L2CSNet_gaze360.pkl"
}

download_component() {
    case "$1" in
        hf-core) download_hf_core ;;
        temporal-ground) download_temporal_ground ;;
        depth) download_depth ;;
        action) download_action ;;
        moment) download_moment ;;
        grounding-dino) download_grounding_dino ;;
        sam) download_sam ;;
        sam2) download_sam2 ;;
        audio) download_audio ;;
        optical-flow) download_optical_flow ;;
        diarization) download_diarization ;;
        emotion) download_emotion ;;
        insightface) download_insightface ;;
        panns) download_panns ;;
        ocr) download_ocr ;;
        gaze) download_gaze ;;
        all)
            download_hf_core
            download_grounding_dino
            download_sam
            download_sam2
            download_audio
            download_optical_flow
            download_diarization
            download_emotion
            download_insightface
            download_panns
            download_ocr
            download_gaze
            ;;
        *)
            echo "error: unknown component: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
}

components=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --venv-root)
            if [[ "$#" -lt 2 || -z "$2" || "$2" == -* ]]; then
                echo "error: --venv-root requires a directory" >&2
                exit 2
            fi
            VENV_ROOT="$2"
            shift
            ;;
        --venv-root=*)
            VENV_ROOT="${1#*=}"
            if [[ -z "$VENV_ROOT" ]]; then
                echo "error: --venv-root requires a directory" >&2
                exit 2
            fi
            ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) components+=("$1") ;;
    esac
    shift
done

if [[ "${#components[@]}" -eq 0 ]]; then
    components=(all)
fi

# Validate the entire request before an earlier component can download anything.
for component in "${components[@]}"; do
    case "$component" in
        hf-core|temporal-ground|depth|action|moment|grounding-dino|sam|sam2|audio|optical-flow|diarization|emotion|insightface|panns|ocr|gaze|all) ;;
        *) echo "error: unknown component: $component" >&2; exit 2 ;;
    esac
done

if [[ -n "$VENV_ROOT" && "$VENV_ROOT" != /* ]]; then
    VENV_ROOT="$PROJECT_ROOT/$VENV_ROOT"
fi

echo "Project root: $PROJECT_ROOT"
echo "Model directory: $MODEL_DIR"
echo "Hugging Face cache: $HF_CACHE_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run: commands will be printed but not executed"
fi

for component in "${components[@]}"; do
    download_component "$component"
done

echo "Model preparation finished. See MODEL_SETUP.md for environment variables and licenses."
