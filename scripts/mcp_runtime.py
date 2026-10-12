#!/usr/bin/env python3
"""Prepare model/cache policy before importing any MCP or model libraries."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    'GDINO_CHECKPOINT': 'GroundingDINO/groundingdino_swinb_cogcoor.pth',
    'GDINO_BERT_PATH': 'bert-base-uncased',
    'SAM_CHECKPOINT': 'sam_vit_h_4b8939.pth',
    'SAM2_CHECKPOINT': 'sam2.1_hiera_large.pt',
    'PANNS_CHECKPOINT': 'Cnn14_mAP=0.431.pth',
    'L2CS_WEIGHTS': 'L2CSNet_gaze360.pkl',
}


def prepare_environment(env, root=ROOT):
    allow = env.get('MCP_ALLOW_MODEL_DOWNLOAD', '0').lower() in ('1', 'true', 'yes', 'y')
    # Explicit policy must be set before huggingface_hub/transformers import.
    env['HF_HUB_OFFLINE'] = '0' if allow else '1'
    env['TRANSFORMERS_OFFLINE'] = '0' if allow else '1'
    env.setdefault('HF_HUB_ETAG_TIMEOUT', '10')
    env.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')
    env.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    env.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    for key in ('HF_HOME', 'TORCH_HOME', 'INSIGHTFACE_HOME'):
        if env.get(key):
            p = Path(env[key]).expanduser()
            env[key] = str(p if p.is_absolute() else root / p)
    model_dir = Path(env.get('MODEL_DIR', str(root / 'models'))).expanduser()
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    env['MODEL_DIR'] = str(model_dir)
    for key, relative in MODEL_PATHS.items():
        raw = env.get(key)
        # Migrate defaults only; custom checkpoint paths remain authoritative.
        is_default = not raw or raw in ('models/' + relative, './models/' + relative)
        path = model_dir / relative if is_default else Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        if is_default and not path.exists() and key in ('SAM_CHECKPOINT', 'SAM2_CHECKPOINT'):
            legacy = model_dir / ('SAM' if key == 'SAM_CHECKPOINT' else 'SAM2') / relative
            if legacy.is_file():
                path = legacy
        env[key] = str(path)


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: mcp_runtime.py path/to/server.py [args...]')
    prepare_environment(os.environ)
    server = Path(sys.argv[1])
    if not server.is_absolute():
        server = ROOT / server
    os.chdir(ROOT)
    sys.path.insert(0, str(server.parent))
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(server), *sys.argv[2:]]
    runpy.run_path(str(server), run_name='__main__')


if __name__ == '__main__':
    main()
