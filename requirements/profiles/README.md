# MCP dependency profiles

These manifests define the isolated MCP environments. Run all installation
commands through `bash scripts/install_profiles.sh` from the repository root.
The `base` environment also installs the editable project and its runtime
dependencies from `pyproject.toml` in the same pip invocation as `base.txt`.

```bash
bash scripts/install_profiles.sh base
bash scripts/install_profiles.sh base audio --torch-backend cpu
bash scripts/install_profiles.sh all

# Optional project extras, installed only into base
bash scripts/install_profiles.sh base --dev
bash scripts/install_profiles.sh base --evaluation
bash scripts/install_profiles.sh base --dev --evaluation --dry-run
```

`--dev` installs build, pytest, black, and ruff; `--evaluation` installs pandas, PyArrow, and
Matplotlib. Both options require selecting `base` or `all`. There is no separate
root requirements installation step.

| Profile | NumPy | Torch | Purpose |
|---|---:|---:|---|
| `base` | 1.26.4 | no | Agent and lightweight MCP servers |
| `vision` | 1.26.4 | 2.11.0 | Detection, segmentation, tracking and vision models |
| `audio` | 1.26.4 | 2.11.0 | Transcription, diarization and audio models |
| `ocr` | 1.26.4 | no | PaddleOCR |
| `music` | 2.2.6 | no | Shazamio and AcoustID |

Both `grounding-dino` and `interaction-recognition` use `vision`: interaction
recognition imports the local GroundingDINO implementation. `base` includes
scikit-image for frame differences and aiohttp for the demo; `music` includes
PyYAML for the MCP framework.

`common.txt` is included by profiles that share the NumPy 1.x/OpenCV runtime.
The OCR and music profiles intentionally do not include one another.

Tool-local `requirements.txt` files are compatibility references only, not
additional installation steps. They must contain a single `-r` reference to one
profile and no package versions.

The installer selects CPU or CUDA PyTorch wheels before installing the vision
or audio manifest, selects CPU or GPU PaddlePaddle for OCR, runs `pip check`, and
then runs `scripts/verify_profile.py`.

Base verification also imports the actual frame-diff implementation and checks
SSIM on identical and changed in-memory images. After updating requirements,
re-run `bash scripts/install_profiles.sh base --skip-config` to update an existing
environment, then restart the demo to rediscover previously skipped tools.

Torch builds are pinned with `+cpu` or `+cu126` so re-running the installer can
switch backends. Verification checks the installed CUDA build, not GPU driver
availability. `--skip-sam2` skips only SAM2, retaining other vision checks.
By default, an existing output MCP config is reused as the source to preserve
custom endpoints and proxies. An explicit `--config-source` overrides this.

Generated service commands include the runtime wrapper once. Backend choices also
update runtime devices for selected vision/audio backends and OCR, including
composite services using those backends. Unselected vision/audio settings stay
unchanged. Repeat `--disable-server NAME` to omit services from the
generated configuration; see [runtime options](../../install.md#2-use-the-generated-mcp-configuration).

Checks cover dependency metadata and imports, not full model inference. Install
FFmpeg and other system tools separately, and use [MODEL_SETUP.md](../../MODEL_SETUP.md)
to prepare weights in the matching profile environment.
