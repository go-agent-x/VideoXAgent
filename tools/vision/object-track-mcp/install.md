# Installation
Run the centralized vision-profile installer from the repository root. It installs
the matching Torch/TorchVision build first and then installs a pinned SAM2 commit:

```bash
bash scripts/install_profiles.sh vision --torch-backend cpu
```

Use `--torch-backend cu126` on the supported CUDA 12.6 stack.
