# Benchmark Data Preparation

These scripts prepare locally supplied benchmark annotations and videos for VideoXAgent batch inference. They copy files, extract archives, and export JSONL records containing a question, answer, local video path, and Agent prompt. They do not download complete datasets or run inference.

The supported preparation entry point is `scripts/data_process/prepare_all_data.sh`. Use the exporters in this directory; the older `scripts/export_*.py` and `scripts/prepare_*.sh` files are separate legacy workflows and should not be run as additional preparation steps.

## Installation

Run the following command from the repository root:

```bash
bash scripts/install_profiles.sh base --evaluation
```

The evaluation extra supplies `pandas` and `pyarrow` for the Parquet exporters. The preparation script uses `.venvs/base/bin/python` by default. Set `PYTHON_BIN` to use another Python environment with the required dependencies. Archive extraction also requires `tar`.

## Prepare Data from a Local Source Directory

Arrange the datasets you want to prepare under one source directory:

```text
/absolute/path/to/prepared-datasets/
├── longvideobench/
│   ├── videos.tar
│   ├── subtitles.tar
│   └── lvb_val.json
├── longvideo/
│   └── LVBench.tar
├── minerva/
│   ├── minerva.json
│   └── youtube_videos/youtube_cache/<video_id>.mp4
└── videomme/
    ├── videomme.tar
    └── Video-MME-v2.tar
```

Only the inputs for the selected datasets are required. The archive layout must match the destination paths below:

- LongVideoBench archives are extracted into `data/LongVideoBench/`; they must contain `videos/` and `subtitles/`, respectively.
- `LVBench.tar`, `videomme.tar`, and `Video-MME-v2.tar` are extracted into `data/`; they must contain the corresponding top-level dataset directory, including annotations and a `videos/` directory.
- Minerva annotations and MP4 files are copied into `data/minerva/` and `data/youtube_cache/`.

For example, prepare only Minerva and VideoMME:

```bash
bash scripts/data_process/prepare_all_data.sh \
  --data-src /absolute/path/to/prepared-datasets minerva videomme
```

Alternatively, supply the source directory through `DATA_SRC`:

```bash
DATA_SRC=/absolute/path/to/prepared-datasets \
  bash scripts/data_process/prepare_all_data.sh minerva
```

Omitting dataset names selects all five datasets. Supported names are `longvideobench`, `lvbench`, `minerva`, `videomme`, and `videomme_v2`.

```bash
# Prepare all datasets using a custom Python environment.
PYTHON_BIN=/absolute/path/to/python \
  bash scripts/data_process/prepare_all_data.sh \
    --data-src /absolute/path/to/prepared-datasets

# Re-run preparation for one dataset, even if its output already exists.
bash scripts/data_process/prepare_all_data.sh \
  --data-src /absolute/path/to/prepared-datasets minerva --force

bash scripts/data_process/prepare_all_data.sh --help
```

`--data-src` takes precedence over `DATA_SRC`. Options may appear before or after dataset names. Relative source-directory and custom `PYTHON_BIN` paths are resolved from the caller's working directory. The script then resolves the repository root from its own location and runs preparation there.

The script checks required inputs and stops on errors. By default, it skips a dataset when all expected JSONL output files exist and no incomplete-preparation marker remains. This is an output-presence check, not a check that the dataset or its videos are complete. `--force` repeats copying/extraction and overwrites generated JSONL files; it does not delete existing dataset directories or remove stale files.

Before preparing a dataset, the script creates `data/.prepare_state/<dataset>.incomplete` and removes it only after success. If copying, extraction, or export fails, the marker remains so the next invocation retries that dataset even if partial output files exist. Preparation is not transactional: a failure can leave copied files or partial JSONL output. Existing outputs from earlier workflows without a marker still use the presence-based skip check; use `--force` to rebuild them when needed.

## Local Inputs and Outputs

All paths in this table are relative to the repository root.

| Dataset | Annotation file | Video directory | JSONL output |
| --- | --- | --- | --- |
| VideoMME | `data/videomme/test-00000-of-00001.parquet` | `data/videomme/videos/` | `data/videomme/test/{short,medium,long}_duration_data.jsonl` |
| Video-MME-v2 | `data/Video-MME-v2/test.parquet` | `data/Video-MME-v2/videos/` | `data/Video-MME-v2/test/test.jsonl` |
| LVBench | `data/LVBench/video_info.meta.jsonl` | `data/LVBench/videos/` | `data/LVBench/test/all_data.jsonl` |
| LongVideoBench | `data/LongVideoBench/lvb_val.json` | `data/LongVideoBench/videos/` | `data/LongVideoBench/test/{short,medium,long}_duration_data.jsonl` |
| Minerva | `data/minerva/minerva.json` | `data/youtube_cache/` | `data/minerva/test/minerva_all_data.jsonl` |

Exported prompts contain absolute local video paths. Re-export after moving the repository or data to another machine. Review exporter output before inference: LVBench skips questions whose videos are missing; Minerva fails if any selected record references a missing video. VideoMME, Video-MME-v2, and LongVideoBench generate paths without checking each video file.

### Minerva: Local Video Workflow

Minerva requires a local file named `data/youtube_cache/<video_id>.mp4` for every selected video. Its prompt uses this file path:

```text
Given a video: /absolute/path/to/VideoXAgent/data/youtube_cache/<video_id>.mp4
Question: ...
Options:
...
```

The exporter reports missing video paths and exits before opening the output file, preserving any existing JSONL output. It does not ask the Agent to download a YouTube video. The legacy root-level Minerva exporter instead generates a YouTube-ID prompt and can write to the same output file; do not mix the two workflows.

Use the full, unfiltered Minerva export as the source for later sampling or filtering. The current exporter assigns `question_id` values using the row position **after** applying `--split` or `--question_type`. Exporting a subset directly can therefore change IDs for the same questions and invalidate saved sample lists or completion records. This workflow preserves the existing ID scheme.

## Export Data Already Placed in the Repository

If annotations and videos are already in the locations above, run the exporters directly from the repository root. This avoids the copying and archive-extraction steps; `DATA_SRC` is not needed.

```bash
for category in short medium long; do
  .venvs/base/bin/python scripts/data_process/export_videomme_data.py "$category"
  .venvs/base/bin/python scripts/data_process/export_longvideobench.py "$category"
done

.venvs/base/bin/python scripts/data_process/export_videomme_v2_data.py
.venvs/base/bin/python scripts/data_process/export_lvbench.py all
.venvs/base/bin/python scripts/data_process/export_minerva_data.py
```

Run only the commands for datasets you have prepared. Direct exporter calls overwrite their selected JSONL output files; they do not perform the shell script's existing-output skip check. Each exporter creates its output directory as needed.

## Optional Sampling and Analysis

The initial data-preparation migration consists of this README, `prepare_all_data.sh`, the five `export_*.py` files, and the regression tests in `tests/`. Sampling and analysis are separate, optional additions:

- `sampling_test/`: sample question-ID lists from exported records.
- `sampling_val/` and `sampling_val_splits/`: construct validation subsets and splits.
- `stat_video_duration.py`: inspect local video durations.

The Video-MME-v2 sampling workflow requires externally prepared files under `data/Video-MME-v2/sampled/`, including `all_data.jsonl` and `videommev2_level{1,2,3}_sampled_120.txt`. The preparation script does not generate these files, and they are not automatically included with the core migration. Prepare and inspect these inputs before using the optional sampling scripts.

Keep raw annotations, videos, generated JSONL files, and sampling outputs out of the code migration commit.

## Verification

Run the regression tests from the repository root:

```bash
.venvs/base/bin/python -B -m unittest discover -s scripts/data_process/tests -v
```

The tests use temporary archives and small annotation fixtures. They exercise
preparation, retry behavior, and the real Minerva exporter; other exporters are
stubbed when checking shell dispatch. They do not download datasets, decode real
videos, or validate full Parquet dataset conversion.
