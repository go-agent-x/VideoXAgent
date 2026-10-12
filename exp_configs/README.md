# Experiment Configuration Examples

These examples are adapted from the experiment files in LVideoXAgent. They retain
the source orchestrator settings while replacing deployment endpoints and dataset
paths. Each example starts with **one question and one worker** so you can check
your installation before increasing the workload.

The source files selected fixed 150-question subsets. These public examples leave
that filter unset and use the JSONL exports produced by this repository. They are
starting points for running experiments, not the exact configurations that produced
the paper's reported results.

## Available Configurations

| Dataset | Example | Orchestrator | Source file in LVideoXAgent |
| --- | --- | --- | --- |
| Video-MME-Long | [videomme_long_claude.yaml](examples/videomme_long_claude.yaml) | Claude Opus 4.6 | `exp_configs/videomme_exp_long_opus.yaml` |
| LongVideoBench-Long | [longvideobench_long_claude.yaml](examples/longvideobench_long_claude.yaml) | Claude Opus 4.6 | `exp_configs/longvideobench_exp.yaml` |
| LVBench | [lvbench_gemini.yaml](examples/lvbench_gemini.yaml) | Gemini 3.1 Pro Preview | `exp_configs/lvbench_exp.yaml` |
| MINERVA | [minerva_gemini.yaml](examples/minerva_gemini.yaml) | Gemini 3.1 Pro Preview via an OpenAI-compatible API | `exp_configs/minerva_exp.yaml` |

The original YAMLs did not pin the VLM-tool model; it came from the local MCP
configuration. Set `vlm_model_name` and `VLM_API_URL` for your deployment. The
Video-MME example also replaces a deployment-specific Claude alias with
`claude-opus-4-6`; confirm the model ID accepted by your provider.

The LVBench and MINERVA orchestrators above differ from the paper's main result
tables. In particular, the LVBench example does **not** enforce the paper's
vision-only setting. Reproducing that setting requires an MCP configuration with
subtitle and all audio-capable tools disabled, including audio modes in composite
tools such as emotion analysis. `disabled_builtin_tools` affects the agent's built-in
tools, not MCP servers. No exact MINERVA Claude/GLM + Qwen configuration was found
among the source experiment files used for these examples.

## Prepare Your Environment and Data

Run commands from the repository root. Install the base environment and evaluation
dependencies, plus the tool profiles and model weights required by your selected
MCP services:

```bash
bash scripts/install_profiles.sh base --evaluation
```

See [dependency profiles](../requirements/profiles/README.md) and
[model setup](../MODEL_SETUP.md) for the tool environments. Prepare your locally
supplied datasets with the supported entry point:

```bash
bash scripts/data_process/prepare_all_data.sh \
  --data-src /absolute/path/to/prepared-datasets \
  videomme longvideobench lvbench minerva
```

Select only the datasets you have. The required archive layout and JSONL paths are
listed in the [data preparation guide](../scripts/data_process/README.md). Re-export
on the machine that will run inference, because prompts contain absolute video paths.

Before launching an example:

1. Create a local `key.yaml` from `key.yaml.example` if it does not exist, and fill
   in `LLM_API_KEY` and `VLM_API_KEY`.
2. Edit the example's `model` provider, model ID, and endpoint, and replace the
   `YOUR_VLM_MODEL` and VLM endpoint placeholders. YAML strings are passed literally;
   `${ENV_VAR}` substitution is not supported.
3. Review the MCP file selected by `mcp_json_file`. Existing server-level
   `VLM_MODEL_NAME`, `VLM_API_URL`, and API-key values take precedence over the
   experiment YAML and `key.yaml`. Remove stale overrides or set them consistently;
   also review `VLM_MODEL_FALLBACKS` if the experiment must use one model.
4. Enable only MCP services with available server files, installed environments,
   and required model weights.
5. Check `data_files`, `system_prompt_file`, and the unique `exp_name` in the YAML.
   All paths in these examples are relative to the repository root.

## Launch an Experiment

Use the existing batch launcher with one configuration. For example:

```bash
bash auto_deepagents.sh exp_configs/examples/videomme_long_claude.yaml --log
```

The other examples use the same entry point:

```bash
bash auto_deepagents.sh exp_configs/examples/longvideobench_long_claude.yaml --log
bash auto_deepagents.sh exp_configs/examples/lvbench_gemini.yaml --log
bash auto_deepagents.sh exp_configs/examples/minerva_gemini.yaml --log
```

Choose the experiment you want to run. `--log` also saves console output under
`logs/`. The launcher uses `.venvs/base/bin/python` by default; set `PYTHON_BIN`
to select another interpreter with the required dependencies. The launcher runs
with Hugging Face offline mode enabled, so download required weights first.

Adjust `max_samples`, `sample_rate`, and `workers` in the YAML after the initial
sample succeeds. `max_samples` limits records considered from each JSONL file;
set it high enough to include your intended split. To use a fixed subset, add a
`qid_file` alongside the relevant `data_files.path`; that file must contain one
matching `question_id` per line and is not included with these examples. With a
QID filter, increase `max_samples` enough to scan the complete source export.

The shell launcher accepts the configuration path and `--log`, not worker or
sample-count flags. One worker runs a separate agent process per question;
multiple workers reuse MCP sessions through `batch_worker.py`.

For a direct `video-agent.py` run, `--disable-tools read_file write_file execute`
adds built-in tool exclusions to the YAML's `disabled_builtin_tools` list. The
lists are combined and deduplicated, so CLI options cannot re-enable a tool
disabled by the experiment. This option does not filter MCP tools; use the MCP
configuration or `scripts/configure_mcp_profiles.py --disable-server NAME` to
exclude MCP services. Batch runs and the web demo use the YAML setting.

Session indexes are written to `conversations/all_data_path_<exp_name>.json`, with
transcripts under `conversations/transcripts/`. The launcher prompts when that
index already exists. `skip_completed` only skips entries whose `is_correct` is
already a Boolean; inference completion alone does not mark a question as scored.
Inspect failures and dataset coverage before interpreting an experiment's accuracy.

These examples have been checked for YAML structure and local configuration
references; real provider calls and benchmark accuracy have not been validated.
