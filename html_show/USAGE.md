# VideoXAgent Dashboard Guide

## Tool Workflow Dashboard

```bash
sh html_show/start_session_server_tool.sh
```

By default, the server listens only on **127.0.0.1:8923**. Open the `Tool Workflow` URL shown in the startup log.
The script supports macOS and Linux and automatically selects an available Python 3 interpreter.
Use `PYTHON_BIN=/path/to/python3 sh html_show/start_session_server_tool.sh` to select an interpreter explicitly.

### Access Control

When running on a Linux server, keep the loopback binding and access the dashboard through SSH forwarding:

```bash
# Run on your own computer; replace the username and server address.
ssh -N -L 8923:127.0.0.1:8923 user@linux-host
```

Then open the `127.0.0.1:8923` URL from the startup log in your local browser.
To bind directly to another network interface, first set a separate random access token, then specify the listening address:

```bash
export VIDEOX_WEB_TOKEN='replace-with-a-long-random-token'
sh html_show/start_session_server_tool.sh --ip 0.0.0.0 --data /path/to/results.json
```

The script refuses to start a non-loopback listener without a token and leaves the previous service running.
For browser login, use `videoxagent` as the username and `VIDEOX_WEB_TOKEN` as the password.
API clients can also use `Authorization: Bearer ...`. Do not put the token in URLs or launch arguments, or reuse a model API key.
Use SSH or an HTTPS reverse proxy for remote access to avoid sending traffic over plain HTTP on the public Internet.
You can also set this token for a loopback listener to restrict access on a shared machine.

The page and API must share the same origin. The server checks `Host` and the browser's `Origin` and does not enable open CORS.
Static serving is limited to the generated dashboard and designated frontend assets; directory browsing is disabled.
The trace API reads only JSONL files registered in the current results index. The media API reads only images, video, and audio in allowed directories.
Keys, configuration files, source code, hidden paths, and symlinks that escape the allowed directories are not served.

To view new data, rerun the script without manually killing the service:

```bash
sh html_show/start_session_server_tool.sh --data /path/to/new_data.json
```

After generating the new page successfully, the script automatically stops previous visualization services
that it started for this project, including instances on other ports. It waits for them to exit before starting a new service.
It sends TERM first and force-stops processes that do not exit within 3 seconds. If the startup health check fails, it also cleans up the new process.
If page generation fails, the previous service remains running. Processes from other projects or programs are left alone.
If one of those processes occupies the target port, the script reports an error; use `--port 9000` to choose another port.
`sh html_show/start_session_server_tool.sh --port 8923 --stop` still stops only this project's service on the specified port.

Styling and chart dependencies are stored in `tailwind.min.js` and `chart.min.js`. Newly generated
HTML embeds both libraries, so the browser does not need to access an external CDN. Existing HTML pages can still
load these dependencies from the two files in the same directory. See [THIRD_PARTY.md](THIRD_PARTY.md) for versions, sources, and licenses.

If styling is missing or charts are blank, first check the server log for 404 errors for these two JS files.
Restore the files and refresh the existing page, or rerun the startup script to generate a new page.

Image previews in tool records are 288×288 pixels (three times the previous size). They show the entire image
against a frosted-glass background made from the same image. Clicking an image opens an in-page preview that preserves its aspect ratio.
On first opening, it fits within the window without exceeding its original size. Use the circled plus and minus buttons
in the upper right to zoom, then scroll to explore the enlarged image. Click ❎, click the background outside the image, or press Esc to close it.
The image viewer's styles and scripts are also embedded in the generated HTML. Regenerate the page after changing them.

### Run Metadata

The header reads metadata from the results JSON and corresponding JSONL files instead of using a fixed update time:

- `Data` shows the actual results filename. `Model` first uses model configuration from `run_metadata`, `metadata`, or
  `experiment_config`, falling back to the model recorded in the session or trace.
- If `started_at` / `ended_at` are explicitly recorded, the header shows `Run started` / `Run ended`.
  Otherwise, it shows `First event` / `Last event` from the JSONL and identifies the timestamp source.
  If the JSONL has no timestamps, it falls back to session times.
- Timestamps without a recorded time zone are explicitly marked; no time zone is assumed. The time when configuration was written is shown separately as
  `Config recorded`. The file's `Data modified` time is shown only when no recorded timestamps are available,
  so a file copy time is not presented as a run time. Missing model, dataset, configuration, and other fields are not inferred.
- The optional `benchmark`, `config_file`, and `run_id` fields are supported. Only the configuration filename is displayed,
  not its contents, system prompts, or key fields.

New results can record the following information in JSON. Existing results do not need to be rewritten; their timestamps can still be extracted from traces:

```json
{
  "run_metadata": {
    "benchmark": "Video-MME-Long",
    "model": "your-actual-model",
    "run_id": "experiment-01",
    "config_file": "configs/experiment-01.yaml",
    "started_at": "2026-09-24T10:00:00+08:00",
    "ended_at": "2026-09-24T11:00:00+08:00"
  },
  "sessions": []
}
```

### Moving or Downloading Benchmark Results

Save the results JSON, JSONL files, and referenced images together, preserving the directory structure. For example:

```text
downloaded-run/
  results.json
  transcripts/session-001.jsonl
  output/frames/frame-001.png
  data/videomme/videos/video-001.mp4   # Optional local video
```

In `results.json`, use a relative `transcript_path` such as `transcripts/session-001.jsonl`.
Tool image paths can use `output/frames/frame-001.png`. Start the server with:

```bash
sh html_show/start_session_server_tool.sh \
  --data /path/to/downloaded-run/results.json \
  --data-root /path/to/downloaded-run \
  --tag experiment-01
```

`--data-root` specifies the root directory of the results bundle, which can be outside the repository. When omitted,
the repository root is used if the results JSON is inside the repository; otherwise, the JSON file's directory is used.
A running server uses one data root and one set of path mappings. Restart it when switching to a separate results bundle,
and open the page generated for that bundle.
If an older page belongs to a different results index, restart with the corresponding `--data` before reading its traces.

Newly generated HTML stores trace references relative to the data root instead of embedding the generating machine's repository root.
After moving the entire results bundle, start the server from its new location. Standard `conversations/`,
`transcripts/`, `output/`, and `data/` paths in traces are resolved against the bundle directory. Other relative image paths are resolved first
against the current JSONL file's directory, then against the data root and the results JSON file's directory.

For older JSONL files containing absolute paths, common directory prefixes are relocated using the directory structure above.
Map custom legacy directories explicitly without performing bulk replacements in the original JSONL files:

```bash
sh html_show/start_session_server_tool.sh \
  --data /path/to/downloaded-run/results.json \
  --data-root /path/to/downloaded-run \
  --path-map '/old/machine/project=.' \
  --path-map '/old/custom-media=/path/to/downloaded-media' \
  --tag experiment-01
```

The left side of a mapping is the old absolute path prefix. The right side can be a new absolute directory or a directory relative to the data root.
Repeat the option for multiple mappings; the longest matching prefix takes precedence. Explicit mappings take precedence over old files that still exist on the original machine.
File access is limited to the selected data root and explicitly mapped destination directories. Symlinks cannot escape these directories.
Download the actual image and trace files too; paths in HTML or JSON alone cannot restore missing files.

By default, the startup script only displays results and **does not enrich or rewrite the results JSON / JSONL**.
Add `--enrich` explicitly to enable answer and metadata enrichment. This feature still uses this repository's dataset configuration.

Use the same path options when generating HTML alone or starting the server manually:

```bash
python3 html_show/gen_dashboard.py \
  --data /path/to/downloaded-run/results.json \
  --data-root /path/to/downloaded-run \
  --out /path/to/downloaded-run/dashboard.html

python3 html_show/server.py 8923 /path/to/downloaded-run \
  --data /path/to/downloaded-run/results.json \
  --data-root /path/to/downloaded-run \
  --dashboard /path/to/downloaded-run/dashboard.html
# Open http://127.0.0.1:8923/dashboard.html
```

If you used `--path-map` during generation, pass the same old prefixes and current destination directories when starting the server manually.
`--out` writes only the specified file. Use `--dashboard` to explicitly select a custom HTML filename.
Open the page through the running HTTP server; traces and images are read from APIs on the same origin.
`api_base` accepts only same-origin reverse proxy paths; cross-origin overrides are ignored.

## Current Page Features

- **Evaluation** (default): evaluation metrics, session filtering and pagination, answer comparison, video playback, interaction traces, and tool image previews.
- **Tool Usage**: filter tool statistics by video duration or task type. Click a chart or `View Sessions` to view the corresponding sessions in Evaluation.
- Supports JSONL traces in Agentic, Claude Code, and Codex formats.

## File Structure

| File | Description |
|------|------|
| `start_session_server_tool.sh` | Current entry point; generates pages and manages the service lifecycle |
| `gen_dashboard.py` | Generates HTML with Evaluation and Tool Usage views |
| `server.py` | Static files, trace and media APIs, with support for video Range requests |
| `benchmark_paths.py` | Data roots, legacy path mappings, and file access boundaries |
| `web_files.py` | Traces registered in the current index, plus media and static asset allowlists |
| `run_metadata.py` | Extracts run metadata and timestamps with their sources |
| `image_viewer.js` / `image_viewer.css` | In-page image previews, zoom, and keyboard interaction |
| `brand.css` / `videoxagent-icon.png` | Page branding styles and icon |
| `tailwind.min.js` / `chart.min.js` | Embedded third-party styling and chart dependencies |
| `add_gt_answer.py` | Manually enriches answers and evaluation information; modifies the selected data |
| `tests/` | Tests for paths, metadata, generated pages, and the service lifecycle |

## API Endpoints

| Endpoint | Description |
|------|------|
| `GET /api/read?path=...&base=...` | Reads JSONL traces registered in the current index; `base` is optional |
| `GET /api/file?path=...&base=...` | Reads images, video, and audio in approved directories; supports Range; `base` is optional |
| `GET /healthz` | Service health check; credentials are also required when authentication is enabled |

These routes support HEAD with the same access checks.

## Validation

```bash
python3 -m unittest discover -s html_show/tests -v
bash -n html_show/start_session_server_tool.sh
```

JavaScript checks and interaction tests for generated pages require Node.js.
