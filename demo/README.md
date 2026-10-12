# VideoXAgent Web Demo

A browser interface for video questions, streamed Agent responses, tool calls,
and follow-up conversations. The backend uses aiohttp and WebSockets, and runs
the Agent and MCP services on the server machine.

## Install and Configure

Run all commands from the repository root with Python 3.11 available.

```bash
bash scripts/install_profiles.sh base
```

The base profile includes the Agent dependencies and aiohttp. Install additional
profiles and prepare model weights only for the MCP services you enable. See the
[installation guide](../install.md) and [model setup guide](../MODEL_SETUP.md).

For a small first run, edit [`exp_configs/minimal.yaml`](../exp_configs/minimal.yaml):

- Replace the LLM model name and `model.base_url` with your provider's values.
- Replace `vlm_model_name` and `VLM_API_URL` with a vision-capable model and endpoint.
- Select the MCP file with `mcp_json_file` and keep only the services whose
  environments and weights are prepared. The example selects `.mcp.profiles.json`.

Create a valid YAML file named `key.yaml` at the repository root:

```yaml
LLM_API_KEY: "YOUR_LLM_API_KEY"
VLM_API_KEY: "YOUR_VLM_API_KEY"
```

Replace the placeholders with your credentials and keep this file out of version
control. The current demo requires `key.yaml` to exist even when credentials are
provided through environment variables; in that case, use a valid mapping such
as `{}` rather than an empty file.

## Start the Demo

Start the server with an explicit default experiment configuration:

```bash
AGENT_EXP_CONFIG=exp_configs/minimal.yaml \
  .venvs/base/bin/python demo/agent_server.py 8080
```

Open **http://localhost:8080/agent_runner.html**. The root URL redirects to the
same page. Keep the terminal open and press `Ctrl+C` to stop the server.

The shell launcher is an alternative:

```bash
bash demo/start_agent_server.sh 8080 --config exp_configs/minimal.yaml
```

The Python entry point accepts a positional port, not a `--config` argument.
Use `AGENT_EXP_CONFIG` with it. The shell launcher accepts `--config` and an
optional port. Both default to port `8080`; without an explicit configuration,
they still select `exp_configs/validation/videomme_v2_gemini.yaml`.

For a different base environment, the launcher accepts `PYTHON_BIN`:

```bash
PYTHON_BIN=/absolute/path/to/base/bin/python \
  bash demo/start_agent_server.sh 8081 --config exp_configs/minimal.yaml
```

The server binds to `127.0.0.1` by default. For a Linux host, keep this binding and
use an SSH tunnel from your computer:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@linux-host
```

For a non-loopback bind, set a separate random `VIDEOX_WEB_TOKEN` and use `--host`:

```bash
export VIDEOX_WEB_TOKEN='replace-with-a-long-random-token'
bash demo/start_agent_server.sh 8080 --config exp_configs/minimal.yaml --host 0.0.0.0
```

Remote binding without a token is rejected. Browser authentication uses username
`videoxagent` and the token as password; API clients may use a Bearer header.
Do not put the token in URLs or reuse a model API key. Use SSH or an HTTPS reverse
proxy for remote connections. Set the token on loopback too when the machine is
shared with other users. `VIDEOX_WEB_HOST` is an alternative to `--host`.

Every HTTP route and WebSocket handshake checks authentication and browser origin.
Clients must use the same origin as the page. Authenticated users are trusted Agent
operators: they share conversation history and can run the Agent's configured tools.

The media endpoint serves only images, video, and audio under `output/`, `uploads/`,
and `data/`. To display media elsewhere, explicitly add its directory:

```bash
bash demo/start_agent_server.sh 8080 --config exp_configs/minimal.yaml \
  --media-root /absolute/path/to/videos
```

Repeat `--media-root` for several directories, or set `VIDEOX_MEDIA_ROOTS` using
the platform path separator (`:` on Linux). Configuration, source, hidden files,
HTML/SVG, and symlink escapes are not served. Static access exposes only the Demo
page. External media roots should contain data you intend to share with these users.
Default media directories that are symlinks are not implicitly trusted; add their
intended external target with `--media-root`. The upload directory must remain a
real directory inside the repository.

## Ask a Video Question

1. Select `exp_configs/minimal.yaml` in **Experiment Config**.
2. Choose a system prompt. **Custom Text** must contain a nonempty prompt, such as
   "Analyze the video using the available tools and ground your answer in their
   results." **File Path** accepts existing root `SYSTEM_PROMPT*.md` files or
   Markdown files under `prompts/`, such as `SYSTEM_PROMPT.md`; arbitrary local
   files are rejected.
3. Upload a video using the file picker or drag and drop. Alternatively, enter a
   video path that already exists on the **server**, not just on your browser's
   machine. Uploads are limited to 2 GiB per file.
4. Ask a question, for example: "Read the video's duration, then describe what
   happens during its first 20 seconds." An uploaded video's server path is
   inserted into the prompt automatically.
5. Send the message and inspect the streamed response and tool events. Send a
   follow-up in the same conversation to reuse its history.

The optional tool checkboxes list **MCP service names** from the selected
configuration. Checking them adds instructions to prioritize those services; it
does not restrict which tool functions are loaded. To control availability,
edit the experiment's MCP configuration.

Start **New chat** after changing the experiment configuration or system prompt.
An existing conversation retains the Agent initialized for that conversation.
The Stop button requests cancellation of the current turn. Closing or reloading
the page also closes its WebSocket and can cancel the running turn.

## Configuration and Startup Behavior

| Setting | Purpose |
|---|---|
| `AGENT_EXP_CONFIG` | Default experiment YAML shown by the server; paths may be repository-relative or absolute. |
| `PYTHON_BIN` | Interpreter used by `start_agent_server.sh`; defaults to `./.venvs/base/bin/python`. |
| `CONVERSATIONS_DIR` | Optional parent directory for the demo's `demo/` conversation store; prefer an absolute path. |
| Experiment `model` | Agent provider, model, and endpoint settings. |
| Experiment `mcp_json_file` | MCP configuration used for actual Agent runs; defaults to `.mcp.profiles.json` when omitted. |
| Experiment `vlm_model_name`, `VLM_API_URL` | Override corresponding MCP service settings in the Web demo. |

The demo injects keys from `key.yaml` only when the corresponding key is absent
from an MCP service's `env`. Avoid placeholder keys in that `env`, because they
would prevent the configured key from being injected.

MCP sessions are cached and reused. Restart the server after changing credentials
or editing an MCP JSON file in place to ensure those changes take effect.

At startup, background preloading currently tries `.mcp.profiles.json`
independently of `AGENT_EXP_CONFIG`. It can therefore report preload errors for
services outside the minimal configuration. Actual conversation initialization
uses the selected experiment's `mcp_json_file`; a failed background preload does
not by itself mean that the selected workflow is unavailable.

## Saved Data

| Default location | Contents |
|---|---|
| `uploads/` | Uploaded videos, stored with timestamp-prefixed filenames. |
| `conversations/demo/<conversation_id>.json` | Recent conversation records and rendered history. |
| `conversations/demo/_agents/<agent_id>.json` | Saved messages and configuration references used to reconstruct an Agent conversation. |

`CONVERSATIONS_DIR` changes the conversation store location; it does not relocate
`uploads/`. Treat these directories as local runtime data.

Completed turns can be continued after a server restart by replaying saved
messages, provided the configuration, credentials, and media remain available.
This does not restore an interrupted in-flight task. Deleting an entry from
Recents removes its conversation record, but does not delete uploaded media,
generated files, or the separate raw Agent transcript.

## Files

| File | Responsibility |
|---|---|
| [`agent_server.py`](agent_server.py) | HTTP routes, WebSocket handling, uploads, and conversation APIs. |
| [`agent_runner.py`](agent_runner.py) | Agent initialization, MCP session caching, and streamed execution. |
| [`agent_runner.html`](agent_runner.html) | Browser interface, including its styles and JavaScript. |
| [`conversation_store.py`](conversation_store.py) | Persistent conversation records and resume data. |
| [`event_serialization.py`](event_serialization.py) | JSON-safe serialization of tool events. |
| [`start_agent_server.sh`](start_agent_server.sh) | Shell launcher with port, configuration, and interpreter selection. |

The demo also uses the repository's `create_deep_agent.py` and shared `utils/`
modules. It is separate from the experiment-results viewer under `html_show/`.

## Troubleshooting

- **Agent Python or aiohttp is missing:** install the base profile and start with
  its interpreter. Use `PYTHON_BIN` when the launcher needs a custom environment.
- **Configuration is missing or tools cannot be listed:** verify the selected
  YAML and its `mcp_json_file` path. Run from the repository root and keep only
  available services in that MCP configuration.
- **The page opens but a run fails:** check `key.yaml`, provider endpoints, model
  names, MCP environments, and any required model weights. Loading the page does
  not validate an Agent inference request.
- **A system prompt fails to load:** provide an existing file, or switch to
  Custom Text and enter a nonempty prompt.
- **The shell launcher prints a `hostname -i` error:** use the direct Python
  launch command and open `localhost`; host detection differs across platforms.
- **A media preview is denied:** browser previews are limited to the repository
  and allowed output directories. Upload the video through the demo when needed.

## Development Checks

```bash
bash scripts/install_profiles.sh base --dev
.venvs/base/bin/python -m pytest \
  _base/tests/test_demo_tool_config.py \
  _base/tests/test_demo_event_serialization.py
```

These tests check configuration selection and event serialization. They do not
validate provider credentials, model inference, or an end-to-end browser session.
