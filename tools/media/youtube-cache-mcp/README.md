# YouTube Cache MCP Installation Guide

## Dependency Installation

### 1. Python Dependencies

Run from the repository root:

```bash
bash scripts/install_profiles.sh base
```

### 2. EJS Challenge Solver (Required)

yt-dlp requires EJS scripts to solve YouTube's JavaScript challenges. See: https://github.com/yt-dlp/yt-dlp/wiki/EJS

The base profile already includes `yt-dlp[default]` and its EJS dependencies, so no additional pip installation is needed.
Use the tools in the base environment for the commands below:

```bash
source .venvs/base/bin/activate
```

### 3. JavaScript Runtime (Required)

yt-dlp requires a JavaScript runtime to parse YouTube pages. Deno is recommended:

**Method A: Use the Installation Script (Requires unzip)**

apt-get install unzip -y

```bash
curl -fsSL https://deno.land/install.sh -o /tmp/deno-install.sh
CI=1 sh /tmp/deno-install.sh
export DENO_INSTALL="$HOME/.deno"
export PATH="$DENO_INSTALL/bin:$PATH"
deno --version
```

`CI=1` skips the installer's optional JSR shell configuration helper; PATH is explicitly set by the commands above.
If the installer reports that Deno was installed successfully but JSR then returns a 403 error, first run
`~/.deno/bin/deno --version`. If the version is displayed correctly, there is no need to reinstall Deno.

**Add to PATH (Add to ~/.bashrc or ~/.zshrc)**
```bash
export PATH="$HOME/.deno/bin:$PATH"
```

**Verify the Installation**
```bash
deno --version
```

### 4. Proxy Configuration

Configure the proxy environment variables in `.mcp.json`:

```json
"youtube-cache": {
  "command": "./.venvs/base/bin/python",
  "args": ["./tools/media/youtube-cache-mcp/server.py"],
  "env": {
    "HTTP_PROXY": "http://your-proxy:port",
    "HTTPS_PROXY": "http://your-proxy:port"
  }
}
```

## Troubleshooting

### Issue 1: "Sign in to confirm you're not a bot"

YouTube has detected bot activity and requires browser cookies:

**Option A: Export Cookies from the Browser (Recommended)**

```bash
# Use browser cookies (sign in to YouTube in the browser first)
yt-dlp --cookies-from-browser chrome "https://www.youtube.com/watch?v=VIDEO_ID"

# Supported browsers: chrome, firefox, edge, safari, opera, brave
```

**Option B: Export a Cookies File**

1. Install the browser extension "Get cookies.txt LOCALLY"
2. Visit youtube.com and sign in
3. Export the cookies to `cookies.txt`
4. Use: `yt-dlp --cookies cookies.txt URL`


For details, see: https://github.com/yt-dlp/yt-dlp/wiki/EJS

## Usage Examples

```bash
# Download using a cookies file (recommended, verified to work)
outnet yt-dlp --cookies tools/media/youtube-cache-mcp/www.youtube.com_cookies.txt \
  -f "best[ext=mp4]/best" \
  -o "data/youtube_cache/%(id)s.%(ext)s" \
  "https://www.youtube.com/watch?v=VIDEO_ID"

# If the local browser can access YouTube directly
yt-dlp --cookies-from-browser chrome \
  -f "best[ext=mp4]/best" \
  -o "data/youtube_cache/%(id)s.%(ext)s" \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Obtaining a Cookies File

1. Sign in to YouTube in your local browser
2. Install the extension "Get cookies.txt LOCALLY"
3. Export cookies from the youtube.com page
4. Upload the file to `tools/media/youtube-cache-mcp/www.youtube.com_cookies.txt` on the server
5. Filter the cookies with `filter_cookies.py` (optional):
   ```bash
   python tools/media/youtube-cache-mcp/filter_cookies.py cookies.txt tools/media/youtube-cache-mcp/www.youtube.com_cookies.txt
   ```

## MCP Tools

| Tool | Function |
|------|------|
| `get_video` | Get the local path to a video, downloading it automatically if it is not present |
| `check_cache` | Check whether a video is cached (without triggering a download) |
