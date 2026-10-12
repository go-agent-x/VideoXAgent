# Dashboard browser dependencies

These files are bundled locally so dashboard styles and charts do not require
access to a CDN. `gen_dashboard.py` embeds them in newly generated HTML files;
older HTML files can still load them from the same directory over HTTP.
Each bundled file includes its MIT license notice.

| Local file | Version | Upstream download |
| --- | --- | --- |
| `tailwind.min.js` | Tailwind CSS 3.4.17 (browser build) | https://cdn.tailwindcss.com/3.4.17 |
| `chart.min.js` | Chart.js 4.4.8 (UMD build) | https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.js |

Upstream SHA-256 values, before prepending the license notice:

```text
176e894661aa9cdc9a5cba6c720044cbbf7b8bd80d1c9a142a7c24b1b6c50d15  tailwind.min.js
e4cf4d144b222634f2e64ff707cd57b953f8de4b65b231cd5ac6bde114648e4d  chart.min.js
```

Licenses:
- https://github.com/tailwindlabs/tailwindcss/blob/v3.4.17/LICENSE
- https://github.com/chartjs/Chart.js/blob/v4.4.8/LICENSE.md

The generator uses the Tailwind 3 `tailwind.config` browser API. Keep this
dependency on version 3 unless that configuration is migrated as well.
