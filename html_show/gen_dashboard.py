#!/usr/bin/env python3
"""Generate tool_workflow.html - complete self-contained dashboard.

Usage:
    python gen_dashboard.py                           # default: reads all_data_path.json, outputs tool_workflow.html
    python gen_dashboard.py --tag exp1                # output: tool_workflow_exp1.html, with tag shown in header
    python gen_dashboard.py --data /path/to/data.json # use custom data JSON
    python gen_dashboard.py --out /path/to/out.html   # custom output path (overrides --tag naming)
"""
import os, json, re, base64, argparse
from pathlib import Path
from benchmark_paths import BenchmarkPaths
from run_metadata import collect_run_metadata

def _parse_args():
    parser = argparse.ArgumentParser(description="Generate tool_workflow dashboard HTML")
    parser.add_argument("--tag", default="", help="Experiment tag shown in HTML header and used in output filename")
    parser.add_argument("--data", default="", help="Path to data JSON (default: conversations/all_data_path.json)")
    parser.add_argument("--out", default="", help="Custom output HTML path (overrides default naming)")
    parser.add_argument("--data-root", default=None, help="Root directory of the benchmark results, transcripts and media")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Map an old path prefix to a directory under the data root (repeatable)")
    return parser.parse_args()

_ARGS = _parse_args()
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _inline_script(filename):
    """Bundle local dependencies so exported dashboards work without a CDN."""
    asset = Path(_SCRIPT_DIR) / filename
    if not asset.is_file():
        raise SystemExit(f"Missing dashboard asset: {asset}. Restore this file before generating HTML.")
    source = asset.read_text(encoding="utf-8")
    # An HTML parser recognizes closing script tags even inside JS strings.
    source = re.sub(r"</script", lambda m: "<\\/" + m.group(0)[2:], source, flags=re.I)
    source = re.sub(r"(?m)^//# sourceMappingURL=.*$", "", source)
    return f'<script data-dashboard-asset="{filename}">\n{source}\n</script>'

_DASHBOARD_SCRIPTS = "\n".join(_inline_script(name) for name in (
    "tailwind.min.js", "chart.min.js", "image_viewer.js",
))
_IMAGE_VIEWER_CSS = (Path(_SCRIPT_DIR) / "image_viewer.css").read_text(encoding="utf-8")
_BRAND_CSS = (Path(_SCRIPT_DIR) / "brand.css").read_text(encoding="utf-8")

if _ARGS.out:
    OUT = _ARGS.out
elif _ARGS.tag:
    OUT = os.path.join(_SCRIPT_DIR, f"tool_workflow_{_ARGS.tag}.html")
else:
    OUT = os.path.join(_SCRIPT_DIR, "tool_workflow.html")
_IMG_PATH = os.path.join(_SCRIPT_DIR, "videoxagent-icon.png")
with open(_IMG_PATH, "rb") as _f:
    _LOGO_B64 = base64.b64encode(_f.read()).decode()

def esc(s):
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

_THIS_DIR = Path(__file__).resolve().parent

# A fresh code checkout may keep all benchmark results outside the repository.
_REPO_ROOT = _THIS_DIR.parent
_DATA_PATH = Path(_ARGS.data).expanduser().resolve() if _ARGS.data else _REPO_ROOT / "conversations" / "all_data_path.json"
if _ARGS.data and not _DATA_PATH.is_file():
    raise SystemExit("Benchmark data JSON not found: {}".format(_DATA_PATH))
if _ARGS.data_root and not Path(_ARGS.data_root).expanduser().is_dir():
    raise SystemExit("Benchmark data root not found: {}".format(_ARGS.data_root))
if _ARGS.data_root:
    _ARGS.data_root = str(Path(_ARGS.data_root).expanduser().resolve())
_PATHS = BenchmarkPaths(_REPO_ROOT, data_file=_DATA_PATH, data_root=_ARGS.data_root, path_maps=_ARGS.path_map)

def _resolve_transcript(tp_raw):
    """Resolve a transcript using the same path rules as the server."""
    if not tp_raw:
        return None
    try:
        return _PATHS.resolve(str(tp_raw).strip())
    except PermissionError:
        return None

def _parse_iso_ts(ts):
    """Parse ISO-like timestamps from Codex/Claude jsonl.

    Expected examples:
      - 2026-03-11T16:28:51.168Z
      - 2026-03-11T16:28:51.168+00:00
    """
    if not ts:
        return None
    try:
        from datetime import datetime

        s = str(ts).strip()
        # Python's datetime.fromisoformat does not accept 'Z' suffix.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None

def compute_eval_stats():
    """Read all_data_path.json and transcripts, return stats + slim sessions."""
    data_path = _DATA_PATH
    if not os.path.exists(data_path):
        return None
    # Load session data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sessions = data.get('sessions', [])
    # Calculate basic statistics
    total = len(sessions)
    has_answer = sum(1 for s in sessions if s.get('model_answer'))
    correct = sum(1 for s in sessions if s.get('is_correct'))
    accuracy = round(correct / has_answer * 100, 1) if has_answer else 0
    # Parse transcripts to extract tool usage and timing
    tool_set = set()
    solve_times = []
    for sess in sessions:
        tp_raw = sess.get('transcript_path', '')
        tp = _resolve_transcript(tp_raw)
        if tp is None:
            continue
        t0 = None
        t1 = None
        with open(tp, 'r', encoding='utf-8') as ftx:
            for line in ftx:
                try:
                    e = json.loads(line)
                    dt = _parse_iso_ts(e.get('timestamp', ''))
                    if dt is not None:
                        if t0 is None:
                            t0 = dt  # First timestamp
                        t1 = dt  # Last timestamp
                    # Track tool names from function calls
                    if e.get('type') == 'response_item':
                        pl = e.get('payload', {})
                        if pl.get('type') == 'function_call':
                            tool_set.add(pl.get('name', ''))
                    # Track tool names from agentic-style tool entries
                    elif e.get('type') == 'tool':
                        tool_set.add(e.get('message', {}).get('name', ''))
                except:
                    pass
        if (t0 is not None and t1 is not None
                and (t0.utcoffset() is None) == (t1.utcoffset() is None)
                and t1 >= t0):
            solve_times.append((t1 - t0).total_seconds())
    avg_solve = round(sum(solve_times) / len(solve_times), 1) if solve_times else 0
    tools_used = len(tool_set)
    # Build slim session objects
    slim = []
    for s in sessions:
        tp_raw = s.get('transcript_path', '') or ''
        sid = s.get('session_id', '') or ''
        tp = _resolve_transcript(tp_raw)
        tp_stat = tp.stat() if tp is not None else None
        slim.append({
            'session_id': sid,
            # The server resolves this reference against its current data root.
            'transcript_path': _PATHS.reference(tp) if tp is not None else str(tp_raw).strip(),
            'transcript_exists': tp is not None,
            'transcript_size': tp_stat.st_size if tp_stat else 0,
            'transcript_mtime': tp_stat.st_mtime if tp_stat else 0,
            'init_user_query': s.get('init_user_query', ''),
            'gt_answer': s.get('gt_answer', ''),
            'model_answer': s.get('model_answer', ''),
            'is_correct': s.get('is_correct', False),
            'duration': s.get('duration', ''),
            'domain': s.get('domain', ''),
            'sub_category': s.get('sub_category', ''),
            'task_type': s.get('task_type', ''),
            'options': s.get('options', []),
            'videoID': s.get('videoID', ''),
            'question_id': s.get('question_id', ''),
        })
    # Keep sessions with available transcripts first to avoid immediate 404s in the viewer.
    slim.sort(key=lambda x: (not x.get('transcript_exists', False), -(x.get('transcript_mtime') or 0)))
    # Extract unique category values for filtering
    domains = sorted(set(s.get('domain', '') for s in sessions if s.get('domain')))
    sub_categories = sorted(set(s.get('sub_category', '') for s in sessions if s.get('sub_category')))
    task_types = sorted(set(s.get('task_type', '') for s in sessions if s.get('task_type')))
    return {'total': total, 'has_answer': has_answer, 'correct': correct,
            'accuracy': accuracy, 'avg_solve_time': avg_solve, 'tools_used': tools_used,
            'solve_time_samples': len(solve_times),
            'domains': domains, 'sub_categories': sub_categories, 'task_types': task_types,
            'sessions_json': json.dumps(slim, ensure_ascii=False).replace('</','<\\/')}

def compute_tool_usage_stats():
    """Read all_data_path.json transcripts, return per-tool usage stats."""
    data_path = _DATA_PATH
    if not os.path.exists(data_path):
        return None
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sessions = data.get('sessions', [])
    total_sessions = len(sessions)
    overall = {}
    by_duration = {}
    by_task_type = {}
    task_types_list = set()
    for sess in sessions:
        qid = sess.get('question_id', '')
        dur = sess.get('duration', '')
        tt = sess.get('task_type', '')
        if tt:
            task_types_list.add(tt)
        tp_raw = sess.get('transcript_path', '')
        tp = _resolve_transcript(tp_raw)
        if tp is None:
            continue
        stool = {}
        with open(tp, 'r', encoding='utf-8') as ftx:
            for line in ftx:
                try:
                    e = json.loads(line)
                    # Codex-style logs
                    if e.get('type') == 'response_item':
                        pl = e.get('payload', {})
                        if pl.get('type') == 'function_call':
                            raw = pl.get('name', '')
                            name = raw.split('__')[-1] if '__' in raw else raw
                            if name:
                                stool[name] = stool.get(name, 0) + 1
                    # Claude Code-style logs
                    elif e.get('type') == 'assistant':
                        content = e.get('message', {}).get('content')
                        if isinstance(content, list):
                            for ci in content:
                                if isinstance(ci, dict) and ci.get('type') == 'tool_use':
                                    raw = ci.get('name', '')
                                    name = raw.split('__')[-1] if '__' in raw else raw
                                    if name:
                                        stool[name] = stool.get(name, 0) + 1
                    # Agentic-style logs (separate tool entries)
                    elif e.get('type') == 'tool':
                        raw = e.get('message', {}).get('name', '')
                        name = raw.split('__')[-1] if '__' in raw else raw
                        if name:
                            stool[name] = stool.get(name, 0) + 1
                except:
                    pass
        for name, cnt in stool.items():
            if name not in overall:
                overall[name] = {'c': 0, 'q': set()}
            overall[name]['c'] += cnt
            overall[name]['q'].add(qid)
            if dur:
                by_duration.setdefault(dur, {})
                if name not in by_duration[dur]:
                    by_duration[dur][name] = {'c': 0, 'q': set()}
                by_duration[dur][name]['c'] += cnt
                by_duration[dur][name]['q'].add(qid)
            if tt:
                by_task_type.setdefault(tt, {})
                if name not in by_task_type[tt]:
                    by_task_type[tt][name] = {'c': 0, 'q': set()}
                by_task_type[tt][name]['c'] += cnt
                by_task_type[tt][name]['q'].add(qid)
    def fin(td, tot):
        r = {}
        for n in sorted(td.keys(), key=lambda x: td[x]['c'], reverse=True):
            d = td[n]; sc = len(d['q'])
            r[n] = {'total_calls': d['c'], 'session_count': sc,
                     'avg_per_session': round(d['c'] / sc, 1) if sc else 0,
                     'usage_rate': round(sc / tot * 100, 1) if tot else 0,
                     'qids': list(d['q'])}
        return r
    tools = fin(overall, total_sessions)
    ds = {}
    for dk, td in by_duration.items():
        dt = sum(1 for s in sessions if s.get('duration') == dk)
        ds[dk] = fin(td, dt)
    ts = {}
    for tk, td in by_task_type.items():
        tt2 = sum(1 for s in sessions if s.get('task_type') == tk)
        ts[tk] = fin(td, tt2)
    result = {'tools': tools, 'by_duration': ds, 'by_task_type': ts,
              'total_sessions': total_sessions, 'task_types': sorted(task_types_list)}
    result['json'] = json.dumps(result, ensure_ascii=False, default=list).replace('</','<\\/')
    return result

# Build
p = []
def w(s): p.append(s)
print("Computing evaluation statistics...")
eval_stats = compute_eval_stats()
if eval_stats:
    print(
        f"  Eval: {eval_stats['total']} sessions, {eval_stats['accuracy']}% accuracy, "
        f"{eval_stats['tools_used']} tools, {eval_stats['avg_solve_time']}s avg solve "
        f"({eval_stats.get('solve_time_samples', 0)} samples)"
    )
else:
    print("  Eval: no data found")
print("Computing tool usage statistics...")
tool_usage_stats = compute_tool_usage_stats()
if tool_usage_stats:
    print(f"  Tool Usage: {len(tool_usage_stats['tools'])} unique tools across {tool_usage_stats['total_sessions']} sessions")
else:
    print("  Tool Usage: no data found")

_tag_label = _ARGS.tag or ""
_tag_title_suffix = f" [{_tag_label}]" if _tag_label else ""
_run_data = json.loads(_DATA_PATH.read_text(encoding="utf-8")) if _DATA_PATH.is_file() else {}
_run_transcripts = []
for _session in _run_data.get("sessions", []):
    _tp = _resolve_transcript(_session.get("transcript_path", ""))
    if _tp is not None:
        _run_transcripts.append(_tp)
_run_metadata = collect_run_metadata(_run_data, _DATA_PATH, list(dict.fromkeys(_run_transcripts)))

# HEAD
w('<!DOCTYPE html>')
w('<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">')
w(f'<title>VideoXAgent - Video Understanding Dashboard{esc(_tag_title_suffix)}</title>')
w('<link rel="icon" type="image/png" href="data:image/png;base64,' + _LOGO_B64 + '">')
w(_DASHBOARD_SCRIPTS)
w('<style data-dashboard-asset="image_viewer.css">\n' + _IMAGE_VIEWER_CSS + '\n</style>')
w('<style data-dashboard-asset="brand.css">\n' + _BRAND_CSS + '\n</style>')
w('''
<script>tailwind.config={theme:{extend:{fontFamily:{sans:['Inter','system-ui','sans-serif']},colors:{brand:{50:'#f0f4ff',100:'#dbe4ff',200:'#bac8ff',300:'#91a7ff',400:'#748ffc',500:'#5c7cfa',600:'#4c6ef5',700:'#4263eb',800:'#3b5bdb',900:'#364fc7'}}}}}</script>
<style>
.tab-panel{display:none}.tab-panel.active{display:block}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:3px}
.yt-inline-player iframe,.yt-inline-player video{width:100%;aspect-ratio:16/9;border:none;display:block;border-radius:8px}
.yt-inline-player:empty{display:none}
.session-details{display:none}
.session-details.expanded{display:block}
.interaction-content{display:none}
.interaction-content.expanded{display:block}
.trace-item-details{display:none}
.trace-item-details.expanded{display:block}
.trace-timeline{position:relative;padding-left:25px}
.trace-timeline::before{content:'';position:absolute;left:8px;top:0;bottom:0;width:2px;background:linear-gradient(to bottom,#7c3aed,#3b82f6,#10b981)}
.trace-item-viewer{position:relative;margin-bottom:8px;overflow:hidden}
.trace-item-viewer::before{content:attr(data-index);position:absolute;left:-25px;top:10px;width:16px;height:16px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:bold;color:#fff;z-index:1}
.trace-item-viewer.assistant-trace::before{background:#10b981}
.trace-item-viewer.tool-trace::before{background:#f59e0b}
</style></head>''')

# NAV + HERO
w('''<body class="bg-gray-50 font-sans text-gray-800 antialiased min-h-screen">
<nav class="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
<div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
<div class="videoxagent-brand">
<img src="data:image/png;base64,''' + _LOGO_B64 + '''" class="videoxagent-logo" alt="">
<span class="videoxagent-wordmark">VideoXAgent</span></div>
<div class="hidden sm:flex items-center gap-6 text-sm font-medium text-gray-500">
<a href="javascript:void(0)" onclick="switchTab('evaluation')" class="hover:text-brand-600 transition-colors">Evaluation</a>
<a href="javascript:void(0)" onclick="switchTab('toolusage')" class="hover:text-brand-600 transition-colors">Tool Usage</a>
</div></div></nav>
<header class="bg-white border-b border-gray-100">
<div class="max-w-3xl mx-auto px-6 py-14 text-center">
<h1 class="text-4xl sm:text-5xl font-extrabold text-gray-900 leading-tight tracking-tight">Online Video Agent Harness for Long Video Understanding<br></h1>''')
if _tag_label:
    w(f'<div class="mt-3 inline-block px-4 py-1.5 rounded-full text-sm font-bold bg-brand-100 text-brand-700 border border-brand-200">{esc(_tag_label)}</div>')
w(f'''<p class="mt-4 text-lg text-gray-500 leading-relaxed max-w-2xl mx-auto">Query-driven video exploration with expert tools and multimodal evidence, all on demand.</p>
<div id="run-metadata" class="mt-5 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-sm text-gray-500">''')
for _label, _value in _run_metadata["fields"]:
    w(f'<span class="break-all"><span class="font-medium">{esc(str(_label))}:</span> {esc(str(_value))}</span>')
w('''</div></div></header>
<main class="max-w-7xl mx-auto px-6 py-10">
<div class="flex gap-2 mb-8 flex-wrap" id="tabBtns">
<button class="tab-btn px-5 py-2.5 rounded-lg text-sm font-medium border transition-all" data-tab="evaluation" onclick="switchTab('evaluation')">01 Evaluation</button>
<button class="tab-btn px-5 py-2.5 rounded-lg text-sm font-medium border transition-all" data-tab="toolusage" onclick="switchTab('toolusage')">02 Tool Usage</button>
</div>''')

# EVALUATION TAB
if eval_stats:
    es = eval_stats
    eval_metrics = [
        (str(es['total']), 'Total Sessions', 'text-brand-600'),
        (f"{es['accuracy']}%", 'Overall Accuracy', 'text-emerald-600'),
        (f"{es['avg_solve_time']}s", 'Avg Solve Time', 'text-amber-500'),
        (str(es['has_answer']), 'Sessions with Answer', 'text-brand-600'),
        (str(es['correct']), 'Correct Answers', 'text-emerald-600'),
        (str(es['tools_used']), 'Tools Used', 'text-purple-600'),
    ]
    w('<div id="panel-evaluation" class="tab-panel active">')
    w('<section class="mb-10"><div class="bg-gray-50 border-l-4 border-emerald-500 rounded-r-lg p-6">')
    w('<h2 class="text-sm font-semibold text-emerald-700 uppercase tracking-wider mb-4">Evaluation Metrics</h2>')
    w('<div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">')
    for v, l, c in eval_metrics:
        w(f'<div class="text-center py-3"><div class="text-3xl font-bold {c}">{v}</div><div class="text-sm text-gray-500 mt-1">{l}</div></div>')
    w('</div></div></section>')
    # Build pre-generated option tags from Python data
    dom_opts = ''.join(f'<option value="{esc(d)}">{esc(d)}</option>' for d in es['domains'])
    sc_opts = ''.join(f'<option value="{esc(s)}">{esc(s)}</option>' for s in es['sub_categories'])
    tt_opts = ''.join(f'<option value="{esc(t)}">{esc(t)}</option>' for t in es['task_types'])
    sel_cls = 'px-3 py-1.5 rounded-md border border-gray-200 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-brand-200'
    w(f'''<section class="mb-10">
<h2 class="text-2xl font-bold text-gray-900 mb-5 flex items-center gap-2"><span class="w-1.5 h-6 bg-emerald-500 rounded-full"></span>Session Browser</h2>
<div class="bg-white rounded-xl border border-gray-200 p-6">
<div class="space-y-3 mb-4">
<div class="flex flex-wrap gap-3 items-center">
<label class="text-sm font-medium text-gray-500 uppercase">Duration:</label>
<select id="evalDurFilter" onchange="evalFilter()" class="{sel_cls}"><option value="all">All Durations</option><option value="short">Short</option><option value="medium">Medium</option><option value="long">Long</option></select>
<label class="text-sm font-medium text-gray-500 uppercase">Domain:</label>
<select id="evalDomFilter" onchange="evalFilter()" class="{sel_cls}"><option value="all">All Domains</option>{dom_opts}</select>
<label class="text-sm font-medium text-gray-500 uppercase">Sub-category:</label>
<select id="evalScFilter" onchange="evalFilter()" class="{sel_cls}"><option value="all">All Sub-categories</option>{sc_opts}</select>
<label class="text-sm font-medium text-gray-500 uppercase">Task Type:</label>
<select id="evalTtFilter" onchange="evalFilter()" class="{sel_cls}"><option value="all">All Task Types</option>{tt_opts}</select>
<label class="text-sm font-medium text-gray-500 uppercase">Result:</label>
<select id="evalResFilter" onchange="evalFilter()" class="{sel_cls}"><option value="all">All Results</option><option value="correct">Correct Only</option><option value="incorrect">Incorrect Only</option></select>
</div>
<div class="flex gap-2">
<input type="text" id="evalSearchInput" placeholder="Search by question or question_id..." onkeyup="if(event.key==='Enter')evalFilter()" class="flex-1 min-w-[250px] px-4 py-2 rounded-lg border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-200 focus:border-brand-400">
<button onclick="evalFilter()" class="px-5 py-2 rounded-lg text-sm font-medium text-white transition-all" style="background:linear-gradient(135deg,#4c6ef5,#7c3aed);">Search</button>
</div>
</div>
<div id="evalStats" class="bg-brand-50/50 text-sm text-gray-600 px-4 py-2.5 rounded-lg mb-4"></div>
<div id="evalScopeBadge" class="hidden border border-purple-100 bg-purple-50/60 text-sm text-purple-700 px-4 py-2.5 rounded-lg mb-4 flex items-center justify-between gap-3">
  <div class="min-w-0">
    <span class="font-semibold">Scoped by Tool Usage:</span>
    <span id="evalScopeLabel" class="font-mono break-all"></span>
    <span id="evalScopeMeta" class="text-xs text-purple-600 ml-2"></span>
  </div>
  <button onclick="evalClearScope()" class="px-3 py-1.5 rounded border border-purple-200 bg-white text-xs font-semibold hover:bg-purple-50">Clear</button>
</div>
<div id="evalList" class="space-y-4 max-h-[800px] overflow-y-auto"></div>
<div id="evalPagination" class="flex flex-wrap items-center justify-center gap-2 mt-5"></div>
</div></section>''')
    w('<script>var evalSessions=' + es['sessions_json'] + ';')
    w('''var evalFiltered=evalSessions.slice(),evalPage=1,evalPerPage=5,evalScope=null;
evalFilter();
function evalEsc(s){if(!s)return'';var d=document.createElement('div');d.textContent=s;return d.innerHTML.replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
function evalJsString(s){return evalEsc(JSON.stringify(s==null?'':String(s)));}
function evalNormPath(p){return (p||'').toString().replace(/\\\\/g,'/');}
function evalSafeUrl(u){try{if(u&&u.indexOf('/api/')>=0)return u;return encodeURI(u);}catch(e){return u;}}
function evalLooksLikeImage(p){return /\\.(png|jpe?g|webp|bmp|gif)$/i.test(p||'');}
function evalApiBase(){
  var override=new URLSearchParams(window.location.search||'').get('api_base')||'';
  if(!override)return '';
  try{
    var target=new URL(override,window.location.href);
    if(target.protocol!=='http:'&&target.protocol!=='https:')return '';
    if(window.location.origin==='null')return '';
    if(target.origin!==window.location.origin)return '';
    return (target.origin+target.pathname).replace(/\\/+$/,'');
  }catch(e){return '';}
}
function evalPathToUrl(p,base){
  p=evalNormPath(p);
  if(!p) return '';
  if(/^https?:\\/\\//i.test(p)) return p;
  // Resolve paths on the serving machine, never against the exporting machine.
  return evalApiBase()+'/api/file?path='+encodeURIComponent(p)+(base?'&base='+encodeURIComponent(base):'');
}
function evalCollectStrings(obj,out,depth){
  if(depth>5||obj==null) return;
  var t=typeof obj;
  if(t==='string'){out.push(obj);return;}
  if(t!=='object') return;
  if(Array.isArray(obj)){
    for(var i=0;i<obj.length;i++) evalCollectStrings(obj[i],out,depth+1);
    return;
  }
  for(var k in obj){
    if(!Object.prototype.hasOwnProperty.call(obj,k)) continue;
    evalCollectStrings(obj[k],out,depth+1);
  }
}
function evalExtractImageUrls(item,base){
  var strs=[];
  evalCollectStrings((item&&item.input)||{},strs,0);
  var outRaw=(item&&item.output)||'';
  if(typeof outRaw==='string' && outRaw){
    var parsed=null;
    try{parsed=JSON.parse(outRaw);}catch(e){}
    if(parsed) evalCollectStrings(parsed,strs,0);
    else{
      var re=/([A-Za-z0-9_\\-\\/\\.]+\\.(?:png|jpe?g|webp|bmp|gif))/ig, m=null, cap=0;
      while((m=re.exec(outRaw))!==null){strs.push(m[1]);cap++;if(cap>50)break;}
    }
  } else {
    evalCollectStrings(outRaw,strs,0);
  }
  var urls=[],seen={};
  for(var i=0;i<strs.length;i++){
    var s=evalNormPath(strs[i]);
    if(!evalLooksLikeImage(s)) continue;
    var u=evalPathToUrl(s,base);
    if(!u) continue;
    u=evalSafeUrl(u);
    if(seen[u]) continue;
    seen[u]=1;
    urls.push(u);
    if(urls.length>=8) break;
  }
  return urls;
}
function evalRenderImageStrip(urls){
  if(!urls||!urls.length) return '';
  var imgs=urls.map(function(u,i){
    var eu=evalEsc(u).replace(/"/g,'&quot;');
    return '<button type="button" class="eval-image-preview" data-eval-image="'+eu+'" aria-haspopup="dialog" aria-label="Open tool image '+(i+1)+'">'+
      '<img loading="lazy" src="'+eu+'" class="eval-image-backdrop" alt="" aria-hidden="true">'+
      '<span class="eval-image-frost" aria-hidden="true"></span>'+
      '<img loading="lazy" src="'+eu+'" class="eval-image-thumbnail" alt="Tool image '+(i+1)+'">'+
      '<span class="eval-image-unavailable">Image unavailable</span></button>';
  }).join('');
  return '<div class="eval-image-strip">'+imgs+'</div>';
}
function evalExtractQuestion(query){if(!query)return'No question';var m=query.match(/Question:\\s*([\\s\\S]+?)(?:\\nOptions:|$)/);return m?m[1].trim().substring(0,300):query.substring(0,300);}
function evalUpdateScopeBadge(){
var badge=document.getElementById('evalScopeBadge');
if(!badge)return;
if(!evalScope||!evalScope.toolName){badge.classList.add('hidden');return;}
var label=document.getElementById('evalScopeLabel');
var meta=document.getElementById('evalScopeMeta');
if(label)label.textContent=evalScope.toolName;
if(meta){
var m=[];
if(evalScope.sourceType==='duration'&&evalScope.sourceValue)m.push('Duration: '+evalScope.sourceValue);
if(evalScope.sourceType==='task_type'&&evalScope.sourceValue)m.push('Task: '+evalScope.sourceValue);
m.push((evalScope.count||0)+' sessions');
meta.textContent='('+m.join(' | ')+')';
}
badge.classList.remove('hidden');
}
function evalSetScope(toolName,qids,sourceType,sourceValue){
var map={};
if(Array.isArray(qids)){for(var i=0;i<qids.length;i++){var q=qids[i];if(q)map[q]=1;}}
evalScope={toolName:toolName||'',qidsMap:map,count:Array.isArray(qids)?qids.length:0,sourceType:sourceType||'all',sourceValue:sourceValue||''};
evalUpdateScopeBadge();
}
function evalClearScope(skipFilter){
evalScope=null;
evalUpdateScopeBadge();
if(!skipFilter)evalFilter();
}
function evalFilterBy(type, val, ev){
if(ev)ev.stopPropagation();
if(type==='duration'){var sel=document.getElementById('evalDurFilter');if(sel)sel.value=val;}
else if(type==='domain'){var sel=document.getElementById('evalDomFilter');if(sel)sel.value=val;}
else if(type==='sub_category'){var sel=document.getElementById('evalScFilter');if(sel)sel.value=val;}
else if(type==='task_type'){var sel=document.getElementById('evalTtFilter');if(sel)sel.value=val;}
evalFilter();
}
function evalFilter(){
var dur=document.getElementById('evalDurFilter').value;
var dom=document.getElementById('evalDomFilter').value;
var sc=document.getElementById('evalScFilter').value;
var tt=document.getElementById('evalTtFilter').value;
var res=document.getElementById('evalResFilter').value;
var q=document.getElementById('evalSearchInput').value.toLowerCase().trim();
evalFiltered=evalSessions.filter(function(s){
if(dur!=='all'&&s.duration!==dur)return false;
if(dom!=='all'&&s.domain!==dom)return false;
if(sc!=='all'&&s.sub_category!==sc)return false;
if(tt!=='all'&&s.task_type!==tt)return false;
if(res==='correct'&&!s.is_correct)return false;
if(res==='incorrect'&&s.is_correct)return false;
if(evalScope&&evalScope.qidsMap&&!evalScope.qidsMap[s.question_id])return false;
if(q&&evalExtractQuestion(s.init_user_query).toLowerCase().indexOf(q)<0&&s.question_id.toLowerCase().indexOf(q)<0)return false;
return true;
});
evalPage=1;evalRenderStats();evalRenderList();evalRenderPag();
}
function evalRenderStats(){
var c=evalFiltered.filter(function(s){return s.is_correct;}).length;
var t=evalFiltered.length;
var acc=t>0?(c/t*100).toFixed(1):0;
var missing=evalFiltered.filter(function(s){return s.transcript_exists===false;}).length;
var missHtml=missing?` | Missing transcripts: <strong style="color:#f59e0b">${missing}</strong>`:'';
document.getElementById('evalStats').innerHTML=`Showing <strong>${t}</strong> sessions | \\u2713 Correct: <strong style="color:#059669">${c}</strong> | \\u2717 Incorrect: <strong style="color:#dc2626">${t-c}</strong> | Accuracy: <strong>${acc}%</strong>${missHtml}`;
}
function evalRenderList(){
var start=(evalPage-1)*evalPerPage;
var page=evalFiltered.slice(start,start+evalPerPage);
var el=document.getElementById('evalList');
if(!page.length){el.innerHTML='<p class="text-center text-gray-400 py-10">No sessions found.</p>';return;}
el.innerHTML=page.map(function(s,i){
var idx=start+i+1;
var bcls=s.is_correct?'border-l-emerald-500':'border-l-red-400';
var badgeClass=s.is_correct?'bg-emerald-100 text-emerald-700':'bg-red-100 text-red-600';
var tExists=(s.transcript_exists!==false);
var tBadge=tExists?'':'<div class="mt-2"><span class="inline-block px-2 py-0.5 rounded-md text-xs font-semibold bg-amber-100 text-amber-700">Transcript missing</span></div>';
var opts=(s.options||[]).map(function(o,oi){
var letter=String.fromCharCode(65+oi);
var isGt=letter===s.gt_answer,isMa=letter===s.model_answer;
var cls='bg-gray-50 text-gray-600 border border-transparent';
var suffix='';
if(isGt){cls='bg-emerald-50 border-l-2 border-emerald-500 text-emerald-800';suffix=' (GT)';}
else if(isMa){cls=s.is_correct?'bg-blue-50 border-l-2 border-blue-400 text-blue-700':'bg-red-50 border-l-2 border-red-400 text-red-600';suffix=' \\u2190 Model';}
return `<div class="px-3 py-2 rounded text-sm ${cls}">${evalEsc(o)}${suffix}</div>`;
}).join('');
var ytSection=s.videoID?`<div class="px-5 pt-4" id="yt-section-${evalEsc(s.session_id)}"><button class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-50 text-red-500 text-xs font-semibold hover:bg-red-100 transition-colors" onclick="evalToggleVideo(${evalJsString(s.videoID)},${evalJsString(s.session_id)},event)">&#9654; YouTube</button><div class="yt-inline-player mt-3 rounded-lg overflow-hidden bg-black" style="min-height:300px;position:relative;" id="eval-yt-${evalEsc(s.session_id)}"></div></div>`:'';
var qEsc=evalEsc(evalExtractQuestion(s.init_user_query));
var qText=qEsc.length>=200?qEsc.substring(0,200)+'...':qEsc;
return `<div class="bg-white rounded-xl border border-gray-200 overflow-hidden border-l-4 ${bcls}" data-sid="${evalEsc(s.session_id)}" data-tp="${evalEsc(s.transcript_path)}" data-exists="${tExists}" data-tsize="${evalEsc(String(s.transcript_size||0))}">
<div class="px-5 py-4 cursor-pointer hover:bg-gray-50/50 transition-colors flex justify-between items-start gap-4" onclick="evalToggleSession(this)">
<div class="flex-1 min-w-0">
<p class="text-base text-gray-800 leading-snug mb-2">#${idx}. ${qText}</p>
<div class="flex flex-wrap gap-x-4 gap-y-1 text-sm text-gray-500">
<span>Duration: <strong class="text-gray-600 cursor-pointer hover:text-brand-600 hover:underline" onclick="evalFilterBy('duration',${evalJsString(s.duration)},event)">${evalEsc(s.duration||'N/A')}</strong></span>
<span>Domain: <strong class="text-gray-600 cursor-pointer hover:text-brand-600 hover:underline" onclick="evalFilterBy('domain',${evalJsString(s.domain)},event)">${evalEsc(s.domain||'N/A')}</strong></span>
<span>Sub-category: <strong class="text-gray-600 cursor-pointer hover:text-brand-600 hover:underline" onclick="evalFilterBy('sub_category',${evalJsString(s.sub_category)},event)">${evalEsc(s.sub_category||'N/A')}</strong></span>
<span>Task: <strong class="text-gray-600 cursor-pointer hover:text-brand-600 hover:underline" onclick="evalFilterBy('task_type',${evalJsString(s.task_type)},event)">${evalEsc(s.task_type||'N/A')}</strong></span>
<span>ID: <strong class="text-gray-600">${evalEsc(s.question_id||'N/A')}</strong></span>
</div>
</div>
<div class="text-center flex-shrink-0">
<span class="inline-block px-3 py-1 rounded-md text-sm font-semibold ${badgeClass}">${s.is_correct?'Correct':'Wrong'}</span>
${tBadge}
<div class="text-xs text-gray-500 mt-1 font-mono">GT:${evalEsc(s.gt_answer)} | M:${evalEsc(s.model_answer||'?')}</div>
</div>
</div>
<div class="session-details border-t border-gray-100">
${ytSection}
<div class="px-5 py-4">
<p class="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-2">Options</p>
<div class="space-y-1.5">${opts}</div>
</div>
<div class="interaction-viewer px-5 pb-4" id="eval-interaction-${evalEsc(s.session_id)}"></div>
<div class="mx-5 mb-4 px-4 py-2.5 rounded-lg text-sm flex flex-wrap items-center gap-4" style="background:linear-gradient(135deg,rgba(76,110,245,0.06),rgba(124,58,237,0.06));border:1px solid rgba(76,110,245,0.1);" id="eval-solve-time-${evalEsc(s.session_id)}">
<span class="font-semibold text-gray-700">Solve Time</span><span class="text-gray-400">Expand to load...</span>
</div>
</div></div>`;
}).join('');
}
function evalRenderPag(){
var total=Math.ceil(evalFiltered.length/evalPerPage);
var el=document.getElementById('evalPagination');
if(total<=1){el.innerHTML='';return;}
var h='';
var dis=evalPage===1?' disabled style="opacity:.4;cursor:not-allowed"':'';
h+=`<button class="px-3 py-1.5 rounded border border-gray-200 text-sm hover:bg-gray-50" onclick="evalGo(${evalPage-1})"${dis}>\\u2190 Prev</button>`;
var st=Math.max(1,evalPage-2),en=Math.min(total,st+4);
if(en-st<4)st=Math.max(1,en-4);
if(st>1)h+=`<button class="px-3 py-1.5 rounded border border-gray-200 text-sm hover:bg-gray-50" onclick="evalGo(1)">1</button>`;
if(st>2)h+='<span class="text-gray-400">...</span>';
for(var i=st;i<=en;i++){
var ac=i===evalPage?'bg-brand-600 text-white border-brand-600':'border-gray-200 hover:bg-gray-50';
h+=`<button class="px-3 py-1.5 rounded border text-sm ${ac}" onclick="evalGo(${i})">${i}</button>`;
}
if(en<total-1)h+='<span class="text-gray-400">...</span>';
if(en<total)h+=`<button class="px-3 py-1.5 rounded border border-gray-200 text-sm hover:bg-gray-50" onclick="evalGo(${total})">${total}</button>`;
var dis2=evalPage===total?' disabled style="opacity:.4;cursor:not-allowed"':'';
h+=`<button class="px-3 py-1.5 rounded border border-gray-200 text-sm hover:bg-gray-50" onclick="evalGo(${evalPage+1})"${dis2}>Next \\u2192</button>`;
h+=`<span class="text-sm text-gray-500">Page ${evalPage} of ${total}</span>`;
el.innerHTML=h;
}
function evalGo(p){
var total=Math.ceil(evalFiltered.length/evalPerPage);
if(p<1||p>total)return;
evalPage=p;evalRenderList();evalRenderPag();
document.getElementById('evalList').scrollIntoView({behavior:'smooth',block:'start'});
}
function evalResetFilters(){
document.getElementById('evalDurFilter').value='all';
document.getElementById('evalDomFilter').value='all';
document.getElementById('evalScFilter').value='all';
document.getElementById('evalTtFilter').value='all';
document.getElementById('evalResFilter').value='all';
document.getElementById('evalSearchInput').value='';
}
function evalScrollToSession(sid){
var idx=-1;
for(var i=0;i<evalFiltered.length;i++){if(evalFiltered[i].session_id===sid){idx=i;break;}}
if(idx<0)return;
var page=Math.floor(idx/evalPerPage)+1;
evalPage=page;evalRenderList();evalRenderPag();
setTimeout(function(){
var card=document.querySelector('[data-sid="'+sid+'"]');
if(!card)return;
card.scrollIntoView({behavior:'smooth',block:'center'});
var header=card.querySelector('[onclick="evalToggleSession(this)"]') || card;
var details=header.nextElementSibling;
if(details && !details.classList.contains('expanded')){evalToggleSession(header);}
},100);
}
function evalJumpToSessionWithScope(sid,toolName,qids,sourceType,sourceValue){
var tuModal=document.getElementById('tuModal');
if(tuModal)tuModal.style.display='none';
switchTab('evaluation');
evalResetFilters();
if(sourceType==='duration'&&sourceValue){document.getElementById('evalDurFilter').value=sourceValue;}
if(sourceType==='task_type'&&sourceValue){document.getElementById('evalTtFilter').value=sourceValue;}
if(toolName && Array.isArray(qids))evalSetScope(toolName,qids,sourceType,sourceValue);
else evalClearScope(true);
evalFilter();
evalScrollToSession(sid);
}
function evalJumpToSession(sid){
evalClearScope(true);
evalJumpToSessionWithScope(sid,null,null,'all','');
}
function evalToggleSession(header){
var details=header.nextElementSibling;
details.classList.toggle('expanded');
if(details.classList.contains('expanded')){
var card=header.closest('[data-sid]');
if(!card)return;
var sid=card.dataset.sid;
var tp=card.dataset.tp;
var exists=card.dataset.exists!=='false';
var ct=document.getElementById('eval-interaction-'+sid);
if(ct&&!ct.dataset.loaded){ct.dataset.loaded='true';evalLoadInteractions(sid,tp,exists);}
}}
function evalToggleVideo(videoID,sid,ev){
if(ev)ev.stopPropagation();
var p=document.getElementById('eval-yt-'+sid);if(!p)return;
if(p.innerHTML.trim()){p.innerHTML='';return;}
var v=document.createElement('video');v.controls=true;v.preload='metadata';
// The API applies the configured data root and old-path mappings to local media.
v.src=evalPathToUrl('data/videomme/videos/'+videoID+'.mp4');
v.onerror=function(){
var frame=document.createElement('iframe');
frame.style.cssText='width:100%;height:100%;border:none;';
frame.src='https://www.youtube.com/embed/'+encodeURIComponent(String(videoID))+'?autoplay=1';
frame.allowFullscreen=true;
frame.allow='accelerometer;autoplay;clipboard-write;encrypted-media;gyroscope;picture-in-picture';
p.innerHTML='';p.appendChild(frame);
};
p.innerHTML='';p.appendChild(v);
}
function evalTruncateText(t,m){if(!t||t.length<=m)return t;return t.substring(0,m)+'... [truncated]';}
function evalParseSessionLog(content){
var lines=content.trim().split('\\n'),entries=[];
for(var i=0;i<lines.length;i++){if(!lines[i].trim())continue;try{entries.push(JSON.parse(lines[i]));}catch(e){}}
return entries;
}
function evalExtractConversationData(entries){
// Support Claude-style logs (assistant.content as array), Codex logs (response_item),
// and agentic-style logs (assistant.content as string + separate tool entries with message.name/input/output).
var isCodex=false;
for(var x=0;x<entries.length;x++){var e0=entries[x];if(e0&&e0.type==='response_item'&&e0.payload&&e0.payload.type){isCodex=true;break;}}
if(isCodex)return evalExtractConversationDataCodex(entries);
var isAgentic=false;
for(var x2=0;x2<entries.length;x2++){if(entries[x2]&&entries[x2].type==='tool'&&entries[x2].message&&entries[x2].message.name){isAgentic=true;break;}}
if(isAgentic)return evalExtractConversationDataAgentic(entries);
var result={interactions:[]};
var seenMsgIds={},seenToolIds={},toolResultMap={},cur=null;
for(var i=0;i<entries.length;i++){
var e=entries[i],et=e.type;
if(et==='user'){
var msg=e.message||{},c=msg.content;
if(e.isMeta)continue;
if(typeof c==='string'){
if(c.indexOf('<local-command-')>=0||c.indexOf('<system-reminder>')>=0)continue;
if(cur)result.interactions.push(cur);
cur={userMessage:c,trace:[]};
}else if(Array.isArray(c)){for(var j=0;j<c.length;j++){
if(c[j].type==='tool_result')toolResultMap[c[j].tool_use_id]=typeof c[j].content==='string'?c[j].content:JSON.stringify(c[j].content);
else if(c[j].type==='text'&&c[j].text&&c[j].text.indexOf('<local-command-')<0&&c[j].text.indexOf('<system-reminder>')<0){
if(cur)result.interactions.push(cur);cur={userMessage:c[j].text,trace:[]};}
}}}
if(et==='assistant'){
var msg2=e.message||{},mid=msg2.id,parts=msg2.content||[];
var seen=mid&&seenMsgIds[mid];if(mid)seenMsgIds[mid]=true;
if(!cur)cur={userMessage:'',trace:[]};
var aText='',aThink='';
for(var k=0;k<parts.length;k++){
if(parts[k].type==='thinking')aThink=parts[k].thinking||'';
else if(parts[k].type==='text')aText=parts[k].text||'';
else if(parts[k].type==='tool_use'){var tid=parts[k].id;if(tid&&!seenToolIds[tid]){seenToolIds[tid]=true;
cur.trace.push({type:'tool_use',name:parts[k].name||'unknown',input:parts[k].input||{},output:'',toolUseId:tid});}}}
if(aText||aThink){if(seen){for(var m2=cur.trace.length-1;m2>=0;m2--){if(cur.trace[m2].type==='assistant'){cur.trace[m2].text=aText||cur.trace[m2].text;cur.trace[m2].thinking=aThink||cur.trace[m2].thinking;break;}}}
else cur.trace.push({type:'assistant',text:aText,thinking:aThink});}
}}
if(cur)result.interactions.push(cur);
for(var n2=0;n2<result.interactions.length;n2++){var tr=result.interactions[n2].trace;for(var p2=0;p2<tr.length;p2++){if(tr[p2].type==='tool_use'&&tr[p2].toolUseId&&toolResultMap[tr[p2].toolUseId])tr[p2].output=toolResultMap[tr[p2].toolUseId];}}
return result;
}
function evalExtractConversationDataAgentic(entries){
var result={interactions:[]},cur=null;
for(var i=0;i<entries.length;i++){
var e=entries[i],et=e.type;
if(et==='user'){
var msg=e.message||{},c=msg.content;
if(e.isMeta)continue;
var txt='';
if(typeof c==='string')txt=c;
else if(Array.isArray(c)){for(var j=0;j<c.length;j++){if(c[j].type==='text'&&c[j].text){txt=c[j].text;break;}}}
if(!txt)continue;
if(txt.indexOf('<local-command-')>=0||txt.indexOf('<system-reminder>')>=0)continue;
if(cur)result.interactions.push(cur);
cur={userMessage:txt,trace:[]};
}
if(et==='assistant'){
var msg2=e.message||{},c2=msg2.content,thk=msg2.thinking||'';
if(!cur)cur={userMessage:'',trace:[]};
var aText='',aThink='';
if(typeof c2==='string'){aText=c2;}
else if(Array.isArray(c2)){for(var k=0;k<c2.length;k++){
if(c2[k].type==='thinking')aThink=c2[k].thinking||'';
else if(c2[k].type==='text')aText=c2[k].text||'';
else if(c2[k].type==='tool_use'){cur.trace.push({type:'tool_use',name:c2[k].name||'unknown',input:c2[k].input||{},output:'',toolUseId:c2[k].id||''});}
}}
if(thk)aThink=thk;
if(aText||aThink)cur.trace.push({type:'assistant',text:aText,thinking:aThink});
}
if(et==='tool'){
var msg3=e.message||{},toolName=msg3.name||'unknown',toolInput=msg3.input||{},toolOutput=msg3.output||'';
if(!cur)cur={userMessage:'',trace:[]};
var outStr='';
if(typeof toolOutput==='string'&&toolOutput){
try{var parsed=JSON.parse(toolOutput);if(Array.isArray(parsed)){var texts=[];for(var pi=0;pi<parsed.length;pi++){if(parsed[pi].type==='text'&&parsed[pi].text)texts.push(parsed[pi].text);}outStr=texts.join('\\n');if(!outStr)outStr=toolOutput;}else{outStr=toolOutput;}}catch(ex){outStr=toolOutput;}
}else if(toolOutput){outStr=typeof toolOutput==='string'?toolOutput:JSON.stringify(toolOutput);}
cur.trace.push({type:'tool_use',name:toolName,input:typeof toolInput==='object'?toolInput:{raw:JSON.stringify(toolInput)},output:outStr,toolUseId:''});
}}
if(cur)result.interactions.push(cur);
return result;
}
function evalExtractConversationDataCodex(entries){
var result={interactions:[]};
var toolResultMap={},cur=null;
function safeJsonParse(s){try{return JSON.parse(s);}catch(e){return null;}}
function joinContent(content,kind){
if(!Array.isArray(content))return '';
var out=[];
for(var i=0;i<content.length;i++){var c=content[i];if(!c||!c.type)continue;
if(kind==='input'&&c.type==='input_text'&&c.text)out.push(c.text);
else if(kind==='output'&&c.type==='output_text'&&c.text)out.push(c.text);
}
return out.join('');
}
for(var i=0;i<entries.length;i++){
var e=entries[i];if(!e||e.type!=='response_item')continue;
var pl=e.payload||{},pt=pl.type;
if(pt==='message'){
if(pl.role==='user'){
var txt=joinContent(pl.content,'input');
if(!txt)continue;
if(cur)result.interactions.push(cur);
cur={userMessage:txt,trace:[]};
}else if(pl.role==='assistant'){
if(!cur)cur={userMessage:'',trace:[]};
var aText=joinContent(pl.content,'output');
if(aText)cur.trace.push({type:'assistant',text:aText,thinking:''});
}
}else if(pt==='function_call'){
if(!cur)cur={userMessage:'',trace:[]};
var args=pl.arguments||'';var inp={};
if(typeof args==='string'&&args){var parsed=safeJsonParse(args);inp=parsed!==null?parsed:{arguments:args};}
else if(args)inp=args;
cur.trace.push({type:'tool_use',name:pl.name||'unknown',input:inp||{},output:'',toolUseId:pl.call_id||''});
}else if(pt==='function_call_output'){
var cid=pl.call_id||'';
if(cid)toolResultMap[cid]=typeof pl.output==='string'?pl.output:JSON.stringify(pl.output);
}else if(pt==='web_search_call'){
if(!cur)cur={userMessage:'',trace:[]};
cur.trace.push({type:'tool_use',name:'web_search',input:pl.action||{},output:JSON.stringify({status:pl.status||'',action:pl.action||{}}),toolUseId:pl.call_id||('web_'+i)});
}}
if(cur)result.interactions.push(cur);
for(var n2=0;n2<result.interactions.length;n2++){var tr=result.interactions[n2].trace;for(var p2=0;p2<tr.length;p2++){if(tr[p2].type==='tool_use'&&tr[p2].toolUseId&&toolResultMap[tr[p2].toolUseId])tr[p2].output=toolResultMap[tr[p2].toolUseId];}}
return result;
}
function evalUpdateSolveTime(sid,entries){
var ct=document.getElementById('eval-solve-time-'+sid);if(!ct)return;
var tss=[];for(var i=0;i<entries.length;i++){var t=entries[i].timestamp;if(t){var d=new Date(t);if(!isNaN(d.getTime()))tss.push(d);}}
if(tss.length<2){ct.innerHTML='<span class="font-semibold text-gray-700">Solve Time</span><span class="text-gray-400">Insufficient data</span>';return;}
var mn=new Date(Math.min.apply(null,tss.map(function(d){return d.getTime();}))),mx=new Date(Math.max.apply(null,tss.map(function(d){return d.getTime();})));
var dur=((mx-mn)/1000).toFixed(1);
function fmt(d){return d.getFullYear()+'/'+String(d.getMonth()+1).padStart(2,'0')+'/'+String(d.getDate()).padStart(2,'0')+' '+String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0')+':'+String(d.getSeconds()).padStart(2,'0');}
ct.innerHTML='<span class="font-semibold text-gray-700">Solve Time</span><span class="text-gray-500">Start: <span class="font-mono text-gray-700">'+fmt(mn)+'</span></span><span class="text-gray-500">End: <span class="font-mono text-gray-700">'+fmt(mx)+'</span></span><span class="text-gray-500">Duration: <span class="font-mono font-bold text-amber-500">'+dur+'s</span></span>';
}
async function evalLoadInteractions(sid,tp,exists){
var ct=document.getElementById('eval-interaction-'+sid);
var card=document.querySelector('[data-sid=\"'+sid.replace(/\"/g,'')+'\"]');
if(!tp){ct.innerHTML='<p class="text-gray-400 text-sm text-center py-3">No transcript path.</p>';return;}
ct.innerHTML='<div class="text-center text-gray-400 text-sm py-5">Loading interactions...</div>';
async function fetchWithTimeout(url, ms){
  var controller=null, timer=null;
  if(typeof AbortController !== 'undefined'){
    controller=new AbortController();
    timer=setTimeout(function(){try{controller.abort();}catch(_e){}}, ms);
  }
  try{
    return await fetch(url, controller?{signal:controller.signal}:undefined);
  } finally {
    if(timer) clearTimeout(timer);
  }
}
function fmtBytes(n){
  n = Number(n||0);
  if(!isFinite(n) || n<=0) return '0 B';
  var u=['B','KB','MB','GB'],i=0;
  while(n>=1024 && i<u.length-1){n/=1024;i++;}
  return (i===0?String(Math.round(n)):n.toFixed(1))+' '+u[i];
}
if(!window.__evalTranscriptCache) window.__evalTranscriptCache = {};
async function fetchTranscript(){
  // Path relocation is handled by the server's BenchmarkPaths resolver.
  var base=evalApiBase();
  var url=base+'/api/read?path='+encodeURIComponent(tp);
  var cacheKey=base+'|'+tp;
  if(window.__evalTranscriptCache[cacheKey]){
    return window.__evalTranscriptCache[cacheKey];
  }
  var r=null;
  try{
    // Timeout only for connection/headers. Large transcripts may legitimately take a while to download.
    var declaredSize=0;
    try{declaredSize=card?Number(card.dataset.tsize||0):0;}catch(_e){}
    var sizeHint=declaredSize?(' | size '+fmtBytes(declaredSize)):'';
    ct.innerHTML='<div class="text-center text-gray-400 text-sm py-5">Loading interactions...<div class="text-xs text-gray-500 mt-2 font-mono break-all">Trying '+evalEsc(base)+sizeHint+'</div></div>';
    r=await fetchWithTimeout(url, 8000);
  }catch(netErr){
    // Normalize AbortError into a clearer message (some browsers provide empty reasons).
    var msg = (netErr && netErr.message) ? netErr.message : String(netErr||'');
    if((netErr && netErr.name==='AbortError') || (msg && msg.toLowerCase().indexOf('aborted')>=0)){
      throw new Error('Connect timeout after 8s: '+url+' (open the dashboard through its running server)');
    }
    throw netErr;
  }
  if(!r.ok){
    var err='';
    try{err=await r.text();}catch(_e){}
    throw new Error('HTTP '+r.status+(err?': '+err.replace(/<[^>]*>/g,'').trim().slice(0,300):''));
  }
  // Stream download to show progress on large files.
  try{
    if(r.body && r.body.getReader && typeof TextDecoder !== 'undefined'){
      var reader=r.body.getReader();
      var decoder=new TextDecoder();
      var out=[];
      var received=0;
      while(true){
        var chunk=await reader.read();
        if(chunk.done) break;
        if(chunk.value){
          received += chunk.value.byteLength || 0;
          out.push(decoder.decode(chunk.value, {stream:true}));
          ct.innerHTML='<div class="text-center text-gray-400 text-sm py-5">Downloading transcript...<div class="text-xs text-gray-500 mt-2 font-mono break-all">From '+evalEsc(base)+' | '+fmtBytes(received)+'</div></div>';
        }
      }
      out.push(decoder.decode());
      var text=out.join('');
      window.__evalTranscriptCache[cacheKey]=text;
      return text;
    }
  }catch(streamErr){
    // Fall back to non-streaming text() below.
  }
  var text2=await r.text();
  window.__evalTranscriptCache[cacheKey]=text2;
  return text2;
}
try{
var content=await fetchTranscript();
ct.innerHTML='<div class="text-center text-gray-400 text-sm py-5">Parsing...</div>';
setTimeout(function(){try{var entries=evalParseSessionLog(content);evalUpdateSolveTime(sid,entries);
var data=evalExtractConversationData(entries);
if(!data.interactions.length){ct.innerHTML='<p class="text-center text-gray-400 text-sm py-4">No interactions found.</p>';return;}
ct.innerHTML=evalRenderInteractions(data.interactions,tp);
}catch(e2){ct.innerHTML='<p class="text-red-500 text-sm text-center py-4">Parse error: '+evalEsc(e2.message)+'</p>';}},50);
}catch(e3){
if(exists===false){
ct.innerHTML='<div class="text-center py-5"><p class="text-amber-600 text-sm font-semibold">Transcript file not found</p><p class="text-gray-500 text-xs mt-2 font-mono break-all">'+evalEsc(tp)+'</p></div>';
return;
}
ct.innerHTML='<div class="text-center py-5"><p class="text-red-500 text-sm">Failed: '+evalEsc(e3.message)+'</p><p class="text-gray-400 text-xs mt-2">Start server: <code class="bg-gray-100 px-1.5 py-0.5 rounded">sh html_show/start_session_server_tool.sh</code></p></div>';}}
function evalRenderInteractions(interactions,transcriptPath){
return interactions.map(function(ix,idx){var tr=ix.trace||[];
var tc=tr.filter(function(t){return t.type==='tool_use';}).length;
var ac=tr.filter(function(t){return t.type==='assistant';}).length;
return '<div class="border border-gray-200 rounded-lg overflow-hidden mb-3"><div class="px-4 py-2.5 flex items-center justify-between cursor-pointer" style="background:linear-gradient(135deg,rgba(76,110,245,0.08),rgba(124,58,237,0.08))" onclick="this.nextElementSibling.classList.toggle(\\\'expanded\\\')"><h5 class="text-base font-semibold text-gray-800">Interaction #'+(idx+1)+'</h5><span class="text-sm text-gray-500">'+tr.length+' traces ('+tc+' tools, '+ac+' responses)</span></div><div class="interaction-content"><div class="p-4 bg-blue-50/40 border-l-4 border-blue-400"><p class="text-sm font-semibold text-blue-600 mb-1">User Message</p><p class="text-base text-gray-700 whitespace-pre-wrap leading-relaxed">'+(evalEsc(ix.userMessage)||'<em class="text-gray-400">No message</em>')+'</p></div>'+(tr.length?'<div class="p-4 bg-gray-50/50"><div class="flex items-center justify-between mb-3"><p class="text-sm font-semibold text-violet-600">Execution Trace</p><button class="px-2.5 py-1 text-xs font-medium text-gray-500 bg-white border border-gray-200 rounded hover:bg-gray-50" onclick="evalToggleAllTraces(this,event)">Expand All</button></div><div class="trace-timeline">'+evalRenderTraceItems(tr,transcriptPath)+'</div></div>':'')+'</div></div>';}).join('');
}
function evalRenderTraceItems(trace,transcriptPath){return trace.map(function(item,idx){if(item.type==='assistant')return evalRenderAssistantTrace(item,idx);if(item.type==='tool_use')return evalRenderToolTrace(item,idx,transcriptPath);return '';}).join('');}
function evalRenderAssistantTrace(item,idx){
var ht=item.thinking&&item.thinking.trim(),pv=(item.text||'').substring(0,60);
return '<div class="trace-item-viewer assistant-trace bg-white border border-gray-200 rounded-lg" data-index="'+(idx+1)+'"><div class="px-3 py-2 flex items-center gap-2 cursor-pointer hover:bg-gray-50" onclick="evalToggleTraceDetails(this)"><span class="text-xs font-bold px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700">Assistant</span><span class="flex-1 text-sm text-gray-500 truncate">'+evalEsc(pv)+(pv.length>=60?'...':'')+'</span></div><div class="trace-item-details border-t border-gray-100 p-3">'+(ht?'<div class="mb-3 p-3 bg-violet-50 rounded-lg border-l-4 border-violet-400"><p class="text-sm font-semibold text-violet-500 mb-1">Thinking</p><p class="text-sm text-gray-600 whitespace-pre-wrap leading-relaxed" style="max-height:300px;overflow-y:auto">'+evalEsc(evalTruncateText(item.thinking,5000))+'</p></div>':'')+(item.text&&item.text.trim()?'<div><p class="text-sm font-semibold text-gray-500 mb-1">Response</p><p class="text-base text-gray-700 whitespace-pre-wrap leading-relaxed">'+evalEsc(item.text)+'</p></div>':'<em class="text-gray-400 text-sm">No text response</em>')+'</div></div>';
}
function evalRenderToolTrace(item,idx,transcriptPath){
var dn=item.name?item.name.replace(/^mcp__/,''):'unknown';
var inp=JSON.stringify(item.input||{},null,2),out=item.output||'';
var urls=evalExtractImageUrls(item,transcriptPath);
var imgHtml=urls.length?('<div><p class="text-xs font-semibold text-gray-500 mb-1">Key Images</p>'+evalRenderImageStrip(urls)+'</div>'):'';
return '<div class="trace-item-viewer tool-trace bg-white border border-gray-200 rounded-lg" data-index="'+(idx+1)+'"><div class="px-3 py-2 flex items-center gap-2 cursor-pointer hover:bg-gray-50" onclick="evalToggleTraceDetails(this)"><span class="text-xs font-bold px-1.5 py-0.5 rounded bg-amber-100 text-amber-700">Tool</span><span class="font-mono text-sm text-blue-600 font-semibold">'+dn+'</span></div><div class="trace-item-details border-t border-gray-100 p-3 space-y-3"><div><p class="text-sm font-semibold text-gray-500 mb-1"><span class="inline-block w-1.5 h-1.5 bg-blue-400 rounded-sm mr-1"></span>Input</p>'+imgHtml+'<pre class="bg-gray-900 text-gray-300 p-3 rounded-lg text-xs overflow-x-auto whitespace-pre-wrap break-all border-l-4 border-blue-400" style="max-height:300px;overflow-y:auto">'+evalEsc(inp)+'</pre></div><div><p class="text-sm font-semibold text-gray-500 mb-1"><span class="inline-block w-1.5 h-1.5 bg-emerald-400 rounded-sm mr-1"></span>Output</p><pre class="bg-gray-900 text-gray-300 p-3 rounded-lg text-xs overflow-x-auto whitespace-pre-wrap break-all border-l-4 border-emerald-400" style="max-height:300px;overflow-y:auto">'+(out?evalEsc(evalTruncateText(out,3000)):'<span class="text-gray-500 italic">No output</span>')+'</pre></div></div></div>';
}
function evalToggleTraceDetails(h){h.nextElementSibling.classList.toggle('expanded');}
function evalToggleAllTraces(btn,ev){ev.stopPropagation();var sec=btn.closest('.p-4');var ds=sec.querySelectorAll('.trace-item-details');var allExp=Array.from(ds).every(function(d){return d.classList.contains('expanded');});ds.forEach(function(d){if(allExp)d.classList.remove('expanded');else d.classList.add('expanded');});btn.textContent=allExp?'Expand All':'Collapse All';}
</script>''')
    w('</div>')
else:
    w('<div id="panel-evaluation" class="tab-panel active"><div class="text-center py-20 text-gray-400"><p class="text-lg">No evaluation data available.</p><p class="text-sm mt-2">Place all_data_path.json in conversations/ directory.</p></div></div>')

# TOOL USAGE TAB
if tool_usage_stats:
    tus = tool_usage_stats
    tu_unique = len(tus['tools'])
    tu_total_calls = sum(t['total_calls'] for t in tus['tools'].values())
    w('<div id="panel-toolusage" class="tab-panel">')
    w(f'<section class="mb-10"><div class="bg-gray-50 border-l-4 border-purple-500 rounded-r-lg p-6"><h2 class="text-sm font-semibold text-purple-700 uppercase tracking-wider mb-4">Tool Usage Metrics</h2><div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4"><div class="text-center py-3"><div class="text-3xl font-bold text-purple-600">{tu_unique}</div><div class="text-sm text-gray-500 mt-1">Unique Tools Used</div></div><div class="text-center py-3"><div class="text-3xl font-bold text-brand-600">{tu_total_calls}</div><div class="text-sm text-gray-500 mt-1">Total Tool Calls</div></div><div class="text-center py-3"><div class="text-3xl font-bold text-emerald-600">{tus["total_sessions"]}</div><div class="text-sm text-gray-500 mt-1">Total Sessions</div></div><div class="text-center py-3"><div class="text-3xl font-bold text-amber-500">{len(tus["task_types"])}</div><div class="text-sm text-gray-500 mt-1">Task Types</div></div></div></div></section>')
    w('''<section class="mb-10">
<h2 class="text-2xl font-bold text-gray-900 mb-5 flex items-center gap-2"><span class="w-1.5 h-6 bg-purple-500 rounded-full"></span>Tool Call Frequency</h2>
<div class="bg-white rounded-xl border border-gray-200 p-6">
<div class="flex flex-wrap gap-3 items-center mb-4">
<label class="text-sm font-medium text-gray-600">Filter by:</label>
<select id="tuFilterType" onchange="tuUpdateFilter()" class="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white">
<option value="all">ALL</option>
<option value="duration">By Duration</option>
<option value="task_type">By Task Type</option>
</select>
<select id="tuFilterValue" onchange="tuRender()" class="px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white" style="display:none"></select>
</div>
<div style="position:relative;height:700px"><canvas id="tuChart"></canvas></div>
</div></section>''')
    w('''<section class="mb-10">
<h2 class="text-2xl font-bold text-gray-900 mb-5 flex items-center gap-2"><span class="w-1.5 h-6 bg-purple-500 rounded-full"></span>Detailed Tool Statistics</h2>
<div class="bg-white rounded-xl border border-gray-200 overflow-x-auto">
<table class="w-full text-sm">
<thead><tr class="border-b border-gray-200 bg-gray-50">
<th class="text-left py-3 px-4 font-semibold text-gray-700">Tool Name</th>
<th class="text-right py-3 px-4 font-semibold text-gray-700">Total Calls</th>
<th class="text-right py-3 px-4 font-semibold text-gray-700">Sessions Used</th>
<th class="text-right py-3 px-4 font-semibold text-gray-700">Avg / Session</th>
<th class="text-right py-3 px-4 font-semibold text-gray-700">Usage Rate</th>
<th class="text-center py-3 px-4 font-semibold text-gray-700">Details</th>
</tr></thead>
<tbody id="tuTableBody"></tbody>
</table></div></section>''')
    w('''<div id="tuModal" class="fixed inset-0 z-[100] flex items-center justify-center" style="display:none;background:rgba(0,0,0,.5)">
<div class="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col mx-4">
<div class="flex items-center justify-between px-6 py-4 border-b border-gray-200">
<h3 id="tuModalTitle" class="text-lg font-bold text-gray-900"></h3>
<button onclick="tuCloseModal()" class="text-gray-400 hover:text-gray-600 text-2xl leading-none">&times;</button>
</div>
<div id="tuModalBody" class="px-6 py-4 overflow-y-auto flex-1"></div>
</div></div>''')
    w('<script>var tuData=' + tus['json'] + ';')
    w('''var tuChart=null;
var tuModalContext=null;
function tuUpdateFilter(){
var type=document.getElementById('tuFilterType').value;
var valSel=document.getElementById('tuFilterValue');
valSel.innerHTML='';
if(type==='all'){valSel.style.display='none';}
else if(type==='duration'){
valSel.style.display='';
['short','medium','long'].forEach(function(v){var o=document.createElement('option');o.value=v;o.textContent=v.charAt(0).toUpperCase()+v.slice(1);valSel.appendChild(o);});
}else if(type==='task_type'){
valSel.style.display='';
tuData.task_types.forEach(function(v){var o=document.createElement('option');o.value=v;o.textContent=v;valSel.appendChild(o);});
}
tuRender();
}
function tuGetCurrentData(){
var type=document.getElementById('tuFilterType').value;
if(type==='all') return tuData.tools;
var val=document.getElementById('tuFilterValue').value;
if(type==='duration') return (tuData.by_duration[val]||{});
if(type==='task_type') return (tuData.by_task_type[val]||{});
return tuData.tools;
}
function tuEsc(s){if(!s)return'';var d=document.createElement('div');d.textContent=s;return d.innerHTML.replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
function tuRender(){
var data=tuGetCurrentData();
var names=Object.keys(data).sort(function(a,b){return data[b].total_calls-data[a].total_calls;});
var top=names.slice(0,30);
var labels=top.map(function(n){return n.length>28?n.substring(0,25)+'...':n;});
var values=top.map(function(n){return data[n].total_calls;});
var colors=top.map(function(_,i){return 'hsla('+(260-i*6)+',55%,55%,.75)';});
if(tuChart){tuChart.destroy();}
var ctx=document.getElementById('tuChart').getContext('2d');
tuChart=new Chart(ctx,{type:'bar',data:{labels:labels,datasets:[{label:'Total Calls',data:values,backgroundColor:colors,borderRadius:4}]},
options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
plugins:{legend:{display:false},tooltip:{callbacks:{title:function(items){return top[items[0].dataIndex];}}}},
scales:{x:{grid:{color:'#f3f4f6'},title:{display:true,text:'Total Calls'}},y:{grid:{display:false},ticks:{font:{size:11,family:'monospace'}}}},
onClick:function(e,els){if(els.length){tuOpenModal(top[els[0].index]);}}}});
var tbody=document.getElementById('tuTableBody');
tbody.innerHTML=names.map(function(n){
var t=data[n];
var rc=t.usage_rate>=50?'bg-emerald-50 text-emerald-700':t.usage_rate>=20?'bg-blue-50 text-blue-700':'bg-gray-100 text-gray-600';
return '<tr class="border-b border-gray-100 hover:bg-gray-50/50">'+
'<td class="py-2.5 px-4 font-mono text-brand-700 text-sm">'+tuEsc(n)+'</td>'+
'<td class="py-2.5 px-4 text-right font-semibold">'+t.total_calls+'</td>'+
'<td class="py-2.5 px-4 text-right">'+t.session_count+'</td>'+
'<td class="py-2.5 px-4 text-right">'+t.avg_per_session+'</td>'+
'<td class="py-2.5 px-4 text-right"><span class="inline-block px-2 py-0.5 rounded text-xs font-semibold '+rc+'">'+t.usage_rate+'%</span></td>'+
'<td class="py-2.5 px-4 text-center"><button data-tn="'+tuEsc(n)+'" class="text-brand-600 hover:text-brand-800 text-xs font-medium cursor-pointer">View Sessions</button></td></tr>';
}).join('');
}
function tuOpenModal(toolName){
var data=tuGetCurrentData();
var info=data[toolName];
if(!info)return;
var qids=info.qids||[];
var ftype=document.getElementById('tuFilterType')?document.getElementById('tuFilterType').value:'all';
var fval=(ftype!=='all'&&document.getElementById('tuFilterValue'))?document.getElementById('tuFilterValue').value:'';
tuModalContext={toolName:toolName,qids:qids,type:ftype,value:fval};
document.getElementById('tuModalTitle').textContent='Sessions using: '+toolName;
var h='<p class="text-sm text-gray-500 mb-4">'+qids.length+' sessions | '+info.total_calls+' total calls | '+info.avg_per_session+' avg/session</p>';
h+='<div class="space-y-2" style="max-height:55vh;overflow-y:auto">';
qids.forEach(function(qid){
var sess=null;
if(typeof evalSessions!=='undefined'){sess=evalSessions.find(function(s){return s.question_id===qid;});}
if(sess){
h+='<div data-session-id="'+tuEsc(sess.session_id)+'" onclick="tuJumpToSession(this.dataset.sessionId)" class="cursor-pointer px-3 py-2 rounded border border-gray-200 text-sm hover:bg-gray-100 transition-colors"><div class="font-mono text-brand-600 text-xs mb-1">ID: '+tuEsc(sess.session_id)+'</div><div class="text-gray-600 text-xs truncate">'+tuEsc(sess.init_user_query?evalExtractQuestion(sess.init_user_query).substring(0,120):'')+'</div></div>';
}else{
h+='<div class="px-3 py-2 rounded border border-gray-200 text-sm font-mono text-gray-700 hover:bg-gray-50">'+tuEsc(qid)+'</div>';
}
});
h+='</div>';
document.getElementById('tuModalBody').innerHTML=h;
document.getElementById('tuModal').style.display='flex';
}
function tuJumpToSession(sid){
if(!tuModalContext){evalJumpToSession(sid);return;}
evalJumpToSessionWithScope(sid,tuModalContext.toolName,tuModalContext.qids,tuModalContext.type,tuModalContext.value);
}
function tuCloseModal(){document.getElementById('tuModal').style.display='none';}
document.getElementById('tuModal').addEventListener('click',function(e){if(e.target===this)tuCloseModal();});
document.getElementById('tuTableBody').addEventListener('click',function(e){var btn=e.target.closest('[data-tn]');if(btn)tuOpenModal(btn.dataset.tn);});
document.addEventListener('keydown',function(e){if(e.key==='Escape'&&document.getElementById('tuModal').style.display!=='none')tuCloseModal();});
tuRender();
</script>''')
    w('</div>')
else:
    w('<div id="panel-toolusage" class="tab-panel"><div class="text-center py-20 text-gray-400"><p class="text-lg">No tool usage data available.</p><p class="text-sm mt-2">Place all_data_path.json in conversations/ directory.</p></div></div>')

# FOOTER
w('''</main>
<footer class="bg-white border-t border-gray-200 mt-16">
<div class="max-w-7xl mx-auto px-6 py-8 text-center text-sm text-gray-400">
<p>Online Video Agent Harness for Long Video Understanding &middot; Baidu, Inc.</p>
</div></footer>''')

# Tab navigation
w('''<script>
function switchTab(id){
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    document.getElementById('panel-'+id).classList.add('active');
    document.querySelectorAll('.tab-btn').forEach(b=>{
        if(b.dataset.tab===id){b.style.background='linear-gradient(135deg,#4c6ef5,#7c3aed)';b.style.color='#fff';b.style.borderColor='transparent';}
        else{b.style.background='#fff';b.style.color='#4b5563';b.style.borderColor='#e5e7eb';}
    });
}
document.addEventListener('DOMContentLoaded',function(){
    switchTab('evaluation');
});
</script></body></html>''')

html = "\n".join(p)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Written {len(html)} bytes to {OUT}")
print(f"Total lines: {html.count(chr(10))+1}")
