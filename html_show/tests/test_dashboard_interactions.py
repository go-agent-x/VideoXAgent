"""Exercise transcript parsing, rendering, and loading in generated JavaScript."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


HTML_DIR = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

# A small DOM/clock adapter runs the real generated script without a browser or
# network. Assertions below inspect rendered content and requests, not internals.
HARNESS = r'''
const vm = require('vm'), assert = require('assert');
const {TextDecoder} = require('util');
function makeDashboard(search = '') {
  const nodes = {}, cards = {}, requests = [], timers = new Map();
  let nextTimer = 0;
  function node() {
    const n = {value:'all', dataset:{}, style:{}, innerHTML:'',
      classList:{add(){}, remove(){}}, scrollIntoView(){}};
    Object.defineProperty(n, 'textContent', {set(value) {
      n.innerHTML = String(value).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }});
    return n;
  }
  const ctx = {URL, URLSearchParams, TextDecoder,
    window:{location:{search, origin:'http://localhost:8923', href:'http://localhost:8923/html_show/dashboard.html'}},
    document:{
      getElementById(id) {
        if (!nodes[id]) {
          nodes[id] = node();
          if (id === 'evalSearchInput') nodes[id].value = '';
        }
        return nodes[id];
      },
      createElement:node,
      querySelector(selector) {
        const match = selector.match(/^\[data-sid="(.*)"\]$/);
        return match ? cards[match[1]] || null : null;
      }
    },
    setTimeout(callback, delay) {
      const id = ++nextTimer;
      timers.set(id, {callback, delay});
      return id;
    },
    clearTimeout(id) {timers.delete(id);},
    AbortController:class {
      constructor() {this.signal = {aborted:false};}
      abort() {this.signal.aborted = true;}
    },
    fetch:async (url, options) => {
      requests.push({url, options});
      return h.respond(url, options);
    }
  };
  const h = {ctx, nodes, cards, requests,
    respond() {throw new Error('Unexpected network request');},
    async load(sid, path, exists = true) {
      cards[sid] = {dataset:{tp:path, tsize:'512'}};
      await ctx.evalLoadInteractions(sid, path, exists);
      // Parsing is deferred so the loading indicator can paint. Flush the
      // remaining callbacks after the fetch settles, without a real delay.
      let count = 0;
      while (timers.size) {
        assert(++count < 10, 'Loader left recurring timers');
        const [id, timer] = [...timers].sort((a,b) => a[1].delay-b[1].delay)[0];
        timers.delete(id);
        timer.callback();
        await Promise.resolve();
      }
      return nodes['eval-interaction-' + sid].innerHTML;
    }
  };
  vm.createContext(ctx);
  vm.runInContext(SCRIPT, ctx);
  return h;
}
function jsonl(records) {return records.map(record => JSON.stringify(record)).join('\n');}
function parse(h, records) {
  return h.ctx.evalExtractConversationData(h.ctx.evalParseSessionLog(jsonl(records))).interactions;
}
function textResponse(text, status = 200) {
  return {ok:status >= 200 && status < 300, status, text:async () => text};
}
function streamResponse(text) {
  const bytes = Buffer.from(text, 'utf8');
  let offset = 0;
  return {ok:true, status:200, body:{getReader() {return {async read() {
    if (offset >= bytes.length) return {done:true};
    const value = bytes.subarray(offset, offset + 7);
    offset += value.length;
    return {done:false, value};
  }};}}, text:async () => {throw new Error('Streaming response was read twice');}};
}
'''


@unittest.skipUnless(NODE, "Node.js is needed for dashboard interaction checks")
class DashboardInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="dashboard interactions ")
        root = Path(cls.temporary.name)
        data = root / "results.json"
        data.write_text(json.dumps({"sessions": [{
            "session_id": "fixture", "question_id": "q1",
            "init_user_query": "Question: What happens?",
            "transcript_path": "transcripts/fixture.jsonl",
            "model_answer": "A", "gt_answer": "A", "is_correct": True,
        }]}), encoding="utf-8")
        output = root / "index.html"
        try:
            subprocess.run([
                sys.executable, str(HTML_DIR / "gen_dashboard.py"),
                "--data", str(data), "--data-root", str(root), "--out", str(output),
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True)
            scripts = re.findall(r"<script\b[^>]*>(.*?)</script>",
                                 output.read_text(encoding="utf-8"), re.S)
            cls.script = next(script for script in scripts if "var evalSessions=" in script)
            cls.tool_usage_script = next(script for script in scripts if "var tuData=" in script)
        except Exception:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_javascript(self, assertions):
        code = "const SCRIPT = " + json.dumps(self.script) + ";\n"
        code += "const TU_SCRIPT = " + json.dumps(self.tool_usage_script) + ";\n" + HARNESS
        code += "\n(async () => {\n" + assertions + "\n})().catch(error => {console.error(error);process.exitCode=1;});"
        result = subprocess.run([NODE], input=code, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True,
                                timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_claude_partial_messages_and_tool_results_render_once(self):
        self.run_javascript(r'''
const h = makeDashboard();
const records = [
  {type:'user', isMeta:true, message:{content:'hidden metadata'}},
  {type:'user', message:{content:'Question <sample> & details'}},
  {type:'assistant', message:{id:'message-1', content:[
    {type:'text', text:'Partial answer'},
    {type:'tool_use', id:'call-1', name:'mcp__inspect', input:{query:'find <person>'}}
  ]}},
  {type:'assistant', message:{id:'message-1', content:[
    {type:'thinking', thinking:'Compare the frames'},
    {type:'text', text:'Final answer'},
    {type:'tool_use', id:'call-1', name:'mcp__inspect', input:{query:'find <person>'}}
  ]}},
  {type:'user', message:{content:[{type:'tool_result', tool_use_id:'call-1', content:'Detected <person>'}]}}
];
const entries = h.ctx.evalParseSessionLog('\ninvalid json\n' + jsonl(records) + '\n');
const interactions = h.ctx.evalExtractConversationData(entries).interactions;
assert.strictEqual(interactions.length, 1);
assert.strictEqual(interactions[0].userMessage, 'Question <sample> & details');
const trace = interactions[0].trace;
assert.strictEqual(trace.length, 2);
const tool = trace.find(item => item.type === 'tool_use');
assert.strictEqual(tool.name, 'mcp__inspect');
assert.strictEqual(tool.input.query, 'find <person>');
assert.strictEqual(tool.output, 'Detected <person>');
const answer = trace.find(item => item.type === 'assistant');
assert.strictEqual(answer.text, 'Final answer');
assert.strictEqual(answer.thinking, 'Compare the frames');
const html = h.ctx.evalRenderInteractions(interactions, 'transcripts/fixture.jsonl');
for (const text of ['Question &lt;sample&gt; &amp; details', 'Final answer',
                    'Compare the frames', 'Detected &lt;person&gt;', '1 tools, 1 responses']) {
  assert(html.includes(text), text);
}
assert(!html.includes('Partial answer'));
assert(!html.includes('hidden metadata'));
''')

    def test_codex_function_results_and_web_search_are_preserved(self):
        self.run_javascript(r'''
const h = makeDashboard();
const item = payload => ({type:'response_item', payload});
const interactions = parse(h, [
  item({type:'message', role:'user', content:[{type:'input_text', text:'Find the event'}]}),
  item({type:'function_call', call_id:'call-1', name:'inspect', arguments:'{"frame":3}'}),
  item({type:'function_call_output', call_id:'call-1', output:{result:'Found <event>'}}),
  item({type:'web_search_call', status:'completed', action:{query:'event details'}}),
  item({type:'message', role:'assistant', content:[
    {type:'output_text', text:'The event '}, {type:'output_text', text:'is confirmed.'}
  ]}),
  item({type:'message', role:'user', content:[{type:'input_text', text:'Inspect again'}]}),
  item({type:'function_call', call_id:'call-2', name:'inspect', arguments:'raw arguments'}),
  item({type:'function_call_output', call_id:'call-2', output:'Second result'})
]);
assert.strictEqual(interactions.length, 2);
assert.strictEqual(interactions[0].userMessage, 'Find the event');
const trace = interactions[0].trace;
assert.strictEqual(trace[0].input.frame, 3);
assert.deepStrictEqual(JSON.parse(trace[0].output), {result:'Found <event>'});
assert.strictEqual(trace[1].name, 'web_search');
assert.strictEqual(trace[1].input.query, 'event details');
assert.strictEqual(JSON.parse(trace[1].output).status, 'completed');
assert.strictEqual(trace[2].text, 'The event is confirmed.');
assert.strictEqual(interactions[1].trace[0].input.arguments, 'raw arguments');
assert.strictEqual(interactions[1].trace[0].output, 'Second result');
const html = h.ctx.evalRenderInteractions(interactions, 'transcripts/fixture.jsonl');
for (const text of ['Found &lt;event&gt;', 'web_search', 'event details',
                    'The event is confirmed.', 'Interaction #2', 'Second result']) {
  assert(html.includes(text), text);
}
''')

    def test_agentic_text_blocks_and_structured_tool_output_are_preserved(self):
        self.run_javascript(r'''
const h = makeDashboard();
const interactions = parse(h, [
  {type:'user', message:{content:[{type:'text', text:'Describe the clip'}]}},
  {type:'assistant', message:{content:'Checking the scene', thinking:'Follow the moving object'}},
  {type:'tool', message:{name:'inspect', input:{frame:7}, output:JSON.stringify([
    {type:'text', text:'A person enters'}, {type:'text', text:'A door closes'}
  ])}},
  {type:'tool', message:{name:'score', input:'clip-7', output:{score:0.9}}},
  {type:'assistant', message:{content:[{type:'text', text:'The person entered.'}]}}
]);
assert.strictEqual(interactions.length, 1);
assert.strictEqual(interactions[0].userMessage, 'Describe the clip');
const trace = interactions[0].trace;
assert.strictEqual(trace.length, 4);
assert.strictEqual(trace[0].text, 'Checking the scene');
assert.strictEqual(trace[0].thinking, 'Follow the moving object');
assert.strictEqual(trace[1].input.frame, 7);
assert.strictEqual(trace[1].output, 'A person enters\nA door closes');
assert.strictEqual(JSON.parse(trace[2].input.raw), 'clip-7');
assert.deepStrictEqual(JSON.parse(trace[2].output), {score:0.9});
const html = h.ctx.evalRenderInteractions(interactions, 'transcripts/fixture.jsonl');
for (const text of ['Describe the clip', 'Follow the moving object', 'A person enters',
                    'A door closes', 'The person entered.', '2 tools, 2 responses']) {
  assert(html.includes(text), text);
}
''')

    def test_loader_streams_same_origin_and_reuses_cached_transcript(self):
        self.run_javascript(r'''
const h = makeDashboard();
const path = 'transcripts/clip café #1 & detail.jsonl';
const content = jsonl([
  {type:'user', message:{content:'Explain the person’s actions in the video.'}},
  {type:'tool', message:{name:'inspect', input:{image:'frame café.png'}, output:'The person is walking — slowly.'}}
]);
h.respond = () => streamResponse(content);
const html = await h.load('one', path);
assert(html.includes('Explain the person’s actions in the video.'));
assert(html.includes('The person is walking — slowly.'));
assert(html.includes('&amp;base=' + encodeURIComponent(path)));
assert.strictEqual(h.requests.length, 1);
assert.strictEqual(h.requests[0].url, '/api/read?path=' + encodeURIComponent(path));
assert(h.requests[0].options.signal, 'Fetch should support cancellation');
const cached = await h.load('two', path);
assert.strictEqual(cached, html);
assert.strictEqual(h.requests.length, 1, 'Reopening a loaded transcript must use its cache');
''')

    def test_loader_only_accepts_same_origin_backend_and_keeps_caches_separate(self):
        self.run_javascript(r'''
const h = makeDashboard('?api_base=' + encodeURIComponent('http://localhost:8923///'));
const path = 'transcripts/session.jsonl';
h.respond = () => textResponse(jsonl([{type:'user', message:{content:'From remote backend'}}]));
assert((await h.load('remote', path)).includes('From remote backend'));
assert.strictEqual(h.requests[0].url, 'http://localhost:8923/api/read?path=' + encodeURIComponent(path));
await h.load('remote-again', path);
assert.strictEqual(h.requests.length, 1);
h.ctx.window.location.search = '?api_base=' + encodeURIComponent('http://attacker.example');
h.respond = () => textResponse(jsonl([{type:'user', message:{content:'From local backend'}}]));
const local = await h.load('local', path);
assert(local.includes('From local backend'));
assert(!local.includes('From remote backend'));
assert.strictEqual(h.requests.length, 2);
assert.strictEqual(h.requests[1].url, '/api/read?path=' + encodeURIComponent(path));
for (const override of ['//attacker.example', 'https://localhost:8923',
                         'http://localhost:8923@attacker.example', 'javascript:alert(1)']) {
  h.ctx.window.location.search = '?api_base=' + encodeURIComponent(override);
  assert.strictEqual(h.ctx.evalApiBase(), '', override);
}
h.ctx.window.location.origin = 'null';
h.ctx.window.location.href = 'file:///tmp/export/index.html';
h.ctx.window.location.search = '?api_base=file%3A%2F%2F%2Ftmp%2Fexport';
assert.strictEqual(h.ctx.evalApiBase(), '', 'Opaque file origins cannot authorize an API override');
''')

    def test_loader_reports_http_missing_empty_and_timeout_failures(self):
        self.run_javascript(r'''
const h = makeDashboard();
const emptyPath = await h.load('no-path', '');
assert(emptyPath.includes('No transcript path.'));
assert.strictEqual(h.requests.length, 0);
h.respond = () => textResponse('<h1>Access denied</h1>', 403);
const denied = await h.load('denied', 'private.jsonl');
assert(denied.includes('HTTP 403: Access denied'));
assert(!denied.includes('<h1>'));
h.respond = () => textResponse('Not found', 404);
const missing = await h.load('missing', 'missing.jsonl', false);
assert(missing.includes('Transcript file not found'));
assert(missing.includes('missing.jsonl'));
h.respond = () => textResponse('');
assert((await h.load('empty', 'empty.jsonl')).includes('No interactions found.'));
h.respond = () => {const error = new Error('Request aborted'); error.name='AbortError'; throw error;};
assert((await h.load('timeout', 'slow.jsonl')).includes('Connect timeout after 8s'));
// A failed request must be retryable, rather than cached as an empty success.
h.respond = () => textResponse(jsonl([{type:'user', message:{content:'Retry worked'}}]));
assert((await h.load('retry', 'private.jsonl')).includes('Retry worked'));
assert.strictEqual(h.requests.filter(r => r.url.includes('private.jsonl')).length, 2);
''')

    def test_untrusted_text_cannot_break_out_of_html_attributes(self):
        self.run_javascript(r'''
const h = makeDashboard();
const payload = 'frame" onerror="alert(1)\' data-unsafe=\'yes';
const escaped = h.ctx.evalEsc(payload);
assert(!escaped.includes('"'), escaped);
assert(!escaped.includes("'"), escaped);
assert(escaped.includes('&quot;'), escaped);
assert(escaped.includes('&#39;') || escaped.includes('&#x27;'), escaped);
const transcript = 'transcripts/' + payload + '.jsonl';
const media = 'https://example.test/' + payload + '.png';
const html = h.ctx.evalRenderInteractions([{userMessage:'Question',trace:[{
  type:'tool_use', name:'inspect', input:{image:media}, output:''
}]}],transcript);
assert(!html.includes(' onerror="alert(1)'), html);
assert(!html.includes(" data-unsafe='yes"), html);
''')

    def test_benchmark_values_cannot_inject_inline_click_handlers(self):
        self.run_javascript(r'''
const h = makeDashboard();
const attack = "');globalThis.__injected=true;//";
h.ctx.evalFiltered = [{session_id:attack, videoID:attack, duration:attack,
  domain:attack, sub_category:attack, task_type:attack, question_id:'q1',
  transcript_path:'transcripts/fixture.jsonl', init_user_query:'Question',
  is_correct:true, gt_answer:'A', model_answer:'A'}];
h.ctx.evalRenderList();
const html = h.nodes.evalList.innerHTML;
const decode = text => text.replace(/&quot;/g,'"').replace(/&#(?:39|x27);/gi,"'")
  .replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
h.ctx.evalToggleVideo = () => {};
h.ctx.evalFilterBy = () => {};
let checked = 0;
for (const tag of html.match(/<[^>]+>/g) || []) {
  const handler = tag.match(/\bonclick="([^"]*)"/);
  if (!handler || !/evalToggleVideo|evalFilterBy/.test(handler[1])) continue;
  const dataset = {};
  for (const attribute of tag.matchAll(/\bdata-([a-z-]+)="([^"]*)"/g)) {
    dataset[attribute[1].replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = decode(attribute[2]);
  }
  h.ctx.__element = {dataset};
  vm.runInContext('(function(event){' + decode(handler[1]) + '\n}).call(__element, {})', h.ctx);
  checked++;
}
assert(checked >= 5, 'Video and benchmark filter handlers should remain usable');
assert.strictEqual(h.ctx.__injected, undefined, 'Dataset text executed as JavaScript');
''')

    def test_video_fallback_treats_untrusted_identifier_as_url_data(self):
        self.run_javascript(r'''
const h = makeDashboard(), appended = [];
const panel = {innerHTML:'', appendChild(node){appended.push(node);}};
h.nodes['eval-yt-fixture'] = panel;
const identifier = 'clip" onload="alert(1)<script>';
h.ctx.evalToggleVideo(identifier, 'fixture');
assert.strictEqual(appended.length, 1);
appended[0].onerror();
assert.strictEqual(appended.length, 2);
assert.strictEqual(appended[1].src, 'https://www.youtube.com/embed/' + encodeURIComponent(identifier) + '?autoplay=1');
assert.strictEqual(panel.innerHTML, '', 'Fallback should create DOM nodes, not parse untrusted HTML');
''')

    def test_tool_usage_names_and_session_ids_remain_data(self):
        self.run_javascript(r'''
const h = makeDashboard();
h.ctx.Chart = class {destroy(){}};
h.ctx.document.addEventListener = () => {};
const originalGet = h.ctx.document.getElementById;
h.ctx.document.getElementById = function(id) {
  const element = originalGet(id);
  element.getContext = () => ({});
  element.addEventListener = () => {};
  return element;
};
vm.runInContext(TU_SCRIPT, h.ctx);
const tool = 'inspect" onmouseover="globalThis.__toolInjected=true" data-extra="';
const session = "');globalThis.__sessionInjected=true;//";
h.ctx.tuData = {tools:{[tool]:{total_calls:1, session_count:1, avg_per_session:1,
  usage_rate:100, qids:['q1']}}, by_duration:{}, by_task_type:{}, task_types:[]};
h.ctx.evalSessions = [{question_id:'q1', session_id:session, init_user_query:'Question: Safe display'}];
h.ctx.tuRender();
const decode = text => text.replace(/&quot;/g,'"').replace(/&#(?:39|x27);/gi,"'")
  .replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
const button = h.nodes.tuTableBody.innerHTML.match(/<button\b[^>]*>/)[0];
assert(!/\bonmouseover="/.test(button), button);
assert(!/\bdata-extra="/.test(button), button);
const encodedTool = button.match(/\bdata-tn="([^"]*)"/)[1];
assert.strictEqual(decode(encodedTool), tool, 'The tool remains selectable by its exact name');
h.ctx.tuOpenModal(decode(encodedTool));
const modal = h.nodes.tuModalBody.innerHTML;
const clickable = modal.match(/<div\b[^>]*\bonclick="[^"]*"[^>]*>/)[0];
const handler = clickable.match(/\bonclick="([^"]*)"/)[1];
const dataset = {};
for (const attribute of clickable.matchAll(/\bdata-([a-z-]+)="([^"]*)"/g)) {
  dataset[attribute[1].replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = decode(attribute[2]);
}
let selected;
h.ctx.tuJumpToSession = value => {selected = value;};
h.ctx.__element = {dataset};
vm.runInContext('(function(){' + decode(handler) + '\n}).call(__element)', h.ctx);
assert.strictEqual(h.ctx.__sessionInjected, undefined, 'Session identifier executed as JavaScript');
assert.strictEqual(selected, session, 'The exact session remains reachable from the modal');
''')


if __name__ == "__main__":
    unittest.main()
