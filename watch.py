"""
watch.py -- local control panel. Runs on your machine, not in Colab.

WHAT THIS CAN AND CANNOT DO, stated plainly:

  CAN   generate the one-file notebook, open it in Colab, and read the
        structured report the notebook writes to Drive (once you give it a
        Google Drive token, or once you drop the report into runs/).
  CANNOT programmatically run a Colab notebook. Google ships no such public
        API. Anything claiming to do it is poking private endpoints that break
        without notice and would need your session cookie.

So the loop stays: you press Run in Colab, and this app watches the report so
you (and any agent) can read exactly what failed instead of scrolling a
terminal. The report is plain JSON in runs/ -- drop it there and everything
works with no token at all.

    python watch.py            start the panel on http://127.0.0.1:8765
    python watch.py --build    just regenerate rsl_run.ipynb and exit
"""

import http.server
import json
import os
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path

D = Path(__file__).parent
RUNS = D / "runs"
NOTEBOOK = D / "rsl_all.ipynb"
PORT = 8765

# Colab's new-notebook URL. Harmless if it 404s -- the point is that the
# notebook itself is uploaded from disk, not fetched from a URL.
COLAB = "https://colab.research.google.com/"

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>rsl-coder-7b &middot; control panel</title>
<style>
  :root {
    --bg:#14100d; --panel:#1e1815; --line:#33291f; --txt:#f0e6da;
    --dim:#a2917f; --ok:#7fbf7f; --warn:#e0c060; --err:#e4687c; --acc:#e8924c;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
    font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { padding:20px 26px; border-bottom:1px solid var(--line);
    display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; letter-spacing:.04em; }
  .sub { color:var(--dim); font-size:12px; }
  main { padding:22px 26px; max-width:1000px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }
  button { background:var(--panel); color:var(--txt); border:1px solid var(--line);
    padding:9px 15px; border-radius:5px; cursor:pointer; font:inherit; }
  button:hover { border-color:var(--acc); color:var(--acc); }
  button.p { background:var(--acc); color:#1a1208; border-color:var(--acc);
    font-weight:600; }
  a.btn { display:inline-block; background:var(--panel); color:var(--txt);
    border:1px solid var(--line); padding:9px 15px; border-radius:5px;
    text-decoration:none; font:inherit; }
  a.btn:hover { border-color:var(--acc); color:var(--acc); }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.1em;
    color:var(--dim); margin:26px 0 9px; font-weight:600; }
  .card { background:var(--panel); border:1px solid var(--line);
    border-radius:7px; padding:15px 17px; }
  .stage { font-size:19px; font-weight:600; }
  .ok { color:var(--ok); } .bad { color:var(--err); } .warn { color:var(--warn); }
  pre { background:#100d0b; border:1px solid var(--line); border-radius:6px;
    padding:13px; overflow:auto; max-height:340px; font-size:12px;
    white-space:pre-wrap; word-break:break-word; color:#d8cbbc; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  td,th { text-align:left; padding:5px 9px; border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:600; text-transform:uppercase;
    letter-spacing:.07em; font-size:11px; }
  .muted { color:var(--dim); }
  a { color:var(--acc); }
  code { background:#100d0b; padding:1px 5px; border-radius:3px; font-size:12px; }
  .steps li { list-style:none; margin:2px 0; }
  .steps { padding:0; margin:0; }
  .steps li:before { content:"\\2022"; color:var(--acc); margin-right:9px; }
</style>

<header>
  <h1>rsl-coder-7b</h1>
  <span class="sub">phase 0 + 1 &middot; control panel</span>
</header>
<main>
  <div class="row">
    <button class="p" onclick="go('/api/build')">1 &middot; Build notebook</button>
    <a class="btn" href="https://colab.research.google.com/"
       target="_blank" rel="noopener">2 &middot; Open Colab</a>
    <button onclick="refresh()">3 &middot; Refresh</button>
  </div>

  <h2>How to run</h2>
  <div class="card muted">
    1. Build the notebook (button above)<br>
    2. Open Colab, <code>File &rarr; Upload notebook</code>, pick
       <code>rsl_all.ipynb</code> &mdash; <b>that is the only file</b>; all 21
       sources are embedded in it<br>
    3. Run <b>cell 1</b> (setup + selftest, no download), wait for
       <code>SELFTEST PASSED</code><br>
    4. Run <b>cell 2</b> (status), then the phase cells <b>3&ndash;9</b> in order<br>
    5. Press <b>Refresh</b> here to read the result<br><br>
    <span class="warn">No Colab run API exists.</span> You press Run; this page
    reads the <code>report.json</code> the notebook writes. Drop that file into
    <code>runs/</code> and it appears with no setup at all.
  </div>

  <h2>Status</h2>
  <div class="card" id="status">no report yet</div>

  <h2>Phases</h2>
  <div class="card"><table id="phases"><tr><th>#</th><th>phase</th><th>ran</th>
    <th>gate</th><th>detail</th></tr></table></div>

  <h2>Steps</h2>
  <div class="card"><ul class="steps" id="steps"></ul></div>

  <h2>Results</h2>
  <div class="card" id="results">&mdash;</div>

  <h2>Traceback</h2>
  <pre id="tb">none</pre>
</main>

<script>
function esc(s){ return String(s==null?'':s).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function go(u){ fetch(u).then(r=>r.text()).then(t=>{ if(t.trim()) alert(t); refresh(); }); }

function refresh(){
  fetch('/api/report').then(r=>r.json()).then(d=>{
    if(!d.exists){ document.getElementById('status').innerHTML =
      '<span class="warn">no report yet</span> &mdash; run the notebook, then Refresh';
      return; }

    const s=d.report;
    const cls = s.ok===true?'ok':(s.ok===false?'bad':'warn');
    let h = '<div class="stage '+cls+'">'+esc(s.stage||'unknown')+'</div>';
    h += '<div class="muted">ok='+esc(s.ok)+' &middot; '+esc(d.age)+' &middot; '+esc(d.source)+'</div>';
    if(s.error) h += '<div class="bad" style="margin-top:9px">'+esc(s.error)+'</div>';
    document.getElementById('status').innerHTML = h;

    document.getElementById('steps').innerHTML = (s.steps||[])
      .map(x=>'<li>'+esc(x.name)+(x.secs?' ('+x.secs+'s)':'')+
             (x.gate===false?' <span class="bad">gate FAIL</span>':'')+'</li>')
      .join('') || '<li class="muted">none</li>';

    const r=s.results;
    if(r){
      let t='<table><tr><th>metric</th><th>value</th></tr>';
      const add=(k,v)=>{ t+='<tr><td>'+esc(k)+'</td><td>'+esc(v)+'</td></tr>'; };
      add('coding (phase 0)', r.baseline_code!=null? r.baseline_code+'/'+r.n : '-');
      add('coding (phase 1)', r.abliterated_code!=null? r.abliterated_code+'/'+r.n : '-');
      add('refusal before', r.baseline_refusal!=null? Math.round(r.baseline_refusal*100)+'%' : '-');
      add('refusal after', r.abliterated_refusal!=null? Math.round(r.abliterated_refusal*100)+'%' : '-');
      add('capability cost', r.capability_cost!=null? r.capability_cost+' tasks' : '-');
      add('gate', r.verdict||'-');
      document.getElementById('results').innerHTML = t+'</table>';
    } else document.getElementById('results').innerHTML='<span class="muted">&mdash;</span>';

    document.getElementById('tb').textContent = s.tb || 'none';

    if(d.phases && d.phases.length){
      let t='<tr><th>#</th><th>phase</th><th>ran</th><th>gate</th><th>detail</th></tr>';
      for(const p of d.phases){
        const g = p.ran ? (p.passed?'<span class="ok">PASS</span>'
                              :'<span class="bad">FAIL</span>')
                        : '<span class="muted">-</span>';
        t+='<tr><td>'+p.n+'</td><td>'+esc(p.label)+'</td>'
         +'<td>'+(p.ran?'yes':'<span class="muted">-</span>')+'</td>'
         +'<td>'+g+'</td><td class="muted">'+esc(p.detail)+'</td></tr>';
      }
      document.getElementById('phases').innerHTML = t;
    } else document.getElementById('phases').innerHTML =
      '<tr><td colspan="5" class="muted">no phase data yet</td></tr>';
  });
}
refresh(); setInterval(refresh, 5000);
</script>
"""

BANNER = "=" * 64


def find_report():
    """Look in runs/ first (dropped manually), then in the unpacked report."""
    RUNS.mkdir(exist_ok=True)
    cands = sorted(RUNS.glob("report*.json"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return cands[0] if cands else None


def human_age(p):
    import datetime as dt
    secs = (dt.datetime.now().timestamp() - p.stat().st_mtime)
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs//60)}m ago"
    return f"{int(secs//3600)}h ago"


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")

        if self.path == "/api/report":
            p = find_report()
            if not p:
                return self._send(200, json.dumps({"exists": False,
                                                   "phases": []}))
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                unreadable = {"stage": "unreadable", "ok": False,
                              "error": f"report.json is corrupt: {e}",
                              "tb": None, "steps": []}
                # Still derive the phase table from the summary files. They are
                # written independently, so an unreadable report must not also
                # erase the progress view -- that is exactly when it is needed.
                return self._send(200, json.dumps(
                    {"exists": True, "age": human_age(p), "source": p.name,
                     "phases": self._phase_table(p.parent),
                     "report": unreadable}))

            return self._send(200, json.dumps(
                {"exists": True, "age": human_age(p), "source": p.name,
                 "phases": self._phase_table(p.parent),
                 "report": rep}))

        if self.path == "/api/build":
            r = subprocess.run([sys.executable, str(D / "build.py")],
                               capture_output=True, text=True)
            if r.returncode:
                return self._send(200, json.dumps(
                    {"error": (r.stderr or r.stdout)[-1500:]}))
            webbrowser.open(NOTEBOOK.as_uri())
            return self._send(200, json.dumps(
                {"ok": True, "notebook": NOTEBOOK.name,
                 "kb": round(NOTEBOOK.stat().st_size / 1024),
                 "msg": "Built and opened locally. "
                        "Upload rsl_all.ipynb to Colab."}))

        if self.path == "/api/open":
            # The page uses a plain <a href> for this, because
            # webbrowser.open() called from an HTTP handler thread can fail
            # silently on Windows. This endpoint stays as a fallback.
            opened = False
            try:
                if sys.platform == "win32":
                    os.startfile(COLAB)          # noqa: S606
                    opened = True
                else:
                    opened = webbrowser.open(COLAB)
            except Exception as e:
                return self._send(200, json.dumps(
                    {"ok": False, "msg": f"could not launch a browser: {e}",
                     "url": COLAB}))
            return self._send(200, json.dumps(
                {"ok": opened, "url": COLAB,
                 "msg": ("Opened Colab. Upload rsl_all.ipynb."
                         if opened else
                         f"Could not open a browser -- go to {COLAB} and "
                         f"upload rsl_all.ipynb.")}))

        return self._send(404, json.dumps({"error": "no such endpoint"}))

    @staticmethod
    def _phase_table(folder):
        """Derive the 8-row phase table from whatever summary files exist.

        Independent of report.json, so progress survives a corrupt or missing
        report."""
        names = {
            0: ("phase0_summary.json", "baseline"),
            1: ("phase1_summary.json", "abliteration"),
            2: ("phase02_ranks.json", "fisher rank table"),
            3: ("phase03_summary.json", "QLoRA reference"),
            4: ("phase04_summary.json", "elastic MoE"),
            5: ("phase05_summary.json", "autonomous harvester"),
            6: ("phase06_summary.json", "self-play + RSL"),
            7: ("phase07_summary.json", "finalize"),
        }
        out = []
        for n, (fn, label) in sorted(names.items()):
            sp = Path(folder) / fn
            ran = sp.exists()
            ok, detail = False, "not run"
            if ran:
                try:
                    s = json.loads(sp.read_text(encoding="utf-8"))
                except Exception:
                    s, detail = {}, "unreadable"
                if n == 7:
                    ok = bool(s.get("gate_code")) and bool(s.get("gate_refusal"))
                    detail = (f"code={s.get('gate_code')} "
                              f"refusal={s.get('gate_refusal')}")
                elif "verdict" in s:
                    ok, detail = bool(s["verdict"]), f"verdict={s['verdict']}"
                elif "gate" in s:
                    ok, detail = bool(s["gate"]), f"gate={s['gate']}"
                else:
                    ok, detail = True, "ran"
            out.append({"n": n, "label": label, "ran": ran, "passed": ok,
                        "detail": detail})
        return out



def main():
    if "--build" in sys.argv:
        raise SystemExit(subprocess.run(
            [sys.executable, str(D / "build.py")]).returncode)

    RUNS.mkdir(exist_ok=True)
    if not NOTEBOOK.exists():
        subprocess.run([sys.executable, str(D / "build.py")],
                       capture_output=True)

    class S(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with S(("127.0.0.1", PORT), H) as httpd:
        print(BANNER)
        print(f"panel  http://127.0.0.1:{PORT}")
        print(f"notebook {NOTEBOOK.name} ({NOTEBOOK.stat().st_size/1024:.0f} KB)")
        print(f"reports {RUNS}")
        print(BANNER)
        print("Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()