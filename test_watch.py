"""
test_watch.py -- exercise the panel against realistic report/summary states.

The panel is only useful when something has gone wrong, so the states that
matter most are: a mid-run crash, a gate that failed, a partially-completed
run, and a corrupt report. A panel that renders only the happy path is
decoration.
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

D = Path(__file__).parent
RUNS = D / "runs"
PORT = 8766
fails = []


def chk(label, got, want=True):
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"          got={got!r} want={want!r}")
        fails.append(label)


def wipe():
    for p in RUNS.glob("*"):
        p.unlink()


def write(name, obj):
    (RUNS / name).write_text(json.dumps(obj, indent=2))


# Check the port is free BEFORE starting, and after. An already-running
# watch.py holds the port and serves STALE html, so a fix appears not to work
# and you debug the wrong copy. That happened here: two processes, and the
# browser kept getting the old broken stylesheet.
import socket  # noqa: E402


def port_busy(p):
    s = socket.socket()
    s.settimeout(0.4)
    busy = s.connect_ex(("127.0.0.1", p)) == 0
    s.close()
    return busy


HINT = ('port {} is in use -- close any running watch.py first:\n'
        '  Get-CimInstance Win32_Process -Filter "Name=\'python.exe\'" | '
        'Where-Object {{ $_.CommandLine -match \'watch.py\' }} | '
        'ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}')

if port_busy(PORT):
    raise SystemExit(HINT.format(PORT))

proc = subprocess.Popen(
    [sys.executable, "-c",
     f"import sys;sys.argv=['watch.py'];sys.path.insert(0,{str(D)!r});"
     f"import watch;watch.PORT={PORT};watch.main()"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)
base = f"http://127.0.0.1:{PORT}"

if not port_busy(PORT):
    proc.terminate()
    raise SystemExit("watch.py did not bind its port -- cannot test")


def api(p):
    return json.loads(urllib.request.urlopen(base + p, timeout=8).read())


try:
    RUNS.mkdir(exist_ok=True)
    print("no report at all")
    wipe()
    d = api("/api/report")
    chk("exists False", d["exists"], False)
    chk("empty phase list", d["phases"], [])

    print("\nphase 0 and 1 done, gate passed")
    write("phase0_summary.json", {"baseline_code": 8, "n": 10,
                                  "baseline_refusal": 0.4})
    write("phase1_summary.json", {"abliterated_code": 7, "n": 10,
                                  "abliterated_refusal": 0.0,
                                  "baseline_code": 8, "capability_cost": 1,
                                  "verdict": "PASS"})
    write("report.json", {"stage": "done", "ok": True, "error": None, "tb": None,
                          "steps": [{"name": "phase0_baseline", "secs": 300,
                                     "gate": True, "detail": "ran"}],
                          "results": {"baseline_code": 8, "n": 10,
                                      "abliterated_code": 7}})
    d = api("/api/report")
    chk("8 phase rows", len(d["phases"]), 8)
    ran = {p["n"]: p["ran"] for p in d["phases"]}
    chk("phase 0 ran", ran[0])
    chk("phase 1 ran", ran[1])
    chk("phase 3 not run", ran[3], False)
    g1 = [p for p in d["phases"] if p["n"] == 1][0]
    chk("phase 1 gate passed", g1["passed"])
    chk("detail names the field", "verdict" in g1["detail"])

    print("\nphase 3 ran but its gate FAILED")
    write("phase03_summary.json", {"phase3_code": 3, "reference": 7,
                                   "gate": False})
    d = api("/api/report")
    g3 = [p for p in d["phases"] if p["n"] == 3][0]
    chk("phase 3 ran", g3["ran"])
    chk("phase 3 gate failed", g3["passed"], False)
    chk("phase 3 detail", "gate=False" in g3["detail"])

    print("\nphase 7 with two independent gates")
    write("phase07_summary.json", {"code": 7, "refusal": 0.0,
                                   "gate_code": True, "gate_refusal": False})
    d = api("/api/report")
    g7 = [p for p in d["phases"] if p["n"] == 7][0]
    chk("phase 7 ran", g7["ran"])
    chk("phase 7 gate failed on refusal", g7["passed"], False)
    chk("phase 7 detail names both", "refusal=False" in g7["detail"])

    print("\nmid-run crash report")
    write("report.json", {
        "stage": "phase6", "ok": False, "error": "self-play + RSL raised",
        "tb": ("Traceback (most recent call last):\n"
               '  File "runner.py", line 118, in run_phase\n'
               "    exec(compile(path.read_text(...)))\n"
               '  File "/content/rsl/phases/phase06_selfplay_rsl.py", line 210\n'
               "    model = PeftModel.from_pretrained(model, str(ADAPTER))\n"
               "RuntimeError: Expected directory or file"),
        "steps": [{"name": "phase5_harvester", "secs": 900, "gate": True},
                  {"name": "phase6_self-play", "secs": 120, "gate": False}],
    })
    d = api("/api/report")
    chk("crash ok=False", d["report"]["ok"], False)
    chk("traceback preserved", "RuntimeError" in d["report"]["tb"])
    chk("error names the phase", d["report"]["error"] == "self-play + RSL raised")

    print("\ncorrupt report.json -- progress view must survive")
    (RUNS / "report.json").write_text("{{{ not json")
    d = api("/api/report")
    chk("server survives", d["exists"])
    chk("stage unreadable", d["report"]["stage"], "unreadable")
    chk("error explains corruption", "corrupt" in (d["report"]["error"] or ""))
    chk("phases still derived", len(d["phases"]), 8)
    chk("phase 3 gate still visible",
        [p for p in d["phases"] if p["n"] == 3][0]["passed"], False)

    print("\nhtml: the stylesheet must actually parse")
    html = urllib.request.urlopen(base + "/", timeout=8).read().decode()
    chk("no doubled braces", "{{" not in html and "}}" not in html)
    chk("style tag present", "<style>" in html)
    # a doubled brace anywhere inside <style> invalidates the whole sheet, and
    # the page silently renders unstyled -- which is exactly what happened
    style = html.split("<style>")[1].split("</style>")[0] if "<style>" in html else ""
    chk("style block non-empty", len(style) > 200)
    chk("css vars intact", "--acc:#e8924c" in style)
    chk("no f-string escape residue", "{{" not in style)

    print("\nOpen Colab must not depend on the server")
    # webbrowser.open() from an HTTP handler thread can fail silently on
    # Windows, so the button is a real <a href>. Assert that, not the endpoint.
    chk("colab is an anchor", 'href="https://colab.research.google.com/"' in html)
    chk("anchor opens in new tab", 'target="_blank"' in html)
    chk("anchor styled like a button", "a.btn {" in style)
    chk("no stale go('/api/open') button", "go('/api/open')" not in html)

    print("\npage text mentions the current notebook")
    chk("references rsl_all", "rsl_all.ipynb" in html)
    chk("no old notebook name", "rsl_run.ipynb" not in html)
    chk("phase table present", 'id="phases"' in html)

    print("\n/api/build is wired (button 1 must not 404)")
    code = None
    try:
        urllib.request.urlopen(base + "/api/build", timeout=20)
        code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    chk("build endpoint responds", code, 200)

finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    wipe()

print(f"\n{len(fails)} failure(s)")
for f in fails:
    print("  FAIL", f)
raise SystemExit(1 if fails else 0)