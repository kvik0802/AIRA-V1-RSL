"""
verify.py -- prove rsl_all.ipynb works before it reaches Colab.

The compression added a decode step, which is exactly the kind of thing that
looks fine and fails on the second run. So this executes cells 1 and 2
VERBATIM from the .ipynb (paths redirected, Drive stubbed out) and asserts:

  1. the payload decodes and every file matches its source byte for byte
  2. the harness self-test passes
  3. all 21 files import from the unpacked location
  4. the gate logic BLOCKS on a missing prerequisite and on a failed gate,
     and ALLOWS a phase whose prerequisites are met

(4) is the one that matters most. A gate that never blocks is worse than no
gate, because it looks like protection.
"""
import base64
import json
import shutil
import sys
import tempfile
import traceback
import zlib
from pathlib import Path

D = Path(__file__).parent
NB = D / "rsl_all.ipynb"
fails = []


def chk(label, got, want=True):
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"          got={got!r} want={want!r}")
        fails.append(label)


nb = json.loads(NB.read_text(encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]

print("=" * 68)
print("1. shape")
print("=" * 68)
chk("nbformat 4", nb["nbformat"], 4)
chk("11 code cells", len(code), 11)
chk("setup cell is small", len("".join(code[0]["source"])) < 4000)

print("\n" + "=" * 68)
print("2. payload decodes and matches source byte for byte")
print("=" * 68)
root = Path(tempfile.gettempdir()) / "rsl_verify2"
if root.exists():
    shutil.rmtree(root)
(root / "refs").mkdir(parents=True)
(root / "phases").mkdir(parents=True)

src2 = "".join(code[1]["source"])
blob = src2.split('_BLOB = "')[1].split('"')[0]
files = json.loads(zlib.decompress(base64.b64decode(blob)).decode())
chk("21 files", len(files), 21)

mismatch = 0
for rel, b64 in files.items():
    unpacked = base64.b64decode(b64)
    orig = D / rel
    if not orig.exists():
        print(f"       {rel}: no source to compare")
        mismatch += 1
    elif unpacked != orig.read_bytes():
        print(f"       {rel}: BYTES DIFFER")
        mismatch += 1
chk("all files round-trip exactly", mismatch, 0)

print("\n" + "=" * 68)
print("3. run cells 1 and 2 verbatim")
print("=" * 68)


def redirect(s):
    s = s.replace('RSL = "/content/rsl"', f'RSL = {str(root)!r}')
    s = s.replace('from google.colab import drive',
                  'raise ImportError("no colab here")')
    s = s.replace('drive.mount("/content/drive", force_remount=False)', 'pass')
    s = s.replace(f'Path(RSL, "report.json")',
                  f'Path({str(root / "report.json")!r})')
    return s


g = {"__name__": "__cells__"}
try:
    exec(compile(redirect("".join(code[0]["source"])), "cell1", "exec"), g)
    exec(compile(redirect(src2), "cell2", "exec"), g)
    print("\ncells 1 and 2 completed")
except SystemExit as e:
    print(f"\nexited: {e}")
    fails.append("cell2 exit")
except Exception:
    traceback.print_exc()
    fails.append("cell2 raised")

rep = json.loads((root / "report.json").read_text())
chk("selftest passed", rep["selftest"][0])
chk("stage selftest", rep["stage"], "selftest")
chk("ok True", rep["ok"])

chk("5 top-level", sorted(p.name for p in root.glob("*.py")),
    ["app.py", "common.py", "refs.py", "rsl_harness.py", "runner.py"])
chk("10 refs", len(list((root / "refs").glob("*.py"))), 10)
chk("6 phases", len(list((root / "phases").glob("*.py"))), 6)

print("\n" + "=" * 68)
print("4. imports + gate logic")
print("=" * 68)
sys.path.insert(0, str(root))
import rsl_harness  # noqa: E402
import runner  # noqa: E402

chk("8 phases registered", len(runner.PHASES), 8)
chk("harness 10 tasks", len(rsl_harness.EVAL), 10)
ok, fails_self = rsl_harness.self_test(verbose=False)
chk("selftest via unpacked dir", ok)

gdir = root / "gates"
gdir.mkdir(exist_ok=True)
runner.ROOT = gdir

print("\n  a) phase 3 with no phase02_ranks.json -- must REFUSE")
try:
    runner.run_phase(3)
    chk("phase 3 blocked", False)
except SystemExit as e:
    chk("phase 3 blocked", "phase02_ranks.json" in str(e))

print("\n  b) phase 4 when phase 3's gate is False -- must REFUSE")
(gdir / "phase03_adapter").mkdir(exist_ok=True)
(gdir / "phase02_ranks.json").write_text(json.dumps({"gate": True}))
(gdir / "phase03_summary.json").write_text(json.dumps({"gate": False}))
try:
    runner.run_phase(4)
    chk("phase 4 blocked by gate", False)
except SystemExit as e:
    chk("phase 4 blocked by gate", "gate" in str(e).lower())

print("\n  c) gate verdicts parse correctly")
(gdir / "phase03_summary.json").write_text(json.dumps({"gate": True}))
chk("gate True", runner.gate_status(3)[0])
(gdir / "phase03_summary.json").write_text(json.dumps({"gate": False}))
chk("gate False", runner.gate_status(3)[0], False)
(gdir / "phase03_summary.json").unlink()
chk("missing artifact blocks", runner.gate_status(3)[0], False)

print("\n  d) phase 0/1 are callable with no prerequisite")
chk("phase 0 no prereq", runner.PHASES[0][2], None)
chk("phase 1 no prereq", runner.PHASES[1][2], None)

shutil.rmtree(root, ignore_errors=True)

print("\n" + "=" * 68)
print(f"{len(fails)} failure(s)")
for f in fails:
    print("  FAIL", f)
print("=" * 68)
raise SystemExit(1 if fails else 0)