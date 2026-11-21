"""
build.py -- generate ONE self-contained Colab notebook containing all 8 phases.

    python build.py

Output: rsl_all.ipynb -- the only file uploaded. Every source file is embedded
as base64, so nothing else has to be attached.

WHY THE PAYLOAD IS SPLIT ACROSS TWO CELLS
------------------------------------------
The first version put all 21 files into a single 157 KB cell. It worked, but a
cell that large is fragile: Colab can truncate or corrupt it, and the failure
mode is a confusing error with no partial recovery. Splitting it means:

  cell 1  imports, dirs, Drive mount, report scaffolding   (small, readable)
  cell 2  the payload + unpack                             (one job, obvious)

If cell 2 is ever truncated, cell 1 still ran and the failure names the file
list instead of looking like the whole notebook being broken.

DESIGN RULE, learned three times the hard way:
  Nothing is measured until the harness has proved itself. self_test() scores
  10/10 against reference solutions and 0/10 against logic-bugged ones, with a
  stub model and zero GPU seconds. A broken harness is indistinguishable from a
  weak model, so it must be impossible to spend GPU time on one.
"""

import base64
import json
import zlib
from pathlib import Path

D = Path(__file__).parent

TOP_FILES = ["rsl_harness.py", "refs.py", "common.py", "app.py", "runner.py"]
REFS_DIR = D / "refs"
PHASES_DIR = D / "phases"
OUT = D / "rsl_all.ipynb"

# --------------------------------------------------------------- cell 1

CELL_SETUP = '''\
# ============================================================================
# CELL 1 -- SETUP: dependencies, directories, Drive     (no model download)
# ============================================================================
import base64, json, os, subprocess, sys, traceback
from pathlib import Path

# --- dependencies ----------------------------------------------------------
# BUG THIS CELL HAD: the pip install was dropped when setup and payload were
# split into two cells. Colab ships no bitsandbytes, so 4-bit loading died with
# "requires bitsandbytes>=0.46.1" AFTER the 15GB config/tokenizer download.
# bitsandbytes is deliberately NOT pinned: the pin drifted below the minimum
# transformers now demands, and the version that matters is dictated by
# transformers, not by us.
print("installing dependencies...")
_rc = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "bitsandbytes>=0.46.1", "accelerate>=0.34", "peft>=0.13"],
    capture_output=True, text=True)
print(_rc.stdout[-400:] or "")
if _rc.returncode:
    print("pip stderr:", _rc.stderr[-800:])
else:
    import bitsandbytes, accelerate, peft
    print(f"bitsandbytes {bitsandbytes.__version__}")
    print(f"accelerate {accelerate.__version__}")
    print(f"peft {peft.__version__}")
import transformers
print(f"transformers {transformers.__version__}")

RSL = "/content/rsl"
for d in ("", "refs", "phases"):
    Path(RSL + "/" + d if d else RSL).mkdir(parents=True, exist_ok=True)

REPORT = {"stage": "setup", "ok": None, "error": None, "tb": None, "steps": []}

# Drive is best-effort. Pre-emption means results must persist, but a mount
# failure must not kill the run -- that would waste the whole session.
DRIVE_ROOT = None
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive/rsl_model")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"drive ok -> {DRIVE_ROOT}")
except Exception as e:
    print(f"WARNING: no Drive ({e}). Results live only in this session and "
          f"will be LOST if it ends.")


def step(name, **kw):
    REPORT["steps"].append(dict(name=name, **kw))
    print(f"[step] {name}", flush=True)


def flush():
    """Write the report locally always, and to Drive when available."""
    blob = json.dumps(REPORT, indent=2)
    try:
        Path(RSL, "report.json").write_text(blob)
    except Exception:
        pass
    if DRIVE_ROOT:
        try:
            (DRIVE_ROOT / "report.json").write_text(blob)
        except Exception as e:
            print(f"  (drive write failed: {e})")
    print(f"[report] stage={REPORT['stage']} ok={REPORT['ok']}", flush=True)


sys.path.insert(0, RSL)
print("\\ncell 1 ready")
'''

# --------------------------------------------------------------- cell 2

CELL_PAYLOAD = '''\
# ============================================================================
# CELL 2 -- UNPACK + SELFTEST     (~10s, no model, no download)
# All source files are embedded below as zlib+base64. Nothing else to upload.
# ============================================================================
import base64, json, sys, traceback, zlib
from pathlib import Path

RSL = "/content/rsl"

# zlib then base64, reversed here. Python source compresses ~4x, which is what
# keeps this cell a manageable size instead of the 155 KB of raw base64.
_BLOB = "__BLOB__"
_files = json.loads(zlib.decompress(base64.b64decode(_BLOB)).decode())
for _rel, _b64 in _files.items():
    _dest = Path(RSL) / _rel
    _dest.parent.mkdir(parents=True, exist_ok=True)
    _dest.write_bytes(base64.b64decode(_b64))
print(f"unpacked {len(_files)} files into {RSL}")
sys.path.insert(0, RSL)

REPORT["stage"] = "selftest"
step("selftest")
try:
    import rsl_harness
    passed, fails = rsl_harness.self_test(verbose=True)
    REPORT["selftest"] = [passed, fails]
    if not passed:
        REPORT.update(ok=False, error="harness self-test failed",
                      tb="\\n".join(fails))
        flush()
        raise SystemExit("HARNESS BROKEN -- do not download a model")
except SystemExit:
    flush()
    raise
except Exception:
    REPORT.update(ok=False, error="selftest raised",
                  tb=traceback.format_exc())
    flush()
    raise

REPORT["ok"] = True
flush()
print("\\nHARNESS VERIFIED. Run CELL 3, then the phase cells in order.")
'''

# --------------------------------------------------------------- cell 3

CELL_STATUS = '''\
# ============================================================================
# CELL 3 -- STATUS: which phases have run and whether their gates passed
# ============================================================================
from runner import print_table, phases_table, all_done

print_table()
print()
print("Run the phase cells below in order. Each refuses to start if the")
print("previous phase did not pass its gate.")
if all_done():
    print("\\nall phases have run.")
'''

PHASE_BLURB = {
    0: ("baseline",
        "Loads the 7B base in 4-bit and scores 10 tasks with real unit tests.\n"
        "**Expect 7-9/10.** This is the baseline every later phase is measured "
        "against, so it is not optional. ~15 min including the download."),
    1: ("abliteration",
        "Finds each layer's refusal direction from dual-use vs benign prompts and\n"
        "projects it out of the residual stream. **Expect 0% refusal, <=2 tasks "
        "lost.** ~5 min."),
    2: ("fisher rank table",
        "Calibration only, no training. Produces the per-layer LoRA rank table from\n"
        "the eFIM diagonal on LoRA-B. Uniform rank is a design flaw; this is the\n"
        "fix. Score LoRA-B only -- at init B=0 so dL/dA is exactly zero. ~15 min."),
    3: ("QLoRA reference",
        "**Mandatory.** The reference QLoRA run at fixed rank. Nothing after this\n"
        "is measurable without it. **2-4 hours.**"),
    4: ("elastic MoE",
        "Three elasticity mechanisms: MoE experts that grow and are pruned (Net2Net\n"
        "init, so growing costs no accuracy), rank reallocated mid-training, and\n"
        "per-parameter LR. **Several hours.**"),
    5: ("autonomous harvester",
        "arXiv -> clone -> **execute**. The information supply, and the thing that\n"
        "stops the loop plateauing. Keep only code that ran and beat the baseline.\n"
        "**Expect ~5% yield; that is normal, not a bug.** ~1 hour."),
    6: ("self-play + RSL",
        "The loop: the model writes tasks, execution decides, it trains on the\n"
        "winners, then rewrites its own harness. **Multi-hour.** Expect a plateau\n"
        "at 4-6 rounds; that is the correct outcome, not a failure."),
    7: ("finalize",
        "Re-abliterate (SFT reinstalls refusals), merge to fp16, write a model card\n"
        "containing only measured numbers. ~30 min."),
}


def phase_cell(n):
    return f'''\
# --- phase {n}: {PHASE_BLURB[n][0]} ---
from runner import run_phase
run_phase({n}, REPORT)
'''


def build_payload():
    """Per-file base64, then the whole dict zlib-compressed and base64'd again."""
    files = {}
    for f in TOP_FILES:
        p = D / f
        if not p.exists():
            raise SystemExit(f"missing {f}")
        files[f] = base64.b64encode(p.read_bytes()).decode()
    for p in sorted(REFS_DIR.glob("*.py")):
        files[f"refs/{p.name}"] = base64.b64encode(p.read_bytes()).decode()
    for p in sorted(PHASES_DIR.glob("*.py")):
        files[f"phases/{p.name}"] = base64.b64encode(p.read_bytes()).decode()
    raw = json.dumps(files).encode()
    blob = base64.b64encode(zlib.compress(raw, 9)).decode()
    return files, blob, len(raw)


def build():
    files, blob, raw_len = build_payload()
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# rsl-coder-7b -- all phases\n",
            "\n",
            "Self-contained. **Upload nothing else.**\n",
            "\n",
            "1. Run **cell 1** (setup) then **cell 2** (unpack + selftest)\n",
            "2. Wait for `HARNESS VERIFIED`\n",
            "3. Run **cell 3** (status)\n",
            "4. Run the phase cells in order\n",
            "\n",
            "**Do not use `Runtime -> Run all` for phases 3 and beyond.** It\n",
            "would start the 2-4 hour phases back to back. The gates stop a\n",
            "broken chain, but you would rather see each result before starting\n",
            "the next.\n",
            "\n",
            f"Embedded: {len(files)} files.\n",
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## Setup\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [CELL_SETUP]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Unpack + selftest\n", "\n",
            "Downloads **nothing**. Scores the harness against 10 reference\n",
            "solutions and 10 logic-bugged ones with a stub model. If this\n",
            "aborts, the reward signal is broken and a 15GB download would only\n",
            "prove it. ~10 seconds.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [CELL_PAYLOAD.replace("__BLOB__", blob)]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## Status\n", "\n", "Which phases have run and whether their gates passed.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [CELL_STATUS]},
    ]

    for n in range(8):
        title, body = PHASE_BLURB[n]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [
            f"## Phase {n} -- {title}\n", "\n", body + "\n"]})
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {}, "outputs": [],
                      "source": [f"# --- phase {n} ---\n", phase_cell(n)]})

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "name": "python3"},
                       "accelerator": "GPU"},
          "cells": cells}
    return nb, files, blob, raw_len


if __name__ == "__main__":
    nb, files, blob, raw_len = build()
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    json.loads(OUT.read_text(encoding="utf-8"))

    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    sizes = [len("".join(c["source"])) for c in code]
    print(f"wrote {OUT.name}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  {len(nb['cells'])} cells: {len(code)} code, "
          f"{len(nb['cells'])-len(code)} markdown")
    print(f"  payload: {len(files)} files, {raw_len//1024} KB raw "
          f"-> {len(blob)//1024} KB compressed "
          f"({raw_len/max(len(blob),1):.1f}x)")
    print(f"  code cell sizes: {', '.join(f'{s//1000}K' for s in sizes)}")

    big = max(sizes)
    print(f"  largest cell: {big//1000}K")
    if big >= 100000:
        raise SystemExit(f"a cell is still {big//1000}K; split or compress further")
    print(f"\nUpload THIS to Colab. Nothing else. One file.")