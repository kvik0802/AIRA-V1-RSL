"""
runner.py -- drives all 7 phases from one notebook.

Two jobs:

1. Model/process reuse. The 7B base is loaded ONCE and shared across phases
   that need it. Re-downloading or reloading seven times wastes ~15 min and,
   on a 12h Colab cap, is the difference between finishing and not.

2. Gating. Each phase declares what it needs and what artifact proves the
   previous phase passed. run_phase() refuses to continue on a failed gate and
   says which gate failed and why.

   The gate check matters more than it looks: phases 4-7 are multi-hour. Running
   phase 6 after phase 5 harvested nothing wastes a day and, worse, produces a
   number that looks like a result.
"""

import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# number -> (label, how to run it, prerequisite artifact, gate key in that file)
PHASES = {
    0: ("baseline", "app:phase0", None, None),
    1: ("abliteration", "app:phase1", None, "verdict"),
    2: ("fisher rank table", "file:phases/phase02_fisher.py", None, "gate"),
    3: ("QLoRA reference", "file:phases/phase03_qlora.py",
        "phase02_ranks.json", "gate"),
    4: ("elastic MoE", "file:phases/phase04_elastic_moe.py",
        "phase03_adapter", "gate"),
    5: ("autonomous harvester", "file:phases/phase05_harvester.py",
        "phase03_seed_tasks.json", "gate"),
    6: ("self-play + RSL", "file:phases/phase06_selfplay_rsl.py",
        "phase05_archive.json", "gate"),
    7: ("finalize", "file:phases/phase07_finalize.py",
        "phase03_adapter", None),
}

SUMMARY = {
    0: "phase0_summary.json",
    1: "phase1_summary.json",
    2: "phase02_ranks.json",
    3: "phase03_summary.json",
    4: "phase04_summary.json",
    5: "phase05_summary.json",
    6: "phase06_summary.json",
    7: "phase07_summary.json",
}

import app as _app  # noqa: E402
from common import ROOT  # noqa: E402


def _load(num):
    p = ROOT / SUMMARY[num]
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def gate_status(num):
    """(passed, detail). Missing artifact counts as not-passed, because a
    phase that produced nothing has not earned the next one."""
    label, how, needs, key = PHASES[num]
    if needs is None and key is None:
        return True, "no gate"
    s = _load(num)
    if not s:
        return False, f"phase {num} produced no {SUMMARY[num]}"
    if key and key in s:
        ok = bool(s[key])
        return ok, f"{key}={s[key]}"
    if num == 7:
        # phase 7 reports two gates
        c, r = s.get("gate_code"), s.get("gate_refusal")
        if c is None and r is None:
            return False, "phase 7 wrote no gate fields"
        return (bool(c) and bool(r)), f"gate_code={c} gate_refusal={r}"
    return True, "ran"


def phases_table():
    rows = []
    for num in sorted(PHASES):
        label, how, needs, key = PHASES[num]
        ran = (ROOT / SUMMARY[num]).exists()
        ok, detail = gate_status(num)
        rows.append({"n": num, "label": label, "ran": ran,
                     "passed": ok, "detail": detail,
                     "needs": needs or "-"})
    return rows


def print_table():
    print(f"{'#':<3}{'phase':<24}{'ran':<6}{'gate':<26}detail")
    print("-" * 92)
    for r in phases_table():
        print(f"{r['n']:<3}{r['label']:<24}"
              f"{'yes' if r['ran'] else '-':<6}"
              f"{('PASS' if r['passed'] else 'FAIL') if r['ran'] else '-':<26}"
              f"{r['detail']}")


def run_phase(num, report=None):
    """Run one phase. Returns True on success.

    On a failed prerequisite or a failed gate it refuses and says why rather
    than burning hours producing a meaningless number.
    """
    if num not in PHASES:
        raise SystemExit(f"no such phase {num}")

    label, how, needs, _key = PHASES[num]

    # prerequisite artifact from an EARLIER phase
    if needs:
        if not (ROOT / needs).exists():
            msg = (f"phase {num} ({label}) needs {needs}, which does not exist."
                   f"\nRun the earlier phase first. Refusing to start.")
            print(msg)
            if report is not None:
                report.update(stage=f"phase{num}", ok=False,
                              error=f"missing prerequisite {needs}")
            raise SystemExit(msg)

    # gate from the immediately preceding phase
    if num > 0:
        ok, detail = gate_status(num - 1)
        if not ok:
            msg = (f"phase {num - 1} did not pass its gate ({detail}).\n"
                   f"Phase {num} ({label}) is skipped: continuing would spend "
                   f"hours to produce a number that means nothing.\n"
                   f"Fix phase {num - 1} first -- fix its cause, do not relax "
                   f"its threshold.")
            print(msg)
            if report is not None:
                report.update(stage=f"phase{num}", ok=False,
                              error=f"blocked: phase {num-1} {detail}")
            raise SystemExit(msg)

    print("\n" + "#" * 70)
    print(f"# PHASE {num} -- {label}")
    print("#" * 70)
    t0 = time.time()

    if report is not None:
        report["stage"] = f"phase{num}"
        report["ok"] = None

    try:
        if how.startswith("app:"):
            getattr(_app, how.split(":", 1)[1])()
        else:
            path = HERE / how.split(":", 1)[1]
            if not path.exists():
                raise SystemExit(f"missing phase source {path}")
            g = {"__name__": "__main__", "__file__": str(path)}
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), g)
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        if report is not None:
            report.update(stage=f"phase{num}", ok=False,
                          error=f"{label} raised", tb=tb)
        raise

    secs = round(time.time() - t0, 1)
    ok, detail = gate_status(num)
    print(f"\n-- phase {num} finished in {secs}s -- gate: "
          f"{'PASS' if ok else 'FAIL'} ({detail})")

    if report is not None:
        report["steps"].append({"name": f"phase{num}_{label.replace(' ', '_')}",
                                "secs": secs, "gate": ok, "detail": detail})
        report["ok"] = ok
        report["results"] = _load(num)
        rep = ROOT / "report.json"
        try:
            rep.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception:
            pass
    return ok


def all_done():
    rows = phases_table()
    return all(r["ran"] for r in rows)