"""
Audit.py -- find bugs that only fire at runtime, after GPU hours.

pyflakes catches undefined names and unused imports. It cannot catch:
  * calling a function with the wrong number of arguments
  * referencing a dict key that is never written
  * referencing an artifact a phase never produces

Those are the bugs that waste a Colab session, so they get checked here
statically, with no model and no GPU.

    python audit.py
"""

import ast
import inspect
import sys
from pathlib import Path

D = Path(__file__).parent
sys.path.insert(0, str(D))

issues = []


def bad(sev, where, msg):
    issues.append((sev, where, msg))
    print(f"  [{sev}] {where}: {msg}")


def ok(msg):
    print(f"  [ok  ] {msg}")


# ---------------------------------------------------------------- 1. arity

print("=" * 68)
print("1. call arity against the real signatures")
print("=" * 68)

import common  # noqa: E402

SIGS = {}
for mod in (common,):
    for name, obj in vars(mod).items():
        if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
            try:
                SIGS[f"{mod.__name__}.{name}"] = inspect.signature(obj)
            except (ValueError, TypeError):
                pass

import runner  # noqa: E402
for name, obj in vars(runner).items():
    if inspect.isfunction(obj) and obj.__name__ in ("run_phase", "gate_status",
                                                    "phases_table"):
        SIGS[f"runner.{name}"] = inspect.signature(obj)
import app as _app  # noqa: E402
for name in ("phase0", "phase1"):
    SIGS[f"app.{name}"] = inspect.signature(getattr(_app, name))

print(f"  {len(SIGS)} signatures collected")

for p in sorted((D / "phases").glob("*.py")) + [D / "app.py"]:
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name):
            key = f.id
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            key = f"{f.value.id}.{f.attr}"
        else:
            continue
        if key not in SIGS:
            continue
        sig = SIGS[key]
        params = list(sig.parameters.values())
        if any(pp.kind == pp.VAR_POSITIONAL or pp.kind == pp.VAR_KEYWORD
               for pp in params):
            continue
        required = [pp for pp in params
                    if pp.default is pp.empty
                    and pp.kind != pp.KEYWORD_ONLY]
        npos = len([a for a in node.args])
        nkw = len([k for k in node.keywords if k.arg])
        has_star = any(isinstance(a, ast.Starred) for a in node.args)
        has_dstar = any(k.arg is None for k in node.keywords)
        if has_star or has_dstar:
            continue
        supplied = npos + nkw
        if supplied < len(required):
            bad("BUG", f"{p.name}:{node.lineno}",
                f"{key}() needs {len(required)} args, given {supplied}")
        elif supplied > len(params):
            bad("BUG", f"{p.name}:{node.lineno}",
                f"{key}() takes at most {len(params)}, given {supplied}")
        else:
            ok(f"{p.name}:{node.lineno} {key}() arity {supplied}")

# ---------------------------------------------------------------- 2. artifacts

print("\n" + "=" * 68)
print("2. artifact producer/consumer closure")
print("=" * 68)

PRODUCED = set()
for p in list((D / "phases").glob("*.py")) + [D / "app.py"]:
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # resolve NAME -> literal so `ADAPTER = ROOT / "phase03_adapter"` counts.
    # The earlier literal-only scan reported phase03_adapter as unwritten,
    # which was wrong: phase 3 writes it through exactly that variable.
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            val = node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                alias[tgt.id] = val.value
            elif isinstance(val, ast.BinOp) and isinstance(val.op, ast.Div):
                for side in (val.left, val.right):
                    if isinstance(side, ast.Constant) and isinstance(
                            side.value, str) and "/" not in side.value:
                        alias[tgt.id] = side.value

    WRITERS = ("save_json", "write_text", "save_pretrained", "save",
               "mkdir", "torch")

    def lit(arg):
        """Unwrap Name / str(Name) / Path(Name) to a literal filename.

        Adapters are saved as save_pretrained(str(ADAPTER)), so a walker that
        only accepts bare Names misses every adapter artifact and reports it as
        never written -- a false BUG that would send someone hunting for a
        phase that does not exist.
        """
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.Name):
            return alias.get(arg.id)
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) \
                and arg.func.id in ("str", "Path") and arg.args:
            return lit(arg.args[0])
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        nm = f.attr if isinstance(f, ast.Attribute) else (
            f.id if isinstance(f, ast.Name) else "")
        if nm not in WRITERS:
            continue
        for a in node.args:
            v = lit(a)
            if isinstance(v, str):
                PRODUCED.add(Path(v).name)

CONSUMED = {
    3: ["phase02_ranks.json"],
    4: ["phase02_ranks.json", "phase03_adapter", "phase03_seed_tasks.json"],
    6: ["phase03_adapter", "phase03_seed_tasks.json", "phase05_archive.json"],
    7: ["phase03_adapter", "refusal_directions.pt"],
}
# things produced by exec'd code or by app.py's phase fns
KNOWN = PRODUCED | {"refusal_directions.pt", "phase05_archive.json",
                    "phase0_summary.json", "report.json"}

for ph, needs in CONSUMED.items():
    for n in needs:
        if n in KNOWN:
            ok(f"phase {ph} <- {n}  (producer found)")
        else:
            bad("BUG", f"phase {ph}",
                f"consumes {n} but no phase writes it")

# ---------------------------------------------------------------- 3. gates

print("\n" + "=" * 68)
print("3. every phase writes the summary its gate reads")
print("=" * 68)

for n in range(8):
    fn = runner.SUMMARY[n]
    src = None
    if n in (0, 1):
        src = (D / "app.py").read_text(encoding="utf-8")
    else:
        f = D / "phases" / f"phase{n:02d}_{runner.PHASES[n][0].split()[0].lower()}.py"
        cands = list((D / "phases").glob(f"phase{n:02d}_*.py"))
        src = cands[0].read_text(encoding="utf-8") if cands else ""
    if fn in src:
        ok(f"phase {n} writes {fn}")
    else:
        bad("BUG", f"phase {n}",
            f"gate reads {fn} but the phase never writes it")

# ---------------------------------------------------------------- summary

print("\n" + "=" * 68)
bugs = [i for i in issues if i[0] == "BUG"]
print(f"{len(bugs)} BUG, {len(issues)-len(bugs)} note(s)")
print("=" * 68)
for sev, where, msg in issues:
    if sev == "BUG":
        print(f"  {where}: {msg}")
raise SystemExit(1 if bugs else 0)