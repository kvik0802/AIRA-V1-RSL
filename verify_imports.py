"""Verify every `from common import X` names something common.py actually defines.

Catches the classic failure: rename a helper in common.py, forget the phase,
and the notebook dies 40 minutes into a Colab run.
"""
import ast
import pathlib
import sys

D = pathlib.Path(__file__).parent

# Every phase is now a plain exec-able module under phases/, so nothing needs
# skipping. Nothing here starts with `!pip`, so ast can parse all of it.

src = ast.parse((D / "common.py").read_text(encoding="utf-8"))

defined = set()
for n in ast.walk(src):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        defined.add(n.name)
    elif isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name):
                defined.add(t.id)
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
            defined.add(a.asname or a.name.split(".")[0])

missing = 0
phase_files = sorted((D / "phases").glob("*.py")) if (D / "phases").is_dir() \
    else sorted(D.glob("phase*.py"))
if not phase_files:
    print("  WARNING: no phase files found")

for p in phase_files:
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "common":
            for a in n.names:
                if a.name not in defined:
                    print(f"  MISSING  {p.name}: {a.name}")
                    missing += 1
                else:
                    print(f"  ok       {p.name}: {a.name}")

# reverse: helpers defined in a phase but never imported anywhere are dead
print(f"\ncommon.py exports {len(defined)} names; {missing} missing references")

# check cross-phase file dependencies (phase4 reads phase3's seed tasks, etc.)
EXPECT = {
    "phase02_fisher.py": [],
    "phase03_qlora.py": ["phase02_ranks.json"],
    "phase04_elastic_moe.py": ["phase02_ranks.json", "phase03_adapter",
                               "phase03_seed_tasks.json"],
    "phase05_harvester.py": [],
    "phase06_selfplay_rsl.py": ["phase03_adapter", "phase03_seed_tasks.json",
                                "phase05_archive.json", "phase06_harness.json"],
    "phase07_finalize.py": ["phase03_adapter", "refusal_directions.pt"],
}
PRODUCERS = {"phase02_ranks.json": "phase02", "phase03_adapter": "phase03",
             "phase03_seed_tasks.json": "phase03", "phase05_archive.json": "phase05",
             "phase06_harness.json": "phase06", "refusal_directions.pt": "phase01"}

print("\nartifact chain:")
for f, needs in EXPECT.items():
    pn = f.replace(".py", "").replace("phase0", "phase 0")
    for n in needs:
        who = PRODUCERS[n]
        print(f"  {pn:26s} needs {n:26s} <- {who}")

sys.exit(1 if missing else 0)