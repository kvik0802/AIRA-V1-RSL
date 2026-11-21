"""
Reference solutions for the eval set, as REAL code in real files.

Why: embedding 10 multi-line programs as escaped \\n strings inside a dict
produced silent IndentationErrors and logic bugs I could only see by running
them. Here each is a normal .py file that can be read, edited and executed
directly. gen_refs() reads the directory and returns {task_name: source}.

If a reference solution fails its own test, the eval set is broken -- not the
model -- so self_test() checks all 10 before anything is downloaded.
"""

from pathlib import Path

REFS = Path(__file__).parent / "refs"

# task name -> filename
FILES = {
    "reverse a linked list": "01_reverse.py",
    "merge two sorted lists": "02_merge.py",
    "balanced parens": "03_parens.py",
    "longest common prefix": "04_lcp.py",
    "LRU cache O(1)": "05_lru.py",
    "matrix exponentiation": "06_mpow.py",
    "count islands in a grid": "07_islands.py",
    "parse JSON literal": "08_json.py",
    "kth largest in a stream": "09_kth.py",
    "run-length encode + decode": "10_rle.py",
}


def gen_refs():
    out = {}
    missing = []
    for name, fn in FILES.items():
        p = REFS / fn
        if not p.exists():
            missing.append(fn)
            continue
        out[name] = p.read_text(encoding="utf-8")
    if missing:
        raise FileNotFoundError(
            "missing reference solution file(s): " + ", ".join(missing))
    return out