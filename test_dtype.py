"""dtype_kwargs must pick the right kwarg name for the INSTALLED transformers.

Getting this wrong is a silent trap both ways:
  dtype=      only consumed by transformers >= 4.56; older passes it to
              Model.__init__ -> "unexpected keyword argument 'dtype'"
  torch_dtype= still accepted everywhere but deprecated on new versions

So decide from the version string rather than guessing. I guessed wrong once and
it cost a Colab run.
"""
import sys
from pathlib import Path

D = Path(__file__).parent
sys.path.insert(0, str(D))
import common  # noqa: E402

fails = []


def chk(label, got, want=True):
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"          got={got!r} want={want!r}")
        fails.append(label)


print("=" * 66)
print("dtype_kwargs picks the right name per version")
print("=" * 66)

CASES = [
    ("4.48.3", "torch_dtype"),   # pinned era: dtype= would raise
    ("4.55.0", "torch_dtype"),
    ("4.56.0", "dtype"),         # dtype= introduced
    ("4.57.6", "dtype"),
    ("5.0.0", "dtype"),
]

import types  # noqa: E402
for ver, want in CASES:
    common._TFORMERS_VERSION = ver          # what the helper actually reads
    import torch
    got = next(iter(common.dtype_kwargs(torch)))
    chk(f"transformers {ver} -> {want}", got, want)

print("\n" + "=" * 66)
print("installed version is handled")
print("=" * 66)
import torch  # noqa: E402
common._TFORMERS_VERSION = None            # force a real read
got = next(iter(common.dtype_kwargs(torch)))
import transformers  # noqa: E402
print(f"  local transformers {transformers.__version__} -> {got}")
chk("returns exactly one kwarg", got in ("dtype", "torch_dtype"))

print("\n" + "=" * 66)
print("no file still hardcodes a dtype kwarg")
print("=" * 66)
for name in ("app.py", "common.py"):
    src = (D / name).read_text(encoding="utf-8")
    has = "from_pretrained(" in src
    uses_helper = "dtype_kwargs(" in src
    chk(f"{name} uses dtype_kwargs", uses_helper, True)
    # a bare torch_dtype=/dtype= directly in a from_pretrained call
    bad = []
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if "torch_dtype=torch.float16)" in ln or ", dtype=torch.float16" in ln:
            bad.append(i + 1)
    chk(f"{name} has no hardcoded load-dtype kwarg", bad, [])

print(f"\n{len(fails)} failure(s)")
for f in fails:
    print("  FAIL", f)
raise SystemExit(1 if fails else 0)