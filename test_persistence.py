"""Two fixes that need regression tests.

1. Drive detection. Phase 0 wrote results to /content/rsl/out -- session-local,
   silently lost on disconnect -- because out_dir() keyed off a marker file
   that was not present. Anything that reports a non-persistent path must
   SAY SO, loudly, or the loss is invisible until it happens.

2. Error visibility. run_tests used a fixed-size tail slice, so failures read
   as a bare "Traceback" with the cause cut off. The exception line must always
   be in the message.
"""
import sys
from pathlib import Path

D = Path(__file__).parent
sys.path.insert(0, str(D))

import common  # noqa: E402
from rsl_harness import run_tests  # noqa: E402

fails = []


def chk(label, got, want=True):
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"          got={got!r} want={want!r}")
        fails.append(label)


print("=" * 68)
print("1. drive detection")
print("=" * 68)
# on this machine google.colab is absent, so we must NOT claim Drive
chk("not in colab here", common.in_colab(), False)
chk("no drive claimed locally", common.drive_root(), None)
chk("falls back to ./out", common.out_dir().name, "out")

warn = common.report_warning()
chk("warns when not persisting", "LOST" in warn and "Drive" in warn)

print("\n  simulating a mounted Drive:")
common.drive_root = lambda: Path("/content/drive/MyDrive")
chk("reports a warning when mounted", common.report_warning(), "")
common.drive_root = lambda: None
chk("warning returns when absent", "LOST" in common.report_warning())

print("\n" + "=" * 68)
print("2. failures must show their cause")
print("=" * 68)

CASES = [
    ("attribute error on None",
     "def f(x):\n    return x.v\n",
     "from sol import f\ndef t():\n    assert f(None) is None\nt()\n",
     "AttributeError"),
    ("wrong number of args",
     "def f(a):\n    return a\n",
     "from sol import f\ndef t():\n    assert f(1,2)==1\nt()\n",
     "TypeError"),
    ("assertion failure",
     "def f(a):\n    return a + 1\n",
     "from sol import f\ndef t():\n    assert f(1)==3\nt()\n",   # 1+1 is 2, not 3
     "AssertionError"),
    ("import error in model code",
     "import nonexistent_module_xyz\n",
     "import nonexistent_module_xyz\ndef t():\n    assert True\nt()\n",
     "ModuleNotFoundError"),
]

for label, code, tests, want in CASES:
    ok, err = run_tests(code, tests)
    chk(f"{label}: scored fail", ok, False)
    chk(f"{label}: message names the cause", want in err, True)

print("\n  a genuinely long traceback must still carry the exception:")
long_tests = ("from sol import f\ndef t():\n"
              + "".join(f"    _ = f({i})\n" for i in range(30))
              + "    assert False, 'boom'\nt()\n")
ok, err = run_tests("def f(a):\n    return a\n", long_tests)
chk("still fails", ok, False)
chk("still shows AssertionError", "AssertionError" in err, True)
chk("error is bounded", len(err) <= 700, True)

print("\n  passing code must not be reported as an error:")
ok, err = run_tests("def f(a):\n    return a + 1\n",
                    "from sol import f\ndef t():\n    assert f(1)==2\nt()\n")
chk("passes", ok, True)
chk("no error text", err, "ok")

print("\n" + "=" * 68)
print(f"{len(fails)} failure(s)")
for f in fails:
    print("  FAIL", f)
print("=" * 68)
raise SystemExit(1 if fails else 0)