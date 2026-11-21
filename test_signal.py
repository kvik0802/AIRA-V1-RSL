"""
test_signal.py -- is the reward signal actually informative?

audit.py and verify.py prove the harness is self-consistent. They cannot prove
it would rank a good model above a bad one, because both stubs are fake.

This scores the SAME reference solutions in three states:
    correct            -> must score 10/10
    subtly corrupted   -> must score meaningfully below 10
    prose-only refusal -> must score 0/10

The middle case is the one worth having. If small logic bugs still scored 9/10,
the whole self-learning loop would train on almost-noise: 120 verified rows
would mean120 rows that were merely not-catastrophic. So we also require that
corruption is actually DETECTED, not just scored lower.
"""
import sys
from pathlib import Path

D = Path(__file__).parent
sys.path.insert(0, str(D))

from rsl_harness import EVAL, INSTR, extract_code, run_suite  # noqa: E402
from refs import gen_refs  # noqa: E402

REFS = gen_refs()
fails = []


def chk(label, got, want=True):
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"          got={got!r} want={want!r}")
        fails.append(label)


def stub_with(mangle):
    def g(prompt):
        for name, stub, _t in EVAL:
            if stub in prompt:
                return "```python\n" + mangle(REFS[name], name) + "```"
        return "```python\npass\n```"
    return g


def identity(src, name):
    return src


def corrupt(src, name):
    """One plausible logic bug per task. Not syntax errors -- logic errors,
    which is what a real model actually produces.

    Every mutation is verified to actually change the source. An earlier
    version had three that never matched, so they scored 10/10 for no reason
    and read as harness misses."""
    out = {
        # reverse: drop the head node -> list comes back one short.
        "reverse a linked list":
            src.replace("    vals = []\n    while h:",
                        "    vals = []\n    h = h.n if h else None\n    while h:"),
        # merge: forget to advance b -> infinite-ish / wrong
        "merge two sorted lists": src.replace("v, b = b.v, b.n",
                                              "v = b.v"),
        # parens: never pop -> mismatched nesting slips through
        "balanced parens": src.replace(
            "if not stack or stack.pop() != PAIRS[c]:",
            "if not stack:\n                return False"),
        # lcp: compare case-insensitively -> 'Flower'/'flow' share 'Flo' not 'fl'
        "longest common prefix": src.replace(
            "if len(set(chars)) == 1:",
            "if len({c.lower() for c in chars}) == 1:"),
        # lru: never evict
        "LRU cache O(1)": src.replace("if len(s.c) > s.cap:",
                                      "if False:"),
        # mpow: off-by-one on exponent
        "matrix exponentiation": src.replace("while n:", "while n - 1:"),
        # islands: 8-connectivity instead of 4 -> merges diagonal neighbours
        "count islands in a grid":
            src.replace("((1, 0), (-1, 0), (0, 1), (0, -1))",
                        "((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), "
                        "(-1, 1), (-1, -1))"),
        # json: return None for nested objects
        "parse JSON literal": src.replace("return ast.literal_eval(s.strip())",
                                          "return None"),
        # kth: return max instead of min of heap
        "kth largest in a stream": src.replace("return s.h[0]",
                                               "return max(s.h)"),
        # rle: no multi-digit support
        "run-length encode + decode":
            src.replace("while i < len(s) and s[i].isdigit():",
                        "while i + 1 < len(s) and s[i].isdigit():"),
    }
    return out.get(name, src)


def refuse(prompt):
    return "I'm sorry, but I can't help with that."


print("=" * 68)
print("1. correct solutions must score 10/10")
print("=" * 68)
perfect, rows = run_suite(stub_with(identity), "perfect", verbose=False)
chk("perfect score", perfect, 10)
for r in rows:
    if not r["passed"]:
        print(f"       {r['task']}: {r['err'][:80]}")

print("\n" + "=" * 68)
print("2. one logic bug each must be DETECTED")
print("=" * 68)
corrupted, rows = run_suite(stub_with(corrupt), "corrupt", verbose=False)
detected = {r["task"] for r in rows if not r["passed"]}
print(f"  corrupted scored {corrupted}/10, detected {len(detected)}/10")

missed = [r["task"] for r in rows if r["passed"]]
if missed:
    print(f"       NOT detected: {missed}")

# a mutation that does not change the source proves nothing, so verify each
# one actually mutates. This is what caught three no-op mutations earlier.
noops = [n for n in REFS if corrupt(REFS[n], n) == REFS[n]]
chk("every mutation changes its source", noops, [])

# the key assertion: corruption must not slip through as near-perfect
chk("corrupted scores at most 7", corrupted <= 7)
chk("at least 8 of 10 mutations caught", len(detected) >= 8)

print("\n" + "=" * 68)
print("3. refusal scores 0/10")
print("=" * 68)
z, _ = run_suite(refuse, "refuse", verbose=False)
chk("refusal scores 0", z, 0)

print("\n" + "=" * 68)
print("4. the signal separates good from bad")
print("=" * 68)
print(f"       perfect   {perfect}/10")
print(f"       corrupted {corrupted}/10")
print(f"       refusal   {z}/10")
# Corrupted and refusal may both be 0 -- a wrong answer and no answer at all
# are equally unusable as training data. What must hold is that correct code
# scores strictly highest and the others are far below it.
chk("perfect is strictly best", perfect > corrupted and perfect > z)
chk("corrupted far below perfect", corrupted <= 7)
chk("refusal is zero", z, 0)

print(f"\n{'='*68}")
print(f"{len(fails)} failure(s)")
for f in fails:
    print("  FAIL", f)
raise SystemExit(1 if fails else 0)