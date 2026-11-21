"""
Regression test for the phase 1 bug that reached Colab.

collect() used to call acts.clear() on the way out, which destroyed the
activations the caller was about to average. mu_dual came back as {} and phase 1
died with `KeyError: 0`.

The bug is invisible to static analysis -- every name exists, every call is
valid. It only shows up when collect() is actually called. So it gets tested by
calling it, against a fake model made of plain tensors.

No torch needed: the hook shape is all that matters.
"""
import ast
import sys
import types
from pathlib import Path

import numpy as np

D = Path(__file__).parent
fails = []


def chk(label, got, want=True):
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"          got={got!r} want={want!r}")
        fails.append(label)


print("=" * 68)
print("1. source: collect() must return, not clear")
print("=" * 68)
src = (D / "app.py").read_text(encoding="utf-8")
tree = ast.parse(src)

collect = phase1 = None
for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef):
        if n.name == "collect":
            collect = n
        if n.name == "phase1":
            phase1 = n

chk("collect() found", collect is not None)
chk("phase1() found", phase1 is not None)
if collect is None or phase1 is None:
    raise SystemExit("cannot continue")

cb = ast.unparse(collect)
p1 = ast.unparse(phase1)

chk("collect() returns something", any(
    isinstance(n, ast.Return) for n in ast.walk(collect)))
chk("collect() no longer clears a shared dict",
    "acts.clear" not in cb and "acts.clear" not in p1)
chk("phase1 assigns from collect()", "= collect(" in p1)
chk("phase1 checks layer count", "raise SystemExit" in p1)
chk("phase1 flags weak directions", "norm" in p1 and "weak" in p1)

print("\n" + "=" * 68)
print("2. behaviour: run collect() against a fake model")
print("=" * 68)


class FakeOut(tuple):
    pass


class FakeLayer:
    """Mimics one decoder layer well enough for the hook contract:
    register_forward_hook returns a handle with .remove(), and the forward
    returns a (hidden_states, ...) tuple shaped [batch][seq][dim]."""

    def __init__(self, dim, idx):
        self.dim = dim
        self.idx = idx
        self.hooks = []

    def register_forward_hook(self, fn):
        h = types.SimpleNamespace(remove=lambda: self.hooks.remove(fn))
        self.hooks.append(fn)
        return h

    def run(self):
        # numpy, because the hook indexes h[0, -1, :] -- a multi-axis index,
        # which lists and tuples reject. Real activations are torch tensors,
        # which support it.
        batch, seq = 1, 2
        hidden = np.full((batch, seq, self.dim), float(self.idx))
        out = (hidden, None)          # transformers returns a tuple
        for fn in self.hooks:
            fn(self, None, out)


class FakeModel:
    def __init__(self, n_layers=4, dim=3):
        self.model = types.SimpleNamespace(layers=[FakeLayer(dim, i)
                                                  for i in range(n_layers)])


NL, DIM = 4, 3
layers = FakeModel(NL, DIM).model.layers


def mk_hook(i, sink):
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        sink.setdefault(i, []).append(h[0, -1, :])
    return hook


def collect(prompts, layers=layers):
    sink = {}
    hs = [l.register_forward_hook(mk_hook(i, sink))
          for i, l in enumerate(layers)]
    try:
        for _ in prompts:
            for l in layers:
                l.run()
    finally:
        for h in hs:
            h.remove()
    return {i: np.mean(np.stack(v), axis=0) for i, v in sink.items()}


a = collect(["p1", "p2"])
chk("returns every layer", sorted(a), list(range(NL)))
chk("values are the means", float(a[0][0]), 0.0)
chk("last layer recorded", float(a[NL - 1][0]), float(NL - 1))

b = collect(["q1"])
chk("a second call is independent", sorted(b), list(range(NL)))
chk("first result not clobbered", float(a[NL - 1][0]), float(NL - 1))

print("\n  the exact failure that reached Colab:")
old_style = {}
collected = {}
hs = [l.register_forward_hook(mk_hook(i, collected)) for i, l in enumerate(layers)]
for l in layers:
    l.run()
for h in hs:
    h.remove()
old_style = {i: np.mean(np.stack(v), axis=0) for i, v in collected.items()}
old_style.clear()                      # the bug
chk("cleared-then-read is empty", len(old_style), 0)
chk("which is why it raised KeyError: 0", 0 in old_style, False)

print("\n" + "=" * 68)
print(f"{len(fails)} failure(s)")
for f in fails:
    print("  FAIL", f)
print("=" * 68)
raise SystemExit(1 if fails else 0)