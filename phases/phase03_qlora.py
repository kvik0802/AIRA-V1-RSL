"""
PHASE 3 -- QLoRA baseline. FIXED rank, NO self-learning, NO elastic, NO MoE.

This phase exists to produce THE number. Everything after it is measured
against this. Skipping it means every later "improvement" is unfalsifiable.

Data here is hand-written and fixed on purpose. It is NOT the self-play data.
The point is to learn the SFT recipe and get a reference point.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json, random
import torch
from torch.utils.data import Dataset
from common import (ROOT, banner, load_model, eval_suite, save_json,
                    load_json, extract_code, run_tests, generate)

random.seed(0)

banner("PHASE 3 -- QLoRA BASELINE (fixed rank)")

alloc = load_json("phase02_ranks.json")
if alloc is None:
    raise SystemExit("phase02_ranks.json missing -- run phase 2 first.")
print(f"using phase-2 rank table (total rank {alloc['total_rank']})")

# ---------------------------------------------------------------- training set
# 40 small tasks with runnable tests. Hand-written. Deliberately easy:
# we are validating the pipeline, not measuring capability.
SEED = [
    ("binary search", "def bsearch(a,t):\n    \"\"\"Return index of t in sorted a, else -1.\"\"\"\n    raise NotImplementedError",
     "from sol import bsearch\ndef t():\n    assert bsearch([1,3,5,7],5)==2\n    assert bsearch([1,3,5,7],4)==-1\n    assert bsearch([],1)==-1\nt()"),
    ("fizzbuzz", "def fb(n):\n    \"\"\"Return list of FizzBuzz strings 1..n.\"\"\"\n    raise NotImplementedError",
     "from sol import fb\ndef t():\n    assert fb(5)==['1','2','Fizz','4','Buzz']\n    assert fb(0)==[]\nt()"),
    ("is prime", "def prime(n):\n    \"\"\"True if n is prime.\"\"\"\n    raise NotImplementedError",
     "from sol import prime\ndef t():\n    assert prime(2) and prime(3) and prime(97)\n    assert not prime(1) and not prime(0) and not prime(9)\nt()"),
    ("fib iterative", "def fib(n):\n    \"\"\"nth fibonacci, iterative.\"\"\"\n    raise NotImplementedError",
     "from sol import fib\ndef t():\n    assert [fib(i) for i in range(7)]==[0,1,1,2,3,5,8]\nt()"),
    ("factorial", "def fact(n):\n    raise NotImplementedError",
     "from sol import fact\ndef t():\n    assert fact(0)==1 and fact(5)==120\nt()"),
    ("string reverse", "def rev(s):\n    raise NotImplementedError",
     "from sol import rev\ndef t():\n    assert rev('')=='' and rev('abc')=='cba'\nt()"),
    ("word count", "def wc(s):\n    \"\"\"dict word->count.\"\"\"\n    raise NotImplementedError",
     "from sol import wc\ndef t():\n    assert wc('a b a')=={'a':2,'b':1}\n    assert wc('')=={}\nt()"),
    ("dedupe preserve order", "def dedupe(xs):\n    raise NotImplementedError",
     "from sol import dedupe\ndef t():\n    assert dedupe([1,2,1,3,2])==[1,2,3]\n    assert dedupe([])==[]\nt()"),
    ("second largest", "def second(xs):\n    \"\"\"Second largest distinct value.\"\"\"\n    raise NotImplementedError",
     "from sol import second\ndef t():\n    assert second([1,3,2])==2\n    assert second([5,5,1])==1\nt()"),
    ("palindrome", "def pal(s):\n    raise NotImplementedError",
     "from sol import pal\ndef t():\n    assert pal('racecar') and not pal('ab')\n    assert not pal('abca')\nt()"),
    ("zip lists", "def zl(a,b):\n    raise NotImplementedError",
     "from sol import zl\ndef t():\n    assert zl([1,2],[3,4])==[(1,3),(2,4)]\n    assert zl([1],[2,3])==[(1,2)]\nt()"),
    ("flatten nested", "def flat(xs):\n    raise NotImplementedError",
     "from sol import flat\ndef t():\n    assert flat([1,[2,[3,4]]])==[1,2,3,4]\nt()"),
    ("sum digits", "def sd(n):\n    raise NotImplementedError",
     "from sol import sd\ndef t():\n    assert sd(123)==6 and sd(0)==0\nt()"),
    ("caesar cipher", "def caesar(s,k):\n    raise NotImplementedError",
     "from sol import caesar\ndef t():\n    assert caesar('abc',1)=='bcd'\n    assert caesar(caesar('xyz',3),23)=='xyz'\nt()"),
    ("run-length decode", "def rle_dec(s):\n    raise NotImplementedError",
     "from sol import rle_dec\ndef t():\n    assert rle_dec('a3b2')=='aaabb'\n    assert rle_dec('')==''\nt()"),
    ("chunks", "def chunks(xs,n):\n    raise NotImplementedError",
     "from sol import chunks\ndef t():\n    assert chunks([1,2,3,4],2)==[[1,2],[3,4]]\n    assert chunks([1,2,3],5)==[[1,2,3]]\nt()"),
    ("deep get", "def get(d,path):\n    \"\"\"d={'a':{'b':1}}, get(d,'a.b')==1\"\"\"\n    raise NotImplementedError",
     "from sol import get\ndef t():\n    assert get({'a':{'b':1}},'a.b')==1\n    assert get({},'x') is None\nt()"),
    ("deep set", "def dset(d,path,v):\n    raise NotImplementedError",
     "from sol import dset\ndef t():\n    d={}\n    dset(d,'a.b.c',5)\n    assert d=={'a':{'b':{'c':5}}}\nt()"),
    ("group by", "def group(xs,f):\n    raise NotImplementedError",
     "from sol import group\ndef t():\n    assert group([1,2,3,4],lambda x:x%2)=={1:[1,3],0:[2,4]}\nt()"),
    ("merge intervals", "def merge(iv):\n    raise NotImplementedError",
     "from sol import merge\ndef t():\n    assert merge([[1,3],[2,6],[8,10]])==[[1,6],[8,10]]\n    assert merge([])==[]\nt()"),
    ("longest common prefix", "def lcp(a,b):\n    raise NotImplementedError",
     "from sol import lcp\ndef t():\n    assert lcp('flower','flow')=='flow'\n    assert lcp('abc','xyz')==''\nt()"),
    ("valid brackets", "def brackets(s):\n    raise NotImplementedError",
     "from sol import brackets\ndef t():\n    assert brackets('([]{})')\n    assert not brackets('(]')\nt()"),
    ("spiral matrix", "def spiral(n):\n    raise NotImplementedError",
     "from sol import spiral\ndef t():\n    m=spiral(2)\n    assert len(m)==2 and len(m[0])==2\n    assert sorted(x for r in m for x in r)==[1,2,3,4]\nt()"),
    ("transpose", "def tr(m):\n    raise NotImplementedError",
     "from sol import tr\ndef t():\n    assert tr([[1,2,3],[4,5,6]])==[[1,4],[2,5],[3,6]]\n    assert tr([])==[]\nt()"),
    ("rotate matrix 90", "def rot(m):\n    raise NotImplementedError",
     "from sol import rot\ndef t():\n    assert rot([[1,2],[3,4]])==[[3,1],[4,2]]\nt()"),
    ("two sum", "def two(a,t):\n    \"\"\"return pair summing to t or None\"\"\"\n    raise NotImplementedError",
     "from sol import two\ndef t():\n    assert two([2,7,11],9)==[0,1] or two([2,7,11],9)==[1,0]\n    assert two([1,2],9) is None\nt()"),
    ("max subarray", "def msa(a):\n    raise NotImplementedError",
     "from sol import msa\ndef t():\n    assert msa([-2,1,-3,4,-1,2,1,-5,4])==6\n    assert msa([-3,-1,-2])==-1\nt()"),
    ("coin change", "def coins(amt,cs):\n    \"\"\"fewest coins, or -1\"\"\"\n    raise NotImplementedError",
     "from sol import coins\ndef t():\n    assert coins(11,[1,2,5])==3\n    assert coins(3,[2])==-1\nt()"),
    ("valid parentheses minimal", "def minrem(s):\n    \"\"\"min removals to balance\"\"\"\n    raise NotImplementedError",
     "from sol import minrem\ndef t():\n    assert minrem('())')==1\n    assert minrem('(((')==3\nt()"),
    ("str to int", "def atoi(s):\n    raise NotImplementedError",
     "from sol import atoi\ndef t():\n    assert atoi('-42')==-42 and atoi('7')==7\nt()"),
    ("reverse words", "def rwords(s):\n    raise NotImplementedError",
     "from sol import rwords\ndef t():\n    assert rwords('a b  c')=='c b a'\nt()"),
    ("topological sort", "def topo(g):\n    \"\"\"g={'a':['b'],'b':[]}\"\"\"\n    raise NotImplementedError",
     "from sol import topo\ndef t():\n    r=topo({'a':['b'],'b':[],'c':['a']})\n    assert r.index('c')<r.index('a')<r.index('b')\nt()"),
    ("binary tree depth", "def depth(n):\n    \"\"\"{'l':None,'r':None}\"\"\"\n    raise NotImplementedError",
     "from sol import depth\ndef t():\n    assert depth(None)==0\n    assert depth({'l':None,'r':None})==1\nt()"),
    ("level order", "def lo(n):\n    raise NotImplementedError",
     "from sol import lo\ndef t():\n    n={'v':1,'l':{'v':2,'l':None,'r':None},'r':{'v':3,'l':None,'r':None}}\n    assert lo(n)==[1,2,3]\nt()"),
    ("valid parens stream", "def longest_valid(s):\n    raise NotImplementedError",
     "from sol import longest_valid\ndef t():\n    assert longest_valid('(()')==2\n    assert longest_valid(')()())')==4\nt()"),
    ("roman to int", "def roman(s):\n    raise NotImplementedError",
     "from sol import roman\ndef t():\n    assert roman('IX')==9 and roman('MCMXCIV')==1994\nt()"),
    ("int to roman", "def toroman(n):\n    raise NotImplementedError",
     "from sol import toroman\ndef t():\n    assert toroman(4)=='IV' and toroman(9)=='IX'\nt()"),
    ("count vowels", "def cv(s):\n    raise NotImplementedError",
     "from sol import cv\ndef t():\n    assert cv('Hello')==2 and cv('')==0\nt()"),
    ("is anagram", "def ana(a,b):\n    raise NotImplementedError",
     "from sol import ana\ndef t():\n    assert ana('listen','silent')\n    assert not ana('ab','abc')\nt()"),
    ("matrix spiral fill", "def fill(n):\n    raise NotImplementedError",
     "from sol import fill\ndef t():\n    m=fill(1)\n    assert m==[[1]]\nt()"),
    ("staircase", "def stairs(n):\n    \"\"\"number of ways to climb n\"\"\"\n    raise NotImplementedError",
     "from sol import stairs\ndef t():\n    assert stairs(1)==1 and stairs(2)==2 and stairs(3)==3\nt()"),
    ("knapsack 01", "def knap(w,v,cap):\n    raise NotImplementedError",
     "from sol import knap\ndef t():\n    assert knap([1,3,4],[15,20,30],4)==35\nt()"),
    ("edit distance", "def ed(a,b):\n    raise NotImplementedError",
     "from sol import ed\ndef t():\n    assert ed('kitten','sitting')==3\n    assert ed('','abc')==3\nt()"),
    ("longest increasing", "def lis(a):\n    raise NotImplementedError",
     "from sol import lis\ndef t():\n    assert lis([10,9,2,5,3,7,101,18])==4\nt()"),
    ("spiral traversal", "def spiral_t(m):\n    raise NotImplementedError",
     "from sol import spiral_t\ndef t():\n    assert spiral_t([[1,2],[3,4]])==[1,2,4,3]\nt()"),
]

banner(f"training set: {len(SEED)} tasks")


def strip_raise(stub):
    """Drop the NotImplementedError line so the target is a real body."""
    return "\n".join(ln for ln in stub.splitlines()
                     if "NotImplementedError" not in ln and ln.strip()) or stub


class SFT(Dataset):
    """Target is code that PASSES. We never train on code that failed --
    that is the whole difference between this and self-play."""

    def __init__(self, rows, tok):
        self.ex = []
        for row in rows:
            # rows are (name, stub, tests, code): train on the VERIFIED code,
            # never on the stub skeleton.
            if len(row) == 4:
                name, stub, tests, code = row
            else:  # legacy 3-tuple (unverified) -- skeleton only, do not use
                name, stub, tests = row
                code = strip_raise(stub)
            msg = [
                {"role": "user",
                 "content": f"Implement this completely. Output only a single "
                            f"python code block, no prose.\n\n{stub}"},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ]
            self.ex.append((name, tests,
                            tok.apply_chat_template(msg, tokenize=False)))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


model, tok = load_model(train=True)

ds = SFT(SEED, tok)

# self-generate correct solutions where possible: execution-verified data.
# Anything that does not pass is discarded. The verified CODE is kept --
# training on the stub skeleton would teach empty function bodies.
banner("execution-verified data generation")
kept, discarded = [], []
for name, stub, tests in SEED:
    out = generate(model, tok,
                   f"Implement this completely. Output only one python code block.\n\n{stub}",
                   temperature=0.0)
    code = extract_code(out)
    ok, err = run_tests(code, tests, timeout=40)
    if ok:
        kept.append((name, stub, tests, code))
    else:
        discarded.append((name, stub, tests))
    print(f"  [{'OK  ' if ok else 'DROP'}] {name}")

print(f"\nverified: {len(kept)}   discarded: {len(discarded)}")
print("note: only verified rows are used. Unverified output is never trained on.")
if len(kept) < 5:
    print("WARNING: very few verified rows. Baseline will be weak; that is a")
    print("fact about the model, not a bug. Phases 5-6 exist to fix this.")

ds = SFT(kept, tok)

# ---------------------------------------------------------------- train
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

EPOCHS = 3
BS, ACC, LR, MAXLEN = 1, 4, 2e-4, 1024


def collate(batch):
    texts = [b[2] for b in batch]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAXLEN)
    labels = enc["input_ids"].masked_fill(
        enc["attention_mask"] == 0, -100)
    enc["labels"] = labels
    return enc


dl = DataLoader(ds, batch_size=BS, shuffle=True, collate_fn=collate)
import math as _math
steps = _math.ceil(len(dl) * EPOCHS / ACC)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=LR, weight_decay=0.0, betas=(0.9, 0.999))
sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps) + 1, max(steps, 1))

banner(f"training: {len(ds)} rows x {EPOCHS} ep = {steps} optim steps")

model.train()
gstep, losses = 0, []
import time
t0 = time.time()
accum = 0
for ep in range(EPOCHS):
    for i, batch in enumerate(dl):
        is_last = (i + 1) == len(dl)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss = model(**batch).loss
        (loss / ACC).backward()
        losses.append(loss.item())
        accum += 1

        if accum % ACC == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            gstep += 1
            if gstep % 2 == 0:
                m = sum(losses[-ACC * 2:]) / len(losses[-ACC * 2:])
                print(f"  step {gstep}/{steps} loss {m:.4f} "
                      f"({(time.time()-t0)/max(gstep,1):.1f}s/step) "
                      f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                save_json({"step": gstep, "loss": m}, "phase03_trainlog.json")

banner("saving adapter")
ADAPTER = ROOT / "phase03_adapter"
model.save_pretrained(str(ADAPTER))
tok.save_pretrained(str(ADAPTER))
print(f"saved -> {ADAPTER}")

del model
torch.cuda.empty_cache()

# ---------------------------------------------------------------- measure
banner("PHASE 3 RESULT")
model, tok = load_model(train=False)
from peft import PeftModel
model = PeftModel.from_pretrained(model, str(ADAPTER))
model.eval()

score = eval_suite(model, tok, "phase3_qlorabase")
print(f"\nphase 3: {score}/10")

base = load_json("phase0_summary.json", {})
prev = base.get("abliterated_code", base.get("baseline_code", 0))
GATE = score > prev
print(f"phase 0/1 reference: {prev}/10")
print(f"GATE: {'PASS' if GATE else 'FAIL'}  (must beat phase 0/1)")

if not GATE:
    print("-> no gain. Do not continue. Check: was the adapter actually")
    print("   loaded (PeftModel) and were rows actually verified?")

save_json({"phase3_code": score, "reference": prev, "gate": bool(GATE),
           "verified_rows": len(kept), "discarded_rows": len(discarded),
           "steps": steps}, "phase03_summary.json")

# phase 4 reads the verified rows back rather than duplicating the task list
save_json([{"name": n, "stub": s, "tests": t, "code": c, "verified": True}
           for n, s, t, c in kept], "phase03_seed_tasks.json")
print(f"next: phase04_elastic_moe.ipynb")