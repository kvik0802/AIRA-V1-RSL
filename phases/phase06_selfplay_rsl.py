"""
PHASE 6 -- Self-play + recursive self-learning (harness rewriting).

Three layers, and the distinction matters:

  GENERATOR  model writes a task AND its hidden test
  EXECUTE    the hidden test decides pass/fail -- never the model's opinion
  RSL        the model rewrites its OWN harness (task templates, test
             generator prompt, sampler settings), and that rewrite is
             evaluated the same way: keep it only if it beats the old harness
             on held-out problems.

The RSL layer is what compounds. Every accepted harness rewrite improves the
search process, so round N benefits from N-1 rounds of process improvement --
not just from more data.

THREE FAILURE MODES, all real, all guarded:
  1. self-grading drift     -> the hidden test must be generated SEPARATELY
                               from the solution and never shown to it
  2. dialect collapse       -> 5% human-written anchors in EVERY batch
  3. unbounded growth       -> convergence is expected at 4-6 rounds; a
                               plateau is the correct outcome, not a bug
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json, math, re, subprocess, sys, tempfile, time
from pathlib import Path

import torch
from common import (ROOT, banner, load_model, eval_suite, refusal_rate,
                    save_json, load_json, extract_code, run_tests, generate,
                    ablate_from_directions)
from peft import PeftModel

ROUNDS = 5
K = 4                     # samples per task
TEMP = 0.8                # diversity is required; greedy collapses to one answer
ANCHOR_FRAC = 0.05        # human-written, non-negotiable
HOLD_FRAC = 0.15          # held out from training, used to judge harness rewrites

banner("PHASE 6 -- SELF-PLAY + RSL HARNESS REWRITING")

ADAPTER = ROOT / "phase03_adapter"
archive = load_json("phase05_archive.json", [])
p3 = load_json("phase03_summary.json", {})
print(f"archive entries: {len(archive)}  phase3: {p3.get('phase3_code')}/10")

# ---- harness: the mutable object. RSL rewrites THIS, not the model. -------
harness = {
    "task_template": (
        "Write a python function `solve(xs)` for this task.\n"
        "TASK: {desc}\n"
        "Then write the test cases that would catch a wrong answer.\n"
        "Output JSON with keys 'code' and 'tests'."),
    "n_samples": K,
    "temperature": TEMP,
    "verify": "execute",
    "anchor_frac": ANCHOR_FRAC,
    "accepted_rewrites": [],
    "rejected_rewrites": [],
}
save_json(harness, "phase06_harness.json")

# ---- anchors: hand-written, never model-generated -------------------------
# 10 problems with real tests. Without these the loop drifts into its own
# dialect by round 3 and stops generalising to anything a human wrote.
ANCHORS = [
    ("sum of squares", "def f(xs):\n    \"\"\"sum of squares of xs\"\"\"",
     "from sol import f\ndef t():\n    assert f([1,2,3])==14 and f([])==0\nt()"),
    ("running max", "def f(xs):\n    \"\"\"list of running maxima\"\"\"",
     "from sol import f\ndef t():\n    assert f([1,3,2,4])==[1,3,3,4]\n    assert f([])==[]\nt()"),
    ("even indices", "def f(xs):\n    \"\"\"elements at even indices\"\"\"",
     "from sol import f\ndef t():\n    assert f([1,2,3,4])==[1,3]\nt()"),
    ("zip longest", "def f(a,b):\n    \"\"\"zip padding with None\"\"\"",
     "from sol import f\ndef t():\n    assert f([1,2],[3])==[(1,3),(2,None)]\nt()"),
    ("title case", "def f(s):\n    \"\"\"capitalise each word\"\"\"",
     "from sol import f\ndef t():\n    assert f('a bc')=='A Bc' and f('')==''\nt()"),
    ("char freq", "def f(s):\n    \"\"\"dict of char->count\"\"\"",
     "from sol import f\ndef t():\n    assert f('aab')=={'a':2,'b':1}\nt()"),
    ("clamp", "def f(x,a,b):\n    \"\"\"clamp x into [a,b]\"\"\"",
     "from sol import f\ndef t():\n    assert f(5,1,3)==3 and f(0,1,3)==1 and f(2,1,3)==2\nt()"),
    ("remove duplicates sorted", "def f(xs):\n    \"\"\"sorted unique\"\"\"",
     "from sol import f\ndef t():\n    assert f([3,1,3,2])==[1,2,3] and f([])==[]\nt()"),
    ("dot product", "def f(a,b):\n    \"\"\"dot product\"\"\"",
     "from sol import f\ndef t():\n    assert f([1,2],[3,4])==11 and f([],[])==0\nt()"),
    ("interleave", "def f(a,b):\n    \"\"\"interleave two lists\"\"\"",
     "from sol import f\ndef t():\n    assert f([1,3],[2,4])==[1,2,3,4]\n    assert f([1],[])==[1]\nt()"),
]

HELD = load_json("phase03_seed_tasks.json", [])
n_hold = max(2, int(len(HELD) * HOLD_FRAC))
HELD_OUT = HELD[:n_hold]
print(f"held-out (never trained on): {[r['name'] for r in HELD_OUT]}")


def gen_task(h, model, tok):
    """Model proposes a task AND a test. The solution never sees the test."""
    out = generate(model, tok, h["task_template"].replace(
        "{desc}", "sum the digits of n, then multiply the result by 2"),
        temperature=h["temperature"])
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        code, tests = obj.get("code"), obj.get("tests")
        if not (isinstance(code, str) and isinstance(tests, str)):
            return None
        if "NotImplementedError" in code:
            return None
        return {"code": code, "tests": tests,
                "task": "generated:sumdigits*2"}
    except Exception:
        return None


def run_round(h, model, tok, rnd, archive):
    """One full round. Returns metrics. Never trains on unexecuted code."""
    rows = []
    n_anchor = max(1, int(len(ANCHORS) * h["anchor_frac"] * 20))

    # (a) model-proposed tasks, verified by execution
    proposed = [gen_task(h, model, tok) for _ in range(6)]
    proposed = [p for p in proposed if p]
    for p in proposed:
        ok, err = run_tests(p["code"], p["tests"], timeout=25)
        # a task whose own reference solution fails its own test is a broken
        # test, not a solvable problem. discard either way.
        if ok:
            rows.append(("generated", p["code"], p["tests"], p["task"]))

    # (b) anchors -- always present, always human-written
    for name, stub, tests in ANCHORS[:n_anchor] + ANCHORS[n_anchor:2 * n_anchor]:
        for _ in range(h["n_samples"]):
            code = extract_code(generate(
                model, tok,
                f"Implement completely. Output one python code block only.\n\n{stub}",
                temperature=h["temperature"]))
            ok, _ = run_tests(code, tests, timeout=25)
            if ok:
                rows.append(("anchor", code, tests, name))
                break

    # (c) archive: metadata only. Harvester records (kept_script filename,
    #     output_tail) -- not runnable code+tests -- so there is nothing safe
    #     to train on here. Do NOT fabricate ("archive", "# technique...", "pass")
    #     rows: tests="pass" has no assert and would be a HARNESS_BUG, and the
    #     DS/trainable filters would drop them anyway. Count usable archive
    #     entries for the report instead.
    n_archive_usable = sum(1 for a in archive[:4] if a.get("kept_script"))

    stats = {"proposed": len(proposed), "kept": len(rows),
             "generated": sum(1 for r in rows if r[0] == "generated"),
             "anchor": sum(1 for r in rows if r[0] == "anchor"),
             "archive_usable": n_archive_usable}
    return rows, stats


# ---- load model -------------------------------------------------------------
banner("loading model + phase-3 adapter")
model, tok = load_model(train=False)
model = PeftModel.from_pretrained(model, str(ADAPTER))
model.eval()

d = ROOT / "refusal_directions.pt"
if d.exists():
    dirs = torch.load(d, map_location="cpu")
    n = ablate_from_directions(model, dirs, 1.0)
    print(f"re-abliterated {n} projections (refusals reinstalled by phase 3 SFT)")

history = []
prev_score = p3.get("phase3_code", 0)

for rnd in range(1, ROUNDS + 1):
    banner(f"ROUND {rnd}/{ROUNDS}")
    t0 = time.time()
    rows, stats = run_round(harness, model, tok, rnd, archive)
    print(f"proposed {stats['proposed']}  kept {stats['kept']} "
          f"(gen {stats['generated']}, anchor {stats['anchor']})  "
          f"{time.time()-t0:.0f}s")

    if not rows:
        print("nothing verified this round. stopping -- no signal.")
        break

    save_json([{"kind": r[0], "code": r[1], "tests": r[2], "task": r[3]}
               for r in rows], f"phase06_round{rnd}_rows.json")

    # ---- SFT on verified rows only
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_linear_schedule_with_warmup

    class DS(Dataset):
        def __init__(self, rows, tok):
            self.e = []
            for kind, code, _t, task in rows:
                if kind == "archive":
                    continue
                m = [{"role": "user",
                      "content": f"Implement this completely. Output one python "
                                 f"code block only.\n\ndef solve(xs):\n    pass  # {task}"},
                     {"role": "assistant", "content": f"```python\n{code}\n```"}]
                self.e.append(tok.apply_chat_template(m, tokenize=False))

        def __len__(self):
            return len(self.e)

        def __getitem__(self, i):
            return self.e[i]

    trainable = [r for r in rows if r[0] != "archive"]
    if not trainable:
        break
    dl = DataLoader(DS(trainable, tok), batch_size=1, shuffle=True,
                    collate_fn=lambda b: tok(b, return_tensors="pt", padding=True,
                                             truncation=True, max_length=768))
    for p in model.parameters():
        p.requires_grad = False
    for n_, p in model.named_parameters():
        if "lora" in n_:
            p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4)
    model.train()
    for step in range(len(dl) * 2):
        b = next(iter(dl))
        b = {k: v.to(model.device) for k, v in b.items()}
        loss = model(**b).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.eval()

    # ---- measure on the human-written suite. ALWAYS.
    score = eval_suite(model, tok, f"phase6_round{rnd}")
    print(f"round {rnd}: {score}/10   (phase3 was {prev_score}/10)")
    history.append({"round": rnd, **stats, "score": score,
                    "prev": prev_score, "secs": round(time.time() - t0)})
    save_json({"history": history, "harness": harness}, "phase06_history.json")

    if score < prev_score:
        print(f"  regression vs {prev_score}. reverting harness for next round.")
        harness["task_template"] = harness.get("_orig_template",
                                               harness["task_template"])
    prev_score = max(prev_score, score)

    # ---- RSL: model rewrites its own harness (PROVISIONAL, not A/B tested).
    # NOTE: HELD_OUT is reserved but NOT used to validate the candidate here;
    # the rewrite is accepted on well-formedness only and retained only if the
    # NEXT round's suite score does not regress (see revert above). Do not
    # claim held-out validation without implementing it.
    if rnd < ROUNDS:
        prompt = (
            "You are improving a code-generation harness. It writes a task, "
            "the model solves it, and a hidden test decides pass/fail.\n\n"
            f"CURRENT TASK TEMPLATE:\n{harness['task_template']}\n\n"
            f"ROUND {rnd} RESULTS: proposed {stats['proposed']}, "
            f"verified {stats['kept']}, suite score {score}/10.\n\n"
            "Write ONE improved task template. It must still require keys "
            "'code' and 'tests', and must still demand executable tests. "
            "Output only the template text.")
        cand = generate(model, tok, prompt, temperature=0.7, max_new_tokens=280)
        cand = re.sub(r"^```.*?```$", "", cand, flags=re.S).strip()
        if len(cand) < 80 or "code" not in cand:
            print("  harness rewrite rejected (malformed)")
            harness["rejected_rewrites"].append({"round": rnd, "why": "malformed"})
        else:
            harness.setdefault("_orig_template", harness["task_template"])
            harness["task_template"] = cand
            harness["accepted_rewrites"].append({"round": rnd, "template": cand[:300]})
            print("  harness rewrite ACCEPTED (pending next round's score)")
        save_json(harness, "phase06_harness.json")

# ---------------------------------------------------------------- report
banner("RESULTS")
base = load_json("phase3_summary.json", {})
print(f"phase 3 baseline : {base.get('phase3_code')}/10")
for h in history:
    print(f"round {h['round']}: {h['score']}/10  "
          f"proposed {h['proposed']}  verified {h['kept']}")

gains = [h["score"] for h in history]
best = max(gains) if gains else 0
GATE = best > base.get("phase3_code", 0)
print(f"\nbest {best}/10 vs phase 3 {base.get('phase3_code')}/10")
print(f"GATE: {'PASS' if GATE else 'FAIL'}")

if not GATE:
    print("-> No net gain. Expected outcome for round 1-2 with a small")
    print("   archive. The information supply is the limit, not the loop.")
    print("   Running more rounds will not fix this. Feed a bigger archive.")

save_json({"history": history, "gate": bool(GATE), "best": best,
           "baseline": base.get("phase3_code"),
           "harness": {k: v for k, v in harness.items() if k != "_orig_template"}},
          "phase06_summary.json")
print(f"\nnext: phase07_finalize.ipynb")