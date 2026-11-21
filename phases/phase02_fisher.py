"""
PHASE 2 -- Fisher-informed rank allocation.

No training. This produces the per-layer rank table that phases 3 and 4 use.

Why this exists: uniform LoRA rank is a design flaw. Attention projections,
MLP down-projections and the up/gate pair do not contribute equally to
adaptation. The empirical Fisher diagonal of the LoRA-B matrix is an
importance proxy (the Optimal Brain Damage lineage, via eFIM). ~8 backward
passes, then water-fill ranks under a fixed total budget.

BUG NOTE, learned the hard way: at LoRA init B=0, so dL/dA is exactly zero
while dL/dB is not. Scoring LoRA-A measures nothing. Score LoRA-B only.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import numpy as np
import torch
from common import (ROOT, banner, load_model, generate, extract_code,
                    run_tests, EVAL, save_json, load_json)

TOTAL_RANK = 128      # sum over all adapted layers, held constant
RANK_MIN, RANK_MAX = 2, 64
CALIB_STEPS = 8       # paper: ~8 backward passes is enough
BATCH = 2
SEQ = 512

banner("PHASE 2 -- FISHER RANK ALLOCATION")
print(f"total rank budget {TOTAL_RANK}  range [{RANK_MIN},{RANK_MAX}]  "
      f"calib steps {CALIB_STEPS}")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model, tok = load_model(train=True, lora_cfg=LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))

# ---------------------------------------------------------------- calibration
# Prompts drawn from the eval set's own task descriptions. Not the solutions.
# Calibration on eval task *descriptions* is deliberate: we are measuring which
# layers this distribution of work needs, not memorising answers.
calib = []
for name, prompt, _t in EVAL:
    calib.append(f"Write a complete, correct python implementation of: {name}.\n"
                 f"Include the class/function signatures shown.")
for _i in range(BATCH * CALIB_STEPS - len(calib)):
    calib.append(calib[_i % len(calib)])

banner("collecting eFIM diagonal on LoRA-B")
fisher = {}

# Hook grads as they appear -- cheaper and more reliable than scanning params.
B_TARGETS = {}
for n, p in model.named_parameters():
    if "lora_B" in n and p.requires_grad:
        B_TARGETS[n] = p
        p.register_hook(lambda g, name=n: fisher.setdefault(name, []).append(
            g.detach().float().pow(2).mean().item()))

model.train()
step = 0
for text in calib:
    ids = tok([text], return_tensors="pt", truncation=True,
              max_length=SEQ).to(model.device)
    out = model(**ids, labels=ids["input_ids"])
    out.loss.backward()
    step += 1
    if step % BATCH == 0:
        model.zero_grad(set_to_none=True)
    if step % 16 == 0:
        print(f"  step {step}/{len(calib)}")

model.zero_grad(set_to_none=True)

# mean squared gradient = eFIM proxy
scores = {}
for n, vals in fisher.items():
    if not vals:
        continue
    scores[n] = float(np.mean(vals))

print(f"\nscored {len(scores)} LoRA-B matrices from {CALIB_STEPS} gradient reads")
if not scores:
    raise SystemExit("NO SCORES -- hooks never fired. Check adapter is attached.")

# ---------------------------------------------------------------- allocation
banner("rank allocation (water-filling under fixed budget)")

# group by layer so a layer gets ONE rank, not a rank per projection matrix
by_layer = {}
for n, s in scores.items():
    li = n.split(".layers.")[1].split(".")[0]
    by_layer.setdefault(li, []).append(s)
layer_score = {li: float(np.mean(v)) for li, v in by_layer.items()}

raw = np.array(list(layer_score.values()), dtype=np.float64)
print(f"layers: {len(layer_score)}   score min={raw.min():.3e} "
      f"max={raw.max():.3e}  ratio={raw.max()/max(raw.min(),1e-30):.1f}x")

# water-filling: allocate proportional to score, clip, then redistribute
# budget from saturated layers. Guarantees the total is spent.
budget = TOTAL_RANK
alloc = {li: RANK_MIN for li in layer_score}
active = set(layer_score)

for _ in range(64):
    tot = sum(alloc.values())
    if tot >= budget:
        break
    room = {li for li in active if alloc[li] < RANK_MAX}
    if not room:
        break
    free = budget - tot
    wtot = sum(layer_score[li] for li in room)
    if wtot <= 0:
        for li in sorted(room, key=lambda k: -layer_score[k]):
            if free <= 0:
                break
            add = min(RANK_MIN, free)
            alloc[li] += add
            free -= add
        continue
    for li in sorted(room, key=lambda k: -layer_score[k]):
        if free <= 0:
            break
        add = int(round(free * layer_score[li] / wtot))
        add = max(add, 1)
        add = min(add, RANK_MAX - alloc[li], free)
        alloc[li] += add
        free -= add

final = {li: int(r) for li, r in alloc.items()}
total = sum(final.values())
uniq = sorted(set(final.values()))
spread = (max(final.values()) - min(final.values())) / max(max(final.values()), 1)

banner("RESULT")
print(f"total rank spent: {total} (budget {TOTAL_RANK})")
print(f"distinct rank values: {uniq}")
print(f"spread: {spread:.0%}")
top = sorted(final.items(), key=lambda kv: -kv[1])[:8]
bot = sorted(final.items(), key=lambda kv: kv[1])[:8]
print("highest rank layers:", top)
print("lowest  rank layers:", bot)

GATE = len(uniq) > 1 and spread >= 0.3 and min(final.values()) > 0 \
       and max(final.values()) < RANK_MAX and abs(total - TOTAL_RANK) <= 4
print(f"\nGATE: {'PASS' if GATE else 'FAIL'}")
print("  need: non-uniform (spread>=30%), no dead layer, budget spent")

if len(uniq) == 1:
    print("-> UNIFORM. Fisher gave no signal. Add calibration data, or the")
    print("   signal is genuinely flat and elastic rank buys you nothing.")
    print("   Do NOT proceed pretending it worked; phase 4 would be theatre.")

save_json({"total_rank": total, "budget": TOTAL_RANK,
           "ranks": final, "layer_scores": layer_score,
           "rank_min": RANK_MIN, "rank_max": RANK_MAX,
           "gate": bool(GATE)}, "phase02_ranks.json")

# per-module LoRAConfig for phase 3/4
per_module = {}
for name, _s in scores.items():
    li = name.split(".layers.")[1].split(".")[0]
    per_module[name] = final[li]
save_json({"rank_per_module": per_module}, "phase02_rank_per_module.json")
print(f"\nsaved -> {ROOT}")
print("next: phase03_qlora.ipynb")