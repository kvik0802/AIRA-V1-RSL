"""
PHASE 7 -- Finalize: re-abliterate, merge, save, write the card.

Order matters and it is not arbitrary:
  1. Re-abliterate FIRST. Phase 3/6 SFT partially reinstalls refusal
     directions. Skipping this is the most common way to end up with an
     "uncensored" model that still refuses.
  2. Measure refusal rate + suite. Both, after the re-ablation.
  3. Merge LoRA to bf16 for deployment, save fp16 for reloading.
  4. Write the card with the MEASURED numbers, not the hoped-for ones.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json, shutil
from pathlib import Path

import torch
from common import (ROOT, MODEL_ID, banner, load_model, eval_suite,
                    refusal_rate, save_json, load_json, ablate_from_directions)

banner("PHASE 7 -- FINALIZE")

ADAPTER = ROOT / "phase03_adapter"
DIRS = ROOT / "refusal_directions.pt"

model, tok = load_model(train=False)
from peft import PeftModel
model = PeftModel.from_pretrained(model, str(ADAPTER))
model = model.merge_and_unload()
print("merged LoRA into base weights")

# ---- 1. re-abliterate -------------------------------------------------------
if DIRS.exists():
    dirs = torch.load(DIRS, map_location="cpu")
    n = ablate_from_directions(model, dirs, strength=1.0)
    print(f"re-abliterated {n} projections across {len(dirs)} layers")
    if n == 0:
        print("-> ablate_from_directions matched no layers. Direction dict keys")
        print("   are layer indices as strings; check phase 1 saved ints.")
else:
    print("no refusal_directions.pt -- run phase 0/1 first, or the model stays censored")

# ---- 2. measure -------------------------------------------------------------
banner("final measurement")
code_score = eval_suite(model, tok, "phase7_final")
ref = refusal_rate()

p0 = load_json("phase0_summary.json", {})
p3 = load_json("phase03_summary.json", {})
p6 = load_json("phase06_summary.json", {})

print(f"\ncoding suite : {code_score}/10")
print(f"refusal rate : {ref:.0%}")
print(f"phase 0 base : {p0.get('baseline_code')}/10  refusal {p0.get('baseline_refusal',0):.0%}")
print(f"phase 3      : {p3.get('phase3_code')}/10")
print(f"phase 6 best : {p6.get('best')}/10")

GATE_CODE = code_score >= p0.get("baseline_code", 0)
GATE_REF = ref <= 0.10
print(f"\nGATE code    : {'PASS' if GATE_CODE else 'FAIL'} (>= phase 0 baseline)")
print(f"GATE refusal : {'PASS' if GATE_REF else 'FAIL'} (<=10%)")
if not GATE_REF:
    print("-> SFT reinstalled refusals. Raise strength to 1.5 or add more")
    print("   dual-use calibration prompts to the phase-1 prompt set.")

# ---- 3. save ----------------------------------------------------------------
banner("saving")
FP16 = ROOT / "final_model"
model.save_pretrained(str(FP16), safe_serialization=True)
tok.save_pretrained(str(FP16))
size = sum(f.stat().st_size for f in FP16.rglob("*")) / 1e9
print(f"fp16 -> {FP16}  ({size:.1f} GB)")

# adapters are small, keep them separately
if ADAPTER.exists():
    shutil.copy(ADAPTER, ROOT / "final_adapter_backup")
    print("adapter backed up")

# ---- 4. model card, measured numbers only -----------------------------------
card = f"""# rsl-coder-7b (experimental)

Text-only coding model. Base: `{MODEL_ID}`. Uncensored via abliteration.
Built on Google Colab T4 (15GB).

## MEASURED results -- 10-task human-written suite with real unit tests

| stage | coding | refusal |
|---|---|---|
| phase 0 base | {p0.get('baseline_code')}/10 | {p0.get('baseline_refusal', 0):.0%} |
| phase 1 abliterated | {p0.get('abliterated_code')}/10 | {p0.get('abliterated_refusal', 0):.0%} |
| phase 3 QLoRA | {p3.get('phase3_code')}/10 | - |
| phase 6 self-play best | {p6.get('best')}/10 | - |
| **phase 7 final** | **{code_score}/10** | **{ref:.0%}** |

## What was actually verified

- Phase 0/1: abliteration cost {p0.get('capability_cost','?')} of 10 tasks
- Phase 2: total LoRA rank {load_json('phase02_ranks.json', {}).get('total_rank','?')}
- Phase 3: {p3.get('verified_rows','?')} execution-verified rows trained,
  {p3.get('discarded_rows','?')} discarded for failing their own tests
- Phase 4: {load_json('phase04_summary.json', {}).get('grow',0)} grow /
  {load_json('phase04_summary.json', {}).get('prune',0)} prune events,
  routing entropy {round(load_json('phase04_summary.json', {}).get('routing_entropy', 0), 2)}
- Phase 5: {load_json('phase05_summary.json', {}).get('kept',0)} harvested
  techniques from {load_json('phase05_summary.json', {}).get('fetched',0)} papers
  ({round(load_json('phase05_summary.json', {}).get('yield_pct', 0), 1)}% yield)

## Honest limits

- The 10-task suite is small and human-written. A score of {code_score}/10 is
  not a benchmark result and should not be compared to published leaderboards.
- Self-play gains are bounded by the harvested archive. More rounds do not
  create information.
- Abliteration degrades capability measurably; the delta is in the table above.
- Not trained on LiveCodeBench or any contaminated benchmark.
- Phase 4 MoE state is in `phase04_moe.pt`; it was trained in-session and is
  not merged into this checkpoint.

## Reproduce

Run in order: `phase01` -> `phase02` -> `phase03` -> `phase04` -> `phase05`
-> `phase06` -> `phase07`. Each gates on the one before it.
"""
(ROOT / "MODEL_CARD.md").write_text(card, encoding="utf-8")
print(f"card -> {ROOT/'MODEL_CARD.md'}")

save_json({"code": code_score, "refusal": ref,
           "gate_code": bool(GATE_CODE), "gate_refusal": bool(GATE_REF),
           "fp16_dir": str(FP16), "gb": round(size, 2)},
          "phase07_summary.json")

banner("DONE")
print(f"model -> {FP16}")
print(f"next: Runtime -> Restart, then load from {FP16}")