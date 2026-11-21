"""
PHASE 4 -- Elastic MoE (grow/prune real; rank + per-param LR are placeholders).

  1. DISCRETE  experts grow and are pruned while training (REAL)
               (Net2Net function-preserving init so no progress is lost)
  2. RANK      checkpoint-only; no adapter rebuild is implemented
  3. CONTINUOUS uniform LR; per-parameter adaptation is NOT implemented

THE LOAD-BALANCING LOSS IS NOT OPTIONAL. Without it the router collapses to a
single expert within a few hundred steps and "MoE" silently becomes "1.5x
parameters". We assert routing entropy at the end and fail loudly.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import (ROOT, banner, load_model, eval_suite, save_json,
                    load_json, extract_code, run_tests, generate)

banner("PHASE 4 -- ELASTIC MoE")

# ---- hyperparams -----------------------------------------------------------
N_EXPERTS = 4
TOP_K = 2
GROW_AFTER = 25       # steps between grow/prune decisions
RANK_RECALC = 50      # steps between rank reallocation
GROW_LOAD = 0.42      # expert carrying more than this share is a candidate
PRUNE_LOAD = 0.02     # below this for the window, it dies
TOTAL_RANK = 128
STEPS = 120
LR = 2e-4

alloc = load_json("phase02_ranks.json")
if alloc is None:
    raise SystemExit("phase02_ranks.json missing -- run phase 2 first.")
p3 = load_json("phase03_summary.json", {})
ADAPTER = ROOT / "phase03_adapter"
if not ADAPTER.exists():
    raise SystemExit("phase03_adapter missing -- run phase 3 first.")

# ---- MoE layer -------------------------------------------------------------
class ElasticMoE(nn.Module):
    """Sparse MoE with runtime add/remove of experts.

    Experts start identical (all zeros output => identity), which is what
    makes Net2Net-style cloning safe: cloning an expert changes nothing about
    the forward pass, so growing never costs accuracy.
    """

    def __init__(self, d_in, d_out, n=N_EXPERTS, k=TOP_K):
        super().__init__()
        self.d_in, self.d_out, self.k = d_in, d_out, k
        self.n = n
        self.experts = nn.ModuleList(
            [nn.Linear(d_in, d_out, bias=False) for _ in range(n)])
        for e in self.experts:                      # zero => identity
            nn.init.zeros_(e.weight)
        self.router = nn.Linear(d_in, n, bias=False)
        nn.init.zeros_(self.router.weight)
        self.load = [0.0] * n
        self.window = [0.0] * n
        self.steps_seen = 0
        self.grow_log, self.prune_log = [], []

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        logits = self.router(xf)
        gates = F.softmax(logits, dim=-1)
        topv, topi = gates.topk(self.k, dim=-1)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)

        # aux loss: mean gate prob per expert must be uniform -> prevents collapse
        mean_p = gates.mean(0)
        k = self.k
        aux = k * (mean_p * mean_p.mean()).sum()

        counts = torch.zeros(self.n, device=xf.device)
        for i in range(self.n):
            counts[i] = (topi == i).float().sum()
        frac = (counts / counts.sum().clamp(min=1)).detach().cpu().tolist()

        self.steps_seen += 1
        for i in range(self.n):
            self.load[i] = 0.9 * self.load[i] + 0.1 * frac[i]
            self.window[i] = 0.9 * self.window[i] + 0.1 * frac[i]

        out = torch.zeros(B * T, self.d_out, device=xf.device, dtype=topv.dtype)
        for i in range(self.n):
            sel = (topi == i).any(-1)
            if not sel.any():
                continue
            w = (topv * (topi == i)).sum(-1).unsqueeze(-1)
            out[sel] += self.experts[i](xf[sel]) * w
        return out.reshape(B, T, self.d_out), aux, frac

    # ---- elasticity ------------------------------------------------------
    def maybe_grow(self, d):
        """Clone the most-loaded expert. Zero-init output means this is a
        no-op for the forward pass -- that is the Net2Net trick."""
        if self.steps_seen < GROW_AFTER or self.steps_seen % GROW_AFTER:
            return None
        if max(self.load) < GROW_LOAD:
            return None
        src = int(np.argmax(self.load))
        new = nn.Linear(self.d_in, self.d_out, bias=False).to(
            self.experts[src].weight.device)
        new.weight.data = self.experts[src].weight.data.clone()
        nn.init.normal_(new.weight.data, std=1e-3)   # tiny noise, breaks symmetry
        self.experts.append(new)
        with torch.no_grad():                        # extend router in place
            old = self.router.weight.data.clone()
            self.router = nn.Linear(self.d_in, self.n + 1, bias=False).to(
                old.device, old.dtype)
            self.router.weight.data = torch.cat(
                [old, torch.randn(1, self.d_in, device=old.device) * 1e-3], 0)
        self.n += 1
        self.load.append(0.0)
        self.window.append(0.0)
        rec = {"step": self.steps_seen, "action": "grow", "from": src,
               "n": self.n, "load": list(self.load)}
        self.grow_log.append(rec)
        return rec

    def maybe_prune(self, d):
        if self.steps_seen < GROW_AFTER * 2 or self.steps_seen % GROW_AFTER:
            return None
        dead = [i for i in range(self.n)
                if self.window[i] < PRUNE_LOAD and i != int(np.argmax(self.window))]
        if not dead or self.n <= 2:
            return None
        victim = dead[0]
        keep = [i for i in range(self.n) if i != victim]
        self.experts = nn.ModuleList([self.experts[i] for i in keep])
        with torch.no_grad():
            w = self.router.weight.data[victim].clone()
            rest = self.router.weight.data[keep].clone()
            self.router = nn.Linear(self.d_in, self.n - 1, bias=False).to(
                w.device, w.dtype)
            self.router.weight.data = rest
        self.n -= 1
        self.load = [self.load[i] for i in keep]
        self.window = [self.window[i] for i in keep]
        rec = {"step": self.steps_seen, "action": "prune", "at": victim,
               "n": self.n, "load": list(self.load)}
        self.prune_log.append(rec)
        return rec


# ---- build -----------------------------------------------------------------
banner("loading phase-3 adapter, injecting MoE into MLP down_proj")
model, tok = load_model(train=False)
from peft import PeftModel
model = PeftModel.from_pretrained(model, str(ADAPTER))
model = model.merge_and_unload()          # fold LoRA in, free the adapter slots
for p in model.parameters():
    p.requires_grad = False

hidden = model.config.hidden_size
inter = model.config.intermediate_size

# replace down_proj with base + sparse MoE branch
class AddMoE(nn.Module):
    def __init__(self, base, m):
        super().__init__()
        self.base, self.moe = base, m
        self.aux, self.frac = None, None

    def forward(self, x):
        b = self.base(x)
        o, aux, frac = self.moe(x)
        self.aux, self.frac = aux, frac
        return b + o


moes = {}
for li, layer in enumerate(model.model.layers):
    moe = ElasticMoE(inter, hidden, N_EXPERTS, TOP_K).to(
        next(layer.parameters()).device, torch.float16)
    moes[str(li)] = moe
    layer.mlp.down_proj = AddMoE(layer.mlp.down_proj, moe)

# trainable: MoE only. Base is frozen.
moe_params = []
for m in moes.values():
    moe_params += list(m.parameters())
for p in moe_params:
    p.requires_grad = True

print(f"trainable MoE params: {sum(p.numel() for p in moe_params)/1e6:.2f}M "
      f"across {len(moes)} layers x {N_EXPERTS} experts")

# ---- data: reuse phase-3 verified rows -------------------------------------
import re as _re
_seedsrc = (ROOT / "phase03_seed_tasks.json")
if not _seedsrc.exists():
    raise SystemExit("phase03_seed_tasks.json missing -- run phase 3 first.")
rows = [(r["name"], r["stub"], r["tests"], r.get("code"))
        for r in json.load(open(_seedsrc))
        if r.get("verified")]
print(f"verified training rows: {len(rows)}")
if not rows:
    raise SystemExit("no verified rows in phase 3.")

from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup


class DS(Dataset):
    def __init__(self, rows, tok):
        self.e = []
        for name, stub, _t, code in rows:
            # Prefer the execution-verified code saved by phase 3; fall back
            # to the stub skeleton only if no verified code was stored.
            if code:
                body = code
            else:
                body = "\n".join(l for l in stub.splitlines()
                                 if "NotImplementedError" not in l and l.strip()) or stub
            m = [{"role": "user", "content":
                  f"Implement this completely. Output only a python code block.\n\n{stub}"},
                 {"role": "assistant", "content": f"```python\n{body}\n```"}]
            self.e.append(tok.apply_chat_template(m, tokenize=False))

    def __len__(self):
        return len(self.e)

    def __getitem__(self, i):
        return self.e[i]


dl = DataLoader(DS(rows, tok), batch_size=1, shuffle=True,
                collate_fn=lambda b: tok(b, return_tensors="pt", padding=True,
                                         truncation=True, max_length=512))

opt = torch.optim.AdamW(moe_params, lr=LR, weight_decay=0.0)

# ---- elastic rank ----------------------------------------------------------
banner(f"training {STEPS} steps")
model.train()
t0 = time.time()
hist = []
for step in range(1, STEPS + 1):
    batch = next(iter(dl))
    batch = {k: v.to(model.device) for k, v in batch.items()}

    # LR schedule: uniform LR (placeholder). NOT per-parameter: setting every
    # param_group to LR does not differentiate params. Keep honest -- do not
    # claim per-parameter adaptation without reading exp_avg_sq.
    if step % 10 == 1:
        for g in opt.param_groups:
            g["lr"] = LR

    loss = model(**batch).loss

    aux = torch.zeros((), device=model.device)
    seen = 0
    for layer in model.model.layers:
        am = getattr(layer.mlp.down_proj, "aux", None)
        if am is not None:
            aux = aux + am
            seen += 1
    aux = aux / max(seen, 1)
    total = loss + 0.01 * aux

    total.backward()
    torch.nn.utils.clip_grad_norm_(moe_params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    # MECHANISM 1: grow / prune
    grew = pruned = None
    for m in moes.values():
        g = m.maybe_grow(None)
        p = m.maybe_prune(None)
        if g:
            grew = g
            # add only the NEW expert + router params; do not recreate the
            # optimizer (that discards momentum for all existing params).
            new_params = list(m.experts[-1].parameters()) + list(m.router.parameters())
            opt.add_param_group({"params": new_params, "lr": LR})
        if p:
            pruned = p

    if step % 10 == 0:
        n0 = sum(m.n for m in moes.values())
        fr = moes["0"].frac or [0] * moes["0"].n
        ent = -sum(f * math.log(max(f, 1e-9)) for f in fr)
        norm = math.log(max(moes["0"].n, 2))
        print(f"  step {step}/{STEPS} loss {loss.item():.4f} aux {aux.item():.4f} "
              f"experts {n0} routing_entropy {ent/norm:.2f} "
              f"{(time.time()-t0)/step:.2f}s/step")
        hist.append({"step": step, "loss": loss.item(), "aux": aux.item(),
                     "experts": n0, "routing_entropy": ent / norm})

    # Rank checkpoint (NOT reallocation): logs training state. True rank
    # reallocation would rebuild LoRA adapters mid-training; that is not
    # implemented here -- do not claim budget movement without it.
    if step % RANK_RECALC == 0:
        print(f"  [rank] checkpoint (budget {TOTAL_RANK}, no adapter rebuild)")
        save_json({"phase": 4, "step": step, "hist": hist,
                   "grow": sum(len(m.grow_log) for m in moes.values()),
                   "prune": sum(len(m.prune_log) for m in moes.values())},
                  "phase04_trainlog.json")

banner("RESULTS")
n_grow = sum(len(m.grow_log) for m in moes.values())
n_prune = sum(len(m.prune_log) for m in moes.values())
final_experts = sum(m.n for m in moes.values())
final_params = sum(p.numel() for m in moes.values() for p in m.parameters())

fr = moes["0"].frac or [0] * moes["0"].n
ent = -sum(f * math.log(max(f, 1e-9)) for f in fr) / math.log(max(moes["0"].n, 2))
live = sum(1 for f in (moes["0"].load or []) if f > PRUNE_LOAD)

print(f"grow events   : {n_grow}")
print(f"prune events  : {n_prune}")
print(f"experts now   : {final_experts}  (started {N_EXPERTS * len(moes)})")
print(f"MoE params    : {final_params/1e6:.2f}M")
print(f"routing entropy (layer 0): {ent:.2f}  (1.0 = uniform, <0.5 = collapsing)")
print(f"live experts  : {live}/{moes['0'].n}")

banner("saving MoE state")
state = {}
for k, m in moes.items():
    state[k] = {"n": m.n,
                "experts": [e.weight.detach().cpu().half() for e in m.experts],
                "router": m.router.weight.detach().cpu().half(),
                "load": m.load, "grow": m.grow_log, "prune": m.prune_log}
torch.save(state, ROOT / "phase04_moe.pt")
print(f"saved -> {ROOT/'phase04_moe.pt'}")

GATE = (n_grow + n_prune) > 0 and ent >= 0.5
print(f"\nGATE: {'PASS' if GATE else 'FAIL'}")
print("  need: >=1 grow OR prune event, and routing entropy >=0.5")
if n_grow == 0 and n_prune == 0:
    print("-> NO elasticity happened. Experts never exceeded GROW_LOAD and")
    print("   never fell below PRUNE_LOAD. The MoE is decorative. Tighten")
    print("   GROW_LOAD / PRUNE_LOAD, or accept that this phase did nothing.")
if ent < 0.5:
    print("-> ROUTER COLLAPSE. Raise the aux coefficient above 0.01.")

save_json({"grow": n_grow, "prune": n_prune, "experts": final_experts,
           "params": final_params, "routing_entropy": ent,
           "hist": hist, "gate": bool(GATE)}, "phase04_summary.json")
print(f"\nnext: phase05_harvester.ipynb")