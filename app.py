"""
PHASE 0 + 1 -- baseline, then abliteration. One file, run top to bottom.

    python app.py selftest     # 10s, no GPU, no download. run this FIRST.
    python app.py phase0      # baseline only
    python app.py phase1      # abliteration
    python app.py all         # both

In Colab, open as a notebook and Run All; the selftest cell runs before the
model is touched and aborts if the reward signal is broken.

DESIGN RULE, learned the hard way three times:
  Nothing is measured until the harness has proved itself. selftest() scores
  10/10 with reference solutions and 0/10 with wrong ones, using a stub model
  and zero GPU seconds. A broken harness looks exactly like a weak model, so it
  must be impossible to spend GPU time on one.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from rsl_harness import (  # noqa: E402
    EVAL, INSTR, PROBES, extract_code, is_refusal, refusal_rate,
    run_suite, run_tests, self_test,
)

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
RUNTIME_GPU = "/content/drive/MyDrive/rsl_model"
LOCAL_OUT = HERE / "out"

BANNER = "=" * 64


def out_dir():
    """Drive when mounted, else ./out.

    Requires the mount's PARENT to exist too: on Windows Path('/content/drive')
    resolves to '\\content\\drive', which can look present via a rooted path and
    yield a broken hybrid like '/content\\drive\\MyDrive'.
    """
    from common import out_dir as _od
    return _od()


def save_json(obj, name, d):
    p = d / name
    json.dump(obj, open(p, "w"), indent=2)
    print(f"saved {p}")


# ---------------------------------------------------------------- model

_STATE = {}


def get_model():
    """Load once, reuse. 4-bit NF4 + fp16.

    torch_dtype, NOT dtype: `dtype=` is only understood by transformers>=4.54.
    On older versions it falls through to Model.__init__ and raises
    "unexpected keyword argument 'dtype'". torch_dtype is accepted everywhere.
    """
    if _STATE:
        return _STATE["model"], _STATE["tok"]

    import torch  # noqa: F401
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    print(BANNER)
    print("ENVIRONMENT")
    print(BANNER)
    print("torch     ", torch.__version__)
    assert torch.cuda.is_available(), "no GPU -- are you in a Colab GPU runtime?"
    print("device    ", torch.cuda.get_device_name(0))
    print("bf16 ok   ", torch.cuda.is_bf16_supported(),
          "(False on T4 -- fp16 is used, that is correct)")
    free, total = torch.cuda.mem_get_info()
    print(f"vram       {free/1e9:.1f} / {total/1e9:.1f} GB free")

    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    from common import dtype_kwargs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        **dtype_kwargs(torch))
    model.eval()
    print(f"mem alloc  {torch.cuda.memory_allocated()/1e9:.2f} GB")
    _STATE.update(model=model, tok=tok, torch=torch)
    return model, tok


def make_generate():
    """Return a generate(prompt, max_new_tokens=...) callable bound to the
    model, so run_suite stays model-agnostic and testable with a stub."""
    model, tok = get_model()
    # torch is imported inside get_model(), so it is not a module-level name
    # here. Fetching it from the cache is correct and avoids a second import;
    # without this the first generation raises NameError, after the 15GB
    # download has already completed.
    torch = _STATE["torch"]

    def generate(prompt, max_new_tokens=420):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    return generate


# ---------------------------------------------------------------- phases

def phase0():
    d = out_dir()
    from common import report_warning
    w = report_warning()
    if w:
        print(w)
    gen = make_generate()
    print(BANNER)
    print("PHASE 0 -- BASELINE")
    print(BANNER)
    code, rows = run_suite(gen, "phase0_baseline")
    ref = refusal_rate(gen)
    print(f"\nbaseline: {code}/{len(EVAL)} coding, {ref:.0%} refusal")
    save_json({"baseline_code": code, "n": len(EVAL),
               "baseline_refusal": ref, "tasks": rows}, "phase0_summary.json", d)
    return code, ref


def phase1(strength=1.0):
    """Abliteration from scratch: find each layer's refusal direction as the
    mean-difference of activations on dual-use vs benign prompts, then project
    it out of the residual stream at o_proj and mlp.down_proj.

    Re-run after any fine-tune -- SFT partially reinstalls refusal behaviour.
    """
    import torch

    d = out_dir()
    model, tok = get_model()
    dual = [p for k, p in PROBES if k == "dual"]
    harm = [p for k, p in PROBES if k == "harm"]

    layers = model.model.layers
    nL = len(layers)

    def mk_hook(i, sink):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # last token = where the refuse/answer decision is made
            sink.setdefault(i, []).append(h[0, -1, :].float().cpu())
        return hook

    def collect(prompts):
        """Return {layer_index: mean activation}.

        BUG THIS HAD: it used to clear the shared `acts` dict on the way out,
        which destroyed the very activations the caller was about to average.
        The result was mu_dual == {} and then `KeyError: 0` when phase 1 built
        directions. Each call now owns its own sink and RETURNS the means, so
        there is nothing to clear and nothing to read afterwards.
        """
        sink = {}
        hs = [l.register_forward_hook(mk_hook(i, sink))
              for i, l in enumerate(layers)]
        try:
            for p in prompts:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True)
                ids = tok([text], return_tensors="pt", truncation=True,
                          max_length=256).to(model.device)
                with torch.no_grad():
                    model(**ids)
        finally:
            for h in hs:
                h.remove()
        return {i: torch.stack(v).mean(0) for i, v in sink.items()}

    print(BANNER)
    print(f"PHASE 1 -- ABLITERATION (strength={strength})")
    print(BANNER)
    print(f"collecting activations: {len(dual)} dual-use, {len(harm)} benign")

    mu_dual = collect(dual)
    if len(mu_dual) != nL:
        raise SystemExit(
            f"collected {len(mu_dual)} of {nL} layers' activations -- the "
            f"forward hooks did not fire. Refusing to compute directions from "
            f"partial data.")
    mu_harm = collect(harm)
    if len(mu_harm) != nL:
        raise SystemExit(
            f"collected {len(mu_harm)} of {nL} layers' activations.")

    # A direction that is near-zero means the two prompt sets did not separate
    # in this layer. Projecting it out would be noise.
    weak = []
    directions = {}
    for i in range(nL):
        diff = mu_dual[i] - mu_harm[i]
        norm = diff.norm().item()
        if norm < 1e-6:
            weak.append(i)
        directions[i] = (diff / diff.norm()).to(torch.float16).to(model.device)
    print(f"{nL} directions computed")
    if weak:
        print(f"WARNING: {len(weak)} layer(s) had near-zero separation "
              f"(layers {weak[:8]}...). Those directions are noise.")
    torch.save({i: v.cpu() for i, v in directions.items()},
               d / "refusal_directions.pt")

    @torch.no_grad()
    def project(W, r):
        W = W.to(torch.float32)
        r = r.to(torch.float32)
        W = W - strength * torch.outer(W @ r, r)
        return W.to(torch.float16)

    n = 0
    for i, layer in enumerate(layers):
        r = directions[i]
        for mod in (layer.self_attn.o_proj, layer.mlp.down_proj):
            mod.weight.data = project(mod.weight.data, r)
            n += 1
    print(f"patched {n} projections across {nL} layers")
    torch.cuda.empty_cache()

    gen = make_generate()
    code, rows = run_suite(gen, "phase1_abliterated")
    ref = refusal_rate(gen)
    print(f"\nafter abliteration: {code}/{len(EVAL)} coding, {ref:.0%} refusal")

    torch.save({i: v.cpu() for i, v in directions.items()},
               d / "refusal_directions.pt")
    print(f"saved {d/'refusal_directions.pt'}")

    prev = json.load(open(d / "phase0_summary.json")) if (
        d / "phase0_summary.json").exists() else {}
    base_code = prev.get("baseline_code")
    base_ref = prev.get("baseline_refusal")
    summary = {"abliterated_code": code, "abliterated_refusal": ref,
               "n_layers": nL, "strength": strength, "tasks": rows}
    if base_code is not None:
        cost = base_code - code
        summary.update(baseline_code=base_code, baseline_refusal=base_ref,
                       capability_cost=cost)
        verdict = "PASS" if (ref <= 0.10 and cost <= 0.20) else "FAIL"
        print(BANNER)
        print("RESULTS")
        print(BANNER)
        print(f"refusal   {base_ref:.0%}  ->  {ref:.0%}")
        print(f"coding    {base_code}/{len(EVAL)}  ->  {code}/{len(EVAL)}")
        print(f"cost      {cost} task(s)  (target <= 2)")
        print(f"GATE      {verdict}")
        if ref > 0.10:
            print("-> refusals survived. Try strength=1.5, or add dual prompts.")
        if cost > 0.20:
            print("-> over-ablated. Try strength=0.7 and re-run phase 0 first.")
        summary["verdict"] = verdict
    save_json(summary, "phase1_summary.json", d)
    if base_code is not None:
        save_json(summary, "phase0_summary.json", d)
    return code, ref


# ---------------------------------------------------------------- cli

def main(argv):
    cmd = argv[1] if len(argv) > 1 else "all"

    # ALWAYS selftest first, whatever the command. Costs ~10s.
    print(BANNER)
    print("SELFTEST -- reward signal, stub model, no GPU")
    print(BANNER)
    ok, fails = self_test()
    if not ok:
        print("\nABORTED: the harness is broken. Do NOT download a model.")
        for f in fails:
            print("  -", f)
        return 1
    print("\nselftest passed; harness is trustworthy.\n")

    if cmd == "selftest":
        return 0
    if cmd == "phase0":
        phase0()
        return 0
    if cmd == "phase1":
        phase1()
        return 0
    if cmd in ("all", "phase01"):
        phase0()
        phase1()
        return 0

    print(f"unknown command {cmd!r}. "
          "use: selftest | phase0 | phase1 | all")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))