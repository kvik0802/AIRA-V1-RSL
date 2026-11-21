"""
Shared helpers for phases 2-7. Phase 0/1 use rsl_harness.py instead.

The reward signal lives in rsl_harness.py, NOT here, and is imported rather
than reimplemented. Phases 2-7 previously carried their own copies of
extract_code/run_tests, which is how two divergent versions of the single most
load-bearing function in the project came to exist.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# single source of truth for anything that judges code
from rsl_harness import (  # noqa: F401,E402
    EVAL, INSTR, PROBES, REFUSAL_MARKERS, extract_code, is_refusal,
    refusal_rate, run_suite, run_tests, self_test,
)

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
RUNTIME_GPU = "/content/drive/MyDrive/rsl_model"
LOCAL_OUT = HERE / "out"


def in_colab():
    """True only inside a real Colab runtime.

    Used instead of probing for paths, because path probing is unreliable in
    both directions: on Windows '/content/drive' resolves to 'C:\\content\\drive'
    and can exist by accident, and Colab's own marker file has moved around
    between versions. Checking for the module is unambiguous.
    """
    import importlib.util
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def drive_root():
    """The mounted Drive folder, or None. Checks both the marker file and the
    directory, because which one exists depends on the Colab version."""
    if not in_colab():
        return None
    for p in (Path("/content/drive/MyDrive"), Path("/content/MyDrive")):
        if p.is_dir():
            return p
    return None


def out_dir():
    """Results must survive Colab pre-emption, so prefer Drive.

    Phase 0 once wrote to /content/rsl/out -- session-local, and silently lost
    on disconnect -- because this function keyed off a marker file that was not
    present. Now it asks whether Drive is actually reachable.
    """
    d = drive_root()
    if d is not None:
        target = d / "rsl_model"
        target.mkdir(parents=True, exist_ok=True)
        return target
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    return LOCAL_OUT


def report_warning():
    """Tell the user loudly when results are not being persisted."""
    if drive_root() is None:
        return (f"\n{'!'*68}\n"
                f"WARNING: Drive is not reachable, so results are being written\n"
                f"to {LOCAL_OUT} which is INSIDE the Colab session.\n"
                f"They will be LOST when this runtime disconnects.\n"
                f"Re-run cell 1 and click 'Allow' on the Drive mount prompt.\n"
                f"{'!'*68}\n")
    return ""


ROOT = out_dir()


def banner(msg):
    print("\n" + "=" * 64)
    print(msg)
    print("=" * 64)


def save_json(obj, name):
    p = ROOT / name
    json.dump(obj, open(p, "w"), indent=2)
    print(f"saved {p}")
    return p


def load_json(name, default=None):
    p = ROOT / name
    if not p.exists():
        return default
    try:
        return json.load(open(p))
    except Exception:
        return default


def check_gpu():
    import torch
    banner("ENVIRONMENT")
    print("torch", torch.__version__)
    assert torch.cuda.is_available(), "no GPU -- Colab runtime must be GPU"
    print("device:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported(),
          "(False on T4 is expected -- fp16 is correct)")
    free, total = torch.cuda.mem_get_info()
    print(f"vram free {free/1e9:.1f} / total {total/1e9:.1f} GB")
    return torch.float16


_TFORMERS_VERSION = None


def dtype_kwargs(torch):
    """Return the load-dtype kwarg this transformers version understands.

    Getting this wrong is a silent trap in both directions:
      * `dtype=` is only consumed by transformers >= 4.56. On older versions it
        is passed to Model.__init__ and raises
        "TypeError: Qwen2ForCausalLM.__init__() got an unexpected keyword
        argument 'dtype'".
      * `torch_dtype=` still works on new versions but is deprecated and warns.

    So decide from the installed version rather than guessing -- I guessed wrong
    once already, and it cost a Colab run.

    Reads the version lazily and caches it, so test_dtype.py can pin a version
    without the import living at module scope.
    """
    global _TFORMERS_VERSION
    if _TFORMERS_VERSION is None:
        import transformers
        _TFORMERS_VERSION = transformers.__version__
    try:
        major, minor = (int(x) for x in _TFORMERS_VERSION.split(".")[:2])
    except ValueError:
        return {"torch_dtype": torch.float16}
    if (major, minor) >= (4, 56):
        return {"dtype": torch.float16}
    return {"torch_dtype": torch.float16}


def load_model(model_id=MODEL_ID, train=False, lora_cfg=None):
    """4-bit NF4 base + optional LoRA."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    check_gpu()

    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0},
        **dtype_kwargs(torch))

    if train:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        cfg = lora_cfg or LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        model.gradient_checkpointing_enable()
    else:
        model.eval()

    print(f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
    return model, tok


def generate(model, tok, prompt, max_new_tokens=420, temperature=0.0):
    """Greedy by default. temperature>0 enables sampling, required for self-play
    diversity -- greedy decoding collapses to one answer per prompt."""
    import torch  # local: only needed on the generation path
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok([text], return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens,
                  pad_token_id=tok.eos_token_id)
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kwargs.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**ids, **kwargs)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def make_generate(model, tok):
    """Adapter so run_suite can drive a real model with the same signature it
    uses for stubs."""
    return lambda p, max_new_tokens=420: generate(
        model, tok, p, max_new_tokens=max_new_tokens)


def eval_suite(model, tok, tag):
    p, rows = run_suite(make_generate(model, tok), tag)
    save_json(rows, f"{tag}_results.json")
    return p


def get_layers(model):
    """Reach the decoder layers through any PEFT wrapper.

    PeftModel.model is a LoraModel, not the CausalLM, so `model.model.layers`
    raises AttributeError on an unmerged adapter -- which is exactly the state
    phase 6 abliterates in.
    """
    import torch.nn as nn
    m = model
    for _ in range(6):
        if hasattr(m, "layers") and isinstance(getattr(m, "layers"), nn.ModuleList):
            return m.layers
        for attr in ("model", "base_model", "module"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        else:
            break
    raise AttributeError("could not reach decoder layers")


def ablate_from_directions(model, directions, strength=1.0):
    """Orthogonally remove each layer's refusal direction.

    NOT idempotent in strength: calling twice projects out the same component
    twice. Call exactly once per model object. Keys may be int or str depending
    on how torch saved them, so normalise before lookup.
    """
    import torch

    @torch.no_grad()
    def project(W, r):
        W = W.to(torch.float32)
        r = r.to(torch.float32).to(W.device)
        return (W - strength * torch.outer(W @ r, r)).to(torch.float16)

    dirs = {int(k): v for k, v in directions.items()}
    n = 0
    for i, layer in enumerate(get_layers(model)):
        if i not in dirs:
            continue
        for mod in (layer.self_attn.o_proj, layer.mlp.down_proj):
            mod.weight.data = project(mod.weight.data, dirs[i])
            n += 1
    torch.cuda.empty_cache()
    return n


def require(names, hint=""):
    """Refuse to run if a previous phase's artifacts are missing. A phase that
    silently proceeds on absent inputs produces numbers that mean nothing."""
    missing = [n for n in names if not (ROOT / n).exists()]
    if missing:
        raise SystemExit(
            f"MISSING artifact(s): {', '.join(missing)}\n"
            f"{hint}\nRun the earlier phase first.")
    print("inputs present:", ", ".join(names))