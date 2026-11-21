"""Confirm the fixes are in BOTH the source files AND the built notebook.

Editing app.py but forgetting to rebuild means Colab still gets the old code
embedded in the payload, and the fix silently does nothing. So check the shipped
artifact, not just the source.
"""
import base64
import json
import zlib
from pathlib import Path

D = Path(__file__).parent
fails = []


def chk(label, got, want=True):
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"          got={got!r} want={want!r}")
        fails.append(label)


print("=" * 66)
print("1. source files")
print("=" * 66)
app = (D / "app.py").read_text(encoding="utf-8")
com = (D / "common.py").read_text(encoding="utf-8")
bld = (D / "build.py").read_text(encoding="utf-8")

chk("common.py defines dtype_kwargs", "def dtype_kwargs" in com)
chk("common.py call site uses it", "**dtype_kwargs(torch))" in com)
chk("app.py call site uses it", "**dtype_kwargs(torch))" in app)
chk("app.py has no hardcoded load dtype", "torch_dtype=torch.float16)" not in app)
chk("build.py installs bitsandbytes", "bitsandbytes>=0.46.1" in bld)
chk("build.py no longer pins transformers", "transformers==" not in bld)

print("\n" + "=" * 66)
print("2. the shipped notebook")
print("=" * 66)
nb = json.loads((D / "rsl_all.ipynb").read_text(encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
setup = "".join(code[0]["source"])
payload = "".join(code[1]["source"])

chk("cell 1 pip installs bitsandbytes", "bitsandbytes>=0.46.1" in setup)
chk("cell 1 prints the installed version", "bitsandbytes.__version__" in setup)
chk("cell 1 pins nothing that can drift", "transformers==" not in setup)
chk("cell 1 still creates dirs", 'RSL = "/content/rsl"' in setup)
chk("cell 1 still mounts Drive", "drive.mount" in setup)

marker = '_BLOB = "'
chk("payload cell has the blob", marker in payload)
blob = payload.split(marker)[1].split('"')[0]
files = json.loads(zlib.decompress(base64.b64decode(blob)).decode())
chk("21 files embedded", len(files), 21)

emb_app = base64.b64decode(files["app.py"]).decode()
emb_com = base64.b64decode(files["common.py"]).decode()

print("\n  embedded copies carry the fix:")
chk("embedded app.py uses dtype_kwargs", "dtype_kwargs(torch)" in emb_app)
chk("embedded common.py defines it", "def dtype_kwargs" in emb_com)
chk("embedded app.py clean of hardcoded dtype",
    "torch_dtype=torch.float16)" not in emb_app)

# embedded bytes must equal the files on disk, or the notebook is stale
stale = []
for rel, b64 in files.items():
    src = D / rel
    if src.exists() and base64.b64decode(b64) != src.read_bytes():
        stale.append(rel)
chk("no stale files in the payload", stale, [])

print("\n" + "=" * 66)
print(f"{len(fails)} failure(s)")
for f in fails:
    print("  FAIL", f)
print("=" * 66)
raise SystemExit(1 if fails else 0)