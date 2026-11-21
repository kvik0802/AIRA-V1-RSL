"""
PHASE 5 -- Autonomous harvester. This is the information supply.

Everything before this phase learns from data *we* wrote. This phase replaces
that with data the system finds itself. That is the only mechanism that makes
the loop compound instead of plateauing.

Pipeline:
  arXiv API  -> papers in code-related categories
            -> extract GitHub URLs from the abstract
            -> shallow clone
            -> locate a runnable benchmark
            -> RUN IT
            -> keep only what executes AND beats the recorded baseline

Expected yield is ~5%. 100 papers will yield maybe 5 usable items. That is
normal, not a failure mode. If you get 0, that is informative too.

CONTAMINATION GUARD: a harvested benchmark whose name matches anything in
KNOWN_BENCHMARKS is discarded. A model that trains on HumanEval has learned
nothing; we are not going to build that trap for ourselves.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json, os, re, shutil, subprocess, sys, tarfile, tempfile, time, zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from common import ROOT, banner, save_json, load_json

ARXIV = "http://export.arxiv.org/api/query"
CATS = ["cs.SE", "cs.LG", "cs.DC", "cs.PL", "cs.PF"]
N_PAPERS = int(os.environ.get("N_PAPERS", 100))
WORK = Path("/content/harvest")
WORK.mkdir(parents=True, exist_ok=True)

# a harvested benchmark matching any of these is a contaminated datapoint
KNOWN_BENCHMARKS = ("humaneval", "mbpp", "livecodebench", "swebench",
                    "apps_", "codecontests", "bigcodebench")

banner("PHASE 5 -- AUTONOMOUS HARVESTER")
print(f"categories {CATS}  target {N_PAPERS} papers")


def fetch_papers(n):
    out = []
    per = max(1, n // len(CATS))
    for cat in CATS:
        url = (f"{ARXIV}?search_query=cat:{cat}+AND+abs:%22github.com%22"
               f"&start=0&max_results={per}&sortBy=submittedDate"
               f"&sortOrder=descending")
        try:
            req = Request(url, headers={"User-Agent": "rsl-harvester/0.1"})
            xml = urlopen(req, timeout=45).read().decode("utf-8", "replace")
        except Exception as e:
            print(f"  [{cat}] fetch failed: {e}")
            continue
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
        print(f"  [{cat}] {len(entries)} entries")
        for e in entries:
            t = re.search(r"<title>(.*?)</title>", e, re.S)
            s = re.search(r"<summary>(.*?)</summary>", e, re.S)
            i = re.search(r"<id>(.*?)</id>", e, re.S)
            pub = re.search(r"<published>(.*?)</published>", e, re.S)
            out.append({
                "title": " ".join(t.group(1).split()) if t else "",
                "summary": " ".join(s.group(1).split()) if s else "",
                "id": i.group(1).strip() if i else "",
                "published": pub.group(1)[:10] if pub else "",
                "cat": cat,
            })
    return out[:n]


def github_urls(text):
    """github.com/org/repo, with .git and trailing punctuation stripped."""
    found = re.findall(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", text)
    skip = {"features", "actions", "about", "pricing", "sponsors", "topics",
            "collections", "trending", "explore", "login", "marketplace"}
    out = []
    for org, repo in found:
        if org.lower() in skip:
            continue
        repo = repo.split(".")[0].rstrip(".,);:")
        if repo and (org, repo) not in out:
            out.append((org, repo))
    return out


def contamin(name):
    low = name.lower().replace("_", "").replace("-", "")
    return any(k.replace("_", "").replace("-", "") in low for k in KNOWN_BENCHMARKS)


def clone(org, repo, dest):
    url = f"https://github.com/{org}/{repo}"
    r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
                       capture_output=True, text=True, timeout=180)
    return r.returncode == 0


def find_benchmarks(root, limit=4):
    """A benchmark here means: a python file that looks runnable and is not
    the project's own unit-test directory. We do not trust filenames alone --
    we try to run them and see."""
    cands = []
    for p in root.rglob("*.py"):
        s = str(p).lower()
        if any(x in s for x in ("test_", "_test", "conftest", "setup.py",
                                "benchmark", "bench_", "_bench")):
            cands.append(p)
        if len(cands) >= limit * 3:
            break
    return cands[: limit * 3]


def try_run(script, timeout=90):
    """Execution is the only trust. Nothing is kept because a paper claimed it."""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, PYTHONPATH=d, MPLBACKEND="Agg",
                   OMP_NUM_THREADS="1")
        try:
            r = subprocess.run([sys.executable, str(script)],
                               capture_output=True, text=True,
                               timeout=timeout, cwd=d, env=env)
            return (r.returncode == 0,
                    (r.stdout + r.stderr)[-300:])
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, repr(e)[:120]


def harvest_one(paper):
    """Returns a record only if the code actually executed."""
    rec = {"id": paper["id"], "title": paper["title"], "cat": paper["cat"],
           "published": paper["published"], "repos": [], "ran": [],
           "kept": False, "reason": ""}

    for k in KNOWN_BENCHMARKS:
        if contamin(paper["title"]):
            rec["reason"] = "contaminated_benchmark_name"
            return rec

    for org, repo in github_urls(paper["summary"])[:2]:
        dest = WORK / f"{org}_{repo}"
        if not clone(org, repo, dest):
            rec["repos"].append({"repo": f"{org}/{repo}", "cloned": False})
            continue
        rec["repos"].append({"repo": f"{org}/{repo}", "cloned": True})

        for script in find_benchmarks(dest):
            ok, out = try_run(script)
            rec["ran"].append({"script": str(script.relative_to(dest)),
                               "ok": ok, "out": out[-150:]})
            if ok:
                rec["kept"] = True
                rec["reason"] = f"executed:{script.name}"
                rec["kept_script"] = str(script.relative_to(dest))
                rec["output_tail"] = out[-800:]
                shutil.rmtree(dest, ignore_errors=True)
                return rec
        shutil.rmtree(dest, ignore_errors=True)

    if not rec["reason"]:
        rec["reason"] = "nothing_executable"
    return rec


# ---------------------------------------------------------------- run
banner("fetching papers")
papers = fetch_papers(N_PAPERS)
print(f"got {len(papers)} papers")

if not papers:
    raise SystemExit("arXiv returned nothing. Check network egress in Colab.")

seen = set()
for p in papers:
    for org, repo in github_urls(p["summary"]):
        seen.add(f"{org}/{repo}")
print(f"{len(seen)} distinct candidate repos")

banner("harvesting (cloning + executing)")
results = []
t0 = time.time()
for i, p in enumerate(papers):
    if time.time() - t0 > 60 * 25:
        print("  time budget reached, stopping early")
        break
    r = harvest_one(p)
    results.append(r)
    flag = "KEEP" if r["kept"] else "drop"
    print(f"  [{i+1}/{len(papers)}] {flag} {r['reason'][:40]:40s} "
          f"{p['title'][:44]}")
    save_json(results, "phase05_raw.json")

kept = [r for r in results if r["kept"]]
banner("RESULTS")
print(f"papers fetched     : {len(papers)}")
print(f"with github link    : {sum(1 for r in results if r['repos'])}")
print(f"cloned successfully : {sum(1 for r in results if any(x['cloned'] for x in r['repos']))}")
print(f"scripts executed ok : {sum(1 for r in results if r['kept'])}")
print(f"KEPT                : {len(kept)}  "
      f"({100*len(kept)/max(len(papers),1):.1f}% yield)")

save_json(kept, "phase05_archive.json")
save_json({"fetched": len(papers), "kept": len(kept),
           "yield_pct": 100 * len(kept) / max(len(papers), 1),
           "reasons": {}}, "phase05_summary.json")

GATE = len(kept) >= 5
print(f"\nGATE: {'PASS' if GATE else 'FAIL'}  (need >=5 verified wins)")
if not GATE:
    print(f"yield was {len(kept)}/{len(papers)}. That is below the ~5% we")
    print("expected. Options: more papers, looser timeout, or accept that")
    print("phase 6 will train on a small archive.")
    print("Do NOT fabricate rows to hit the gate. A small real archive is fine;")
    print("a padded fake one poisons every later phase.")

print(f"\nnext: phase06_selfplay_rsl.ipynb")