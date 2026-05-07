import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
ROOT = Path("c:/Users/Dell/Desktop/Sleep-Token")
NAMES = ["BENCHMARKS_VS_PUBLIC.md","DEMO_SCRIPT.md","DEPLOY_HF_SPACE.md","EXECUTIVE_SUMMARY.md",
         "FINAL_DEMO.md","PYTORCH_STORY.md","RESULTS.md","AUDIT_PLAN.md","JUDGES.md",
         "ALIENWARE_KICKOFF.md","DATA_SOURCES.md","EXTERNAL_CREDIBILITY.md","SUPPLYMIND_BLUEPRINT.md"]
skip = ["_dump/", ".venv/", ".git/", "ROLL-main/", "models/", "v3_arcadia/",
        "ShAuRyA_Phoenix/upstream_prs/", "external_data/", "wandb/"]
problems = []
for p in ROOT.rglob("*"):
    if not p.is_file(): continue
    rel = p.relative_to(ROOT).as_posix()
    if any(rel.startswith(s) for s in skip): continue
    if p.suffix not in (".md",".py",".html",".sh",".yml",".yaml",".toml",".ipynb"): continue
    try: text = p.read_text("utf-8", errors="ignore")
    except: continue
    for n in NAMES:
        pat = r"(?<![A-Za-z0-9_/])" + re.escape(n)
        for m in re.finditer(pat, text):
            ls = text.rfind(chr(10), 0, m.start()) + 1
            le = text.find(chr(10), m.end())
            line = text[ls:le if le > 0 else len(text)]
            if "tree" in line.lower(): continue
            if any(c in line for c in "│├└"): continue
            problems.append((rel, n, line.strip()[:140]))
print("STALE BARE REFS:", len(problems))
for r, n, l in problems[:80]:
    print("FILE:", r, "NAME:", n)
    print("  LINE:", l[:120])
