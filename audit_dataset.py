import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ----------------------------
# Config
# ----------------------------
FILES = sys.argv[1:]
if not FILES:
    print("Usage: py audit_dataset.py symoneural_base.jsonl symoneural_v1_1.jsonl")
    raise SystemExit(2)

SHOW_LAST = int(os.getenv("SHOW_LAST", "10"))
FAIL_ON_BAD = os.getenv("FAIL_ON_BAD", "0").strip().lower() in ("1", "true", "yes")

# Coverage checks (case-insensitive; any needle hit counts as OK)
# You can add more rules without changing code.
CHECKS: Dict[str, List[str]] = {
    "offline-first": [
        "offline-first",
        "offline first",
        "self-hosted",
        "self hosted",
        "act as his own cloud",
    ],
    "not neurotech": [
        "not neurotechnology",
        "not a brain interface",
        "not medical",
        "does not read brain signals",
        "reject the false premise",
    ],
    "PowerShell-first": [
        "powershell first",
        "windows powershell",
        "powerShell copy/paste".lower(),
        "label the shell",
        "debian trixie (bash)",
    ],
    "no partial fixes": [
        "no partial file fixes",
        "not doing partial file fixes",
        "provide full file replacements",
        "full file replacements",
        "complete files",
        "full files (not partial)",
    ],
    "report-first diag": [
        "report-first",
        "report first",
        "tool reports",
        "error output",
        "tree.txt",
    ],
    "baselines sacred": [
        "baselines are sacred",
        "baseline is sacred",
        "known-good",
        "authoritative reference",
        "promote the new artifact as the baseline",
    ],
    "no secrets": [
        "never include secrets",
        "do not repeat secret values",
        "do not store secret values",
        "rotate credentials",
        ".env",
        ".gitignore",
    ],
}

# ----------------------------
# Helpers
# ----------------------------
Row = Tuple[str, int, str, str]  # (file, line_no, prompt, completion)

def _safe_open(path: str):
    return open(path, "r", encoding="utf-8", errors="replace")

def _norm(s: str) -> str:
    return (s or "").strip()

def _preview(s: str, n: int = 120) -> str:
    s = _norm(s).replace("\n", " ").replace("\r", " ")
    return s[:n] + ("…" if len(s) > n else "")

# ----------------------------
# Load + Validate
# ----------------------------
rows: List[Row] = []
bad_json = 0
bad_keys = 0
bad_empty = 0

for p in FILES:
    if not os.path.exists(p):
        print(f"[MISSING FILE] {p}")
        raise SystemExit(2)

    with _safe_open(p) as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except Exception as e:
                bad_json += 1
                print(f"[BAD JSON] {p}:{i} {e}")
                continue

            if not isinstance(obj, dict):
                bad_keys += 1
                print(f"[BAD TYPE] {p}:{i} expected JSON object, got {type(obj).__name__}")
                continue

            if "prompt" not in obj or "completion" not in obj:
                bad_keys += 1
                print(f"[BAD KEYS] {p}:{i} missing prompt/completion")
                continue

            prompt = _norm(str(obj["prompt"]))
            completion = _norm(str(obj["completion"]))

            if not prompt or not completion:
                bad_empty += 1
                print(f"[EMPTY] {p}:{i} prompt_len={len(prompt)} completion_len={len(completion)}")
                continue

            rows.append((p, i, prompt, completion))

bad_total = bad_json + bad_keys + bad_empty

print(f"[+] Files: {', '.join(FILES)}")
print(f"[+] Rows: {len(rows)}")
print(f"[+] Bad:  {bad_total}  (json={bad_json}, keys/type={bad_keys}, empty={bad_empty})")

# ----------------------------
# Duplicates
# ----------------------------
prompts = [r[2] for r in rows]
counts = Counter(prompts)

dupes = [(k, v) for k, v in counts.items() if v > 1]
dupes.sort(key=lambda x: x[1], reverse=True)

print(f"[+] Unique prompts: {len(counts)}")
print(f"[+] Duplicate prompts: {len(dupes)}")

if dupes:
    # Also show where duplicates occur
    prompt_locations = defaultdict(list)  # prompt -> list[(file,line)]
    for p, i, pr, _ in rows:
        prompt_locations[pr].append((p, i))

    print("\nTop duplicate prompts (with locations):")
    for pr, v in dupes[:20]:
        locs = ", ".join([f"{fp}:{ln}" for fp, ln in prompt_locations[pr][:6]])
        extra = "…" if len(prompt_locations[pr]) > 6 else ""
        print(f"  x{v:<3} { _preview(pr, 100) }")
        print(f"       {locs}{extra}")

# ----------------------------
# Coverage Scan
# ----------------------------
print("\nCoverage scan:")
all_text = "\n".join([r[2] + "\n" + r[3] for r in rows]).lower()

for name, needles in CHECKS.items():
    hit = any(n.lower() in all_text for n in needles)
    print(f"  {'OK ' if hit else 'MISS'} {name}")

# ----------------------------
# Show tail for sanity
# ----------------------------
print(f"\nLast {SHOW_LAST} examples (by file order):")
for p, i, pr, _ in rows[-SHOW_LAST:]:
    print(f"- {p}:{i}  {_preview(pr, 120)}")

# ----------------------------
# Optional fail
# ----------------------------
if FAIL_ON_BAD and bad_total > 0:
    raise SystemExit(1)