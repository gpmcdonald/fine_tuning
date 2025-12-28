import sys
import re
import subprocess

def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def parse_requirements(path: str):
    pins = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # only enforce exact pins (pkg==ver)
            m = re.match(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)$", s)
            if m:
                pins[m.group(1).lower()] = m.group(2)
    return pins

def parse_pip_freeze(text: str):
    installed = {}
    for line in text.splitlines():
        line = line.strip()
        if "==" in line:
            pkg, ver = line.split("==", 1)
            installed[pkg.lower()] = ver
    return installed

def main():
    req_path = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    pins = parse_requirements(req_path)
    if not pins:
        print(f"[WARN] No exact pins found in {req_path}. Use pkg==ver for clean reproducibility.")
    print(f"[+] Requirements file: {req_path}")
    print(f"[+] Pinned packages: {len(pins)}")

    freeze = run([sys.executable, "-m", "pip", "freeze"])
    installed = parse_pip_freeze(freeze)

    missing = []
    mismatch = []
    ok = []

    for pkg, ver in pins.items():
        inst = installed.get(pkg)
        if inst is None:
            missing.append((pkg, ver))
        elif inst != ver:
            mismatch.append((pkg, ver, inst))
        else:
            ok.append((pkg, ver))

    for pkg, ver in sorted(ok):
        print(f"[OK] {pkg}=={ver}")

    if missing:
        print("\nMissing:")
        for pkg, ver in missing:
            print(f"  - {pkg}=={ver}")

    if mismatch:
        print("\nMismatched:")
        for pkg, want, got in mismatch:
            print(f"  - {pkg}: want {want}, got {got}")

    if missing or mismatch:
        raise SystemExit(1)

    print("\n[✓] Environment matches pinned requirements exactly.")

if __name__ == "__main__":
    main()