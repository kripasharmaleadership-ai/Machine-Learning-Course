"""
Run every lab in order and report pass/fail + runtime.
Use it before recording, after upgrading libraries, or in CI.
The first run trains and caches the TinyGPT models in outputs/ (a few minutes); later runs reuse them.

    python run_all.py            # run all labs (+ the assignment starter)
    python run_all.py 11 15      # run only labs whose file name starts with 11 or 15
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
labs = sorted(p for p in ROOT.glob("[0-9][0-9]_*.py")) + [ROOT / "assignment_starter.py"]
if len(sys.argv) > 1:
    labs = [p for p in labs if p.name.startswith(tuple(sys.argv[1:]))]

results = []
for lab in labs:
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, lab.name], cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    ok = proc.returncode == 0
    warned = "Warning" in proc.stderr
    results.append((lab.name, ok, elapsed))
    print(f"{'PASS' if ok else 'FAIL'}  {elapsed:6.1f}s  {lab.name}{'   (warnings on stderr)' if ok and warned else ''}")
    if not ok:
        print(proc.stderr[-2000:])

failed = [name for name, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} labs passed")
sys.exit(1 if failed else 0)
