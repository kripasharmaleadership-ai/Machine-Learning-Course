"""
Run every module's run_all.py (every lab + the assignment starters) and report pass/fail per module.

    python run_all.py          # all 8 modules (about an hour on a laptop CPU)
    python run_all.py 7 8      # only Modules 7 and 8
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
modules = sorted(ROOT.glob("Module[0-9]*_*"), key=lambda p: int(p.name[6:].split("_")[0]))
if len(sys.argv) > 1:
    modules = [m for m in modules if m.name[6:].split("_")[0] in sys.argv[1:]]

results = []
for module in modules:
    print(f"\n=== {module.name} " + "=" * max(0, 70 - len(module.name)), flush=True)
    start = time.perf_counter()
    ok = subprocess.run([sys.executable, "run_all.py"], cwd=module).returncode == 0
    results.append((module.name, ok, time.perf_counter() - start))

print("\n" + "=" * 78)
for name, ok, elapsed in results:
    print(f"{'PASS' if ok else 'FAIL'}  {elapsed / 60:5.1f} min  {name}")
failed = [name for name, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} modules passed")
sys.exit(1 if failed else 0)
