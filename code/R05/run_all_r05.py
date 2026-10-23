"""Regenerate every R05 output from the final code state and write a hash manifest.

  python -X utf8 run_all_r05.py experiments   # tests + E1-E7 (about one hour on 16 cores)
  python -X utf8 run_all_r05.py report        # figures, tables, number macros, manifest
  python -X utf8 run_all_r05.py all
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R05 = HERE.parent
PY = [sys.executable, "-X", "utf8"]
EXPERIMENTS = [
    ("pytest", PY + ["-m", "pytest", "-q", str(R05 / "tests")]),
    ("audit", PY + ["r05_audit.py"]),
    ("labs", PY + ["r05_labs.py"]),
    ("lopo", PY + ["r05_lopo.py", "lopo"]),
    ("extrap", PY + ["r05_lopo.py", "extrap"]),
    ("circular", PY + ["r05_lopo.py", "circular"]),
    ("scarcity", PY + ["r05_scarcity.py", "scarcity"]),
    ("unit", PY + ["r05_scarcity.py", "unit"]),
    ("theory", PY + ["r05_theory.py"]),
    ("circular_scarcity", PY + ["r05_circular_scarcity.py"]),
]
REPORT = [
    ("figures", PY + ["r05_figures.py"]),
    ("tables", PY + ["r05_tables.py"]),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def run(steps, log):
    for name, cmd in steps:
        t = time.time()
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (R05 / "outputs" / f"log_{name}.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-20000:],
                                                       encoding="utf-8")
        log.append(dict(step=name, returncode=r.returncode, seconds=round(time.time() - t, 1)))
        print(log[-1], flush=True)
        if r.returncode != 0:
            return False
    return True


def manifest(log, t0):
    import joblib, matplotlib, numpy, pandas, scipy, sklearn  # noqa: E401
    sys.path.insert(0, str(HERE))
    import kag_data_r05 as D
    files = sorted([p for p in (R05 / "outputs").glob("*") if p.is_file() and not p.name.startswith("log_")]
                   + [p for p in (R05 / "figures").glob("*.*") if p.is_file()]
                   + [p for p in HERE.glob("*.py")] + [p for p in (R05 / "tests").glob("*.py")])
    gen = R05.parents[1] / "paper" / "manuscript" / "generated"
    files += sorted(gen.glob("*.tex")) if gen.exists() else []
    prev = R05 / "RUN_MANIFEST.json"
    old = json.loads(prev.read_text(encoding="utf-8")).get("steps", []) if prev.exists() else []
    names = {s["step"] for s in log}
    steps = [s for s in old if s["step"] not in names] + log
    m = dict(created_local=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
             python=sys.version.split()[0], platform=platform.platform(),
             packages={x.__name__: x.__version__ for x in (numpy, pandas, scipy, sklearn, joblib, matplotlib)},
             inputs=D.manifest().to_dict(orient="records"), steps=steps,
             files=[dict(path=str(p.relative_to(R05.parents[1])).replace("\\", "/"), bytes=p.stat().st_size,
                         sha256=sha256(p)) for p in files])
    prev.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print("manifest:", len(files), "files")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0, log = time.time(), []
    ok = True
    if mode in ("experiments", "all"):
        ok = run(EXPERIMENTS, log)
    if ok and mode in ("report", "all"):
        ok = run(REPORT, log)
    manifest(log, t0)
