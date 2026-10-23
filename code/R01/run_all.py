"""Regenerate every R01 output from the final code state and write a hash manifest.

  python -X utf8 run_all.py                  (from this scripts folder; ~10 min on 16 cores)
  python -X utf8 run_all.py --manifest-only  (re-hash existing outputs only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R01 = HERE.parent
STEPS = [
    [sys.executable, "-X", "utf8", "-m", "pytest", "-q", str(R01 / "tests")],
    [sys.executable, "-X", "utf8", "r01_confounding.py"],
    [sys.executable, "-X", "utf8", "r01_fewshot.py", "--jobs", "10", "--draws", "40"],
    [sys.executable, "-X", "utf8", "r01_scarcity.py", "--B", "20", "--draws", "10", "--jobs", "12"],
    [sys.executable, "-X", "utf8", "r01_extrapolation.py"],
    [sys.executable, "-X", "utf8", "r01_stress.py"],
    [sys.executable, "-X", "utf8", "r01_evidence_unit.py"],
    [sys.executable, "-X", "utf8", "r01_figures.py"],
    [sys.executable, "-X", "utf8", "r01_tables.py"],
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_manifest(log, t0):
    import joblib, matplotlib, numpy, pandas, scipy, sklearn, statsmodels  # noqa: E401
    files = sorted([p for p in (R01 / "outputs").glob("*") if p.is_file() and not p.name.startswith("log_")]
                   + [p for p in (R01 / "figures").glob("*.*") if p.is_file()]
                   + [p for p in (R01 / "note" / "generated").glob("*.tex") if p.is_file()]
                   + [p for p in HERE.glob("*.py")] + [p for p in (R01 / "tests").glob("*.py")])
    manifest = dict(
        created_local=time.strftime("%Y-%m-%d %H:%M:%S"), total_seconds=round(time.time() - t0, 1),
        python=sys.version.split()[0], platform=platform.platform(),
        packages={m.__name__: m.__version__ for m in (numpy, pandas, scipy, sklearn, statsmodels, joblib, matplotlib)},
        steps=log,
        files=[dict(path=str(p.relative_to(R01)).replace("\\", "/"), bytes=p.stat().st_size, sha256=sha256(p))
               for p in files])
    (R01 / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("manifest written:", len(files), "files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args()
    log, t0 = [], time.time()
    if a.manifest_only:
        prev = R01 / "RUN_MANIFEST.json"
        if prev.exists():
            log = json.loads(prev.read_text(encoding="utf-8")).get("steps", [])
        write_manifest(log, t0)
        return
    for cmd in STEPS:
        t = time.time()
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
        stem = "pytest" if cmd[3] == "-m" else Path(cmd[3]).stem
        (R01 / "outputs" / f"log_{stem}.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-20000:],
                                                         encoding="utf-8")
        log.append(dict(step=stem, returncode=r.returncode, seconds=round(time.time() - t, 1)))
        print(log[-1], flush=True)
        if r.returncode != 0:
            break
    write_manifest(log, t0)


if __name__ == "__main__":
    main()
