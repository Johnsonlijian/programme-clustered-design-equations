"""Scan local CSV headers for test databases that carry a programme/source column.

Reads only the first line of each CSV (plus a couple of rows when needed) under
1_In_Preparation, and reports files whose header contains a provenance-like column
(reference, author, study, lab, source, publication) together with a numeric target
column. Output is a candidate list for manual licence/anchor review, not a decision.

Output: candidate_audits/kag_r03_header_scan.csv
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

ROOT = Path(r"R:\IMUT_WORKPOOL\01_PROJECTS\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation")
OUTDIR = Path(__file__).resolve().parents[1] / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

PROV = ("reference", "author", "source", "study", "lab", "publication", "ref.", "citation",
        "first author", "specimen id", "test id", "programme", "campaign")
SKIP_DIRS = {"node_modules", "__pycache__", ".git", ".venvs", "venv"}
MIN_BYTES = 40_000


def header_of(p: Path):
    for enc in ("utf-8", "latin-1"):
        try:
            with io.open(p, "r", encoding=enc, errors="strict") as f:
                line = f.readline()
            if not line:
                return None, enc
            return line.strip(), enc
        except Exception:
            continue
    return None, None


def main():
    rows = []
    n_seen = 0
    import os
    targets = []
    for dirpath, dirnames, filenames in os.walk(ROOT, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
        if dirpath.count(os.sep) - str(ROOT).count(os.sep) > 7:
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                targets.append(Path(dirpath) / fn)
    for p in targets:
        try:
            if p.stat().st_size < MIN_BYTES:
                continue
        except OSError:
            continue
        n_seen += 1
        hdr, enc = header_of(p)
        if not hdr:
            continue
        low = hdr.lower()
        hits = [k for k in PROV if k in low]
        if not hits:
            continue
        cols = [c.strip().strip('"') for c in hdr.split(",")]
        rows.append(dict(path=str(p), bytes=p.stat().st_size, encoding=enc,
                         provenance_hits=";".join(sorted(set(hits))),
                         n_cols=len(cols), header=hdr[:400]))
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("bytes", ascending=False)
    df.to_csv(OUTDIR / "header_scan.csv", index=False, encoding="utf-8")
    print(f"scanned {n_seen} csv files, {len(df)} carry a provenance-like column")


if __name__ == "__main__":
    main()
