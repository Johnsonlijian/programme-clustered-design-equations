"""Second pass: from the header scan, keep files that look like *test databases*.

Excludes analysis outputs (rounds, outputs, manuscript, derived, qa, logs, figures,
repro, submission) and requires a provenance column together with a measured-target
column. Prints the shortlist with its header for human licence/anchor review.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
BAD = ("\\outputs\\", "\\rounds\\", "\\manuscript", "\\derived\\", "\\qa", "\\logs\\",
       "\\figures\\", "\\repro", "\\submission", "\\analysis\\", "\\literature",
       "\\_qa", "\\tmp", "\\work_temp", "\\cache")
TARGET = ("test", "vmax", "strength", "load", "cap", "defl", "creep", "fatigue", "life",
          "modulus", "strain", "depth", "mpa", "kn", "fck", "fc", "vu", "v_", "peak", "force")


def main():
    df = pd.read_csv(OUT / "header_scan.csv")
    low = df.path.str.lower()
    keep = df[~low.str.contains("|".join(x.replace("\\", "\\\\") for x in BAD), regex=True)]
    hdr = keep.header.fillna("").str.lower()
    hit = keep[hdr.str.contains("|".join(TARGET), regex=True)]
    hit = hit.sort_values("bytes", ascending=False)
    hit.to_csv(OUT / "header_scan_filtered.csv", index=False)
    print(f"{len(hit)} candidate files after filtering\n")
    for _, r in hit.head(45).iterrows():
        p = r.path
        short = p.split("1_In_Preparation")[-1]
        print(f"{r.bytes/1024:8.0f} KB | {short[:150]}")
        print(f"            {r.header[:220]}")


if __name__ == "__main__":
    main()
