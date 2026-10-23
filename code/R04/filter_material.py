"""Third pass: material/process keywords in the path or header."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
KEYS = ("concrete", "mortar", "asphalt", "timber", "steel", "creep", "fatigue",
        "corrosion", "carbonat", "cement", "composite", "frp", "cfst", "weld",
        "bolt", "print", "compaction", "curing", "shrink", "durab", "timber",
        "brick", "masonry", "soil", "liquef", "clay", "sand")


def main():
    df = pd.read_csv(OUT / "header_scan_filtered.csv")
    low = (df.path + " " + df.header.fillna("")).str.lower()
    m = df[low.str.contains("|".join(KEYS), regex=True)].copy()
    m = m.sort_values("bytes", ascending=False)
    print(f"{len(m)} material/process-flavoured candidates\n")
    seen = set()
    for _, r in m.iterrows():
        short = r.path.split("1_In_Preparation")[-1]
        key = short.split("\\")[1] if short.count("\\") > 1 else short
        if key in seen:
            continue
        seen.add(key)
        print(f"{r.bytes/1024:8.0f} KB | {short[:140]}")
        print(f"            {r.header[:200]}")


if __name__ == "__main__":
    main()
