"""R05-E1b: sensitivity of the clustering diagnostics to the programme definition.

The databases identify programmes by publication. Several publications can come from
one laboratory; if laboratory effects persist across them, publication-level grouping
understates the shared error. Here publications are merged into a laboratory proxy --
the first author's surname (shear databases) or the alphabetic prefix of the source code
(carbonation) -- and ICC, design effect and effective sample size are recomputed.

Output: outputs/audit_labs.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kag_data_r05 as D  # noqa: E402
from kag_models_r05 import oneway_reml  # noqa: E402

OUT = HERE.parent / "outputs"


def lab_of(source: str, domain: str) -> str:
    s = str(source)
    if domain.startswith("CARB"):
        m = re.match(r"[A-Za-z]+", s)
        return (m.group(0) if m else s).lower()
    s = re.split(r"\s+(?:et\s+al\.?|and|&)\s+|\(|\d|PhDTh|PhD|MSc", s)[0]
    return re.sub(r"[^A-Za-z\- ]", "", s).strip().lower() or str(source)


def stats(df, groups):
    r = df.y.values - df.m.values
    c, tau, sig = oneway_reml(r, groups)
    n_g = pd.Series(groups).value_counts().values
    icc = tau ** 2 / (tau ** 2 + sig ** 2)
    ntil = float(np.sum(n_g ** 2) / n_g.sum())
    deff = 1 + (ntil - 1) * icc
    return dict(groups=int(n_g.size), tau=tau, sigma=sig, icc=icc, design_effect=deff, effective_n=len(df) / deff)


def main():
    rows = []
    for name in list(D.DOMAINS) + list(D.SENSITIVITY):
        df = D.load(name)
        pub = stats(df, df.source.values)
        labs = df.source.map(lambda s: lab_of(s, name)).values
        lab = stats(df, labs)
        rows.append(dict(domain=name, n=len(df), **{f"pub_{k}": v for k, v in pub.items()},
                         **{f"lab_{k}": v for k, v in lab.items()}))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "audit_labs.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
