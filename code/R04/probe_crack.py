"""Probe: can an EC2 (EN 1992-1-1 7.3.4) crack-width anchor be rebuilt from the
list-encoded crack database, and is the result sane enough to use as a domain?

The database stores per-element values as text lists, e.g. phi = "[12.7,12.7]",
nbl = "[4,4]" (two layers of four 12.7 mm bars). This probe parses them, forms
    As      = sum_k n_k pi phi_k^2 / 4
    d       = first entry of `zs`           (interpreted as effective depth)
    c       = h - d                          (vertical cover)
    Ac,eff  = 2.5 b c                        (EC2 7.3.4 effective tension area)
    rho_eff = As / Ac,eff
    s_r,max = 3.4 c + 0.425 k1 phi/rho_eff   (k1 = 0.8)
    eps_diff= max([sigs - kt fctm/rho_eff (1 + ae rho_eff)]/Es, 0.6 sigs/Es)
    w_pred  = s_r,max * eps_diff
and compares w_pred with the measured maximum crack width (valcat = 4).

If the measured/predicted ratio is not of order one, the interpretation of the
list columns is wrong and the domain is blocked; the probe prints that verdict.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

P = (Path(r"R:\IMUT_WORKPOOL\01_PROJECTS\NAS_DRIVE\IMUT\1-Research_Output\1-Papers"
          r"\1_In_Preparation\2026-JSE-CrackWindowOrderStatistic\data\raw\zenodo_v5")
     / "00_Master_Database.csv")
OUT = Path(__file__).resolve().parents[1] / "outputs"

NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def first(x, default=np.nan):
    if isinstance(x, str):
        m = NUM.findall(x)
        return float(m[0].replace(",", ".")) if m else default
    try:
        return float(x)
    except Exception:
        return default


def as_sum(x):
    if isinstance(x, str):
        vals = [float(v.replace(",", ".")) for v in NUM.findall(x)]
        return float(np.sum(vals)) if vals else np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    d = pd.read_csv(P, low_memory=False, dtype=str)
    w = d[d.valcat.astype(str) == "4"].copy()
    out = {}
    for c in ["h", "b", "zs", "clat", "sigs", "fctm", "Ecm", "Es"]:
        w[c + "_v"] = w[c].map(first)
    w["As"] = w["nbl"].map(as_sum) * np.pi * w["phi"].map(first) ** 2 / 4.0

    h, b, dd = w.h_v, w.b_v, w.zs_v
    c_v = h - dd
    rho_eff = w.As / (2.5 * b * c_v)
    sr = 3.4 * c_v + 0.425 * 0.8 * w.phi.map(first) / rho_eff
    ae = w.Es_v / w.Ecm_v
    eps = (w.sigs_v - 0.6 * w.fctm_v / rho_eff * (1 + ae * rho_eff)) / w.Es_v
    eps = np.maximum(eps, 0.6 * w.sigs_v / w.Es_v)
    w["w_pred"] = sr * eps
    w["ratio"] = pd.to_numeric(w.value, errors="coerce", downcast=None) / w.w_pred

    ok = w[np.isfinite(w.ratio) & (w.w_pred > 0) & (w.value.astype(float) > 0)]
    out["n_valcat4"] = int(len(w))
    out["n_usable"] = int(len(ok))
    out["ratio_median"] = float(ok.ratio.median()) if len(ok) else None
    out["ratio_p10"] = float(ok.ratio.quantile(.10)) if len(ok) else None
    out["ratio_p90"] = float(ok.ratio.quantile(.90)) if len(ok) else None
    out["programmes"] = int(ok.EPID.nunique()) if len(ok) else 0
    out["cover_median_mm"] = float(np.nanmedian(c_v)) if len(w) else None
    out["d_median_mm"] = float(np.nanmedian(dd))
    out["rho_eff_median"] = float(np.nanmedian(rho_eff))
    out["verdict"] = ("USABLE" if out["ratio_median"] and 0.25 <= out["ratio_median"] <= 4.0
                      else "BLOCKED: ratio not of order one")
    (OUT / "crack_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
