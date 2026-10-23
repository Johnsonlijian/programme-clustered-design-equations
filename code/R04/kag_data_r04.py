"""R03 data: a concrete *material* domain with a documented codified exponent.

Domain CARB -- natural and accelerated carbonation of concrete.
Source: RILEM TC 281-CCC derived observations, distributed in the Zenodo archive
10.5281/zenodo.21830461 (derived files CC BY 4.0), itself derived from
Vollpracht et al., Materials and Structures 57 (2024) 206, doi:10.1617/s11527-024-02465-0
(article and supplement CC BY 4.0). Both licences were read from the archive's own
LICENSE_OUTPUTS_AND_DERIVED_DATA / DATA_ATTRIBUTION_AND_CHANGES.md, not inferred.

Codified knowledge
------------------
The diffusion-controlled carbonation law used in fib Model Code / CEB durability
design states that the carbonation depth grows with the square root of exposure
time and with the square root of the CO2 concentration:

    x = k * sqrt(t),        k proportional to sqrt(C_CO2)

so the *codified* content is the exponent 1/2 on the CO2 concentration, while the
absolute rate constant is calibrated on data. In log form the anchor is

    m = 0.5 * ln(C / C_ref),   C_ref = 0.04 % (atmospheric)

and the model estimates deviations from that exponent together with a free exponent
on the water-to-binder ratio:

    y = ln k = m + c + b1 * ln(C/C_ref) + b2 * ln(w/b) + u_programme + e

Programme identity = the publication reference of each observation (RILEM `reference`
column), i.e. the laboratory/campaign that produced it.

Nothing is imputed: rows without a positive rate, a CO2 concentration or a w/b ratio
are dropped and counted in the manifest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R02 = Path(__file__).resolve().parents[1].parent / "R02_crossdomain_theory_2026-10-20" / "scripts"
WS = Path(__file__).resolve().parents[4]
SRC = (WS / "2026-AEI-Jev-Concrete-EvidenceSelection" / "data" / "external" / "zenodo_21830461"
       / "unpacked" / "ccr_natural_carbonation_ml_audit_v2_20260806" / "data"
       / "rilem_carbonation_observations_v2.csv")

C_REF = 0.04          # atmospheric CO2, % by volume
FEATURES = ["x_co2", "x_wb"]


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_carb() -> pd.DataFrame:
    raw = pd.read_csv(SRC, low_memory=False)
    df = pd.DataFrame({
        "source": raw["reference"].astype(str).str.strip(),
        "rate": pd.to_numeric(raw["carbonation_rate_mm_sqrt_d"], errors="coerce"),
        "co2": pd.to_numeric(raw["co2_vol_pct"], errors="coerce"),
        "wb": pd.to_numeric(raw["w_b"], errors="coerce"),
        "fcm28": pd.to_numeric(raw["fcm28_mpa"], errors="coerce"),
        "exposure": raw["exposure_block"].astype(str),
    })
    n0 = len(df)
    df = df[(df.rate > 0) & (df.co2 > 0) & (df.wb > 0) & (df.source != "nan")]
    df = df.reset_index(drop=True)
    df["y"] = np.log(df.rate)
    df["x_co2"] = np.log(df.co2 / C_REF)
    df["m"] = 0.5 * df.x_co2                     # codified sqrt(C) scaling
    df["x_wb"] = np.log(df.wb)
    df.attrs["dropped"] = int(n0 - len(df))
    return df


if __name__ == "__main__":
    d = load_carb()
    g = d.groupby("source").size()
    print(f"CARB n={len(d)} programmes={g.size} ge6={(g >= 6).sum()} "
          f"specimens_in_ge6={g[g >= 6].sum()} dropped={d.attrs['dropped']}")
    print(f"CO2 levels: {sorted(d.co2.round(2).unique())[:6]} ... n_levels={d.co2.nunique()}")
    print(f"rate mm/sqrt(d): median={d.rate.median():.3f} p05={d.rate.quantile(.05):.3f} "
          f"p95={d.rate.quantile(.95):.3f}")
    print(f"w/b: median={d.wb.median():.3f} range={d.wb.min():.2f}-{d.wb.max():.2f}")
    print(f"sha256={sha256(SRC)}")
