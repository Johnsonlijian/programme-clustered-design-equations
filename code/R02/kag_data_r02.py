"""R02 data registry.

FRP and SFRC are the R01 domains, imported unchanged from the frozen R01 loaders
(read-only local copies held under 2026-DecisionAdmissibleUQ-RealRC). `DEEP` is the
R02 boundary domain: reinforced-concrete deep/short-span beams, a regime in which
the codified one-way shear expression is knowingly outside its stated scope.

DOMAINS carries the provenance and scope flags used by the audit table; nothing
here is copied out of the source project, all files are read in place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R01_SCRIPTS = (Path(__file__).resolve().parents[2]
               / "R01_direction_and_pilot_2026-10-20" / "scripts")
sys.path.insert(0, str(R01_SCRIPTS))

import kag_data as K1        # noqa: E402  frozen R01 loaders (FRP, SFRC)

WS = Path(__file__).resolve().parents[4]          # .../1_In_Preparation
DEEP_FILE = (WS / "2026-DecisionAdmissibleUQ-RealRC" / "data" / "processed"
             / "deepbeam_steel_clean.csv")

FEATURES = {"FRP": K1.FEATURES["FRP"], "SFRC": K1.FEATURES["SFRC"],
            "DEEP": ["ln_d", "ln_bw", "ln_fc", "ln_rho", "ln_ad"]}
FEATURE_LABELS = dict(K1.FEATURE_LABELS)

DOMAINS = {
    "FRP": dict(label="FRP-RC beams without shear reinforcement",
                anchor="ACI 440.1R-15", scope="in scope",
                licence="CC BY-NC 4.0", source="doi:10.15131/shef.data.5057527",
                member="slender beam, a/d >= 2.5"),
    "SFRC": dict(label="steel-fibre RC beams without stirrups",
                 anchor="Narayanan & Darwish (1987)", scope="in scope",
                 licence="CC BY 4.0", source="doi:10.5281/zenodo.2578061",
                 member="slender beam, a/d >= 2.5"),
    "DEEP": dict(label="RC deep / short-span beams without stirrups",
                 anchor="ACI 318-19 one-way shear, Av < Av,min", scope="OUT of scope",
                 licence="not stated in repository", source="github.com/kmegahed/Deep-beam-ML-models",
                 member="deep beam, a/d 0.27-2.50"),
}


def aci318_vc_kN(fc, bw, d, rho):
    """ACI 318-19 Table 22.5.5.1, case Av < Av,min (SI, N/mm/MPa):

        Vc = 0.66 lambda_s (rho_w)^(1/3) sqrt(f'c) b_w d,
        lambda_s = sqrt(2 / (1 + 0.004 d_mm)) <= 1   (size effect, d in mm)

    The expression is the code's *slender-member* one-way shear strength. It is
    applied here to a/d <= 2.5 on purpose: the R02 question includes what happens
    when codified knowledge is used outside its stated scope, and the audit flags
    the domain as out of scope for exactly this reason. It is not the code's
    deep-beam (strut-and-tie) provision and is not presented as such.
    """
    lam_s = np.minimum(1.0, np.sqrt(2.0 / (1.0 + 0.004 * d)))
    return 0.66 * lam_s * rho ** (1 / 3) * np.sqrt(fc) * bw * d / 1e3


def load_deep(min_n=1) -> pd.DataFrame:
    """Deep/short-span beams without vertical shear reinforcement; no imputation."""
    raw = pd.read_csv(DEEP_FILE)
    df = pd.DataFrame({
        "source": raw["Author"].astype(str).str.strip(),
        "specimen": raw["Specimen"].astype(str),
        "bw": pd.to_numeric(raw["b"], errors="coerce"),
        "d": pd.to_numeric(raw["d"], errors="coerce"),
        "fc": pd.to_numeric(raw["fck"], errors="coerce"),
        "rho": pd.to_numeric(raw["rho"], errors="coerce"),
        "a_d": pd.to_numeric(raw["a_d"], errors="coerce"),
        "rho_v": pd.to_numeric(raw["rho_v"], errors="coerce"),
        "V_kN": pd.to_numeric(raw["V"], errors="coerce"),
    })
    df = df.dropna(subset=["bw", "d", "fc", "rho", "a_d", "V_kN"])
    df = df[(df.bw > 0) & (df.d > 0) & (df.fc > 0) & (df.rho > 0) & (df.V_kN > 0)]
    df = df[(df.rho_v.fillna(0.0) <= 0.0) & (df.a_d <= 2.5)].reset_index(drop=True)
    df["V_anchor_kN"] = aci318_vc_kN(df.fc, df.bw, df.d, df.rho)
    df["y"] = np.log(df.V_kN)
    df["m"] = np.log(df.V_anchor_kN)
    df["ln_d"], df["ln_bw"], df["ln_fc"] = np.log(df.d), np.log(df.bw), np.log(df.fc)
    df["ln_rho"], df["ln_ad"] = np.log(df.rho), np.log(df.a_d)
    return df


LOADERS = {"FRP": K1.load_frp, "SFRC": K1.load_sfrc, "DEEP": load_deep}


def load(name: str) -> pd.DataFrame:
    return LOADERS[name]()


if __name__ == "__main__":
    for k in LOADERS:
        d = load(k)
        g = d.groupby("source").size()
        r = np.exp(d.y - d.m)
        print(f"{k:5s} n={len(d):4d} programmes={g.size:3d} ge6={(g >= 6).sum():3d} "
              f"anchor ratio mean={r.mean():.2f} cov={r.std(ddof=1) / r.mean():.2f}")
