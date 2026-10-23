"""R06 mechanism-fidelity ladder: anchor rungs fixed from the literature BEFORE any run.

A rung is a codified anchor of a given mechanistic content. Every rung of a domain is
built on IDENTICAL rows (the R05 loader output), so the data-only arm (DATA-RE), which
ignores the anchor, must give identical results across rungs; the runner asserts this.

Shear domains (FRP, SFRC)
  K1  base code equation, as in R05: ACI 440.1R-15 V_c (FRP); Narayanan & Darwish 1987 (SFRC).
      Neither contains a size term.
  K2  K1 x lambda_s(d), lambda_s = sqrt(2 / (1 + 0.004 d)) <= 1, d in mm
      (ACI 318-19 Eq. 22.5.5.1.3, size-effect factor for members without shear
      reinforcement; Bazant-type fracture-mechanics size effect). Applying it to FRP-RC
      and SFRC is the hypothesis that the size effect of the concrete shear contribution
      is a property of the concrete, not of the reinforcement type. Coded/free sets are
      unchanged: the anchor now asserts a depth exponent of 1 + d ln(lambda_s)/d ln(d)
      instead of 1.
Carbonation (CARB; rows of the R05 CARB loader, which require reactive CaO)
  K1  k ~ C^(1/2) (Fick diffusion with a sharp front); coded: x_co2; free: x_cao, x_wb.
  K2  k ~ (C / a)^(1/2), binding capacity a ~ reactive CaO (Papadakis et al. 1991);
      coded: x_co2, x_cao; free: x_wb. This is the R05 CARB anchor.
  K3  k ~ C^(1/2) (w/b) / CaO_r: the CO2-buffer-capacity-per-paste-volume signature of
      Leemann et al. (2015, Cem. Concr. Compos.) and Leemann & Moro (2017, Mater. Struct.
      50:30, doi:10.1617/s11527-016-0917-2), who report the carbonation coefficient to be
      proportional to w/CaO_reactive; coded: x_co2 (+1/2), x_wb (+1), x_cao (-1); no free.
      Added 2026-10-22 after the K1/K2 theory inputs were seen, before any carbonation
      ladder outcome was seen.

Deep beams (DEEP; mechanism switch, added after the author released the asset on 2026-10-22)
  Source: 840-test RC deep-beam database of Megahed (2024, Sci. Rep. 14:14590,
  doi:10.1038/s41598-024-64386-w, CC BY 4.0; file deep_beam2222.xlsx from
  github.com/kmegahed/Deep-beam-ML-models, SHA-256 D9E8BE7D...43EE). Rows used: no web
  reinforcement (rho_v = rho_h = 0), a/d <= 2.5, all inputs positive. Programme = "Author"
  field (publication). Coded exponents in every rung: d, b_w, f'c, rho, a/d; no free variable
  in the primary analysis ("DEEP"); w_bp/d free in the sensitivity analysis ("DEEP-WB").
  BEAM    ACI 318-19 Table 22.5.5.1 (A_v < A_v,min): V_c = 0.66 lambda_s rho^(1/3) sqrt(f'c) b_w d.
          The code's one-way (beam-action) shear strength, used OUTSIDE its scope: it asserts
          no a/d dependence. The "wrong mechanism" rung.
  ARCH    Zsutty (1971, ACI J. 68(2):138-143) short-beam equation, a/d < 2.5:
          v = 2.2 (f'c rho d/a)^(1/3) (2.5 d/a) MPa, i.e. exponents f'c 1/3, rho 1/3, a/d -4/3,
          b_w 1, d 1. Arch action codified as a power law; no size term.
  ARCHS   ARCH x lambda_s (size rung inside deep beams).
  ARCH99  ACI 318-99 Eq. (11-29), deep flexural members: V_c = (3.5 - 2.5 M_u/(V_u d))
          (1.9 sqrt(f'c) + 2500 rho_w V_u d/M_u) b_w d [psi], first factor <= 2.5,
          V_c <= 6 sqrt(f'c) b_w d, critical section at 0.5 a <= d so that
          M_u/(V_u d) = min(0.5 a/d, 1). SI constants 0.158, 17.24, 0.5. Sensitivity rung:
          piecewise, often governed by the cap.

Registered 2026-10-22 before the ladder experiment was run (see ../PREREGISTRATION.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R05 = HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts"
sys.path.insert(0, str(R05))
import kag_data_r05 as D  # noqa: E402

# The deep-beam loader, anchor equations and lambda_s live in the shared registry
# (kag_data_r05) so the R05 identification audit covers the domain; aliases keep
# the R06 call sites (`R.load_deep_rows`, `R.lambda_s`, `R.deep_anchor_kN`).
DEEP_FILE = D.DEEP_FILE
DEEP_FEATURES = D.DEEP_FEATURES
load_deep_rows = D.load_deep_rows
deep_anchor_kN = D.deep_anchor_kN
lambda_s = D.lambda_s


def _deep(rung, plate=False):
    """Coded: the five anchor variables. Primary analysis (domain "DEEP"): no free variable,
    like the slender-beam domains. Sensitivity (domain "DEEP-WB"): the relative support-plate
    width w_bp/d as a free variable, about which none of the anchors makes a statement
    (nodal-zone bearing; Megahed's table reports plate widths for every row, some inferred
    from drawings by the compiler). The red-team seat asked for the unadjusted analysis to be
    primary so that the size signature does not rest on a post hoc covariate."""
    df = load_deep_rows()
    df["m"] = np.log(deep_anchor_kN(df, rung))
    return df, list(DEEP_FEATURES), (["ln_wb"] if plate else [])


def _shear(name, rung):
    s = D.spec(name)
    df = D.load(name)
    if rung == "K2":
        df = df.copy()
        df["m"] = df["m"] + np.log(lambda_s(df["d"]))
    elif rung != "K1":
        raise KeyError(rung)
    return df, list(s["coded"]), list(s["free"])


def _carb(rung):
    df = D.load("CARB")
    if rung == "K1":
        df = df.copy()
        df["m"] = 0.5 * df["x_co2"]
        return df, ["x_co2"], ["x_cao", "x_wb"]
    if rung == "K2":
        return df, ["x_co2", "x_cao"], ["x_wb"]
    if rung == "K3":
        df = df.copy()
        df["m"] = 0.5 * df["x_co2"] + 1.0 * df["x_wb"] - 1.0 * df["x_cao"]
        return df, ["x_co2", "x_wb", "x_cao"], []
    raise KeyError(rung)


RUNGS = {"FRP": ("K1", "K2"), "SFRC": ("K1", "K2"), "CARB": ("K1", "K2", "K3"),
         "DEEP": ("BEAM", "ARCH", "ARCHS", "ARCH99")}
SENSITIVITY = {"DEEP-WB": ("BEAM", "ARCH", "ARCHS", "ARCH99")}     # plate-width sensitivity
LABEL = {
    ("FRP", "K1"): "ACI 440.1R-15",
    ("FRP", "K2"): "ACI 440.1R-15 x lambda_s (ACI 318-19)",
    ("SFRC", "K1"): "Narayanan-Darwish 1987",
    ("SFRC", "K2"): "Narayanan-Darwish 1987 x lambda_s (ACI 318-19)",
    ("CARB", "K1"): "sqrt(C), binder free",
    ("CARB", "K2"): "sqrt(C/a), a ~ reactive CaO",
    ("CARB", "K3"): "sqrt(C) (w/b)/CaO_r (Leemann-Moro buffer capacity)",
    ("DEEP", "BEAM"): "ACI 318-19 one-way shear (beam action, out of scope)",
    ("DEEP", "ARCH"): "Zsutty 1971 short-beam (arch action)",
    ("DEEP", "ARCHS"): "Zsutty 1971 x lambda_s",
    ("DEEP", "ARCH99"): "ACI 318-99 Eq. 11-29 deep beam",
}


def load(name, rung):
    """Return (df, coded, free) for a domain and a rung."""
    if name in ("FRP", "SFRC"):
        return _shear(name, rung)
    if name == "CARB":
        return _carb(rung)
    if name == "DEEP":
        return _deep(rung)
    if name == "DEEP-WB":
        return _deep(rung, plate=True)
    raise KeyError(name)


if __name__ == "__main__":
    for name, rungs in list(RUNGS.items()) + list(SENSITIVITY.items()):
        for r in rungs:
            df, coded, free = load(name, r)
            res = df.y - df.m
            lab = LABEL.get((name, r), LABEL.get((name.split("-")[0], r), ""))
            print(f"{name:7s} {r}: n={len(df)} programmes={df.source.nunique()} coded={coded} free={free} "
                  f"| residual sd={res.std():.3f} mean={res.mean():+.3f} | {lab}")
