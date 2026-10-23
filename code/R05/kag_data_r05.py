"""R05 domain registry: three multi-programme databases, each with a codified anchor.

Every domain returns a DataFrame with
    source : test-programme identity (publication / test campaign)
    year   : publication year parsed from the source label (for the calibration-
             circularity sensitivity), NaN when absent
    y      : ln(response)
    m      : ln(anchor prediction)
and log-feature columns. Features are split into
    coded  : the anchor makes an assertion about the exponent (including the
             assertion that a variable has no effect); the model learns deviations
             from that assertion under a knowledge-trust prior
    free   : the anchor is silent; the exponent is estimated in every arm

FRP  : Sheffield FRP-RC beams without shear reinforcement, doi:10.15131/shef.data.5057527
       (CC BY-NC 4.0); anchor ACI 440.1R-15 Vc. Loader frozen in R01.
SFRC : Lantsoght SFRC beams without stirrups, doi:10.5281/zenodo.2578061 (CC BY 4.0);
       anchor Narayanan & Darwish (1987). Loader frozen in R01.
CARB : RILEM TC 281-CCC carbonation observations as redistributed in Zenodo
       doi:10.5281/zenodo.21830461 (derived data CC BY 4.0; upstream Vollpracht et al.
       2024, doi:10.1617/s11527-024-02465-0). Anchor: diffusion-binding carbonation
       theory, x = sqrt(2 D C t / a)  =>  k = x / sqrt(t) proportional to
       C^(1/2) a^(-1/2), with the CO2-binding capacity a taken proportional to the
       reactive CaO content per m3 of concrete. Coded exponents: +1/2 on CO2,
       -1/2 on reactive CaO. The diffusivity D is not codified: its w/b dependence is
       a free exponent. Rows without reactive CaO are not imputed (they enter only the
       CO2-only sensitivity domain CARB-CO2).

DEEP : 840-test RC deep-beam database of Megahed (2024, Sci. Rep. 14:14590,
       doi:10.1038/s41598-024-64386-w, CC BY 4.0; deep_beam2222.xlsx from
       github.com/kmegahed/Deep-beam-ML-models). Rows used: no web reinforcement
       (rho_v = rho_h = 0), a/d <= 2.5, all inputs positive => 322 tests in
       48 publications. Programme = "Author" field. Domain anchor: Zsutty 1971
       short-beam (arch action); the ACI 318-19 one-way shear equation is the
       deliberately out-of-scope control rung, not the domain anchor. The asset
       was blocked in R02 while it belonged to an ACI manuscript under review;
       that manuscript was declined on 2026-08-19 and the copyright reverted,
       so the asset became admissible (see 02_overlap_risk_register.md,
       KAG-OV-008). Sensitivity domain DEEP-WB adds the relative support-plate
       width w_bp/d as a free variable (nodal-zone bearing; plate widths are
       reported for every row, some inferred from drawings by the compiler).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R01_SCRIPTS = HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts"
sys.path.insert(0, str(R01_SCRIPTS))
import kag_data as K1  # noqa: E402  frozen R01 loaders

WS = HERE.parents[3]
CARB_FILE = (WS / "2026-AEI-Jev-Concrete-EvidenceSelection" / "data" / "external" / "zenodo_21830461"
             / "unpacked" / "ccr_natural_carbonation_ml_audit_v2_20260806" / "data"
             / "rilem_carbonation_observations_v2.csv")
C_REF = 0.04            # atmospheric CO2, vol-%


def long_path(p: Path) -> str:
    """Windows MAX_PATH guard for deep UNC paths."""
    s = str(p)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(long_path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _year(sources: pd.Series) -> pd.Series:
    return pd.to_numeric(sources.astype(str).str.extract(r"((?:18|19|20)\d\d)")[0], errors="coerce")


def load_frp():
    d = K1.load_frp()
    d["year"] = _year(d.source)
    return d


def load_sfrc():
    d = K1.load_sfrc()
    d["year"] = _year(d.source)
    return d


def _load_carb_raw():
    raw = pd.read_csv(long_path(CARB_FILE), low_memory=False)
    return pd.DataFrame({
        "source": raw["reference"].astype(str).str.strip(),
        "exposure": raw["exposure_block"].astype(str),
        "rate": pd.to_numeric(raw["carbonation_rate_mm_sqrt_d"], errors="coerce"),
        "co2": pd.to_numeric(raw["co2_vol_pct"], errors="coerce"),
        "wb": pd.to_numeric(raw["w_b"], errors="coerce"),
        "cao": pd.to_numeric(raw["reactive_cao_kg_m3"], errors="coerce"),
    })


def load_carb():
    d = _load_carb_raw()
    d = d[(d.rate > 0) & (d.co2 > 0) & (d.wb > 0) & (d.cao > 0) & (d.source != "nan")].reset_index(drop=True)
    d["year"] = np.nan
    d["y"] = np.log(d.rate)
    d["x_co2"] = np.log(d.co2 / C_REF)
    d["x_cao"] = np.log(d.cao / 150.0)
    d["x_wb"] = np.log(d.wb)
    d["m"] = 0.5 * d.x_co2 - 0.5 * d.x_cao
    return d


def load_carb_co2():
    d = _load_carb_raw()
    d = d[(d.rate > 0) & (d.co2 > 0) & (d.wb > 0) & (d.source != "nan")].reset_index(drop=True)
    d["year"] = np.nan
    d["y"] = np.log(d.rate)
    d["x_co2"] = np.log(d.co2 / C_REF)
    d["x_wb"] = np.log(d.wb)
    d["m"] = 0.5 * d.x_co2
    return d


DEEP_FILE = (WS / "2026-DecisionAdmissibleUQ-RealRC" / "data" / "raw"
             / "github_kmegahed_DeepBeamML_deep_beam2222.xlsx")
DEEP_FEATURES = ["ln_d", "ln_bw", "ln_fc", "ln_rho", "ln_ad"]


def load_deep_rows():
    raw = pd.read_excel(long_path(DEEP_FILE))
    df = pd.DataFrame({
        "source": raw["Author"].astype(str).str.strip(),
        "specimen": raw["Specimen"].astype(str),
        "h": pd.to_numeric(raw["h"], errors="coerce"),
        "bw": pd.to_numeric(raw["b"], errors="coerce"),
        "d": pd.to_numeric(raw["d"], errors="coerce"),
        "fc": pd.to_numeric(raw["fck"], errors="coerce"),
        "rho": pd.to_numeric(raw["rho"], errors="coerce"),
        "a_d": pd.to_numeric(raw["a_d"], errors="coerce"),
        "rho_v": pd.to_numeric(raw["rho_v"], errors="coerce"),
        "rho_h": pd.to_numeric(raw["rho_h"], errors="coerce"),
        "w_tp": pd.to_numeric(raw["w_tp"], errors="coerce"),
        "w_bp": pd.to_numeric(raw["w_bp"], errors="coerce"),
        "V_kN": pd.to_numeric(raw["V"], errors="coerce"),
    })
    df = df.dropna(subset=["bw", "d", "fc", "rho", "a_d", "V_kN"])
    df = df[(df.bw > 0) & (df.d > 0) & (df.fc > 0) & (df.rho > 0) & (df.V_kN > 0) & (df.a_d <= 2.5)
            & (df.rho_v.fillna(0) <= 0) & (df.rho_h.fillna(0) <= 0)].reset_index(drop=True)
    df["year"] = np.nan
    df["y"] = np.log(df.V_kN)
    df["ln_d"], df["ln_bw"], df["ln_fc"] = np.log(df.d), np.log(df.bw), np.log(df.fc)
    df["ln_rho"], df["ln_ad"] = np.log(df.rho), np.log(df.a_d)
    df["ln_wb"] = np.log(df.w_bp / df.d)          # relative width of the support bearing plate
    return df


def lambda_s(d_mm):
    """ACI 318-19 size-effect factor, SI form (0.004 per mm corresponds to 1/10 per inch)."""
    return np.minimum(1.0, np.sqrt(2.0 / (1.0 + 0.004 * np.asarray(d_mm, dtype=float))))


def deep_anchor_kN(df, rung):
    fc, b, d, rho, ad = df.fc, df.bw, df.d, df.rho, df.a_d
    if rung == "BEAM":
        v = 0.66 * lambda_s(d) * rho ** (1 / 3) * np.sqrt(fc) * b * d
    elif rung in ("ARCH", "ARCHS"):
        v = 2.2 * (fc * rho / ad) ** (1 / 3) * (2.5 / ad) * b * d
        if rung == "ARCHS":
            v = v * lambda_s(d)
    elif rung == "ARCH99":
        r = np.minimum(0.5 * ad, 1.0)
        f1 = np.minimum(2.5, 3.5 - 2.5 * r)
        v = np.minimum(f1 * (0.158 * np.sqrt(fc) + 17.24 * rho / r), 0.5 * np.sqrt(fc)) * b * d
    else:
        raise KeyError(rung)
    return v / 1e3


def _deep(rung):
    df = load_deep_rows()
    df["m"] = np.log(deep_anchor_kN(df, rung))
    return df


def load_deep():
    return _deep("ARCH")


def load_deep_wb():
    return load_deep()


DOMAINS = {
    "FRP": dict(loader=load_frp, coded=K1.FEATURES["FRP"], free=[], size="ln_d",
                anchor="ACI 440.1R-15", label="FRP-RC beams, shear", anchor_cut=2002,
                licence="CC BY-NC 4.0", doi="10.15131/shef.data.5057527"),
    "SFRC": dict(loader=load_sfrc, coded=K1.FEATURES["SFRC"], free=[], size="ln_d",
                 anchor="Narayanan-Darwish 1987", label="SFRC beams, shear", anchor_cut=1987,
                 licence="CC BY 4.0", doi="10.5281/zenodo.2578061"),
    "CARB": dict(loader=load_carb, coded=["x_co2", "x_cao"], free=["x_wb"], size=None,
                 anchor="diffusion-binding sqrt(C/a)", label="concrete carbonation rate", anchor_cut=None,
                 licence="CC BY 4.0", doi="10.5281/zenodo.21830461"),
    "DEEP": dict(loader=load_deep, coded=DEEP_FEATURES, free=[], size="ln_d",
                 anchor="Zsutty 1971 arch action", label="RC deep beams, shear", anchor_cut=1971,
                 licence="CC BY 4.0", doi="10.1038/s41598-024-64386-w"),
}
SENSITIVITY = {
    "CARB-CO2": dict(loader=load_carb_co2, coded=["x_co2"], free=["x_wb"], size=None,
                     anchor="sqrt(C)", label="carbonation, CO2-only anchor", anchor_cut=None,
                     licence="CC BY 4.0", doi="10.5281/zenodo.21830461"),
    "DEEP-WB": dict(loader=load_deep_wb, coded=DEEP_FEATURES, free=["ln_wb"], size="ln_d",
                    anchor="Zsutty 1971 arch action", label="deep beams, plate-adjusted", anchor_cut=1971,
                    licence="CC BY 4.0", doi="10.1038/s41598-024-64386-w"),
}
FEATURE_LABELS = dict(K1.FEATURE_LABELS, x_co2="CO2 concentration", x_cao="reactive CaO content",
                      x_wb="water/binder ratio", ln_wb="support plate width / d")
FEATURE_TEX = {"ln_d": "$d$", "ln_bw": "$b_w$", "ln_fc": "$f'_c$", "ln_rho": "$\\rho$", "ln_Ef": "$E_f$",
               "ln_ad": "$a/d$", "ln1p_F": "$F$", "x_co2": "CO$_2$", "x_cao": "CaO$_r$", "x_wb": "$w/b$",
               "ln_wb": "$w_{bp}/d$"}
CODE_EXPONENT = {"x_co2": 0.5, "x_cao": -0.5}


def spec(name):
    return DOMAINS[name] if name in DOMAINS else SENSITIVITY[name]


def load(name):
    return spec(name)["loader"]()


def features(name):
    s = spec(name)
    return list(s["coded"]) + list(s["free"])


def manifest():
    rows = []
    for k, p in list(K1.FILES.items()) + [("CARB", CARB_FILE), ("DEEP", DEEP_FILE)]:
        rows.append(dict(dataset=k, file=p.name, sha256=sha256(p)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for nm in list(DOMAINS) + list(SENSITIVITY):
        d = load(nm)
        g = d.groupby("source").size()
        r = np.exp(d.y - d.m)
        print(f"{nm:8s} n={len(d):5d} programmes={g.size:3d} ge6={(g >= 6).sum():3d} "
              f"median/prog={g.median():.0f} years={d.year.notna().mean():.2f} ratio-cov={r.std() / r.mean():.2f}")
    print(manifest().to_string(index=False))
