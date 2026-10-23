"""KAG-SmallData R01 pilot: data loaders with source (test-programme) identity and
codified-knowledge anchors.

Two public, multi-source experimental databases are read *read-only* from the
local copies already downloaded for 2026-DecisionAdmissibleUQ-RealRC:

FRP  - FRP-reinforced concrete beams without shear reinforcement,
       University of Sheffield figshare, doi:10.15131/shef.data.5057527 (CC BY-NC).
       Source identity = the "reference" column (publication / test programme).
       Knowledge anchor = ACI 440.1R-15 concrete shear, Vc = 0.4 sqrt(f'c) b_w c.
SFRC - steel-fibre reinforced concrete beams without stirrups,
       Lantsoght database, Zenodo doi:10.5281/zenodo.2578061 (CC BY).
       Source identity = the two-row "Reference" cell (authors + year).
       Knowledge anchor = Narayanan & Darwish (1987), ACI Struct. J. 84(3).

Each loader returns a DataFrame with: source, y (= ln V_test [kN]), m (= ln V_anchor
[kN]) and log-feature columns listed in FEATURES[name]. Nothing is imputed except
what is stated in the loader docstring.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parents[4]
RAW = WS / "2026-DecisionAdmissibleUQ-RealRC" / "data" / "raw"
FILES = {
    "FRP": RAW / "figshare_5057527_FRP_RC_beams_shear_no_web_reinforcement.xlsx",
    "SFRC": RAW / "Zenodo_2578061_SFRC_shear_beams.xlsx",
}
FEATURES = {
    "FRP": ["ln_d", "ln_bw", "ln_fc", "ln_rho", "ln_Ef", "ln_ad"],
    "SFRC": ["ln_d", "ln_bw", "ln_fc", "ln_rho", "ln_ad", "ln1p_F"],
}
FEATURE_LABELS = {
    "ln_d": "effective depth d",
    "ln_bw": "web width b_w",
    "ln_fc": "concrete strength f'c",
    "ln_rho": "longitudinal reinforcement ratio",
    "ln_Ef": "FRP modulus E_f",
    "ln_ad": "shear span ratio a/d",
    "ln1p_F": "fibre factor F",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def aci440_vc_kN(fc, bw, d, rho, Ef):
    """ACI 440.1R-15 concrete shear strength of FRP-RC members (SI, N/mm/MPa)."""
    Ec = 4700.0 * np.sqrt(fc)
    rn = rho * Ef / Ec
    k = np.sqrt(2.0 * rn + rn ** 2) - rn
    return 0.4 * np.sqrt(fc) * bw * (k * d) / 1e3


def nd1987_v_kN(fc, b, d, rho, a_d, F):
    """Narayanan & Darwish (1987) ultimate shear of SFRC beams (SI)."""
    fcu = fc / 0.8
    fsp = fcu / (20.0 - np.sqrt(F)) + 0.7 + np.sqrt(F)
    e = np.where(a_d > 2.8, 1.0, 2.8 / a_d)
    v = e * (0.24 * fsp + 80.0 * rho / a_d) + 0.41 * 4.15 * F
    return v * b * d / 1e3


def load_frp() -> pd.DataFrame:
    """Rectangular slender beams (type != 'T', a/d >= 2.5, rho < 0.1); no imputation."""
    raw = pd.read_excel(FILES["FRP"])
    raw["source"] = raw["reference"].ffill().str.strip()
    df = pd.DataFrame({
        "source": raw["source"],
        "specimen": raw["name"].astype(str),
        "section": raw["type"].astype(str),
        "bw": pd.to_numeric(raw["bw [mm]"], errors="coerce"),
        "d": pd.to_numeric(raw["d [mm]"], errors="coerce"),
        "fc": pd.to_numeric(raw["fc,cyl [MPa]"], errors="coerce"),
        "rho": pd.to_numeric(raw["ρfl [-]"], errors="coerce"),
        "Ef": pd.to_numeric(raw["Efl [MPa]"], errors="coerce"),
        "a_d": pd.to_numeric(raw["a/d [-]"], errors="coerce"),
        "V_kN": pd.to_numeric(raw["Vu [N]"], errors="coerce") / 1e3,
    })
    df = df[df.section != "T"].dropna(subset=["source", "bw", "d", "fc", "rho", "Ef", "a_d", "V_kN"])
    df = df[(df.a_d >= 2.5) & (df.rho > 0) & (df.rho < 0.1) & (df.V_kN > 0) & (df.d > 0) & (df.bw > 0)]
    df = df.reset_index(drop=True)
    df["V_anchor_kN"] = aci440_vc_kN(df.fc, df.bw, df.d, df.rho, df.Ef)
    df["y"] = np.log(df.V_kN)
    df["m"] = np.log(df.V_anchor_kN)
    df["ln_d"], df["ln_bw"], df["ln_fc"] = np.log(df.d), np.log(df.bw), np.log(df.fc)
    df["ln_rho"], df["ln_Ef"], df["ln_ad"] = np.log(df.rho), np.log(df.Ef), np.log(df.a_d)
    return df


_YEAR = re.compile(r"^(18|19|20)\d\d[a-z]?$")


def _sfrc_sources(col0: pd.Series) -> list[str]:
    """The Reference cell spans two rows: author string, then year, then blanks."""
    name, year, out = None, None, []
    for v in col0:
        s = None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip()
        if s:
            if _YEAR.match(s.replace(".0", "")) and name is not None and year is None:
                year = s.replace(".0", "")
            else:
                name, year = s, None
        out.append(f"{name} ({year})" if year else str(name))
    return out


SFRC_POS = {"b": 2, "h": 3, "d": 4, "rho": 16, "fy": 17, "a_d": 18, "fc": 21,
            "Lf": 24, "Vf": 25, "F": 29, "V_kN": 38}


def load_sfrc() -> pd.DataFrame:
    """Slender beams (a/d >= 2.5) with tabulated fibre factor F; no imputation of
    model inputs (rows lacking b, d, fc, rho, a/d, F or V are dropped)."""
    raw = pd.read_excel(FILES["SFRC"], header=None)
    body = raw.iloc[3:].reset_index(drop=True)
    df = pd.DataFrame({k: pd.to_numeric(body.iloc[:, j], errors="coerce") for k, j in SFRC_POS.items()})
    df.insert(0, "specimen", body.iloc[:, 1].astype(str))
    df.insert(0, "source", _sfrc_sources(body.iloc[:, 0]))
    df = df.dropna(subset=["b", "d", "fc", "rho", "a_d", "F", "V_kN"])
    df = df[(df.a_d >= 2.5) & (df.F >= 0) & (df.V_kN > 0) & (df.d > 0) & (df.rho > 0)].reset_index(drop=True)
    df["V_anchor_kN"] = nd1987_v_kN(df.fc, df.b, df.d, df.rho, df.a_d, df.F)
    df["y"] = np.log(df.V_kN)
    df["m"] = np.log(df.V_anchor_kN)
    df["ln_d"], df["ln_bw"], df["ln_fc"] = np.log(df.d), np.log(df.b), np.log(df.fc)
    df["ln_rho"], df["ln_ad"], df["ln1p_F"] = np.log(df.rho), np.log(df.a_d), np.log1p(df.F)
    return df


LOADERS = {"FRP": load_frp, "SFRC": load_sfrc}


def load(name: str) -> pd.DataFrame:
    return LOADERS[name]()


def manifest() -> pd.DataFrame:
    rows = []
    for k, p in FILES.items():
        rows.append(dict(dataset=k, file=p.name, bytes=p.stat().st_size, sha256=sha256(p)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for k in LOADERS:
        d = load(k)
        g = d.groupby("source").size()
        r = np.exp(d.y - d.m)
        print(f"{k}: n={len(d)} sources={g.size} median/src={g.median():.1f} max/src={g.max()} "
              f"| anchor ratio mean={r.mean():.2f} CoV={r.std(ddof=1) / r.mean():.2f}")
    print(manifest().to_string(index=False))
