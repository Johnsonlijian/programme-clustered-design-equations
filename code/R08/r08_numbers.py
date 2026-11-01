"""R08 manuscript numbers -> paper/manuscript/generated/numbers_r08.tex.

Post hoc revision numbers: calibration diagnostics, SFRC K1-K2 directional
check, split-budget summary, certified-anchor comparisons, prospective
repaired-anchor check and repaired-anchor paired deltas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R08 = HERE.parent
ROOT = R08.parent.parent
OUT = R08 / "outputs"
R06 = ROOT / "rounds" / "R06_mechanism_pivot_2026-10-22"
GEN = ROOT / "paper" / "manuscript" / "generated"
GEN.mkdir(exist_ok=True, parents=True)

MACROS = []


def m(name, value):
    MACROS.append((name, value))


def f(x, d):
    return f"{float(x):.{d}f}"


def sgn(x, d=3):
    return r"\ensuremath{+}" + f(x, d) if float(x) >= 0 else r"\ensuremath{-}" + f(-float(x), d)


# --- 1. calibration diagnostics -------------------------------------------------
cal = pd.read_csv(OUT / "calibration_diag.csv").set_index("set")
eight = cal.loc["8 primary"]
all11 = cal.loc["all 11 (manuscript)"]
m("rCalSlopeEight", f(eight.slope, 2))
m("rCalSlopeEightLo", f(eight.slope_lo, 2))
m("rCalSlopeEightHi", f(eight.slope_hi, 2))
m("rCalRelAll", f(all11.reliability, 2))
m("rCalEivAll", f(all11.slope_eiv, 2))
m("rCalEivAllLo", f(all11.slope_eiv_lo, 2))
m("rCalEivAllHi", f(all11.slope_eiv_hi, 2))
m("rCalRelEight", f(eight.reliability, 2))
m("rCalEivEight", f(eight.slope_eiv, 2))
m("rCalEivEightLo", f(eight.slope_eiv_lo, 2))
m("rCalEivEightHi", f(eight.slope_eiv_hi, 2))

# --- 2. SFRC K1-K2 directional check --------------------------------------------
lp = pd.read_csv(R06 / "outputs" / "ladder_paired.csv", dtype={"S": str})
s = lp[(lp.dataset == "SFRC") & (lp.method == "KA-FIXED") & (lp.k == 0)
       & (lp.S.isin(["3", "6", "12", "24", "all"]))]
m("rSfrcDirMin", sgn(s.diff_K1_minus_K2.min()))
m("rSfrcDirMax", sgn(s.diff_K1_minus_K2.max()))
m("rSfrcDirHiMin", sgn(s.hi.min()))
m("rSfrcDirHiMax", sgn(s.hi.max()))
s3 = s[s.S == "3"].iloc[0]
m("rSfrcDirThreeLo", sgn(s3.lo))
m("rSfrcDirThreeHi", sgn(s3.hi))

# --- 3. split budget ------------------------------------------------------------
bs = pd.read_csv(OUT / "budget_split.csv").set_index(["domain", "rung"])
m("rSplitSfrcKi", f(bs.loc[("SFRC", "K1"), "S_split"], 2))
m("rSplitSfrcKii", f(bs.loc[("SFRC", "K2"), "S_split"], 2))
m("rSplitBeam", f(bs.loc[("DEEP", "BEAM"), "S_split"], 1))
m("rSplitCarbKiiRatio",
  f(bs.loc[("CARB", "K2"), "S_split"] / bs.loc[("CARB", "K2"), "S_obs"], 2))
arch = bs.loc[[("DEEP", "ARCH"), ("DEEP", "ARCHS"), ("DEEP", "ARCH99")]]
m("rSplitArchRatioMin", f((arch.S_split / arch.S_obs).min(), 2))
m("rSplitArchRatioMax", f((arch.S_split / arch.S_obs).max(), 2))
frp = bs.loc[[("FRP", "K1"), ("FRP", "K2")]]
m("rSplitFrpRatioMin", f((frp.S_split / frp.S_obs).min(), 2))
m("rSplitFrpRatioMax", f((frp.S_split / frp.S_obs).max(), 2))
m("rSfrcCKi", f(bs.loc[("SFRC", "K1"), "c_prespec"], 3))
m("rSfrcCKii", f(bs.loc[("SFRC", "K2"), "c_prespec"], 3))

# --- 4. certified anchor ----------------------------------------------------------
cs = pd.read_csv(OUT / "certified_summary.csv")
cs_r = cs.set_index(["rung", "method", "k"])
m("rSfrcCPakaZero", f(cs_r.loc[("K1C", "PAKA-S", 0), "rmse"], 3))
m("rSfrcCFixZero", f(cs_r.loc[("K1C", "KA-FIXED", 0), "rmse"], 3))

cp = pd.read_csv(OUT / "certified_paired.csv").set_index(["k", "a", "b"])
for name, key in (("rSfrcPakaGbm", (0, "PAKA-S[K1]", "GBM[-]")),
                  ("rSfrcCPakaGbm", (0, "PAKA-S[K1C]", "GBM[-]")),
                  ("rSfrcCPakaGbmThree", (3, "PAKA-S[K1C]", "GBM[-]")),
                  ("rSfrcCFixDiff", (0, "KA-FIXED[K1C]", "KA-FIXED[K1]"))):
    r = cp.loc[key]
    m(name, sgn(r["diff"]))
    m(name + "Lo", sgn(r["lo"]))
    m(name + "Hi", sgn(r["hi"]))

cb = pd.read_csv(OUT / "certified_budget.csv").set_index("rung")
m("rSfrcBKi", f(cb.loc["K1", "B"], 3))
m("rSfrcBKic", f(cb.loc["K1C", "B"], 3))
m("rSfrcBRed", str(int(round(100 * (1 - cb.loc["K1C", "B"] / cb.loc["K1", "B"])))))

# --- 5. prospective check ---------------------------------------------------------
pc = pd.read_csv(OUT / "prospective_check.csv")
TAG = {("SFRC", "K1C"): "SfrcKic", ("SFRC", "K2C"): "SfrcKiic",
       ("FRP", "K1C"): "FrpKic", ("FRP", "K2C"): "FrpKiic",
       ("DEEP", "BEAMC"): "DeepBeamc"}
for r in pc.itertuples():
    t = "rP" + TAG[(r.domain, r.rung)]
    m(t + "Pred", f(r.S_pred, 1))
    m(t + "PredLo", f(r.S_pred_lo, 1))
    m(t + "PredHi", f(r.S_pred_hi, 1))
    m(t + "Split", f(r.S_split, 1))
    m(t + "Obs", r"\ensuremath{>}24" if not np.isfinite(r.S_obs) else f(r.S_obs, 1))
    m(t + "ObsLo", f(r.S_obs_lo, 1))
    m(t + "ObsHi", f(r.S_obs_hi, 1))
    m(t + "NoCross", str(int(round(100 * r.no_cross))))
    m(t + "Ratio", "--" if not np.isfinite(r.ratio) else f(r.ratio, 2))
    m(t + "ParentPred", f(r.parent_pred, 1))
    m(t + "ParentObs", "none" if not np.isfinite(r.parent_obs) else f(r.parent_obs, 1))
m("rProsN", str(len(pc)))
m("rProsDirOk", str(int(pc.direction_ok.sum())))
m("rProsMagOk", str(int(pc.magnitude_ok.sum())))

# --- 6. repair paired deltas --------------------------------------------------------
rp = pd.read_csv(OUT / "repair_paired.csv", dtype={"S": str})
rp_i = rp.set_index(["domain", "repair", "S", "k", "method"])
for name, key in (("rFrpKicFixThree", ("FRP", "K1C", "3", 0, "KA-FIXED")),
                  ("rDeepBeamcFixThree", ("DEEP", "BEAMC", "3", 0, "KA-FIXED")),
                  ("rDeepBeamcPakaThree", ("DEEP", "BEAMC", "3", 0, "PAKA-S")),
                  ("rDeepBeamcFixAll", ("DEEP", "BEAMC", "all", 0, "KA-FIXED"))):
    r = rp_i.loc[key]
    m(name, sgn(r["diff"]))
    m(name + "Lo", sgn(r["lo"]))
    m(name + "Hi", sgn(r["hi"]))
for name, meth, kk in (("rDeepPakaGbm", "PAKA-S vs GBM", 0),
                       ("rDeepPakaGbmme", "PAKA-S vs GBM-ME", 0),
                       ("rDeepPakaGbmThree", "PAKA-S vs GBM", 3)):
    r = rp[(rp.method == meth) & (rp.k == kk)].iloc[0]
    m(name, sgn(r["diff"]))
    m(name + "Lo", sgn(r["lo"]))
    m(name + "Hi", sgn(r["hi"]))

# --- 7. repair full library ---------------------------------------------------------
rf = pd.read_csv(OUT / "repair_fulllib.csv").set_index(["dataset", "rung", "k", "method"])
m("rDeepBeamcPakaZero", f(rf.loc[("DEEP", "BEAMC", 0, "PAKA-S"), "rmse"], 3))
m("rDeepGbmZero", f(rf.loc[("DEEP", "BEAMC", 0, "GBM"), "rmse"], 3))
m("rDeepGbmmeZero", f(rf.loc[("DEEP", "BEAMC", 0, "GBM-ME"), "rmse"], 3))

# --- 8. tuned ML baselines (needs tuned_summary.csv + tuned_vs_hier.csv) -------------
_tsum, _tvh = OUT / "tuned_summary.csv", OUT / "tuned_vs_hier.csv"
if _tsum.exists() and _tvh.exists():
    ts = pd.read_csv(_tsum, dtype={"S": str}).set_index(
        ["dataset", "S", "k", "method"])
    for dom, suf in (("FRP", "F"), ("SFRC", "S"), ("CARB", "C"), ("DEEP", "D")):
        m(f"rTunedGbmThree{suf}", f(ts.loc[(dom, "3", 0, "GBM"), "rmse"], 3))
        m(f"rTunedGbmmeThree{suf}", f(ts.loc[(dom, "3", 0, "GBM-ME"), "rmse"], 3))
        m(f"rTunedGbmTfour{suf}", f(ts.loc[(dom, "24", 0, "GBM"), "rmse"], 3))
        m(f"rTunedGbmmeTfour{suf}", f(ts.loc[(dom, "24", 0, "GBM-ME"), "rmse"], 3))
    tvh = pd.read_csv(_tvh)
    m("rTunedNBetter", str(int((tvh.best_ml < tvh.best_hier).sum())))
    m("rTunedMinGap", sgn(tvh.ml_minus_hier.min()))
else:
    print("NOTE: tuned_summary.csv / tuned_vs_hier.csv missing; "
          "tuned macros skipped")

# --- write --------------------------------------------------------------------------
lines = [rf"\providecommand{{\{n}}}{{{v}}}" for n, v in MACROS]
lines += ["", "% --- macro index ---"]
lines += [f"% \\{n} = {v}" for n, v in MACROS]
(GEN / "numbers_r08.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

for n, v in MACROS:
    print(f"\\{n} = {v}")
print(f"\nwrote {GEN / 'numbers_r08.tex'} ({len(MACROS)} macros)")

# --- sanity check (expected values from the revision brief) ---------------------------
EXPECT = {"rCalSlopeEight": "0.61", "rCalSlopeEightLo": "0.21", "rCalSlopeEightHi": "1.00",
          "rCalEivAll": "0.99", "rCalEivAllLo": "0.69", "rCalEivAllHi": "2.33",
          "rCalRelAll": "0.66",
          "rSfrcDirMin": r"\ensuremath{-}0.022", "rSfrcDirMax": r"\ensuremath{-}0.019",
          "rSplitSfrcKi": "8.17", "rSplitSfrcKii": "8.23", "rSplitBeam": "0.6",
          "rSplitCarbKiiRatio": "1.15", "rSplitArchRatioMin": "0.37",
          "rSplitArchRatioMax": "0.55",
          "rSfrcCPakaZero": "0.249",
          "rSfrcCPakaGbm": r"\ensuremath{+}0.008",
          "rSfrcCPakaGbmLo": r"\ensuremath{-}0.046", "rSfrcCPakaGbmHi": r"\ensuremath{+}0.064",
          "rSfrcCFixDiff": r"\ensuremath{-}0.049",
          "rSfrcBKi": "0.037", "rSfrcBKic": "0.015", "rSfrcBRed": "60",
          "rPSfrcKiicPred": "23.2", "rPSfrcKiicObs": "22.1", "rPSfrcKiicRatio": "1.05",
          "rPFrpKiicRatio": "2.06", "rPDeepBeamcPred": "66.3",
          "rProsDirOk": "4", "rProsMagOk": "4",
          "rDeepPakaGbm": r"\ensuremath{-}0.065",
          "rDeepGbmZero": "0.288"}
got = dict(MACROS)
bad = {k: (v, got.get(k)) for k, v in EXPECT.items() if got.get(k) != v}
print("\nsanity check:", "ALL MATCH" if not bad else f"MISMATCHES {bad}")
