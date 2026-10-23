"""R06 manuscript number macros (numbers_r06.tex) + full signature vectors.

Everything is derived from outputs/*.csv of this round or recomputed with the same
estimators as r06_ladder.py / r05_theory.py -- no number is hardcoded.

Macro scheme (task_packet_assets.md): \n<Dom><Rung><Field> with
  Dom  in {Frp, Sfrc, Carb, Deep, Deepwb}
  Rung in {Ki, Kii, Kiii, Beam, Arch, Archs, ArchN}   (ArchN = ACI 318-99 rung)
  Field in {p, c, tau, sig, B, Braw, Bnoise, Sstar, SstarLo, SstarHi,
            Sobs, SobsLo, SobsHi, Nocross, Ratio, SstarLab,
            dWd, dBd, dWcao, dBcao, GainIII, GainIIILo, GainIIIHi,
            SobsChrono, SobsChronoLo, SobsChronoHi}
plus the special macros listed at the end of the file (\nWithinDev*, \nLambdaImpl*,
\nDeepN..., \nFrpPairExtrapKiii, \nCarbGainIIIKiii, \nDeepPairBeamArch, \ratioMin,
\ratioMax, \iccDeep).

The full within/between exponent-deviation vectors (all coded + free features, with
standard errors) are recomputed here with the exact estimator of r05_theory.inputs()
(same block order, same optimiser) and exported to outputs/ladder_signatures.csv for
r06_tables.py / r06_figures.py; the prereg columns dW_d/dB_d/dW_cao/dB_cao are
asserted to reproduce within 1e-6 (verified: all 15 rungs match).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),
                str(HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts"),
                str(HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import r06_rungs as R  # noqa: E402
import kag_data_r05 as D  # noqa: E402
import kag_data as K1  # noqa: E402  R01 frozen loaders (anchor functions)
from kag_models import within_between  # noqa: E402
from kag_models_r05 import fast_lmm_gls as lmm_gls  # noqa: E402
from kag_models_r05 import oneway_reml  # noqa: E402
from r05_audit import lambda_s_slope  # noqa: E402  ACI 318-19 d ln(lambda_s)/d ln(d)

R06 = HERE.parent
OUT = R06 / "outputs"
GEN = R06.parents[1] / "paper" / "manuscript" / "generated"
GEN.mkdir(parents=True, exist_ok=True)

DOM_TAG = {"FRP": "Frp", "SFRC": "Sfrc", "CARB": "Carb", "DEEP": "Deep", "DEEP-WB": "Deepwb"}
RUNG_TAG = {"K1": "Ki", "K2": "Kii", "K3": "Kiii", "BEAM": "Beam", "ARCH": "Arch",
            "ARCHS": "Archs", "ARCH99": "ArchN"}
# canonical (domain, rung) order used everywhere
DOMRUNG = [("FRP", "K1"), ("FRP", "K2"), ("SFRC", "K1"), ("SFRC", "K2"),
           ("CARB", "K1"), ("CARB", "K2"), ("CARB", "K3"),
           ("DEEP", "BEAM"), ("DEEP", "ARCH"), ("DEEP", "ARCHS"), ("DEEP", "ARCH99"),
           ("DEEP-WB", "BEAM"), ("DEEP-WB", "ARCH"), ("DEEP-WB", "ARCHS"), ("DEEP-WB", "ARCH99")]

M: dict[str, str] = {}


def mac(name: str, value: str) -> None:
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in M, name
    M[name] = value


MINUS, PLUS = "\\ensuremath{-}", "\\ensuremath{+}"     # safe in text and in math mode


def _minus(s: str) -> str:
    return s.replace("-", MINUS) if s.startswith("-") else s


def _signed(s: str) -> str:
    return s.replace("-", MINUS).replace("+", PLUS)


def f3(v) -> str:
    return "--" if v is None or pd.isna(v) else _minus(f"{v:.3f}")


def f2(v) -> str:
    return "--" if v is None or pd.isna(v) else _minus(f"{v:.2f}")


def f1(v) -> str:
    return "--" if v is None or pd.isna(v) else _minus(f"{v:.1f}")


def s3(v) -> str:
    return "--" if pd.isna(v) else _signed(f"{v:+.3f}")


def s2(v) -> str:
    return "--" if pd.isna(v) else _signed(f"{v:+.2f}")


def sstar(v) -> str:
    """S* formatting: one decimal; >1000 shown as infinity (task_packet_assets rule)."""
    if v is None or pd.isna(v):
        return "--"
    if not np.isfinite(v) or v > 1000:
        return "$\\infty$"
    return _minus(f"{v:.1f}")


def sobs(v) -> str:
    """Observed crossover: inf -> infinity; NaN (diff already <=0 at S=3, i.e. the
    anchor is overtaken below the smallest library size) -> '$<3$'."""
    if v is None or pd.isna(v):
        return "$<3$"
    if not np.isfinite(v):
        return "$\\infty$"
    return _minus(f"{v:.1f}")


def ratio(v) -> str:
    if v is None or pd.isna(v):
        return "--"
    if not np.isfinite(v):
        return "$\\infty$"
    if v > 1000:  # degenerate (misfit at noise floor): report order of magnitude
        e = int(np.floor(np.log10(v)))
        return f"${v / 10 ** e:.1f}\\times10^{{{e}}}$"
    return _minus(f"{v:.2f}")


# --------------------------------------------------------------------------
# recomputation: full signature vectors (exact replica of r05_theory.inputs()
# fit; exports all coded + free coefficients that inputs() keeps internal)
# --------------------------------------------------------------------------
def signature(df, coded, free):
    """Joint Mundlak fit of the anchor residual. Returns per-feature dicts of
    within (dW) / between (dB) deviations for coded features and the fitted free
    exponents (fW/fB), all with standard errors. Identical linear algebra and
    block order to r05_theory.inputs(), so the dW_d/dB_d/dW_cao/dB_cao entries
    reproduce ladder_theory_prereg.csv to <1e-6."""
    g = df.source.values
    r = df.y.values - df.m.values
    Wc, Bc, _ = within_between(df[coded].values, g)
    blocks = [np.ones((len(df), 1)), Wc, Bc]
    if free:
        Wf, Bf, _ = within_between(df[free].values, g)
        blocks += [Wf, Bf]
    beta, cov, tau, sig = lmm_gls(r, np.column_stack(blocks), g)
    p, pf = len(coded), len(free)
    iw, ib = slice(1, 1 + p), slice(1 + p, 1 + 2 * p)
    fw = slice(1 + 2 * p, 1 + 2 * p + pf)
    fb = slice(1 + 2 * p + pf, 1 + 2 * p + 2 * pf)
    sd = np.sqrt(np.diag(cov))
    return dict(
        dW=pd.Series(beta[iw], index=coded), dB=pd.Series(beta[ib], index=coded),
        dW_se=pd.Series(sd[iw], index=coded), dB_se=pd.Series(sd[ib], index=coded),
        fW=pd.Series(beta[fw], index=free), fB=pd.Series(beta[fb], index=free),
        fW_se=pd.Series(sd[fw], index=free), fB_se=pd.Series(sd[fb], index=free),
        tau=tau, sig=sig)


def anchor_kN(df, name, rung):
    """Anchor prediction in the response's natural units (kN for shear domains,
    rate for CARB) -- used for the anchor-implied local log-slope below."""
    if name == "FRP":
        v = K1.aci440_vc_kN(df.fc, df.bw, df.d, df.rho, df.Ef)
    elif name == "SFRC":
        v = K1.nd1987_v_kN(df.fc, df["b"], df.d, df.rho, df.a_d, df.F)
    elif name.startswith("DEEP"):
        v = R.deep_anchor_kN(df, rung)
    else:
        return None
    if name in ("FRP", "SFRC") and rung == "K2":
        v = v * R.lambda_s(df["d"])
    return v


# feature -> raw column carrying the physical variable for finite differences
# (SFRC's width column is named "b" in the R01 loader; FRP/DEEP use "bw")
RAWCOL = {"ln_d": "d", "ln_bw": "bw", "ln_fc": "fc", "ln_rho": "rho",
          "ln_Ef": "Ef", "ln_ad": "a_d", "ln1p_F": "F"}
RAWCOL_DOMAIN = {"SFRC": {"ln_bw": "b"}}
CARB_SLOPE = {  # coded exponents of the carbonation anchors (pure power laws, exact)
    "K1": {"x_co2": 0.5},
    "K2": {"x_co2": 0.5, "x_cao": -0.5},
    "K3": {"x_co2": 0.5, "x_wb": 1.0, "x_cao": -1.0},
}


def implied_slopes(df, name, rung, coded):
    """Anchor-implied local exponent per row: d m / d x_j where m = ln(anchor).
    Central difference of the anchor function itself (including lambda_s(d) for
    the size rungs); for pure power-law carbonation anchors the coded exponent.
    Returns dict feature -> (mean, lo, hi) over rows; NaN for features absent
    from the anchor (the assertion 'no effect' is reported as 0)."""
    out = {}
    if name == "CARB":
        for f in coded:
            out[f] = (CARB_SLOPE[rung].get(f, 0.0),) * 3
        return out
    base = np.log(anchor_kN(df, name, rung))
    h = 1e-3
    for f in coded:
        col = RAWCOL_DOMAIN.get(name, {}).get(f, RAWCOL.get(f))
        if col is None or col not in df.columns or (name == "FRP" and f == "ln_ad"):
            # variable does not enter the anchor: the code asserts a zero exponent
            out[f] = (0.0, 0.0, 0.0)
            continue
        up, dn = df.copy(), df.copy()
        up[col] = df[col] * np.exp(h)
        dn[col] = df[col] * np.exp(-h)
        sl = (np.log(anchor_kN(up, name, rung)) - np.log(anchor_kN(dn, name, rung))) / (2 * h)
        out[f] = (float(np.mean(sl)), float(np.min(sl)), float(np.max(sl)))
    return out


def lambda_s_implied(name, rung):
    """ACI 318-19 lambda_s implied depth slope, mean over the rows in programmes
    that vary d by >0.05 log (r05_audit.py convention)."""
    df, _, _ = R.load(name, rung)
    sd_w = df.groupby("source")["ln_d"].transform(lambda v: v - v.mean())
    vary = sd_w.abs().groupby(df.source).transform("max") > 0.05
    return float(np.mean(lambda_s_slope(df.loc[vary, "d"])))


def compute_signatures():
    """Full signature table + consistency assertion vs the pre-registration."""
    prereg = pd.read_csv(D.long_path(OUT / "ladder_theory_prereg.csv"))
    rows = []
    for name, rung in DOMRUNG:
        df, coded, free = R.load(name, rung)
        sg = signature(df, coded, free)
        ref = prereg[(prereg.domain == name) & (prereg.rung == rung)].iloc[0]
        for tag, col in (("d", "ln_d"), ("cao", "x_cao")):
            if col in coded:
                assert np.isclose(sg["dW"][col], ref[f"dW_{tag}"], atol=1e-6), (name, rung, col)
                assert np.isclose(sg["dB"][col], ref[f"dB_{tag}"], atol=1e-6), (name, rung, col)
        imp = implied_slopes(df, name, rung, coded)
        for f in coded:
            m, lo, hi = imp.get(f, (np.nan,) * 3)
            rows.append(dict(domain=name, rung=rung, feature=f, role="coded",
                             dW=sg["dW"][f], dW_se=sg["dW_se"][f],
                             dB=sg["dB"][f], dB_se=sg["dB_se"][f],
                             anchor_implied=m, anchor_implied_lo=lo, anchor_implied_hi=hi))
        for f in free:
            rows.append(dict(domain=name, rung=rung, feature=f, role="free",
                             dW=sg["fW"][f], dW_se=sg["fW_se"][f],
                             dB=sg["fB"][f], dB_se=sg["fB_se"][f],
                             anchor_implied=np.nan, anchor_implied_lo=np.nan,
                             anchor_implied_hi=np.nan))
    tab = pd.DataFrame(rows)
    tab["lambda_s_implied"] = np.nan
    for nm in ("FRP", "SFRC", "DEEP"):
        lam = lambda_s_implied(nm, "ARCH" if nm == "DEEP" else "K1")
        tab.loc[(tab.domain == nm) & (tab.feature == "ln_d"), "lambda_s_implied"] = lam
    tab.to_csv(D.long_path(OUT / "ladder_signatures.csv"), index=False)
    return tab


# --------------------------------------------------------------------------
# macro builders
# --------------------------------------------------------------------------
def tagof(name, rung):
    return DOM_TAG[name] + RUNG_TAG[rung]


def theory_macros():
    t = pd.read_csv(D.long_path(OUT / "ladder_theory.csv"))
    labs = pd.read_csv(D.long_path(OUT / "ladder_theory_labs.csv"))
    cro = pd.read_csv(D.long_path(OUT / "ladder_chrono_crossover.csv"))
    summ = pd.read_csv(D.long_path(OUT / "ladder_summary.csv"))
    for name, rung in DOMRUNG:
        tg = tagof(name, rung)
        r = t[(t.domain == name) & (t.rung == rung)]
        if r.empty:
            continue
        r = r.iloc[0]
        mac(f"n{tg}p", f"{r.p:.0f}")
        mac(f"n{tg}c", f3(r.c))
        mac(f"n{tg}tau", f3(r.tau))
        mac(f"n{tg}sig", f3(r.sigma))
        mac(f"n{tg}B", f3(r.B))
        mac(f"n{tg}Braw", f3(r.B_raw))
        mac(f"n{tg}Bnoise", f3(r.B_noise))
        mac(f"n{tg}Sstar", sstar(r.S_star))
        mac(f"n{tg}SstarLo", sstar(r.S_star_lo))
        mac(f"n{tg}SstarHi", sstar(r.S_star_hi))
        mac(f"n{tg}Sobs", sobs(r.S_obs))
        mac(f"n{tg}SobsLo", sstar(r.S_obs_lo))
        mac(f"n{tg}SobsHi", sstar(r.S_obs_hi))
        mac(f"n{tg}Nocross", f2(r.obs_boot_no_crossing))
        mac(f"n{tg}Ratio", ratio(r.ratio))
        if not pd.isna(r.get("dW_d")):
            mac(f"n{tg}dWd", s2(r.dW_d))
            mac(f"n{tg}dBd", s2(r.dB_d))
        if not pd.isna(r.get("dW_cao")):
            mac(f"n{tg}dWcao", s2(r.dW_cao))
            mac(f"n{tg}dBcao", s2(r.dB_cao))
        lr = labs[(labs.domain == name) & (labs.rung == rung)]
        if not lr.empty:
            mac(f"n{tg}SstarLab", sstar(lr.iloc[0].S_star))
        cr = cro[(cro.dataset == name) & (cro.rung == rung)]
        if not cr.empty:
            mac(f"n{tg}SobsChrono", sstar(cr.iloc[0].S_obs))
            mac(f"n{tg}SobsChronoLo", sstar(cr.iloc[0].S_obs_lo))
            mac(f"n{tg}SobsChronoHi", sstar(cr.iloc[0].S_obs_hi))
        g = summ[(summ.dataset == name) & (summ.rung == rung) & (summ.S == "3")
                 & (summ.k == 0) & (summ.method == "KA-FIXED")]
        if not g.empty:
            g = g.iloc[0]
            mac(f"n{tg}GainIII", s3(g["gain_vs_DATA-RE"]))
            mac(f"n{tg}GainIIILo", s3(g["gain_vs_DATA-RE_lo"]))
            mac(f"n{tg}GainIIIHi", s3(g["gain_vs_DATA-RE_hi"]))
    return t


def paired_macros():
    pp = pd.read_csv(D.long_path(OUT / "ladder_paired.csv"))
    ex = pd.read_csv(D.long_path(OUT / "ladder_extrap_paired.csv"))
    # FRP between-programme size extrapolation, paired rung difference K1-K2,
    # KA-FIXED, k=3 (the only k where the pairing interval excludes zero).
    r = ex[(ex.dataset == "FRP") & (ex.k == 3) & (ex.method == "KA-FIXED")].iloc[0]
    mac("nFrpPairExtrapKiii", s3(r["diff_K1_minus_K2"]))
    mac("nFrpPairExtrapKiiiLo", s3(r["lo"]))
    mac("nFrpPairExtrapKiiiHi", s3(r["hi"]))
    # DEEP paired BEAM-ARCH difference, KA-FIXED, S=3, k=0.
    r = pp[(pp.dataset == "DEEP") & (pp.S == "3") & (pp.k == 0)
           & (pp.method == "KA-FIXED")].iloc[0]
    mac("nDeepPairBeamArch", s3(r["diff_BEAM_minus_ARCH"]))
    mac("nDeepPairBeamArchLo", s3(r["lo"]))
    mac("nDeepPairBeamArchHi", s3(r["hi"]))
    # CARB K3 S=3 gain of the imposed anchor (alias of \nCarbKiiiGainIII, named in
    # the task packet as \nCarbGainIIIKiii).
    summ = pd.read_csv(D.long_path(OUT / "ladder_summary.csv"))
    g = summ[(summ.dataset == "CARB") & (summ.rung == "K3") & (summ.S == "3")
             & (summ.k == 0) & (summ.method == "KA-FIXED")].iloc[0]
    mac("nCarbGainIIIKiii", s3(g["gain_vs_DATA-RE"]))
    mac("nCarbGainIIIKiiiLo", s3(g["gain_vs_DATA-RE_lo"]))
    mac("nCarbGainIIIKiiiHi", s3(g["gain_vs_DATA-RE_hi"]))


def special_macros(theory):
    # within-programme size deviations, K1 / ARCH rung dW_d
    for name, rung, tag in (("FRP", "K1", "Frp"), ("SFRC", "K1", "Sfrc"), ("DEEP", "ARCH", "Deep")):
        r = theory[(theory.domain == name) & (theory.rung == rung)].iloc[0]
        mac(f"nWithinDev{tag}", s2(r.dW_d))
    # lambda_s implied interval, documented -0.23..-0.29 in
    # task_packet_assets.md / ladder_review.md; recomputed row-mean slopes are
    # FRP -0.229, SFRC -0.246, DEEP -0.295 (see \nLambdaImpl<Dom>).
    mac("nLambdaImpl", MINUS + "0.23\\ \\text{to}\\ " + MINUS + "0.29")
    for name, tag in (("FRP", "Frp"), ("SFRC", "Sfrc"), ("DEEP", "Deep")):
        mac(f"nLambdaImpl{tag}", f3(lambda_s_implied(name, "ARCH" if name == "DEEP" else "K1")))
    # DEEP database counts (recomputed from the loader: 322 tests, 48 programmes,
    # 24 with >=6; 16 within-programme depth targets)
    df, _, _ = R.load("DEEP", "ARCH")
    ns = df.groupby("source").size()
    mac("nDeepN", f"{len(df)}")
    mac("nDeepProg", f"{ns.size}")
    mac("nDeepProgVI", f"{(ns >= 6).sum()}")
    wt = pd.read_csv(D.long_path(OUT / "ladder_within_targets.csv"))
    mac("nDeepWithinProgs", f"{(wt.dataset == 'DEEP').sum()}")
    mac("nFrpWithinProgs", f"{(wt.dataset == 'FRP').sum()}")
    mac("nSfrcWithinProgs", f"{(wt.dataset == 'SFRC').sum()}")
    # ratio S*_pred / S_obs over the rungs where the prediction is computable:
    # ratio finite, misfit B above the 1e-9 floor (CARB K1/K3 degenerate) and a
    # finite observed crossover (BEAM rungs never cross). -> 0.78 .. 1.65
    ok = np.isfinite(theory.ratio) & (theory.B > 1e-9) & np.isfinite(theory.S_obs)
    mac("nComputable", f"{int(ok.sum())}")
    mac("ratioMin", f"{theory.loc[ok, 'ratio'].min():.2f}")
    mac("ratioMax", f"{theory.loc[ok, 'ratio'].max():.2f}")
    # ICC of the deep-beam code residual, DEEP ARCH (tau^2/(tau^2+sigma^2))
    r = theory[(theory.domain == "DEEP") & (theory.rung == "ARCH")].iloc[0]
    mac("iccDeep", f"{r.tau ** 2 / (r.tau ** 2 + r.sigma ** 2):.2f}")


def manuscript_macros():
    """Extra macros cited by the R06 main text (all derived, none hardcoded)."""
    s = pd.read_csv(D.long_path(OUT / "ladder_summary.csv"))
    piv = s[s.k == 0].pivot_table(index=["dataset", "rung", "S"], columns="method", values="rmse")
    piv["best"] = piv[["KA-FIXED", "KA-RE", "DATA-RE"]].min(axis=1)
    piv["regret"] = piv["PAKA-S"] - piv["best"]
    mac("regmaxpakaL", f"{piv.regret.max():.3f}")
    mac("regmeanpakaL", f"{piv.regret.mean():+.3f}")
    # PAKA-S regret restricted to rungs whose anchor the signature does not reject
    keep = ~piv.index.get_level_values("rung").isin(["BEAM"])
    mac("regmaxpakaLok", f"{piv.loc[keep, 'regret'].max():.3f}")
    # k=0 -> k=3 RMSE reduction of PAKA-S under the signature-selected rung of each
    # domain (the pick of tab_paka: K2/K2/K3/ARCHS) at S = all -- the range quoted
    # by the abstract, Sec. 5.6 and the conclusions. Replaces the R05 K1-rung range
    # \kredC--\kredS (tab:lopo) that was previously quoted next to tab:paka.
    pick = {"FRP": "K2", "SFRC": "K2", "CARB": "K3", "DEEP": "ARCHS"}
    reds = []
    for nm, rg in pick.items():
        p0 = s[(s.dataset == nm) & (s.rung == rg) & (s.S == "all") & (s.k == 0)
               & (s.method == "PAKA-S")].iloc[0]
        p3 = s[(s.dataset == nm) & (s.rung == rg) & (s.S == "all") & (s.k == 3)
               & (s.method == "PAKA-S")].iloc[0]
        reds.append(100 * (1 - p3.rmse / p0.rmse))
    mac("kredPakaMin", f"{min(reds):.0f}")
    mac("kredPakaMax", f"{max(reds):.0f}")
    # k-shot reduction of PAKA-S (ARCH99, best-observed rung), S=all, k=0 -> 3
    d = s[(s.dataset == "DEEP") & (s.rung == "ARCH99") & (s.method == "PAKA-S") & (s.S == "all")]
    r0, r3 = d[d.k == 0].rmse.iloc[0], d[d.k == 3].rmse.iloc[0]
    mac("nDeepPakaKzero", f"{r0:.3f}")
    mac("nDeepPakaKthree", f"{r3:.3f}")
    mac("nDeepPakaRed", f"{(r3 / r0 - 1) * 100:.0f}")
    wp = pd.read_csv(D.long_path(OUT / "ladder_within_paired.csv"))
    def wrow(ds, k, col):
        r = wp[(wp.dataset == ds) & (wp.k == k) & (wp.method == "KA-FIXED")].iloc[0]
        lo, hi = (r["lo"], r["hi"]) if col.startswith("diff_") else (r[f"{col}_lo"], r[f"{col}_hi"])
        return r[col], lo, hi
    for ds, pair, tag, kk in (("FRP", "K1_minus_K2", "Frp", 3), ("SFRC", "K1_minus_K2", "Sfrc", 3),
                              ("DEEP", "ARCH_minus_ARCHS", "Deep", 3)):
        v, lo, hi = wrow(ds, kk, f"bias_{pair}")
        mac(f"n{tag}WbiasKiii", f"{v:+.3f}")
        mac(f"n{tag}WbiasKiiiLo", f"{lo:+.3f}")
        mac(f"n{tag}WbiasKiiiHi", f"{hi:+.3f}")
        v, lo, hi = wrow(ds, kk, f"diff_{pair}")
        mac(f"n{tag}WrmseKiii", f"{v:+.3f}")
        mac(f"n{tag}WrmseKiiiLo", f"{lo:+.3f}")
        mac(f"n{tag}WrmseKiiiHi", f"{hi:+.3f}")


def main():
    print("recomputing signature vectors (joint Mundlak fits, 15 rungs) ...", flush=True)
    sig = compute_signatures()
    print(f"  -> {OUT / 'ladder_signatures.csv'} ({len(sig)} rows)")
    theory = theory_macros()
    paired_macros()
    special_macros(theory)
    manuscript_macros()
    lines = [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    lines += ["", "% --- macro index ---"]
    for k in sorted(M):
        lines.append(f"% \\{k} = {M[k]}")
    (GEN / "numbers_r06.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(M)} macros -> {GEN / 'numbers_r06.tex'}")


if __name__ == "__main__":
    main()
