"""R05-E1: identification audit -- where in each database is the physics identified?

Per domain: programme share of the anchor's error (ICC), between-programme share of
each log-feature (eta^2), how many programmes vary it, and the exponent deviation from
the anchor estimated within programmes and between programmes (Mundlak decomposition,
random-intercept REML, univariate and joint). For the two shear domains the within-
programme size deviation is compared with the size dependence that a *different*
codified source (the ACI 318-19 size factor lambda_s) implies over the same depths.

Outputs: outputs/audit_domains.csv, outputs/audit_features.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import kag_data_r05 as D  # noqa: E402
from kag_models import within_between  # noqa: E402
from kag_models_r05 import fast_lmm_gls as lmm_gls  # noqa: E402
from kag_models_r05 import oneway_reml  # noqa: E402

OUT = HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)


def between_share(x, g):
    s = pd.DataFrame({"x": x, "g": g})
    gm = s.groupby("g").x.transform("mean")
    tot = float(np.sum((s.x - s.x.mean()) ** 2))
    return float(np.sum((gm - s.x.mean()) ** 2) / tot) if tot > 0 else np.nan


def mundlak(t, Phi, g):
    W, B, _ = within_between(Phi, g)
    p = Phi.shape[1]
    X = np.column_stack([np.ones(len(t)), W, B])
    beta, cov, tau, sig = lmm_gls(t, X, g)
    out = []
    for j in range(p):
        iw, ib = 1 + j, 1 + p + j
        gap = beta[ib] - beta[iw]
        se_gap = np.sqrt(cov[ib, ib] + cov[iw, iw] - 2 * cov[ib, iw])
        out.append(dict(within=beta[iw], within_se=np.sqrt(cov[iw, iw]), between=beta[ib],
                        between_se=np.sqrt(cov[ib, ib]), gap=gap, z=gap / se_gap))
    return out


def lambda_s_slope(d_mm):
    """d ln(lambda_s) / d ln(d) for ACI 318-19 lambda_s = sqrt(2/(1+0.004 d)) <= 1."""
    d = np.asarray(d_mm, float)
    active = np.sqrt(2.0 / (1.0 + 0.004 * d)) < 1.0
    return np.where(active, -0.5 * 0.004 * d / (1.0 + 0.004 * d), 0.0)


def main():
    dom_rows, feat_rows = [], []
    for name in list(D.DOMAINS) + list(D.SENSITIVITY):
        s = D.spec(name)
        df = D.load(name)
        feats = D.features(name)
        g = df.source.values
        n_s = df.groupby("source").size()
        r = df.y.values - df.m.values
        c, tau, sig = oneway_reml(r, g)
        rec = dict(domain=name, label=s["label"], anchor=s["anchor"], n=len(df), programmes=int(n_s.size),
                   programmes_ge6=int((n_s >= 6).sum()), singletons=int((n_s == 1).sum()),
                   median_per_programme=float(n_s.median()), tau=tau, sigma=sig,
                   icc=tau ** 2 / (tau ** 2 + sig ** 2), licence=s["licence"], doi=s["doi"])
        # Kish design effect for the programme-clustered sample and the specimen-equivalent size
        nbar = float(np.sum(n_s.values ** 2) / n_s.values.sum())
        rec["design_effect"] = 1.0 + (nbar - 1.0) * rec["icc"]
        rec["effective_n"] = len(df) / rec["design_effect"]
        dom_rows.append(rec)
        uni = {f: mundlak(r, df[[f]].values, g)[0] for f in feats}
        joint = mundlak(r, df[feats].values, g)
        sd_w = df.groupby("source")[feats].transform(lambda v: v - v.mean())
        for j, f in enumerate(feats):
            varying = (sd_w[f].abs().groupby(df.source).max() > 0.05).sum()
            fr = dict(domain=name, feature=f, label=D.FEATURE_LABELS.get(f, f),
                      role="coded" if f in s["coded"] else "free",
                      code_exponent_offset="deviation" if f in s["coded"] else "full exponent",
                      between_share=between_share(df[f].values, g), programmes_varying=int(varying),
                      programmes_total=int(n_s.size))
            fr.update({f"uni_{k}": v for k, v in uni[f].items()})
            fr.update({f"joint_{k}": v for k, v in joint[j].items()})
            if f == s.get("size"):
                vary = sd_w[f].abs().groupby(df.source).transform("max") > 0.05
                fr["lambda_s_implied"] = float(np.mean(lambda_s_slope(df.loc[vary, "d"])))
                fr["d_range_varying"] = f"{df.loc[vary, 'd'].min():.0f}-{df.loc[vary, 'd'].max():.0f}"
            feat_rows.append(fr)
    dom, feat = pd.DataFrame(dom_rows), pd.DataFrame(feat_rows)
    dom.to_csv(OUT / "audit_domains.csv", index=False)
    feat.to_csv(OUT / "audit_features.csv", index=False)
    pd.set_option("display.width", 250)
    print(dom[["domain", "n", "programmes", "programmes_ge6", "tau", "sigma", "icc", "design_effect", "effective_n"]]
          .round(3).to_string(index=False))
    print(feat[["domain", "feature", "role", "between_share", "programmes_varying", "uni_within", "uni_within_se",
                "uni_between", "uni_between_se", "uni_z", "joint_within", "joint_between", "joint_z"]]
          .round(3).to_string(index=False))
    print(feat.loc[feat.lambda_s_implied.notna() if "lambda_s_implied" in feat else [],
                   ["domain", "uni_within", "lambda_s_implied", "d_range_varying"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
