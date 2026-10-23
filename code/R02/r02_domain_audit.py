"""R02 diagnostic 0: domain eligibility and confounding audit.

For every domain the paper could use, report the quantities the design needs and
the provenance flags a reader needs:

  n, programmes, programmes with >= 6 specimens
  anchor ratio (mean, CoV, share unconservative)
  ICC of the code residual (programme share of the marginal residual variance)
  between-programme share eta^2 of the size variable d
  within- vs between-programme exponent gap for d and its z statistic
  scope / licence / source

The audit is the evidence behind the "which domains may carry the claim" question;
a domain with no programme identity, an out-of-scope anchor or an unclear licence
is listed as inadmissible rather than quietly dropped.

Outputs: outputs/domain_audit.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kag_data_r02 as K                                       # noqa: E402
from kag_models import KnowledgeAnchoredLMM, lmm_gls, within_between  # noqa: E402
from kag_models_r02 import ka_fit                              # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def between_share(x, g):
    s = pd.DataFrame({"x": x, "g": g})
    gm = s.groupby("g").x.transform("mean")
    tot = float(np.sum((s.x - s.x.mean()) ** 2))
    return float(np.sum((gm - s.x.mean()) ** 2) / tot) if tot > 0 else np.nan


def mundlak(df, feat):
    W, B, _ = within_between(df[[feat]].values, df.source.values)
    X = np.column_stack([np.ones(len(df)), W[:, 0], B[:, 0]])
    beta, cov, tau, sig = lmm_gls(df.y.values - df.m.values, X, df.source.values)
    gap = beta[2] - beta[1]
    se = np.sqrt(cov[2, 2] + cov[1, 1] - 2 * cov[1, 2])
    return dict(within=float(beta[1]), within_se=float(np.sqrt(cov[1, 1])),
                between=float(beta[2]), between_se=float(np.sqrt(cov[2, 2])),
                gap=float(gap), z=float(gap / se))


def audit(name):
    df = K.load(name)
    meta = K.DOMAINS[name]
    g = df.source.values
    n_s = pd.Series(g).value_counts()
    ratio = np.exp(df.y.values - df.m.values)
    r = df.y.values - df.m.values
    m0 = ka_fit(r, np.zeros((len(df), 0)), g, lam=0.0, ml=False, n_starts=4)
    mk = mundlak(df, "ln_d")
    rec = dict(domain=name, label=meta["label"], member=meta["member"],
               anchor=meta["anchor"], scope=meta["scope"], licence=meta["licence"],
               source_id=meta["source"], n=len(df), programmes=int(n_s.size),
               programmes_ge6=int((n_s >= 6).sum()), specimens_ge6=int(n_s[n_s >= 6].sum()),
               anchor_ratio_mean=float(ratio.mean()),
               anchor_ratio_cov=float(ratio.std(ddof=1) / ratio.mean()),
               anchor_unconservative=float((ratio < 1).mean()),
               tau=float(m0["tau"]), sigma=float(m0["sig"]),
               icc=float(m0["tau"] ** 2 / (m0["tau"] ** 2 + m0["sig"] ** 2)),
               eta2_d=between_share(df.ln_d.values, g),
               d_within=mk["within"], d_within_se=mk["within_se"],
               d_between=mk["between"], d_between_se=mk["between_se"], d_z=mk["z"])
    return rec


def main():
    rows = [audit(k) for k in K.LOADERS]
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "domain_audit.csv", index=False)
    pd.set_option("display.width", 240)
    print(tab[["domain", "n", "programmes", "programmes_ge6", "anchor_ratio_mean",
               "anchor_ratio_cov", "icc", "eta2_d", "d_within", "d_within_se",
               "d_between", "d_between_se", "d_z", "scope"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
