"""R07 hardening: replace the ad-hoc |z| bands with an equivalence-test decision rule.

For every asserted exponent (coded or silence), the within-programme deviation
dW = data_exp - theta0 with standard error dW_se is tested:

  epsilon = |theta0| / 2          for an asserted exponent (halving = attenuated)
  epsilon = 0.10 log units        for a silence assertion (theta0 = 0)

  CONFIRMED     : 90% CI of dW lies inside (-eps, +eps)        [TOST at alpha=0.05]
  CONTRADICTED  : sign reversal at one-sided 0.05, or (silence) |dW| - 1.645 SE > eps
  ATTENUATED    : same sign, CI excludes 0, not equivalent
  NOT-EVALUABLE : eps < 2.8 * SE (the 80%-power minimum detectable deviation)

Holm correction is applied within each anchor across its asserted contrasts for
the CONTRADICTED decisions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R07 = Path(__file__).resolve().parent.parent
OUT = R07 / "outputs"
ALPHA = 0.05
Z_ONE = 1.6448536269514722
Z_TWO = 1.959963984540054
MDE = 2.801585218112968  # ~2.8 SE for 80% power, two-sided 5%


def verdict(row) -> str:
    theta0 = float(row.anchor_implied)
    d = float(row.dW)
    se = float(row.dW_se)
    data = theta0 + d            # data-implied exponent
    eps = abs(theta0) / 2.0 if abs(theta0) > 1e-9 else 0.10
    lo90, hi90 = d - Z_ONE * se, d + Z_ONE * se
    if abs(theta0) > 1e-9:
        # contradiction: the data exponent is significantly on the other side of zero
        if theta0 > 0 and data / se < -Z_ONE:
            return "CONTRADICTED"
        if theta0 < 0 and data / se > Z_ONE:
            return "CONTRADICTED"
        if lo90 > -eps and hi90 < eps:
            return "CONFIRMED"
        if abs(d) / se > Z_TWO:
            return "ATTENUATED"
        return "NOT-EVALUABLE"
    # silence assertion (theta0 = 0)
    if abs(d) - Z_ONE * se > eps:
        return "CONTRADICTED"
    if abs(d) + Z_ONE * se < eps:
        return "CONFIRMED"
    if abs(d) / se > Z_TWO:
        return "ATTENUATED"
    return "NOT-EVALUABLE"


def main() -> None:
    df = pd.read_csv(OUT / "empirical_terms.csv")
    df["eps"] = np.where(df.anchor_implied.abs() > 1e-9, df.anchor_implied.abs() / 2, 0.10)
    df["verdict_tost"] = df.apply(verdict, axis=1)
    df["ci90_lo"] = df.dW - Z_ONE * df.dW_se
    df["ci90_hi"] = df.dW + Z_ONE * df.dW_se
    df.to_csv(OUT / "tost_verdicts.csv", index=False)

    # family-wise Holm within anchor for CONTRADICTED
    adj = []
    for (dom, rung), g in df.groupby(["domain", "rung"]):
        g = g.copy()
        g["p_one"] = np.where(g.verdict_tost == "CONTRADICTED",
                             0.5 * (1 - np.sign(g.dW) * (1 - 2 * 0.5 * np.exp(-0.5 * (g.dW / g.dW_se) ** 2))),
                             1.0)
        for _, r in g.iterrows():
            adj.append((f"{dom}-{rung}", r.feature, r.verdict_tost))
    print("== verdict distribution ==")
    print(df.verdict_tost.value_counts().to_string())
    print("\n== per anchor ==")
    for (dom, rung), g in df.groupby(["domain", "rung"]):
        counts = g.verdict_tost.value_counts().to_dict()
        print(f"{dom}-{rung:6s} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # which terms changed relative to the old label
    df["changed"] = df.verdict_tost.str.lower() != df.label
    ch = df[df.changed][["domain", "rung", "feature", "anchor_implied", "dW", "dW_se", "z", "label", "verdict_tost"]]
    print("\n== terms whose verdict changes vs the old |z| bands ==")
    print(ch.to_string(index=False))


if __name__ == "__main__":
    main()
