"""R07 WP2 -- empirical validity layer + per-anchor dossier.

For every coded term of an anchor, the within-programme contrast gives an
estimate of the data-implied exponent; the anchor asserts a value. The layer
reports, per term, the z-score of the deviation (dW / SE) and a label:

  consistent      |z| <= 2
  inconsistent    2 < |z|, same sign as the assertion
  contradicted    sign of the data-implied exponent differs from the assertion
                  (or the anchor asserts zero and the data say |exp| > 0.1)

The anchor-level verdict joins three layers of the protocol:
  metadata audit (WP1, clean rule application)  -> declaration fidelity
  empirical layer (this file)                   -> data consistency
  priced value S* and PAKA regret (R06 outputs) -> what the prior is worth

Coded exponents come from the signature file's own `anchor_implied` column
(computed by r06_numbers.compute_signatures), so this layer is independent of
the hand-written META table used by the metadata audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R07 = HERE.parent
OUT = R07 / "outputs"
R06OUT = HERE.parents[1] / "R06_mechanism_pivot_2026-10-22" / "outputs"
sys.path[:0] = [str(HERE),
                str(HERE.parents[1] / "R06_mechanism_pivot_2026-10-22" / "scripts"),
                str(HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts"),
                str(HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import r06_rungs as R  # noqa: E402
import r07_audit as A  # noqa: E402
import r07_stress as S  # noqa: E402

ANCHORS = [("FRP", "K1"), ("FRP", "K2"), ("SFRC", "K1"), ("SFRC", "K2"),
           ("CARB", "K1"), ("CARB", "K2"), ("CARB", "K3"),
           ("DEEP", "BEAM"), ("DEEP", "ARCH"), ("DEEP", "ARCHS"),
           ("DEEP", "ARCH99")]


def label_row(coded, dW, se, z):
    data = coded + dW
    if abs(coded) < 0.05:                      # the anchor asserts no dependence
        return ("contradicted" if abs(data) > 0.10 else "consistent"), data
    if np.sign(data) != np.sign(coded):
        return "contradicted", data
    if abs(z) > 2.0:
        return "inconsistent", data
    return "consistent", data


def main():
    sig = pd.read_csv(R06OUT / "ladder_signatures.csv")
    sig = sig[(sig.role == "coded")].copy()
    sig["z"] = sig.dW / sig.dW_se
    sig["data_exp"] = sig.anchor_implied + sig.dW
    sig[["label", "data_exp"]] = sig.apply(
        lambda r: pd.Series(label_row(r.anchor_implied, r.dW, r.dW_se, r.z)),
        axis=1)

    theory = pd.read_csv(R06OUT / "ladder_theory.csv")
    summ = pd.read_csv(R06OUT / "ladder_summary.csv")
    piv = summ[summ.k == 0].pivot_table(index=["dataset", "rung", "S"],
                                        columns="method", values="rmse")
    piv["regret"] = piv["PAKA-S"] - piv[["KA-FIXED", "KA-RE", "DATA-RE"]].min(axis=1)
    regmax = piv.groupby(level=["dataset", "rung"]).regret.max()

    rows = []
    for dom, rung in ANCHORS:
        g = sig[(sig.domain == dom) & (sig.rung == rung)]
        worst = g.loc[g.z.abs().idxmax()]
        n_inc = int((g.label == "inconsistent").sum())
        n_con = int((g.label == "contradicted").sum())
        n_ok = int((g.label == "consistent").sum())

        # metadata layer: clean rule application must pass
        df, _, _ = R.load(dom, rung)
        meta_res = A.audit(dom, rung, df, S.clean_predict(dom, rung))

        tr = theory[(theory.domain == dom) & (theory.rung == rung)].iloc[0]
        rows.append(dict(
            anchor=f"{dom}-{rung}",
            metadata_ok=int(not meta_res["rejected"]),
            n_coded_terms=len(g),
            n_consistent=n_ok, n_inconsistent=n_inc, n_contradicted=n_con,
            worst_term=worst.feature, worst_z=round(float(worst.z), 1),
            worst_label=worst.label,
            S_star=round(float(tr.S_star), 2) if np.isfinite(tr.S_star) else np.inf,
            ratio=round(float(tr.ratio), 3) if np.isfinite(tr.ratio) else None,
            paka_regret_max=round(float(regmax[(dom, rung)]), 3),
        ))

    d = pd.DataFrame(rows)

    # descriptive profile: the protocol grades each asserted term, it does not
    # certify a rule as true or false; the variance budget prices the remainder
    short = {"consistent": "ok", "inconsistent": "att", "contradicted": "con"}
    prof = (sig.sort_values(["domain", "rung"])
               .groupby(["domain", "rung"])
               .apply(lambda g: "; ".join(f"{r.feature}:{short[r.label]}"
                                          for r in g.itertuples())))
    d["term_profile"] = [prof[(dom, rung)] for dom, rung in ANCHORS]
    d["declaration"] = np.where(d.metadata_ok == 1, "valid", "VIOLATED")
    d["priced"] = d.S_star.map(lambda v: "unbounded" if (isinstance(v, float) and v > 1e6) else "finite")
    d.to_csv(OUT / "anchor_dossiers.csv", index=False)
    sig.to_csv(OUT / "empirical_terms.csv", index=False)

    pd.set_option("display.width", 260)
    print(d[["anchor", "n_consistent", "n_inconsistent", "n_contradicted",
             "worst_term", "worst_z", "S_star", "ratio", "paka_regret_max",
             "declaration", "priced"]].to_string(index=False))
    print("\n== per-term profile (ok = |z|<=2; att = 2<|z|, same sign; con = sign contradicted) ==")
    for _, r in d.iterrows():
        print(f"{r.anchor:10s} {r.term_profile}")

    # the two terms R06's main text does not discuss -- reported explicitly
    for term in ("ln1p_F", "ln_ad"):
        sub = sig[sig.feature == term]
        print(f"\n== term {term} ==")
        print(sub[["domain", "rung", "anchor_implied", "data_exp", "z", "label"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
