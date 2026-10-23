"""R04 theory point estimate for CARB (no bootstrap: the result is a negative).

Writes outputs/theory_r04.csv in the same schema as R02.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R02 = HERE.parents[1] / "R02_crossdomain_theory_2026-10-20" / "scripts"
R01 = HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts"
sys.path[:0] = [str(HERE), str(R02), str(R01)]

import kag_data_r04 as D     # noqa: E402
import r02_theory as T       # noqa: E402

OUT = HERE.parent / "outputs"


def main():
    df = D.load_carb()
    feats = D.FEATURES
    th = T.theory_inputs(df, feats)
    summ = pd.read_csv(OUT / "scarcity_r04_summary.csv")
    smap = {"3": 3, "6": 6, "12": 12, "24": 24}
    S, G = [], []
    for lab, so in smap.items():
        blk = summ[(summ.S.astype(str) == lab) & (summ.k == 0)]
        ka = blk[blk.method == "KA-RE"].rmse
        da = blk[blk.method == "DATA-RE"].rmse
        if len(ka) and len(da):
            S.append(so)
            G.append(float(da.iloc[0] ** 2 - ka.iloc[0] ** 2))
    meas = T.fit_gain(np.array(S, float), np.array(G))

    # Only the CO2 exponent is codified; the w/b exponent is free in both arms, so
    # the bias term must use the codified component alone. Both versions are stored.
    delta = th["delta"]
    SB = th["Sigma_B"]
    B_codified = float(delta[0] ** 2 * SB[0, 0])
    rec = dict(domain="CARB", p=th["p"], n_bar=th["n_bar"], tau=th["tau"], sigma=th["sig"],
               A=th["A"], B=th["B"], B_codified_only=B_codified,
               S_star_pred=th["S_star"],
               S_star_pred_codified=th["A"] / B_codified if B_codified > 0 else np.inf,
               S_star_lo=np.nan, S_star_hi=np.nan,
               S_star_meas=meas["S_star_meas"],
               ratio=th["S_star"] / meas["S_star_meas"],
               gain_r2=meas["r2"], A_meas=meas["A_meas"], B_meas=meas["B_meas"])
    pd.DataFrame([rec]).to_csv(OUT / "theory_r04.csv", index=False)
    print(pd.DataFrame([rec]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
