"""R07: turn the four hardening experiments into manuscript fragments.

Writes generated/tab_pglo.tex, generated/tab_tolerance.tex, generated/numbers_r07b.tex.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

R07 = Path(__file__).resolve().parent.parent
OUT = R07 / "outputs"
GEN = R07.parents[1] / "paper" / "manuscript" / "generated"
GEN.mkdir(parents=True, exist_ok=True)

LABEL = {"FRP": "FRP shear", "SFRC": "SFRC shear", "CARB": "Carbonation", "DEEP": "Deep beams"}


def main() -> None:
    pg = pd.read_csv(OUT / "pglo.csv")
    pg["optimism_pct"] = (pg.rmse_lopo / pg.rmse_random - 1.0) * 100.0
    rows = []
    for dom in ("FRP", "SFRC", "CARB", "DEEP"):
        g = pg[pg.domain == dom].set_index("arm")
        rows.append(
            f"{LABEL[dom]} & {g.loc['KA-FIXED', 'rmse_random']:.3f} & {g.loc['KA-FIXED', 'rmse_lopo']:.3f} & "
            f"{g.loc['KA-FIXED', 'optimism_pct']:.1f} & {g.loc['KA-RE', 'optimism_pct']:.1f} & "
            f"{g.loc['DATA-RE', 'optimism_pct']:.1f} \\\\")
    (GEN / "tab_pglo.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    tc = pd.read_csv(OUT / "tolerance_curve.csv")
    rows = []
    for r in tc.to_dict("records"):
        rows.append(f"{r['eps_fraction']:.2f} & {r['CONFIRMED']} & {r['ATTENUATED']} & "
                    f"{r['CONTRADICTED']} & {r['NOT-EVALUABLE']} \\\\")
    (GEN / "tab_tolerance.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    cal = pd.read_csv(OUT / "calibration_summary.csv").iloc[0]
    st = pd.read_csv(OUT / "stacking_pointwise.csv")
    deep = st[st.domain == "DEEP"].iloc[0]
    m = {
        "aPglobMin": f"{pg.optimism_pct.min():.1f}",
        "aPglobMax": f"{pg.optimism_pct.max():.1f}",
        "aCalSlope": f"{cal.slope:.2f}",
        "aCalSlopeLo": f"{cal.slope_lo:.2f}",
        "aCalSlopeHi": f"{cal.slope_hi:.2f}",
        "aCalRtwo": f"{cal.r2:.2f}",
        "aCalMape": f"{cal.mape_pct:.0f}",
        "aCalCrps": f"{cal.crps_budget:.2f}",
        "aCalCrpsBand": f"{cal.crps_factor2:.2f}",
        "aCalCrpsConst": f"{cal.crps_constant:.2f}",
        "aStackPaka": f"{float(st.PAKA.mean()):.3f}",
        "aStackPoint": f"{float(st.POINT.mean()):.3f}",
        "aStackPakaDeep": f"{deep.PAKA:.3f}",
        "aStackPointDeep": f"{deep.POINT:.3f}",
    }
    lines = [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (GEN / "numbers_r07b.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("fragments written")
    for k, v in sorted(m.items()):
        print(f"  \\{k} = {v}")


if __name__ == "__main__":
    main()
