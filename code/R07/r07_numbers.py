"""R07 number macros (numbers_r07.tex) for the merged manuscript.

Every macro is derived from the WP2/WP3/WP4 outputs; nothing is hand-copied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R07 = HERE.parent
OUT = R07 / "outputs"
GEN = HERE.parents[2] / "paper" / "manuscript" / "generated"
GEN.mkdir(parents=True, exist_ok=True)

M: dict[str, str] = {}


def mac(name, value):
    assert name not in M, name
    M[name] = value


def main():
    stress = pd.read_csv(OUT / "stress_results.csv")
    doss = pd.read_csv(OUT / "anchor_dossiers.csv")
    down = pd.read_csv(OUT / "downstream_summary.csv")

    # detection counts
    for key, tag in (("sign", "Sign"), ("formula", "Formula"), ("unit", "Unit"),
                     ("range", "Range")):
        g = stress[stress.corrupt == key]
        mac(f"aAudit{tag}Det", f"{int(g.audit_rejected.sum())}")
        mac(f"aAudit{tag}N", f"{len(g)}")
    inm = stress[stress.group == "in_support"]
    mac("aSupportIn", f"{int(inm.support_flagged.sum())}")
    mac("aSupportInN", f"{len(inm)}")
    unit = stress[stress.corrupt == "unit"]
    mac("aSupportUnit", f"{int(unit.support_flagged.sum())}")
    mac("aSupportUnitN", f"{len(unit)}")

    # worst z per anchor (absolute value, one decimal for the sub-10 cases)
    z = {r.anchor: abs(r.worst_z) for r in doss.itertuples()}
    terms = pd.read_csv(OUT / "empirical_terms.csv")

    def zof(dom, rung, feat):
        r = terms[(terms.domain == dom) & (terms.rung == rung)
                  & (terms.feature == feat)].iloc[0]
        return abs(float(r.z))

    mac("aZBeamAd", f"{z['DEEP-BEAM']:.0f}")
    mac("aZCaoKii", f"{z['CARB-K2']:.0f}")
    mac("aZCaoKiii", f"{z['CARB-K3']:.0f}")
    mac("aZFrpAd", f"{zof('FRP', 'K1', 'ln_ad'):.1f}")
    mac("aZFrpKiiD", f"{zof('FRP', 'K2', 'ln_d'):.1f}")
    mac("aZFsrcF", f"{zof('SFRC', 'K2', 'ln1p_F'):.1f}")

    # downstream: largest damage when the corrupted rule is deployed directly;
    # the audited chain's margin over the stacking pipeline
    ka_damage, margins = [], []
    for dom, rung in (("CARB", "K2"), ("FRP", "K2"), ("SFRC", "K2")):
        g = down[(down.domain == dom) & (down.rung == rung)]

        def m(case, meth):
            return g[(g.case == case) & (g.method == meth)]["mean"].iloc[0]
        ka_damage.append(m("corrupt", "KA-FIXED") - m("clean", "KA-FIXED"))
        margins.append(m("corrupt", "PAKA-S") - m("clean", "DATA-RE"))
    mac("aDownKaMax", f"{max(ka_damage):.2f}")
    mac("aDownPakaMin", f"{min(margins):.3f}")
    mac("aDownPakaMax", f"{max(margins):.3f}")

    # clean rerun vs R06 ladder (max |diff| over the three cases)
    r06 = pd.read_csv(HERE.parents[1] / "R06_mechanism_pivot_2026-10-22" / "outputs"
                      / "ladder_summary.csv")
    diffs = []
    for dom, rung in (("CARB", "K2"), ("FRP", "K2"), ("SFRC", "K2")):
        got = down[(down.domain == dom) & (down.rung == rung)
                   & (down.case == "clean") & (down.method == "PAKA-S")]["mean"].iloc[0]
        ref = r06[(r06.dataset == dom) & (r06.rung == rung)
                  & (r06.S.astype(str) == "all") & (r06.k == 0)
                  & (r06.method == "PAKA-S")].rmse.iloc[0]
        diffs.append(abs(got - ref))
    mac("aHfourDiff", f"{max(diffs) * 1e3:.1f}")

    lines = [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    lines += ["", "% --- macro index ---"]
    lines += [f"% \\{k} = {v}" for k, v in sorted(M.items())]
    (GEN / "numbers_r07.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(M)} macros -> {GEN / 'numbers_r07.tex'}")
    for k in sorted(M):
        print(f"  \\{k} = {M[k]}")


if __name__ == "__main__":
    main()
