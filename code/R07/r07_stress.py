"""R07 WP3 -- pre-registered stress test for the metadata audit.

Corruptions (frozen in PREREGISTRATION.md):
  S1 sign      : CaO exponent flipped (CARB K2/K3); lambda_s multiplied
                 instead of divided (FRP/SFRC K2).      -> in-support
  S2 unit      : depth in cm instead of mm; CO2 in fraction instead of %.
                                                        -> out-of-support
  S3 range     : CaO-anchored rules applied to rows without reactive CaO;
                 deep-beam anchors applied outside a/d <= 2.5.  -> out-of-support
  S4 formula   : CARB K3 given the K1 exponent set (and K1 given K3's).
                                                        -> in-support

Baseline: support-domain gate (training-only kNN density, RealRC-style). A
missing-input check is *not* part of a density gate; that check is itself a
metadata check and belongs to the audit -- noted in the report.

Detection is a batch decision: a rule application on one database is rejected
if any audit check fires. Test labels are not used anywhere in the audit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R07 = HERE.parent
OUT = R07 / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(HERE),
                str(HERE.parents[1] / "R06_mechanism_pivot_2026-10-22" / "scripts"),
                str(HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts"),
                str(HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import r06_rungs as R  # noqa: E402
import kag_data_r05 as D  # noqa: E402
import kag_data as K1  # noqa: E402
import r07_audit as A  # noqa: E402

ANCHORS = [("FRP", "K1"), ("FRP", "K2"), ("SFRC", "K1"), ("SFRC", "K2"),
           ("CARB", "K1"), ("CARB", "K2"), ("CARB", "K3"),
           ("DEEP", "BEAM"), ("DEEP", "ARCH"), ("DEEP", "ARCHS"),
           ("DEEP", "ARCH99")]


def clean_predict(domain, rung):
    if domain == "FRP":
        def f(df):
            m = np.log(K1.aci440_vc_kN(df["fc"], df["bw"], df["d"], df["rho"], df["Ef"]))
            if rung == "K2":
                m = m + np.log(D.lambda_s(df["d"]))
            return m
        return f
    if domain == "SFRC":
        def f(df):
            m = np.log(K1.nd1987_v_kN(df["fc"], df["b"], df["d"], df["rho"],
                                     df["a_d"], df["F"]))
            if rung == "K2":
                m = m + np.log(D.lambda_s(df["d"]))
            return m
        return f
    if domain == "CARB":
        def f(df):
            m = 0.5 * df["x_co2"].to_numpy(dtype=float)
            if rung == "K2":
                m = m - 0.5 * df["x_cao"].to_numpy(dtype=float)
            elif rung == "K3":
                m = (m + df["x_wb"].to_numpy(dtype=float)
                     - df["x_cao"].to_numpy(dtype=float))
            return m
        return f
    if domain == "DEEP":
        return lambda df: np.log(D.deep_anchor_kN(df, rung))
    raise KeyError(domain)


def cases():
    """Yield dicts: id, domain, rung, corrupt, group, df, predict."""
    for domain, rung in ANCHORS:
        df, _, _ = R.load(domain, rung)
        pred = clean_predict(domain, rung)
        base = dict(domain=domain, rung=rung, df=df, predict=pred)
        yield dict(id=f"{domain}-{rung}", corrupt="clean", group="clean", **base)

        # S1 sign (in-support)
        if (domain, rung) in (("CARB", "K2"), ("CARB", "K3")):
            def p(df, rung=rung):
                m = 0.5 * df["x_co2"].to_numpy(dtype=float)
                if rung == "K2":
                    m = m + 0.5 * df["x_cao"].to_numpy(dtype=float)
                else:
                    m = (m + df["x_wb"].to_numpy(dtype=float)
                         + df["x_cao"].to_numpy(dtype=float))
                return m
            yield dict(id=f"{domain}-{rung}", corrupt="sign", group="in_support",
                       df=df, predict=p, domain=domain, rung=rung)
        if (domain, rung) in (("FRP", "K2"), ("SFRC", "K2")):
            def p(df, pred=pred):
                return pred(df) - 2.0 * np.log(D.lambda_s(df["d"]))
            yield dict(id=f"{domain}-{rung}", corrupt="sign", group="in_support",
                       df=df, predict=p, domain=domain, rung=rung)

        # S2 unit (out-of-support)
        if domain in ("FRP", "SFRC", "DEEP"):
            d2 = df.copy()
            d2["d"] = df["d"].to_numpy(dtype=float) * 0.1
            yield dict(id=f"{domain}-{rung}", corrupt="unit", group="out_support",
                       df=d2, predict=pred, domain=domain, rung=rung)
        if domain == "CARB":
            d2 = df.copy()
            d2["x_co2"] = df["x_co2"].to_numpy(dtype=float) - np.log(100.0)
            yield dict(id=f"{domain}-{rung}", corrupt="unit", group="out_support",
                       df=d2, predict=pred, domain=domain, rung=rung)

        # S3 range (out-of-support)
        if (domain, rung) in (("CARB", "K2"), ("CARB", "K3")):
            co2 = D.load_carb_co2().copy()
            co2["x_cao"] = np.where(co2["cao"].to_numpy(dtype=float) > 0,
                                    np.log(co2["cao"].to_numpy(dtype=float) / 150.0),
                                    np.nan)
            if rung == "K2":
                def p(df):
                    return 0.5 * df["x_co2"].to_numpy(dtype=float) - 0.5 * df["x_cao"].to_numpy(dtype=float)
            else:
                def p(df):
                    return (0.5 * df["x_co2"].to_numpy(dtype=float)
                            + df["x_wb"].to_numpy(dtype=float)
                            - df["x_cao"].to_numpy(dtype=float))
            yield dict(id=f"{domain}-{rung}", corrupt="range", group="out_support",
                       df=co2, predict=p, domain=domain, rung=rung)
        if domain == "DEEP":
            d2 = df.copy()
            d2["a_d"] = df["a_d"].to_numpy(dtype=float) * 2.0
            yield dict(id=f"{domain}-{rung}", corrupt="range", group="out_support",
                       df=d2, predict=pred, domain=domain, rung=rung)

        # S4 formula (in-support): CARB K3 <-> K1 exponent sets
        if (domain, rung) == ("CARB", "K3"):
            def p(df):
                return 0.5 * df["x_co2"].to_numpy(dtype=float)
            yield dict(id=f"{domain}-{rung}", corrupt="formula", group="in_support",
                       df=df, predict=p, domain=domain, rung=rung)
        if (domain, rung) == ("CARB", "K1"):
            def p(df):
                return (0.5 * df["x_co2"].to_numpy(dtype=float)
                        + df["x_wb"].to_numpy(dtype=float)
                        - df["x_cao"].to_numpy(dtype=float))
            yield dict(id=f"{domain}-{rung}", corrupt="formula", group="in_support",
                       df=df, predict=p, domain=domain, rung=rung)


def main():
    rows = []
    trains = {}
    for domain, rung in ANCHORS:
        trains[(domain, rung)] = R.load(domain, rung)[0]

    for c in cases():
        dom, rung = c["domain"], c["rung"]
        feats = A.META[(dom, rung)]["feats"]
        res = A.audit(dom, rung, c["df"], c["predict"])
        flagged, dist, thr = A.support_gate(trains[(dom, rung)], c["df"], feats)
        rows.append(dict(
            id=c["id"], domain=dom, rung=rung, corrupt=c["corrupt"], group=c["group"],
            n=len(c["df"]),
            audit_rejected=int(res["rejected"]),
            audit_reasons="|".join(res["reasons"]),
            scope=round(res["checks"]["scope"], 4),
            unit=round(res["checks"]["unit"], 4),
            nonfinite=round(res["checks"]["nonfinite"], 4),
            sign=round(res["checks"]["sign"], 4),
            expo=round(res["checks"]["expo"], 4),
            support_flagged=int(flagged),
            support_dist=round(dist, 4) if np.isfinite(dist) else None,
            support_thr=round(thr, 4) if np.isfinite(thr) else None,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stress_results.csv", index=False)

    # ---- verdicts (pre-registered) ----
    clean = df[df.group == "clean"]
    h1 = df[df.corrupt.isin(["sign", "unit"])]
    h2 = df[df.group == "in_support"]
    h3_range = df[df.corrupt == "range"]
    h3_form = df[(df.corrupt == "formula") & (df.rung == "K3")]

    summary = {
        "clean_false_rejects": int(clean.audit_rejected.sum()),
        "clean_support_false_flags": int(clean.support_flagged.sum()),
        "H1_audit_detection_sign_unit": float(h1.audit_rejected.mean()) if len(h1) else None,
        "H2_support_detection_in_support": float(h2.support_flagged.mean()) if len(h2) else None,
        "H3_audit_detection_range": float(h3_range.audit_rejected.mean()) if len(h3_range) else None,
        "H3_audit_detection_formula_K3": float(h3_form.audit_rejected.mean()) if len(h3_form) else None,
        "n_cases": len(df),
    }
    (OUT / "stress_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(df[["id", "corrupt", "group", "audit_rejected", "audit_reasons",
              "support_flagged"]].to_string(index=False))
    print("\n--- summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
