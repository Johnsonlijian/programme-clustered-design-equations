"""R07 WP4 -- downstream test of the protocol (H4/H5).

H4 (no regression): on clean anchors the audit passes, so the audited chain is
the R06 chain. Verified by re-running the LOPO evaluation at S = all, k = 0 and
comparing with the R06 ladder summary.

H5 (corrupted deployments): for an in-support corruption the support gate does
not fire; a support-gated pipeline therefore deploys the corrupted rule and
pays for it, while the audit rejects the rule and falls back to data-only.

Cases are restricted to corruptions that keep the coded/free partition (sign
flip of the CaO exponent; lambda_s divided instead of multiplied). The
CARB K1<->K3 formula swap changes the partition and stays in the WP3 detection
test, not here.

Targets: programmes with at least six tests (same as the R06 ladder).
Arms: KA-FIXED (deployed anchor), DATA-RE, PAKA-S.
"""
from __future__ import annotations

import sys
import time
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
import r06_ladder as L  # noqa: E402
import kag_bench_r05 as B  # noqa: E402
import kag_data_r05 as D  # noqa: E402

ARMS = ("KA-FIXED", "DATA-RE", "PAKA-S")


def corrupt_m(df, domain, rung, kind):
    """Return the deployed anchor column for the corrupted rule."""
    m = df["m"].to_numpy(dtype=float)
    if kind == "clean":
        return m
    if kind == "sign" and domain == "CARB" and rung == "K2":
        return (0.5 * df["x_co2"].to_numpy(dtype=float)
                + 0.5 * df["x_cao"].to_numpy(dtype=float))
    if kind == "sign" and domain in ("FRP", "SFRC") and rung == "K2":
        return m - 2.0 * np.log(D.lambda_s(df["d"].to_numpy(dtype=float)))
    raise KeyError((domain, rung, kind))


def main():
    r06out = HERE.parents[1] / "R06_mechanism_pivot_2026-10-22" / "outputs"
    summary = pd.read_csv(r06out / "ladder_summary.csv")

    cases = [("CARB", "K2"), ("FRP", "K2"), ("SFRC", "K2")]
    rows = []
    for domain, rung in cases:
        for kind in ("clean", "corrupt"):
            df0, coded, free = R.load(domain, rung)
            coded, free = L._canon_feats(domain, coded, free)
            df = df0.copy()
            df["m"] = corrupt_m(df0, domain, rung, "clean" if kind == "clean" else "sign")
            targets = [s for s in df.source.unique()
                       if (df.source == s).sum() >= 6]
            t0 = time.time()
            for t in targets:
                r, _ = B.run_programme(df, coded, free, t, arms=ARMS, ks=(0,),
                                       n_draws=1, seed=0)
                r["target"] = t
                r["domain"], r["rung"], r["case"] = domain, rung, kind
                rows.append(r)
            print(f"{domain}-{rung} {kind}: {len(targets)} targets "
                  f"in {time.time() - t0:.1f}s", flush=True)

    per = pd.concat(rows, ignore_index=True)
    per.to_csv(OUT / "downstream_per_target.csv", index=False)

    agg = (per[per.k == 0]
           .groupby(["domain", "rung", "case", "method"])
           .rmse.agg(["mean", "std", "count"]).reset_index())
    agg.to_csv(OUT / "downstream_summary.csv", index=False)

    print("\n== mean LOPO RMSE at S = all, k = 0 ==")
    print(agg.pivot_table(index=["domain", "rung"], columns=["case", "method"],
                          values="mean").round(4).to_string())

    # ---- H4: clean chain must reproduce the R06 ladder value ----
    print("\n== H4: clean PAKA-S vs R06 ladder_summary (S=all, k=0) ==")
    for domain, rung in cases:
        got = agg[(agg.domain == domain) & (agg.rung == rung)
                  & (agg.case == "clean") & (agg.method == "PAKA-S")]["mean"].iloc[0]
        ref = summary[(summary.dataset == domain) & (summary.rung == rung)
                      & (summary.S.astype(str) == "all") & (summary.k == 0)
                      & (summary.method == "PAKA-S")].rmse.iloc[0]
        print(f"  {domain}-{rung}: rerun {got:.6f} vs R06 {ref:.6f}  "
              f"|diff| = {abs(got - ref):.2e}")

    # ---- H5: audit chain (data-only fallback) vs support chain (deploy) ----
    print("\n== H5: corrupted rule, two pipelines ==")
    for domain, rung in cases:
        g = agg[(agg.domain == domain) & (agg.rung == rung)]
        data = g[(g.case == "clean") & (g.method == "DATA-RE")]["mean"].iloc[0]
        paka_c = g[(g.case == "corrupt") & (g.method == "PAKA-S")]["mean"].iloc[0]
        paka_0 = g[(g.case == "clean") & (g.method == "PAKA-S")]["mean"].iloc[0]
        ka_c = g[(g.case == "corrupt") & (g.method == "KA-FIXED")]["mean"].iloc[0]
        ka_0 = g[(g.case == "clean") & (g.method == "KA-FIXED")]["mean"].iloc[0]
        print(f"  {domain}-{rung}:")
        print(f"    clean   PAKA {paka_0:.4f}  KA-FIXED {ka_0:.4f}")
        print(f"    corrupt PAKA {paka_c:.4f}  KA-FIXED {ka_c:.4f}")
        print(f"    data-only {data:.4f}")
        print(f"    audit chain (=data-only) {'BETTER' if data < paka_c else 'worse'}"
              f" than support chain by {paka_c - data:+.4f}")


if __name__ == "__main__":
    main()
