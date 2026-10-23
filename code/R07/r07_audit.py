"""R07 WP1 -- label-free rule-validity audit (metadata layer).

The audit checks a *deployed rule application* against what the rule itself
declares, without using any test label:

  C1 scope      : the declared range of each coded variable (e.g. a/d <= 2.5,
                  reactive CaO required) -- violation fraction.
  C2 unit       : physical plausibility bounds (a 10x unit slip falls outside);
                  physical variables are reconstructed from log columns.
  C3 positivity : the implemented anchor is finite on every row.
  C4 sign       : the numeric local exponent of each coded variable has the
                  declared sign (0 = the rule asserts no dependence).
  C5 exponent   : the numeric local exponent equals the declared value
                  (power-law anchors and lambda_s carriers only; equation-form
                  anchors are excluded).

All thresholds come from the rule metadata below (transcribed from the R06
anchor table / 07_source_registry), never from the deployment data. The audit
is a batch decision: a rule application on one database is rejected if any
check fires.

WP2 (the empirical layer: within-programme exponents) is *not* here; it lives
in the R06 signature outputs and is reused by r07_integrate.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),
                str(HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts"),
                str(HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import kag_data_r05 as D  # noqa: E402
import kag_data as K1  # noqa: E402

# --------------------------------------------------------------------------
# declared metadata of the eleven anchor forms (see R06 tab:rungs / S6)
#   scope : (op, value) on a physical variable
#   unit  : plausible physical range (unit-sanity bounds)
#   sign  : declared sign of the local exponent (+1 / -1 / 0)
#   expo  : declared local exponent ("ls" = 1 + d ln(lambda_s)/d ln(d) carrier,
#           with the base exponent given; only for anchors whose coded terms are
#           power laws)
#   log   : columns that are already log(physical); physical values are exp()
# --------------------------------------------------------------------------
def _ls_slope(d):
    """d ln(lambda_s)/d ln(d) for lambda_s = min(1, sqrt(2/(1+0.004 d))), d in mm.

    ACI 318-19 caps lambda_s at 1, so the slope is exactly zero for d <= 250 mm.
    """
    d = np.asarray(d, dtype=float)
    base = -0.5 * (0.004 * d) / (1.0 + 0.004 * d)
    return np.where(d > 250.0, base, 0.0)


META = {
    ("FRP", "K1"): dict(
        feats=["d", "fc", "bw", "rho"],
        scope={"a_d": ("ge", 2.5)},
        unit={"d": (30, 5000), "fc": (5, 200), "bw": (20, 5000), "rho": (1e-4, 0.30)},
        sign={"d": +1, "fc": +1, "bw": +1, "rho": +1},
        expo={"d": 1.0, "bw": 1.0},   # fc enters through E_c too -> equation form
    ),
    ("FRP", "K2"): dict(
        feats=["d", "fc", "bw", "rho"],
        scope={"a_d": ("ge", 2.5)},
        unit={"d": (30, 5000), "fc": (5, 200), "bw": (20, 5000), "rho": (1e-4, 0.30)},
        sign={"d": +1, "fc": +1, "bw": +1, "rho": +1},
        expo={"d": ("ls", 1.0), "bw": 1.0},
    ),
    ("SFRC", "K1"): dict(
        feats=["d", "fc", "b", "rho", "a_d", "F"],
        scope={"a_d": ("ge", 2.5)},
        unit={"d": (30, 5000), "fc": (5, 300), "b": (20, 5000), "rho": (1e-4, 0.30),
              "a_d": (0.3, 10.0), "F": (0.0, 5.0)},
        sign={"d": +1, "fc": +1, "b": +1, "rho": +1, "F": +1},
        expo={"d": 1.0},
    ),
    ("SFRC", "K2"): dict(
        feats=["d", "fc", "b", "rho", "a_d", "F"],
        scope={"a_d": ("ge", 2.5)},
        unit={"d": (30, 5000), "fc": (5, 300), "b": (20, 5000), "rho": (1e-4, 0.30),
              "a_d": (0.3, 10.0), "F": (0.0, 5.0)},
        sign={"d": +1, "fc": +1, "b": +1, "rho": +1, "F": +1},
        expo={"d": ("ls", 1.0)},
    ),
    ("CARB", "K1"): dict(
        feats=["x_co2", "x_cao", "x_wb"],
        scope={"cao": ("gt", 0), "co2": ("gt", 0), "wb": ("gt", 0)},
        unit={"co2": (0.01, 100.0), "wb": (0.15, 1.2), "cao": (20.0, 600.0)},
        sign={"co2": +1},
        expo={"co2": 0.5},
        log={"x_co2": "co2", "x_cao": "cao", "x_wb": "wb"},
        ref={"x_co2": 0.04, "x_cao": 150.0, "x_wb": 1.0},
    ),
    ("CARB", "K2"): dict(
        feats=["x_co2", "x_cao"],
        scope={"cao": ("gt", 0), "co2": ("gt", 0), "wb": ("gt", 0)},
        unit={"co2": (0.01, 100.0), "wb": (0.15, 1.2), "cao": (20.0, 600.0)},
        sign={"co2": +1, "cao": -1},
        expo={"co2": 0.5, "cao": -0.5},
        log={"x_co2": "co2", "x_cao": "cao", "x_wb": "wb"},
        ref={"x_co2": 0.04, "x_cao": 150.0, "x_wb": 1.0},
    ),
    ("CARB", "K3"): dict(
        feats=["x_co2", "x_wb", "x_cao"],
        scope={"cao": ("gt", 0), "co2": ("gt", 0), "wb": ("gt", 0)},
        unit={"co2": (0.01, 100.0), "wb": (0.15, 1.2), "cao": (20.0, 600.0)},
        sign={"co2": +1, "wb": +1, "cao": -1},
        expo={"co2": 0.5, "wb": 1.0, "cao": -1.0},
        log={"x_co2": "co2", "x_cao": "cao", "x_wb": "wb"},
        ref={"x_co2": 0.04, "x_cao": 150.0, "x_wb": 1.0},
    ),
    ("DEEP", "BEAM"): dict(
        feats=["d", "bw", "fc", "rho", "a_d"],
        scope={"a_d": ("le", 2.5)},
        unit={"d": (30, 5000), "bw": (20, 5000), "fc": (5, 300), "rho": (1e-4, 0.30),
              "a_d": (0.3, 10.0)},
        sign={"d": +1, "bw": +1, "fc": +1, "rho": +1, "a_d": 0},
        expo={"d": ("ls", 1.0), "a_d": 0.0},
    ),
    ("DEEP", "ARCH"): dict(
        feats=["d", "bw", "fc", "rho", "a_d"],
        scope={"a_d": ("le", 2.5)},
        unit={"d": (30, 5000), "bw": (20, 5000), "fc": (5, 300), "rho": (1e-4, 0.30),
              "a_d": (0.3, 10.0)},
        sign={"d": +1, "bw": +1, "fc": +1, "rho": +1, "a_d": -1},
        expo={"d": 1.0, "a_d": -4.0 / 3.0, "fc": 1.0 / 3.0, "rho": 1.0 / 3.0,
              "bw": 1.0},   # d: V = v(d/a) * b * d, with a/d held -> exponent 1
    ),
    ("DEEP", "ARCHS"): dict(
        feats=["d", "bw", "fc", "rho", "a_d"],
        scope={"a_d": ("le", 2.5)},
        unit={"d": (30, 5000), "bw": (20, 5000), "fc": (5, 300), "rho": (1e-4, 0.30),
              "a_d": (0.3, 10.0)},
        sign={"d": +1, "bw": +1, "fc": +1, "rho": +1, "a_d": -1},
        expo={"d": ("ls", 1.0), "a_d": -4.0 / 3.0, "fc": 1.0 / 3.0,
              "rho": 1.0 / 3.0, "bw": 1.0},
    ),
    ("DEEP", "ARCH99"): dict(
        feats=["d", "bw", "fc", "rho", "a_d"],
        scope={"a_d": ("le", 2.5)},
        unit={"d": (30, 5000), "bw": (20, 5000), "fc": (5, 300), "rho": (1e-4, 0.30),
              "a_d": (0.3, 10.0)},
        sign={"bw": +1, "fc": +1, "rho": +1},
        expo={},  # piecewise (capped) -- equation form, not a power law
    ),
}


def physical(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Reconstruct physical variables from log columns (copy, index-aligned).

    Log columns are centred on the loader's reference values (e.g. x_co2 =
    ln(co2 / 0.04), x_cao = ln(cao / 150)), so the inverse uses the same refs.
    """
    out = df
    ref = meta.get("ref", {})
    for lc, pv in meta.get("log", {}).items():
        if lc in df.columns:
            if out is df:
                out = df.copy()
            out[pv] = ref.get(lc, 1.0) * np.exp(df[lc].to_numpy(dtype=float))
    return out


def _local_exponents(df, meta, predict, h=1e-4, rows=256):
    """Numeric local exponents of the anchor w.r.t. each coded feature.

    Returns (dict feature -> array, sampled base frame)."""
    cols = [c for c in meta["feats"] if c in df.columns]
    if not cols:
        return {}, None
    idx = np.linspace(0, len(df) - 1, min(rows, len(df))).astype(int)
    base = df.iloc[idx].reset_index(drop=True)
    m0 = np.asarray(predict(base), dtype=float)
    out = {}
    for c in cols:
        b = base.copy()
        if c in meta.get("log", {}):
            b[c] = b[c] + h          # log space: additive step
        else:
            b[c] = b[c] * np.exp(h)  # physical space: multiplicative step
        m1 = np.asarray(predict(b), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            le = (m1 - m0) / h
        out[c] = le
    return out, base


def audit(domain, rung, df, predict, tol=0.01):
    """Run the metadata audit on one rule application. Returns a dict."""
    meta = META[(domain, rung)]
    phys = physical(df, meta)
    n = len(df)
    res = dict(domain=domain, rung=rung, n=n, checks={}, rejected=False, reasons=[])

    # C1 declared scope
    bad = 0
    for col, (op, val) in meta.get("scope", {}).items():
        if col not in phys.columns:
            continue
        v = pd.to_numeric(phys[col], errors="coerce").to_numpy()
        if op == "gt":
            bad += int(np.sum(~(v > val)))
        elif op == "ge":
            bad += int(np.sum(~(v >= val)))
        elif op == "le":
            bad += int(np.sum(~(v <= val)))
    res["checks"]["scope"] = bad / n if n else 0.0

    # C2 unit plausibility
    bad = 0
    for col, (lo, hi) in meta.get("unit", {}).items():
        if col not in phys.columns:
            continue
        v = pd.to_numeric(phys[col], errors="coerce").to_numpy()
        bad += int(np.sum(~((v >= lo) & (v <= hi))))
    res["checks"]["unit"] = bad / n if n else 0.0

    # C3 positivity / finiteness of the implemented anchor
    m = np.asarray(predict(df), dtype=float)
    res["checks"]["nonfinite"] = float(np.mean(~np.isfinite(m)))

    # C4 sign and C5 exponent, on rows where the anchor is finite
    le, base = _local_exponents(df, meta, predict)
    sign_bad = 0
    expo_bad = 0
    seen = 0
    for c, vals in le.items():
        vals = np.asarray(vals, dtype=float)
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        seen += int(finite.sum())
        pname = meta.get("log", {}).get(c, c)   # column name -> physical name
        dec = meta["sign"].get(pname)
        if dec is not None:
            if dec == 0:
                sign_bad += int(np.sum(finite & (np.abs(vals) > 0.05)))
            elif dec > 0:
                sign_bad += int(np.sum(finite & (vals < -0.02)))
            else:
                sign_bad += int(np.sum(finite & (vals > 0.02)))
        de = meta["expo"].get(pname)
        if de is not None:
            if isinstance(de, tuple) and de[0] == "ls":
                dvals = base["d"].to_numpy(dtype=float)
                declared = de[1] + _ls_slope(dvals)
                # the ACI cap makes lambda_s non-differentiable at d = 250 mm;
                # rows within 1 mm of the kink are excluded from the check
                ok_rows = finite & (np.abs(dvals - 250.0) >= 1.0)
            else:
                declared = np.full_like(vals, float(de))
                ok_rows = finite
            expo_bad += int(np.sum(ok_rows & (np.abs(vals - declared) > tol)))
    res["checks"]["sign"] = sign_bad / seen if seen else 0.0
    res["checks"]["expo"] = expo_bad / seen if seen else 0.0

    for k in ("scope", "unit", "nonfinite", "sign", "expo"):
        if res["checks"][k] > 0:
            res["reasons"].append(k)
    res["rejected"] = bool(res["reasons"])
    return res


# --------------------------------------------------------------------------
# support-domain gate (RealRC-style baseline): training-only kNN density
# --------------------------------------------------------------------------
def support_gate(train_df, deploy_df, feats, k=5, q=0.95):
    """Flag a deployment batch when its inputs leave the training support.

    Rows of the deployment with missing features are dropped (a density gate
    does not check required-input presence -- that is a metadata check and
    belongs to the audit). Returns (flagged, median_distance, threshold).
    """
    cols = [c for c in feats if c in train_df.columns and c in deploy_df.columns]
    if not cols:
        return False, np.nan, np.nan
    X = train_df[cols].to_numpy(dtype=float)
    Y = deploy_df[cols].dropna().to_numpy(dtype=float)
    if len(Y) == 0:
        return True, np.nan, np.nan
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    Y = (Y - mu) / sd
    # train self-distances (exclude the point itself)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    kk = min(k, len(X) - 1)
    train_d = np.sort(D, axis=1)[:, :kk].mean(axis=1)
    thr = np.quantile(train_d, q)
    # deployment distance to the k nearest training points
    Dd = np.sqrt(((Y[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    dep_d = np.sort(Dd, axis=1)[:, :kk].mean(axis=1)
    return bool(np.median(dep_d) > thr), float(np.median(dep_d)), float(thr)
