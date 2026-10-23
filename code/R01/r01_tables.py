"""Generate LaTeX tables and number macros for the R01 note from the output tables,
so that every number in the note is traceable to a regenerated CSV/JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R01 = Path(__file__).resolve().parents[1]
OUT, TAB = R01 / "outputs", R01 / "note" / "generated"
TAB.mkdir(parents=True, exist_ok=True)
FEAT = {"ln_d": "$d$", "ln_bw": "$b_w$", "ln_fc": "$f'_c$", "ln_rho": "$\\rho$", "ln_Ef": "$E_f$",
        "ln_ad": "$a/d$", "ln1p_F": "$F$"}
METH = {"ANCHOR": "规范原式", "KA-FIXED": "规范固定+项目效应", "KA-RE": "知识锚定层次模型",
        "KA-WB": "知识锚定(组内/组间)", "DATA-RE": "纯数据层次幂律", "DATA-WB": "纯数据(组内/组间)",
        "GBM": "梯度提升", "GBM-FEAT": "梯度提升+规范特征", "GBM-RES": "梯度提升学残差", "GP-RES": "高斯过程学残差"}
ORDER = list(METH)


def f3(v):
    return "--" if pd.isna(v) else f"{v:.3f}"


def f2(v):
    return "--" if pd.isna(v) else f"{v:.2f}"


def write(name, body):
    (TAB / name).write_text(body, encoding="utf-8")


def t_confounding(macros):
    f = pd.read_csv(OUT / "confounding_features.csv")
    s = json.loads((OUT / "confounding_summary.json").read_text())
    rows = []
    for r in f.itertuples():
        rows.append(f"{r.dataset} & {FEAT[r.feature]} & {f2(r.between_share)} & {r.programmes_varying}/{r.programmes_total} & "
                    f"{r.anchor_resid_beta_within:+.2f} ({r.anchor_resid_se_within:.2f}) & "
                    f"{r.anchor_resid_beta_between:+.2f} ({r.anchor_resid_se_between:.2f}) & {r.anchor_resid_z_gap:+.1f} \\\\")
    write("tab_confounding.tex", "\n".join(rows) + "\n")
    for ds, tag in (("FRP", "F"), ("SFRC", "S")):
        d = s[ds]
        macros[f"n{tag}"] = str(d["n"])
        macros[f"prog{tag}"] = str(d["sources"])
        macros[f"progsix{tag}"] = str(d["sources_ge6"])
        macros[f"icc{tag}"] = f2(d["resid_icc"])
        macros[f"tau{tag}"] = f2(d["resid_tau"])
        macros[f"sig{tag}"] = f2(d["resid_sig"])
        macros[f"ratio{tag}"] = f2(d["anchor_ratio_mean"])
        macros[f"cov{tag}"] = f2(d["anchor_ratio_cov"])
        r = f[(f.dataset == ds) & (f.feature == "ln_d")].iloc[0]
        macros[f"bsd{tag}"] = f2(r.between_share)
        macros[f"vard{tag}"] = str(r.programmes_varying)
        macros[f"wd{tag}"] = f"{r.anchor_resid_beta_within:+.2f}"
        macros[f"wdse{tag}"] = f2(r.anchor_resid_se_within)
        macros[f"bd{tag}"] = f"{r.anchor_resid_beta_between:+.2f}"
        macros[f"bdse{tag}"] = f2(r.anchor_resid_se_between)
        macros[f"zd{tag}"] = f"{r.anchor_resid_z_gap:.1f}"


def t_fewshot(macros):
    s = pd.read_csv(OUT / "fewshot_summary.csv")
    rows = []
    for ds in ("FRP", "SFRC"):
        for m in ORDER:
            g = s[(s.dataset == ds) & (s.method == m)].set_index("k")
            if g.empty:
                continue
            rows.append(f"{ds} & {METH[m]} & {f3(g.rmse.get(0))} [{f2(g.rmse_lo.get(0))}, {f2(g.rmse_hi.get(0))}] & "
                        f"{f3(g.rmse.get(1))} & {f3(g.rmse.get(3))} [{f2(g.rmse_lo.get(3))}, {f2(g.rmse_hi.get(3))}] & "
                        f"{f2(g.cover90.get(3))} & {f2(g.unconservative.get(0))} \\\\")
        rows.append("\\midrule")
    write("tab_fewshot.tex", "\n".join(rows[:-1]) + "\n")
    for ds, tag in (("FRP", "F"), ("SFRC", "S")):
        for m, mt in (("KA-RE", "kare"), ("DATA-RE", "datare"), ("GBM", "gbm"), ("GBM-RES", "gbmres"),
                      ("KA-FIXED", "kafix"), ("GP-RES", "gpres")):
            g = s[(s.dataset == ds) & (s.method == m)].set_index("k")
            macros[f"fs{mt}zero{tag}"] = f3(g.rmse.get(0))
            macros[f"fs{mt}three{tag}"] = f3(g.rmse.get(3))


def t_scarcity(macros):
    s = pd.read_csv(OUT / "scarcity_summary.csv")
    order = ["3", "6", "12", "24", "all"]
    rows = []
    for ds in ("FRP", "SFRC"):
        for k in (0, 3):
            for S in order:
                g = s[(s.dataset == ds) & (s.k == k) & (s.S == S)].set_index("method")
                ka = g.loc["KA-RE"]
                rows.append(f"{ds} & {k} & {S} & {ka.n_train:.0f} & {f3(g.rmse.get('ANCHOR'))} & {f3(g.rmse.get('KA-FIXED'))} & "
                            f"{f3(ka.rmse)} & {f3(g.rmse.get('DATA-RE'))} & {f3(g.rmse.get('GBM-RES'))} & {f3(g.rmse.get('GBM'))} & "
                            f"{ka['gain_vs_DATA-RE']:+.3f} [{ka['gain_vs_DATA-RE_lo']:+.2f}, {ka['gain_vs_DATA-RE_hi']:+.2f}] \\\\")
            rows.append("\\midrule")
    write("tab_scarcity.tex", "\n".join(rows[:-1]) + "\n")
    # programme-equivalent value of the anchor: the training size S_eq at which the data-only
    # curve reaches the anchored model's error at S (log-linear interpolation in S, k = 0)
    eq_rows = []
    for ds, tag in (("FRP", "F"), ("SFRC", "S")):
        d = s[(s.dataset == ds) & (s.k == 0)]
        xs = np.log([3, 6, 12, 24])
        data = np.array([d[(d.S == S) & (d.method == "DATA-RE")].rmse.iloc[0] for S in ("3", "6", "12", "24")])
        for S, st in (("3", "three"), ("6", "six")):
            ka = d[(d.S == S) & (d.method == "KA-RE")].rmse.iloc[0]
            seq = np.nan
            for i in range(3):
                if data[i] >= ka >= data[i + 1]:
                    w = (data[i] - ka) / (data[i] - data[i + 1])
                    seq = float(np.exp(xs[i] + w * (xs[i + 1] - xs[i])))
            macros[f"seq{st}{tag}"] = f"{seq:.1f}"
            macros[f"worth{st}{tag}"] = f"{seq - int(S):.1f}"
            eq_rows.append(dict(dataset=ds, S=int(S), rmse_anchored=ka, S_equivalent_data_only=seq,
                                programmes_worth=seq - int(S)))
    pd.DataFrame(eq_rows).to_csv(OUT / "knowledge_exchange_rate.csv", index=False)
    for ds, tag in (("FRP", "F"), ("SFRC", "S")):
        for S, st in (("3", "three"), ("6", "six"), ("12", "twelve"), ("24", "tfour"), ("all", "all")):
            ka = s[(s.dataset == ds) & (s.k == 0) & (s.S == S) & (s.method == "KA-RE")].iloc[0]
            macros[f"gain{st}{tag}"] = f"{ka['gain_vs_DATA-RE']:+.3f}"
            macros[f"gainlo{st}{tag}"] = f"{ka['gain_vs_DATA-RE_lo']:+.3f}"
            macros[f"gainhi{st}{tag}"] = f"{ka['gain_vs_DATA-RE_hi']:+.3f}"
            macros[f"ntr{st}{tag}"] = f"{ka.n_train:.0f}"
            for m, mt in (("KA-RE", "kare"), ("DATA-RE", "datare")):
                macros[f"sc{mt}{st}{tag}"] = f3(s[(s.dataset == ds) & (s.k == 0) & (s.S == S) & (s.method == m)].rmse.iloc[0])


def t_extrap(macros):
    e = pd.read_csv(OUT / "extrapolation_summary.csv")
    sp = pd.read_csv(OUT / "extrapolation_split.csv")
    rows = []
    for ds in ("FRP", "SFRC"):
        for m in ORDER:
            g = e[(e.dataset == ds) & (e.method == m)].set_index("k")
            if g.empty:
                continue
            rows.append(f"{ds} & {METH[m]} & {f3(g.rmse.get(0))} & {g.bias.get(0):+.3f} & {f2(g.unconservative.get(0))} & "
                        f"{f3(g.rmse.get(3))} \\\\")
        rows.append("\\midrule")
    write("tab_extrap.tex", "\n".join(rows[:-1]) + "\n")
    for r in sp.itertuples():
        tag = "F" if r.dataset == "FRP" else "S"
        macros[f"exthr{tag}"] = f"{r.threshold_mm:.0f}"
        macros[f"extrprog{tag}"] = str(r.train_programmes)
        macros[f"extgt{tag}"] = str(r.targets)
        macros[f"extgtn{tag}"] = str(r.target_n)
        macros[f"extbeyond{tag}"] = f"{100 * r.target_beyond_train_max:.0f}"
        macros[f"extdmax{tag}"] = f"{r.target_d_max:.0f}"
        macros[f"extrdmax{tag}"] = f"{r.train_d_max:.0f}"


def t_stress(macros):
    st = pd.read_csv(OUT / "stress_summary.csv")
    variants = ["none", "size+0.3", "size-0.3", "missing-term"]
    rows = []
    for ds in ("FRP", "SFRC"):
        for S in ("6", "all"):
            for m in ("KA-FIXED", "KA-RE", "KA-WB", "GBM-RES"):
                g = st[(st.dataset == ds) & (st.S_label.astype(str) == S) & (st.method == m) & (st.k == 0)].set_index("variant")
                rows.append(f"{ds} & {S} & {METH[m]} & " + " & ".join(f3(g.rmse.get(v)) for v in variants) + " \\\\")
        rows.append("\\midrule")
    write("tab_stress.tex", "\n".join(rows[:-1]) + "\n")


def t_evidence(macros):
    ev = pd.read_csv(OUT / "evidence_unit_summary.csv")
    rows = []
    for ds in ("FRP", "SFRC"):
        for N in (24, 48, 96):
            g = ev[(ev.dataset == ds) & (ev.N == N) & (ev.k == 0)].set_index("method")
            cells = []
            for m in ("DATA-RE", "KA-RE", "KA-FIXED", "GBM-RES"):
                r = g.loc[m]
                cells.append(f"{r.dispersed_gain:+.3f} [{r.gain_lo:+.2f}, {r.gain_hi:+.2f}]")
            r0 = g.iloc[0]
            rows.append(f"{ds} & {N} & {r0.programmes_concentrated:.1f} / {r0.programmes_dispersed:.1f} & " + " & ".join(cells) + " \\\\")
    write("tab_evidence.tex", "\n".join(rows) + "\n")
    for ds, tag in (("FRP", "F"), ("SFRC", "S")):
        r = ev[(ev.dataset == ds) & (ev.N == 24) & (ev.k == 0) & (ev.method == "DATA-RE")].iloc[0]
        macros[f"evdata{tag}"] = f"{r.dispersed_gain:+.3f}"
        r = ev[(ev.dataset == ds) & (ev.N == 24) & (ev.k == 0) & (ev.method == "KA-RE")].iloc[0]
        macros[f"evka{tag}"] = f"{r.dispersed_gain:+.3f}"


def main():
    macros = {}
    t_confounding(macros)
    t_fewshot(macros)
    t_scarcity(macros)
    t_extrap(macros)
    t_stress(macros)
    t_evidence(macros)
    bad = [k for k in macros if any(ch.isdigit() for ch in k)]
    assert not bad, bad
    write("numbers.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros, tables in {TAB}")


if __name__ == "__main__":
    main()
