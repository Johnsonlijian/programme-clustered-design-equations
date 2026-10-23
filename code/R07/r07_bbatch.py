"""R07 B-batch: references, positioning, identification assumptions, multiplicity,
CACAIE discussion, and moving Fig 7 / Table 8 to the supplement."""
from pathlib import Path

M = Path(r"R:\IMUT_WORKPOOL\01_PROJECTS\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-KAG-SmallData\paper\manuscript")
MAIN = M / "main_r07.tex"
SUP = M / "supplement.tex"
BIB = M / "references.bib"

t = MAIN.read_text(encoding="utf-8")
s = SUP.read_text(encoding="utf-8")


def sub(old, new, label, n_expect=1):
    global t
    n = t.count(old)
    assert n == n_expect, f"{label}: {n} (expected {n_expect})"
    t = t.replace(old, new)
    print("ok:", label)


def cut_block(start_marker, end_marker, label):
    """Remove a float block and return it."""
    global t
    i = t.index(start_marker)
    j = t.index(end_marker, i) + len(end_marker)
    block = t[i:j]
    t = t[:i] + t[j:]
    print("cut:", label, len(block), "chars")
    return block


# ---------- B5a references ----------
BIB_TEXT = r"""
@article{DerKiureghian2009,
  author  = {Der Kiureghian, Armen and Ditlevsen, Ove},
  title   = {Aleatory or epistemic? {Does} it matter?},
  journal = {Structural Safety},
  volume  = {31},
  number  = {2},
  pages   = {105--112},
  year    = {2009},
  doi     = {10.1016/j.strusafe.2008.06.020}
}

@article{KennedyOHagan2001,
  author  = {Kennedy, Marc C. and O'Hagan, Anthony},
  title   = {Bayesian Calibration of Computer Models},
  journal = {Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
  volume  = {63},
  number  = {3},
  pages   = {425--464},
  year    = {2001},
  doi     = {10.1111/1467-9868.00294}
}

@article{Wolpert1992,
  author  = {Wolpert, David H.},
  title   = {Stacked generalization},
  journal = {Neural Networks},
  volume  = {5},
  number  = {2},
  pages   = {241--259},
  year    = {1992},
  doi     = {10.1016/S0893-6080(05)80023-1}
}

@article{Breiman1996,
  author  = {Breiman, Leo},
  title   = {Stacked Regressions},
  journal = {Machine Learning},
  volume  = {24},
  number  = {1},
  pages   = {49--64},
  year    = {1996},
  doi     = {10.1023/A:1018046112532}
}

@inproceedings{Yao2022WildTime,
  author    = {Yao, Huaxiu and Choi, Caroline and Cao, Bochuan and Lee, Yoonho and Lin, Yizhe and Kim, Yong Jae and Finn, Chelsea},
  title     = {{Wild-Time}: A Benchmark of In-the-Wild Distribution Shift over Time},
  booktitle = {Advances in Neural Information Processing Systems 35},
  pages     = {10309--10324},
  year      = {2022},
  doi       = {10.52202/068431-0749}
}
"""
bib = BIB.read_text(encoding="utf-8")
assert "DerKiureghian2009" not in bib
BIB.write_text(bib.rstrip() + "\n" + BIB_TEXT, encoding="utf-8")
print("ok: bib entries appended")

# ---------- B5b positioning paragraph ----------
sub("The econometrics of panel data supplies the language.",
    r"""This paper sits where three literatures meet. Model-error and code-calibration work treats resistance model error as an epistemic random variable and calibrates partial factors on pooled test data, but the error is assumed exchangeable across specimens \citep{DerKiureghian2009,KennedyOHagan2001}; knowledge-guided learning assumes the prior is faithful rather than testing it \citep{vonRueden2023,Karniadakis2021}; and the reproducibility literature shows that evaluations ignoring dependence are optimistic without turning the dependence into an instrument \citep{Kapoor2023,Yao2022WildTime}. The design-equation term as a falsifiable hypothesis---graded inside programmes and priced in programmes---is the object that connects them.

The econometrics of panel data supplies the language.""",
    "positioning paragraph")

# ---------- B5c identification assumptions + B5d multiplicity ----------
sub(r"\subsection{What the check buys}\label{sec:audit-downstream}",
    r"""Three assumptions carry the within-programme estimates, and one convention governs their reading. First, the assignment of a variable inside a programme is unrelated to the residual: experimenters choose levels before testing, and the k-shot protocol never looks at a held-out outcome. Second, measurement conventions shared inside a programme enter the programme effect $u_j$ and cancel in the within contrast; if the measurement error scales with the variable's level, the cancellation is partial and the within exponent is attenuated towards zero. Third, an exponent is read as a power law: for equation-form anchors the same contrast grades the sign only, because the constants absorb scale. Violations of the first two bias the within estimate towards the between estimate. On reading: the ladder, the contrasts and the direction of every test were pre-specified, and no multiplicity adjustment is claimed beyond that pre-registration; the one verdict that would not survive a conservative correction---the FRP $a/d$ term, $z\approx-\aZFrpAd$---is reported as a caveat rather than a confirmed rejection.

\subsection{What the check buys}\label{sec:audit-downstream}""",
    "identification assumptions + multiplicity")

# ---------- B5e CACAIE discussion paragraph ----------
sub(r"\paragraph{Model uncertainty for a new material system}",
    r"""\paragraph{Benchmarking infrastructure models}
Two consequences follow for how data-driven models of infrastructure are benchmarked. First, a random split of a pooled test database measures interpolation inside programmes, not transfer to new ones; the same records evaluated leave-programme-out give a different ranking, and the gap between the two is the programme effect rather than noise. Second, the protocol transfers with its inputs, not with its results: the three layers need only a programme identifier, a declared rule and a program that applies it, so the same check, budget and stack can be run on any pooled infrastructure database---bridges, pavements, geotechnical sites---without new theory. What changes between domains is the estimate of $B$; the procedure does not.

\paragraph{Model uncertainty for a new material system}""",
    "CACAIE benchmark paragraph")

# ---------- B2 move Fig 7 to the supplement ----------
fig_block = cut_block(
    "\\begin{figure}[t]\\centering\n\\includegraphics[width=\\textwidth]{fig4_unit_extrapolation_circularity.pdf}",
    "\\end{figure}",
    "fig:unit block")
for a, b, lab in [
    (r"(Fig.~\ref{fig:unit}a, Table~S1)", r"(Fig.~S3a, Table~S1)", "fig ref a"),
    (r"(Fig.~\ref{fig:unit}b, Table~S2)", r"(Fig.~S3b, Table~S2)", "fig ref b"),
    (r"(Fig.~\ref{fig:unit}c, Table~S3)", r"(Fig.~S3c, Table~S3)", "fig ref c"),
]:
    sub(a, b, lab)

# ---------- B4 move Table 8 to the supplement ----------
tab_block = cut_block(
    "\\begin{table}[t]\\centering\\scriptsize\\renewcommand{\\arraystretch}{0.82}\n\\caption{New-programme RMSE ($k=0$) against library size $S$ for each anchor rung.}\\label{tab:scarcity}",
    "\\end{table}",
    "tab:scarcity block")
sub(r"Table~\ref{tab:scarcity}", r"Table~S9", "tab ref")

# ---------- supplement: append Figure S3 and Table S9 ----------
S9 = "\n" + fig_block.replace("[t]", "[H]").replace("\\label{fig:unit}", "\\label{figS3}") + "\n\n" \
     + tab_block.replace("[t]", "[H]").replace("\\label{tab:scarcity}", "\\label{tabS9}") + "\n"
assert "\\end{document}" in s
s = s.replace("\\end{document}", S9 + "\n\\end{document}")

MAIN.write_text(t, encoding="utf-8")
SUP.write_text(s, encoding="utf-8")
print("written main", len(t), "supp", len(s))
