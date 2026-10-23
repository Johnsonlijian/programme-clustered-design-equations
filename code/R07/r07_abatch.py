"""R07 A-batch: CACAIE mechanism framing (title, abstract, intro opener, 5 conclusions, register)."""
from pathlib import Path

P = Path(r"R:\IMUT_WORKPOOL\01_PROJECTS\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-KAG-SmallData\paper\manuscript\main_r07.tex")
t = P.read_text(encoding="utf-8")


def sub(old, new, label, n_expect=1):
    global t
    n = t.count(old)
    assert n == n_expect, f"{label}: found {n} (expected {n_expect})"
    t = t.replace(old, new)
    print("ok:", label)


# ---------- A1 title (CACAIE mechanism version) ----------
sub(r"\title{Audit, price, allocate: usability and value of codified design equations for unseen test programmes}",
    r"\title{Design equations as testable mechanism hypotheses: programme-level evidence, pricing, and trust allocation}",
    "title")

# ---------- A2 abstract ----------
a0 = t.index(r"\begin{abstract}") + len(r"\begin{abstract}")
a1 = t.index(r"\end{abstract}")
abstract = r"""

A design code is worth most where data are scarcest, and in the pooled databases on which codes and their machine-learning rivals are calibrated the scarce unit is the test programme, not the test: about half of the log-error variance is shared within programmes ($\ICC={\iccMin}$--${\iccMax}$), so {\nAll} tests carry the information of {\neffMin}--{\neffMax} independent observations. Building on that regularity, we treat each term of a design equation as a testable mechanism hypothesis and ask three questions: whether the deployed rule is the rule that was declared, what the rule is worth for a programme not yet in the database, and how it should share trust with data. Within-programme contrasts grade every asserted exponent free of laboratory confounding; a metadata check catches every injected sign, unit and range corruption with no false rejection and no test labels, where a support-domain gate is blind to corruptions that stay inside the training support. The verdicts are graded rather than binary: the ACI 318-19 size factor keeps its sign in all three shear systems and closes the within-programme depth deviation in SFRC and deep beams, but only halves it in FRP; the beam equation's silence on slenderness and a $-1/2$ binding exponent are contradicted. A first-order variance budget, $S^\ast=pc/B$, converts the residual misfit into the number of programmes at which data-only learning overtakes the anchor---within a pre-registered factor of two on all eleven computable rungs---and its own inputs identify three failure regimes. Programme-aware knowledge anchoring then allocates trust among code-imposed, code-as-prior and data-only arms by whole-programme log score, staying within {\regmaxpakaL} log RMSE of the best arm and cutting error by {\kredPakaMin}--{\kredPakaMax}\% with three tests, while capping the cost of a corrupted rule at $\aDownPakaMin$--$\aDownPakaMax$ against $\aDownKaMax$ for direct deployment. Identification, price and allocation are all estimable before new tests exist.
"""
t = t[:a0] + abstract + t[a1:]
print("ok: abstract")

# ---------- A3 intro opener and gap sentences ----------
sub("Codified design equations and the machine-learning models that are increasingly proposed to replace them are calibrated on databases assembled from the literature.",
    "A codified design equation is a hypothesis about mechanism: each coded term asserts a sign, an exponent and a range of applicability, and each omitted variable asserts that the mechanism does not depend on it. These hypotheses are calibrated on databases assembled from the literature.",
    "intro opener")
sub("Two research lines now meet in these databases.",
    "Two literatures bear on these databases.", "intro line 2")
sub("What has not been examined is how the two interact in engineering test databases, where the knowledge is a design code and the dependence is the programme.",
    "Neither literature, however, prices a design code under the dependence structure of test databases, where the prior is a code and the dependence is the programme.",
    "intro gap sentence")
sub("The interaction matters because a code's value should be largest where data are scarce",
    "The interaction is decisive: a code's value should be largest where data are scarce",
    "intro interaction")
sub("Second, we audit a rule before pricing it", "Second, we grade a rule before pricing it", "intro step2 verb")
sub("the audit grades terms individually", "the grading resolves terms individually", "intro grading")

# ---------- A4 conclusions: 7 -> 5 ----------
old_concl_start = t.index(r"\begin{enumerate}")
old_concl_end = t.index(r"\end{enumerate}") + len(r"\end{enumerate}")
new_concl = r"""\begin{enumerate}
\item In four public databases, about half of the codified models' log-error variance is shared within a test programme, so hundreds to a thousand tests carry the information of a few dozen independent ones; the programme, not the test, is the unit of evidence.
\item A rule can be graded before it is priced. Deterministic checks of its declared metadata detect injected sign, unit and range corruptions of a deployed equation with no test labels, where a support-domain gate is blind to every corruption that leaves the inputs inside the training support; within-programme contrasts then resolve the asserted exponents one by one---confirming the sign of the ACI 318-19 size factor in all three shear systems and its magnitude in SFRC and the deep beams, the square-root CO$_2$ law and the buffer-capacity form in carbonation, and arch action in deep beams, and contradicting a $-1/2$ binding-capacity exponent and beam action for deep beams. Between-programme contrasts are attenuated or sign-reversed and cannot support the same judgement.
\item The value of an anchor tracks the fidelity of its exponents to these signatures: across the fifteen anchor rungs it ranges from negative at every library size (beam action in deep beams) through crossovers of five to twenty-three programmes (the shear rungs and the $\sqrt C$ and rejected-exponent carbonation anchors) to an anchor that is never overtaken (the buffer-capacity carbonation anchor).
\item A first-order variance budget, $S^\ast=p\,c/B$, predicts each anchor's crossover from the database's own variance components and misfit; every computable prediction agrees with the observed crossover within a factor of two, and the regimes where the budget cannot apply---misfit at the noise floor, a misspecified mechanism inflating the programme variance, between-programme masking---are flagged by its own inputs.
\item Programme-aware knowledge anchoring stacks the three trust levels of a chosen anchor by whole-programme log score and re-weights them with the first tests of a new programme; its worst shortfall is {\regmaxpakaL} RMSE, three tests reduce its error by {\kredPakaMin}--{\kredPakaMax}\%, and when the rule itself is corrupted, falling back to data-only caps the loss at $\aDownPakaMin$--$\aDownPakaMax$ log RMSE against $\aDownKaMax$ for direct deployment.
\end{enumerate}"""
t = t[:old_concl_start] + new_concl + t[old_concl_end:]
print("ok: conclusions 7->5")

# ---------- A5 register: audit -> check/grading ----------
for old, new, lab in [
    (r"\section{Auditing usability without labels}", r"\section{Grading a rule against the data}", "sec title"),
    ("We separate two audits", "We separate two checks", "two checks"),
    (r"\subsection{The metadata audit}", r"\subsection{The metadata check}", "sub 3.1"),
    ("the audit is a batch decision", "the check is a batch decision", "batch decision"),
    ("The audit cannot check what the rule does not assert", "The metadata check cannot verify what the rule does not assert", "cannot check"),
    (r"\subsection{The empirical audit and the anchor dossier}", r"\subsection{The empirical grading and the anchor dossier}", "sub 3.2"),
    (r"\subsection{What the audit buys}", r"\subsection{What the check buys}", "sub 3.3"),
    ("The audit decides usability", "The check decides usability", "decides usability"),
    ("Metadata audit under four corruptions", "Metadata check under four corruptions", "table caption"),
    ("Corruption & Group & Cases & Audit & Support gate", "Corruption & Group & Cases & Check & Support gate", "table header"),
    (r"The audit of Section~\ref{sec:audit} decides", r"The check of Section~\ref{sec:audit} decides", "bridge"),
    ("which is what the identification audit shows", "which is what the identification check shows", "identification audit 1"),
    ("Identification audit. Deviations", "Identification check. Deviations", "tab audit caption"),
    ("Once the signature audit selects an anchor", "Once the signature grading selects an anchor", "signature audit"),
]:
    sub(old, new, lab)

P.write_text(t, encoding="utf-8")
print("written", len(t), "chars")
