"""One-off: build main_r07.tex from the R06 copy (structural edits, robust anchors)."""
from pathlib import Path

P = Path(r"R:\IMUT_WORKPOOL\01_PROJECTS\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-KAG-SmallData\paper\manuscript\main_r07.tex")
t = P.read_text(encoding="utf-8")


def sub(old, new, label, count_expect=1):
    global t
    n = t.count(old)
    assert n == count_expect, f"{label}: anchor found {n} times (expected {count_expect})"
    t = t.replace(old, new)
    print(f"ok: {label}")


# 1 -- title: brace matching (the file breaks the title over two lines)
i = t.index(r"\title{")
j = t.index("}", i)
t = t[:i] + (r"\title{Audit, price, allocate: usability and value of codified "
             r"design equations for unseen test programmes}") + t[j + 1:]
print("ok: title")

# 2 -- abstract: marker based
A0 = t.index(r"\begin{abstract}") + len(r"\begin{abstract}")
A1 = t.index(r"\end{abstract}")
abstract = r"""

Design equations and the machine-learning models proposed to replace them are calibrated on databases that pool tests from many experimental programmes. We treat each term of a design equation as a prior and ask three questions about it: may the rule be used on this database, what is it worth for a programme that is not yet in the database, and how should the two be combined? Our answer is a protocol whose layers share one object, the misfit of the rule at a new programme. First, a validity audit: deterministic checks of the rule's own metadata---declared range, units, sign and exponent of each coded term---with no test labels, followed by within-programme contrasts that grade every asserted exponent against the data. On four public databases (shear of FRP-reinforced, steel-fibre-reinforced and deep RC beams; carbonation; {\nAll} tests in {\progAll} programmes) the metadata audit catches every sign, unit and range corruption of the deployed rule while a support-domain gate catches none of the in-support corruptions, and the contrasts reject the beam-action equation's silence on slenderness and a $-1/2$ binding exponent while confirming the ACI 318-19 size factor and the square-root CO$_2$ law. Second, a price: a first-order variance budget, $S^\ast=pc/B$, converts the remaining misfit into the number of test programmes at which data-only learning overtakes the anchor, within a factor of two for every rung where the prediction is computable. Third, an allocation: programme-aware knowledge anchoring stacks the three trust levels of an anchor and re-weights them with the first tests of a new programme, staying within {\regmaxpakaL} RMSE of the best single arm, with three tests cutting its error by {\kredPakaMin}--{\kredPakaMax}\%. The audit decides usability, the budget prices the residual, and the stack collects the price; all three are estimable before new tests exist.
"""
t = t[:A0] + abstract + t[A1:]
print("ok: abstract")

# 3 -- numbers_r07 input
sub(r"\input{generated/numbers_r06.tex}",
    "\\input{generated/numbers_r06.tex}\n\\input{generated/numbers_r07.tex}",
    "numbers_r07 input")

# 4 -- intro: two minimal edits
sub("Second, we use within-programme contrasts,",
    r"Second, we audit a rule before pricing it: deterministic checks of its declared metadata (range, units, sign and exponent), which need no test labels, and within-programme contrasts,",
    "intro prefix")
sub("; the signature audit confirms some terms",
    "; the audit confirms some terms", "intro suffix")

# 5 -- keywords
sub(r"\sep shear strength \sep carbonation",
    r"\sep shear strength \sep carbonation \sep prior validity", "keywords")

# 6 -- new audit section before the methods section
SECTION = r"""
\section{Auditing usability without labels}\label{sec:audit}

A codified term can be priced only if the rule that is applied is the rule that was declared. We separate two audits: a deterministic check of the rule's own metadata, which uses no data labels, and the within-programme check of Section~\ref{sec:res-physics}, which asks whether the asserted exponents survive the data. Both grade the rule term by term; neither returns a truth value for the equation as a whole.

\subsection{The metadata audit}\label{sec:audit-meta}
Five checks are read off the rule itself: the declared range of each coded variable (for example $a/d\le2.5$, or a requirement of reactive CaO), the physical plausibility of its units (a tenfold slip leaves the plausible interval), the positivity of the implemented expression, the sign of each asserted exponent, and its value where the term is a power law; equation-form terms are checked by sign only, because their constants absorb scale. Every threshold comes from the rule metadata, never from the deployment data, and the audit is a batch decision: a rule applied to one database is rejected if any check fires.

The checks are calibrated on a stress test with four corruptions of the deployed rule: a sign flip (the CaO exponent, or $\lambda_s$ multiplied instead of divided), a unit slip (depth in centimetres, CO$_2$ in volume fraction), a range drift (a CaO-anchored rule applied to rows without reactive CaO, a deep-beam anchor applied beyond $a/d=2.5$), and a formula swap (the K1 and K3 exponent sets exchanged). On the four databases and eleven anchor forms (Table~\ref{tab:auditdetect}) every sign, unit and range corruption is caught, and none of the eleven valid rules is rejected. The comparison arm is a support-domain gate of the kind used for applicability: a training-only density estimate on the coded features. It fires on none of the in-support corruptions---the rule is wrong while its inputs remain inside the training support---and on only $\aSupportUnit$ of the $\aSupportUnitN$ unit slips, because the depth distribution of a laboratory database is wide enough to absorb a tenfold slip within two standard deviations. The metadata audit is deterministic and has no such weakness.

The audit cannot check what the rule does not assert. Free exponents carry no declared value, so a formula swap on the free side is invisible to it; a unit slip of a few per cent stays inside the plausible interval; and where $\lambda_s$ is capped at one ($d\le250$~mm) multiplying and dividing by it are the same operation. These limits are why a second audit is needed.

\subsection{The empirical audit and the anchor dossier}\label{sec:audit-emp}
Within-programme contrasts give, for every asserted exponent, the deviation of the data-implied exponent from the codified one and its standard error (Section~\ref{sec:res-physics}, Table~\ref{tab:within}). Grading the deviation against its standard error turns the signature analysis into a validity profile (Table~\ref{tab:dossiers}): a term is \emph{consistent} ($|z|\le2$), \emph{attenuated} (same sign, larger deviation) or \emph{contradicted} (opposite sign, or asserted absent while the data show a dependence). The profile is read together with the price: the same table lists the predicted crossover and \PAKA{}'s regret, so a term that the data reject sits next to what the anchor is nevertheless worth.

Three readings organise the ladder. First, $\lambda_s$ carries the depth signature where it is coded: under the anchors that already contain it the residual within-programme deviation is {\nSfrcKiidWd} in SFRC and {\nDeepArchsdWd} in the deep beams, both within two standard errors, and {\nFrpKiidWd} in FRP, about half of the deviation of the plain anchor. Second, the beam-action equation's silence on slenderness is contradicted ($z\approx-\aZBeamAd$), while the arch anchors keep the sign of the dependence and only understate its magnitude. Third, the binding-capacity exponent $-1/2$ is contradicted ($z\approx-\aZCaoKii$), whereas the buffer-capacity exponent $-1$ keeps the sign with a smaller deviation ($z\approx-\aZCaoKiii$), which is why the buffer form prices higher.

Two terms that the earlier analysis did not examine appear in the profile and belong in the limitations: the fibre factor of the SFRC equation has the wrong sign within programmes ($z\approx-\aZFsrcF$), so the anchor's fibre term is not identified in these data even though its depth term is; and the FRP equation's silence on $a/d$ is contradicted by a within-programme exponent near $-0.3$ ($z\approx-\aZFrpAd$), weaker than the deep-beam case by an order of magnitude but of the same kind.

\subsection{What the audit buys}\label{sec:audit-downstream}
The audit decides usability; it does not promise better predictions. On valid rules it changes nothing: the clean re-run reproduces the ladder to ${\aHfourDiff}\times10^{-3}$ in log RMSE. On corrupted deployments it pays in proportion to what the pipeline would otherwise do with the rule (Table~\ref{tab:downstream}): a pipeline that deploys the corrupted rule directly loses up to $\aDownKaMax$ in log RMSE, and even when it stacks the arms, \PAKA{}'s weights shift towards the data-only arm and the loss is $\aDownPakaMin$--$\aDownPakaMax$. The audit chain---reject the rule, fall back to data-only---captures that difference. Stacking and auditing therefore provide overlapping but distinct protection, and the two are complements: the audit says whether the rule may be used, the stack says how much to lean on it.

\begin{table}[t]\centering\footnotesize
\caption{Metadata audit under four corruptions of the deployed rule, with the support-domain gate as comparison arm. A case is one rule applied to one database; it is detected if any check fires (audit) or if the deployment inputs leave the training support (gate). \emph{Checks} lists the audit signal that fired.}\label{tab:auditdetect}
\resizebox{\textwidth}{!}{\begin{tabular}{llcccc}\toprule
Corruption & Group & Cases & Audit & Support gate & Checks\\\midrule
\tabinput{generated/tab_audit_detect.tex}
\bottomrule\end{tabular}}
\end{table}

\begin{table}[t]\centering\footnotesize
\caption{Validity profile of the eleven anchor forms read with the price. \emph{Terms} counts the asserted exponents graded consistent / attenuated / contradicted by the within-programme contrasts ($|z|\le2$ for consistent; a contradiction is a sign reversal or an assertion of absence that the data contradict); \emph{worst} gives the most deviant term and its $z$; $S^\ast$ is the predicted crossover (unbounded where the misfit is at the noise floor) and \emph{regret} is \PAKA{}'s worst shortfall against the best single arm over the library sizes.}\label{tab:dossiers}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccc}\toprule
Anchor & Terms (ok/att/con) & Worst term ($z$) & $S^\ast$ & Regret\\\midrule
\tabinput{generated/tab_dossiers.tex}
\bottomrule\end{tabular}}
\end{table}

\begin{table}[t]\centering\footnotesize
\caption{Downstream cost of an in-support corrupted rule: LOPO log RMSE at the full library, $k=0$, mean over target programmes with at least six tests. \emph{Clean \PAKA} is the valid-rule reference; \emph{corrupt KA-FIXED} deploys the corrupted rule directly; \emph{corrupt \PAKA} stacks it; \emph{data-only} is the audit chain's fallback.}\label{tab:downstream}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccc}\toprule
Domain, rung & Clean \PAKA{} & Corrupt KA-FIXED & Corrupt \PAKA{} & Data-only\\\midrule
\tabinput{generated/tab_downstream.tex}
\bottomrule\end{tabular}}
\end{table}

"""
sub(r"\section{Knowledge-anchored learning across library sizes}\label{sec:method}",
    SECTION + "\\section{Knowledge-anchored learning across library sizes}\\label{sec:method}",
    "audit section")

# 7 -- conclusions: insert one item before the signature item
sub("programmes, not tests, are the unit of evidence.\n\\item Contrasts within programmes",
    r"""programmes, not tests, are the unit of evidence.
\item A rule can be audited before it is priced. Deterministic checks of its declared metadata detect sign, unit and range corruptions of a deployed equation with no test labels, where a support-domain gate is blind to every corruption that leaves the inputs inside the training support; within-programme contrasts then grade every asserted exponent, rejecting the beam equation's silence on slenderness ($z\approx-\aZBeamAd$) and the $-1/2$ binding exponent ($z\approx-\aZCaoKii$) and confirming the size factor where it is coded.
\item Contrasts within programmes""", "conclusion item")

# 8 -- limitations: append two sentences
sub("cannot be assessed with these data.",
    r"""cannot be assessed with these data. Two terms that the earlier analysis did not examine appear in the validity profile: the SFRC equation's fibre factor has the wrong sign within programmes ($z\approx-\aZFsrcF$), so its fibre term is not identified in these data, and the FRP equation's silence on $a/d$ is contradicted by a within-programme exponent near $-0.3$ ($z\approx-\aZFrpAd$). The metadata audit checks only what a rule declares: free exponents, unit slips of a few per cent and the capped region of $\lambda_s$ are outside its reach, and its downstream value is decision-level---a pipeline that stacks the arms already buffers part of a corrupted rule ($\aDownPakaMin$--$\aDownPakaMax$ in log RMSE, against $\aDownKaMax$ when the rule is deployed directly).""",
    "limitations")

P.write_text(t, encoding="utf-8")
print(f"written: {P} ({len(t)} chars)")
