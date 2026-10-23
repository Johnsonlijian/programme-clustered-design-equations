"""R07 §4 reorder: price before allocate, aligned with the three-layer narrative."""
from pathlib import Path

P = Path(r"R:\IMUT_WORKPOOL\01_PROJECTS\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-KAG-SmallData\paper\manuscript\main_r07.tex")
t = P.read_text(encoding="utf-8")

BUD = r"\subsection{When does the code stop paying?}\label{sec:theory}"
PAKA = r"\subsection{Programme-aware knowledge anchoring}"
S5 = r"\section{Data and evaluation protocol}\label{sec:data}"

# 1 -- cut the budget block (it sits between PAKA and section 5)
i0 = t.index(BUD)
i1 = t.index(S5)
budget = t[i0:i1]
assert budget.count(r"\subsection{") == 1, "budget block must hold exactly one subsection"
t = t[:i0] + t[i1:]

# 2 -- re-insert it before PAKA (price before allocate)
j = t.index(PAKA)
t = t[:j] + budget + t[j:]

# 3 -- section title and the bridging paragraph
OLD_SEC = r"\section{Knowledge-anchored learning across library sizes}\label{sec:method}"
NEW_SEC = (r"\section{Pricing an anchor and allocating trust}\label{sec:method}" + "\n\n"
           r"The audit of Section~\ref{sec:audit} decides whether a rule may be used. Two "
           r"questions remain for a rule that passes: what is it worth, and how should the "
           r"arms that use it be combined? Section~\ref{sec:arms} defines the arms, "
           r"Section~\ref{sec:theory} turns the anchor's misfit into a price in programme "
           r"units, and Section~\ref{sec:paka} allocates trust among the arms with the first "
           r"tests of a new programme.")
assert t.count(OLD_SEC) == 1
t = t.replace(OLD_SEC, NEW_SEC)

# 4 -- subsection retitled, PAKA gets a label
assert t.count(BUD) == 1
t = t.replace(BUD, r"\subsection{The price of an anchor}\label{sec:theory}")
assert t.count(PAKA + "\n") == 1
t = t.replace(PAKA + "\n", PAKA + r"\label{sec:paka}" + "\n")

# 5 -- verify the new order
i_arms = t.index(r"\subsection{Three ways to use an anchor}")
i_price = t.index(r"\subsection{The price of an anchor}")
i_paka = t.index(r"\subsection{Programme-aware knowledge anchoring}")
i_s5 = t.index(S5)
assert i_arms < i_price < i_paka < i_s5, (i_arms, i_price, i_paka, i_s5)
print("order ok: arms < price < paka < section5")

P.write_text(t, encoding="utf-8")
print("written", len(t), "chars")
