"""R09 label-normalised table fragments from existing generated bodies."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE.parents[2] / "paper" / "manuscript" / "generated"

MAP = {"Code as published": "code as published",
       "Code exponents imposed": "code imposed",
       "Code as REML prior": "code as prior",
       "Data-only hierarchical": "data only",
       "1-SE trust selection": "one-SE selection",
       r"PAKA (proposed)": r"\PAKA{}",
       "Gradient boosting": "boosting",
       "Boosting on code residual": "boosting on code residual",
       "Mixed-effects boosting": "mixed-effects boosting",
       "GP on code residual": "GP on code residual"}

src = (GEN / "tab_lopo.tex").read_text(encoding="utf-8")
for a, b in MAP.items():
    src = src.replace(a, b)
(GEN / "tab_lopo_r08.tex").write_text(src, encoding="utf-8")

rows = []
for line in (GEN / "tab_crossover.tex").read_text(encoding="utf-8").splitlines():
    if " & " in line and "$\\infty$" in line.split(" & ")[1] or \
       (" & " in line and "$\\infty$" in line.split(" & ")[2]):
        cells = line.split(" & ")
        cells[4] = "-- "
        line = " & ".join(cells)
    rows.append(line)
(GEN / "tab_crossover_r08.tex").write_text("\n".join(rows), encoding="utf-8")

d = (GEN / "tab_domains.tex").read_text(encoding="utf-8")
d = d.replace("diffusion-binding sqrt(C/a)",
              r"diffusion--binding $\sqrt{C/a}$")
d = d.replace("sqrt(C)", r"$\sqrt{C}$")
d = d.replace("Narayanan-Darwish 1987", "Narayanan--Darwish 1987")
(GEN / "tab_domains_r08.tex").write_text(d, encoding="utf-8")
print("written: tab_lopo_r08, tab_crossover_r08, tab_domains_r08")
print("\n".join(r for r in rows if "--" in r and "Carbonation" in r))
