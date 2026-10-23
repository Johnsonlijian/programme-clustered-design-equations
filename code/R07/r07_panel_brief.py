"""R07 WP6 -- build the multi-model panel brief from the merged manuscript."""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

BASE = Path(r"R:\IMUT_WORKPOOL\01_PROJECTS\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-KAG-SmallData")
PDF = BASE / "paper" / "manuscript" / "main_panel.pdf"
TEX = BASE / "paper" / "manuscript" / "main_r07.tex"
OUT = BASE / "rounds" / "R07_merged_protocol_2026-10-22" / "panel_brief.md"

reader = PdfReader(str(PDF))
pages = [(p.extract_text() or "") for p in reader.pages]
full = "\n".join(pages)
print("pages:", len(pages), "chars:", len(full))

# front matter: title + abstract + intro (first ~9k chars)
front = full[:9000]

# conclusions: from the Conclusions heading to Data availability
i_c = full.find("Conclusions")
i_d = full.find("Data availability", i_c if i_c > 0 else 0)
concl = full[i_c:i_d][:4500] if i_c > 0 and i_d > i_c else ""

# captions
caps = re.findall(r"(?:Table|Figure)\s+\d+:[^\n]{0,400}", full)

# structure from the tex
tex = TEX.read_text(encoding="utf-8")
struct = []
for line in tex.splitlines():
    m = re.match(r"\\(section|subsection)\{(.+?)\}", line)
    if m:
        struct.append(("§" if m.group(1) == "section" else "  · ") + m.group(2))

brief = f"""# Panel brief — cold read of a merged manuscript (positive, mechanism-centred; no downgrade)

You are one seat on a four-model review panel. The author asks for a ruthless but constructive
cold read: **how should this paper be written, and where should it be submitted?** Two hard
constraints from the author:

1. **No downgrade.** Do not propose lower-tier, scope-forced or APC-driven venues as the main
   route. If the paper is not ready for a top venue, say what must change, not where to settle.
2. **Not an audit.** The paper must read as a *positive, mechanism-centred* contribution.
   A "data audit" framing (here is what is wrong with these databases) is the failure mode to
   avoid; the paper's own claim is that programme clustering is an *identification instrument*
   and that design-equation terms can be audited, priced and allocated.

## What the paper is

A protocol for using codified design equations as priors in data-driven structural prediction.
Three layers share one object, the misfit B of the rule at a new programme:

- **Audit (validity, no test labels).** Deterministic checks of the rule's declared metadata
  (range, units, sign, exponent) plus within-programme contrasts that grade every asserted
  exponent against the data.
- **Price (value).** A first-order variance budget S* = p c / B converts the residual misfit
  into the number of test programmes at which data-only learning overtakes the anchor.
- **Allocate (PAKA).** Stacking of three trust levels (code imposed / code as prior / data only)
  by whole-programme log score, re-weighted by the first k tests of a new programme.

## Evidence base (for judging claims)

- Four public databases: FRP shear (253 tests / 41 programmes), SFRC shear (353 / 85),
  carbonation (1230 / 48), RC deep beams (322 / 48); 2,158 tests, 222 programmes.
- ICC of the code residual 0.44–0.56; effective independent tests 41–61 (Kish design effect).
- Eleven anchor forms (fifteen domain–rung combinations), pre-registered ladder.
- Metadata audit under four corruptions: sign 4/4 detected, unit 11/11, range 6/6, formula
  (constrained side) 1/1; zero false rejections of the eleven valid rules. A support-domain
  gate detects 0/6 of the in-support corruptions and only 3/11 unit slips.
- Empirical layer: the ACI 318-19 size factor closes the within-programme depth deviation in
  SFRC (+0.06) and deep beams (−0.04) but only halves it in FRP (−0.21, z≈−4); the beam-action
  equation's silence on a/d is contradicted (z≈−21); a −1/2 binding exponent is contradicted
  (z≈−11) while −1 keeps the sign (z≈−5). Two terms are flagged and left as caveats: the SFRC
  fibre factor has the opposite sign within programmes (z≈−6), and the FRP equation's silence
  on a/d is contradicted (z≈−3.7).
- Price: 11 computable crossovers, predicted/observed ratio 0.78–1.65 (pre-registered tolerance
  factor two); three failure regimes identifiable from the budget's inputs.
- Allocation: worst shortfall against the best single arm 0.042 log RMSE; three tests of a new
  programme cut the error by 13–32%. Under a corrupted rule, deploying it directly costs up to
  0.12 log RMSE while stacking already buffers the loss to 0.004–0.008.

## Manuscript facts

- English, Elsevier `elsarticle` review format, 46 pages double-spaced incl. figures/tables,
  plus a 10-page supplement (S1–S8) and a reproducibility archive.
- Section structure:
{chr(10).join('  ' + s for s in struct)}
- Numbered tables: 1 metadata-audit detection, 2 anchor validity dossiers, plus 7 from the
  earlier version (domains, rungs, within/between signatures, crossovers, scarcity, PAKA, LOO).
- Figures: 6 (evidence structure, mechanism signatures, scarcity curves, predicted-vs-observed
  crossovers, unit/extrapolation/circularity, PAKA).

## Your task

Review the manuscript as a candidate *positive mechanism-centred methods paper*. Cover, in this
order, with concrete and actionable judgements:

1. **Title** — does it carry the contribution? Alternatives, if the current one under-sells or
   over-claims.
2. **Abstract** — structure, claim strength, missing numbers, first-sentence impact.
3. **Data** — are four databases enough for the claims? What would a hostile reviewer demand?
4. **Methods** — is the audit + budget + stacking combination coherent, or does it read as three
   papers stapled? Is the identifiability argument sound?
5. **Results and conclusions** — claim-evidence fit; over- or under-claiming; the two caveat
   terms (SFRC fibre factor, FRP a/d); whether the "size factor works in SFRC/deep beams but not
   FRP" nuance is handled honestly.
6. **Figures and tables** — number, order, necessity, captions, and whether each pays for its
   space. Any numbering/ordering problems you can infer from the caption list.
7. **Literature** — what positioning is missing; which literatures must be cited for this to be
   reviewable at a top venue (model uncertainty, hierarchical/mixed models for test databases,
   knowledge-guided ML, code calibration, random-split critiques).
8. **Novelty** — what is genuinely new, what a reviewer will call incremental, and how to frame
   the difference in one paragraph.
9. **Prose** — register, redundancy, defensive hedging, AI-flavoured phrasing, and any place
   where the paper reads like an audit rather than a positive contribution.
10. **Venue** — rank three realistic targets with one-line reasons and the exact reframing each
    would require. Respect the no-downgrade rule.

Finish with: (a) the single biggest threat to acceptance, (b) the three changes with the highest
expected value, (c) a one-sentence verdict on whether this is a strong paper at a top venue.

## Manuscript excerpt (rendered text)

### Front matter and introduction
{front}

### Conclusions
{concl}

### Table and figure captions
{chr(10).join(caps)}
"""

OUT.write_text(brief, encoding="utf-8")
print("brief ->", OUT, len(brief), "chars")
