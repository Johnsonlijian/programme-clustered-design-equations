# Programme-clustered evidence and the value of codified design equations

Reproducibility package for the manuscript *“Design equations as testable mechanism hypotheses:
programme-level evidence, pricing, and trust allocation”* (Lijian Ren, Inner Mongolia University of
Technology).

The paper asks what a codified design equation is worth for predicting a test programme that is not
yet in the database. It treats each coded term as a testable mechanism hypothesis, grades the
asserted exponents with within-programme contrasts, converts the remaining misfit into the number
of programmes at which data-only learning overtakes the anchor, and allocates trust among
code-imposed, code-as-prior and data-only arms.

Archived version: [10.5281/zenodo.23583641](https://doi.org/10.5281/zenodo.23583641)

## What is here

| Path | Contents |
|---|---|
| `code/` | The analysis pipeline, one folder per internal round (R01→R07). Python 3.11+. |
| `derived/` | Small derived tables: ladder theory and signatures, anchor dossiers, stress-test results, downstream comparison, per-term empirical grades. |
| `figures/` | Generated figure masters (PDF) and previews (PNG). |
| `REPRODUCIBLE_RUNBOOK.md` | Order of execution and what each round produces. |
| `DATASETS_AND_LINKS.csv` | The four public databases, their DOIs and licences. |

**Not included**: the raw third-party databases (download from the links in
`DATASETS_AND_LINKS.csv`; licences are non-commercial for the FRP database and CC BY for the
others), the manuscript source, and the internal round notes.

## Quick check without rerunning

`derived/ladder_summary.csv` and `derived/ladder_theory.csv` contain the numbers behind the
scarcity curves and the predicted/observed crossovers; `derived/anchor_dossiers.csv` and
`derived/empirical_terms.csv` contain the per-anchor validity profile; `derived/stress_results.csv`
and `derived/downstream_summary.csv` contain the metadata-check stress test and the downstream
comparison.

## Licence

Code: MIT (see `LICENSE`). Derived tables and figures: CC BY 4.0. The databases keep their own
licences; see `DATASETS_AND_LINKS.csv`.
