# KAG-SmallData — When do design equations pay off? Programme-level uncertainty and code calibration in clustered engineering data

Reproducibility package for the manuscript *“When do design equations pay off? Programme-level
uncertainty and code calibration in clustered engineering data”* (Lijian Ren, Inner Mongolia
University of Technology).

The paper asks what a codified design equation is worth for predicting a test programme that is not
yet in the database. It treats each coded term as a testable mechanism hypothesis, grades the
asserted exponents with within-programme contrasts, converts the remaining misfit into the number
of programmes at which data-only learning overtakes the anchor, and allocates trust among
code-imposed, code-as-prior and data-only arms.

Archived version (concept DOI, always resolves to the latest release):
[10.5281/zenodo.23583640](https://doi.org/10.5281/zenodo.23583640)

## What is here

| Path | Contents |
|---|---|
| `code/` | The analysis pipeline, one folder per internal round (R01→R08, R10). Python 3.11+. |
| `derived/` | Small derived tables: ladder theory and signatures, anchor dossiers, stress-test results, downstream comparison, per-term empirical grades, PGLO leakage, tolerance sensitivity, stacking-level comparison and S* calibration; R08 nested/certified/repaired-anchor and tuned-baseline outputs; R10 PAKA prior-strength sensitivity outputs. |
| `figures/` | Generated figure masters (PDF) and previews (PNG). |
| `REPRODUCIBLE_RUNBOOK.md` | Order of execution and what each round produces. |
| `DATASETS_AND_LINKS.csv` | The four public databases, their DOIs and licences. |
| `FORMULA_VERIFICATION_R10.md` | Clause-by-clause check of the five implemented standard/literature equations against public sources. |
| `PAKA_A0_NESTED_RECORD.md` | Record of the fully nested sensitivity analysis for the PAKA prior strength a0. |

**Not included**: the raw third-party databases (download from the links in
`DATASETS_AND_LINKS.csv`; licences are non-commercial for the FRP database and CC BY for the
others), the manuscript source, and the internal round notes.

## Quick check without rerunning

`derived/ladder_summary.csv` and `derived/ladder_theory.csv` contain the numbers behind the
scarcity curves and the predicted/observed crossovers; `derived/anchor_dossiers.csv` and
`derived/empirical_terms.csv` contain the per-anchor validity profile; `derived/stress_results.csv`
and `derived/downstream_summary.csv` contain the metadata-check stress test and the downstream
comparison. The R07 hardening outputs are `derived/pglo.csv`, `derived/stacking_pointwise.csv`,
`derived/tolerance_curve.csv`, `derived/calibration_rungs.csv`, `derived/calibration_summary.csv`
and `derived/tost_verdicts.csv`.
The R08 revision outputs are `derived/nested_wholedb.csv`, `derived/nested_sub.csv`,
`derived/nested_lopo.csv` (nested grading), `derived/certified_*.csv` (pre-specified anchor
ladder), `derived/repair_*.csv` and `derived/prospective_check.csv` (repaired anchors and the
bounded prospective check), and `derived/tuned_*.csv` (tuned ML baselines).
The R10 outputs are `derived/paka_a0_regret_summary.csv`, `derived/paka_a0_kshot_summary.csv`,
`derived/paka_a0_info.csv` and `derived/paka_a0_per_draw.csv.gz` (nested prior-strength
sensitivity); the identity check that ties `a0=3` to the frozen ladder is reproduced by
`code/R10/paka_a0_nested_r10.py check`.
The hardened figure source is `code/R07/r07_figure_hardening.py`; it regenerates `fig0_concept`,
`fig3_scarcity`, `fig4_pred_vs_obs` and `fig_leakage` from the derived tables.

## Licence

Code: MIT (see `LICENSE`). Derived tables and figures: CC BY 4.0. The databases keep their own
licences; see `DATASETS_AND_LINKS.csv`.
