# Reproducible runbook

The pipeline was developed in internal rounds. Each round has its own `scripts/` folder; scripts
import modules from neighbouring round folders and read/write relative to the project root, so the
cleanest way to rerun them is to restore the original layout:

```
<project>/
  rounds/
    R01_.../scripts/  ...
    R02_.../scripts/  ...
    ...
    R07_.../scripts/  ...
  paper/manuscript/
```

Copy the folders in `code/` back to `rounds/<round>/scripts/`, download the four databases into the
paths the `kag_data_r05` loader expects (see its header), then run:

1. `R01` — data ingestion and the first two databases; produces the per-domain frames.
2. `R02` — theory sketch (superseded; kept for provenance).
3. `R04` — carbonation database admission.
4. `R05` — hierarchical models, arms, PAKA, the first ladder.
5. `R06` — mechanism pivot: `r06_rungs.py` (anchor definitions), `r06_ladder.py` (library-size
   ladder), `r06_numbers.py` (manuscript numbers), `r06_figures.py` (figures),
   `r06_appendix.py` (supplement tables).
6. `R07` — merged protocol: `r07_audit.py` (metadata checks), `r07_stress.py` (corruption stress
   test), `r07_empirical.py` (within-programme grades and dossiers), `r07_downstream.py`
   (corrupted-deployment comparison), `r07_numbers.py` / `r07_tables.py` (core manuscript
   fragments). The hardening pass then runs `r07_exp23.py` (PGLO leakage and pointwise versus
   programme-level stacking), `r07_exp14.py` (tolerance curve and S* calibration),
   `r07_tost.py` (equivalence-test verdicts), and `r07_exp_numbers.py` (R07 table/number
   fragments). These scripts write `pglo.csv`, `stacking_pointwise.csv`,
   `tolerance_curve.csv`, `calibration_rungs.csv`, `calibration_summary.csv` and
   `tost_verdicts.csv` under `derived/` in the public package.

Every number in the manuscript comes from `generated/numbers*.tex`, written by the `*_numbers.py`
scripts; nothing is typed by hand. The derived tables in `derived/` are the small outputs of those
steps.

Environment: Python 3.11, numpy, pandas, scipy, matplotlib, joblib, pypdf. Compile the manuscript
with pdflatex + bibtex. The public package intentionally excludes the active manuscript, raw
third-party databases and internal round logs; use the DOI/links in `DATASETS_AND_LINKS.csv` and
respect each dataset's licence when restoring inputs.
