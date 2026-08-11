# Secondary and exploratory analysis extension

This directory contains the prespecified extension code for the manuscript's
access, presenting-severity, and care-pathway analyses. It reads the unchanged
parent complete-week cohort produced by `analysis/03_build_cohort.py` and writes
all generated files to `analysis_extension_outputs/`, which is ignored by Git.

The public release includes the machine-readable analysis definitions and the
statistical code. It excludes the restricted cohort, all patient- and
week-level derived inputs, results, figures, the signed analysis addendum, and
checksums of restricted files.

Run the extension after the parent pipeline:

```bash
python analysis_extension/run_all.py
```

The scripts perform the following tasks:

1. derive access, severity, and care-pathway variables;
2. fit disease-specific and disease-heterogeneity models;
3. run the prespecified sensitivity analyses;
4. fit supportive encounter-level case-mix models;
5. independently validate the estimates and completeness series;
6. generate the severity and manuscript Figure 5 care-pathway forest plots; and
7. prepare non-patient-level flat tables used for the results workbook.

The default parent-cohort path is
`analysis_outputs/02_cohort/cohort_primary_weeks.pkl`. Authorized researchers
may instead set `NEDIS_EXTENSION_COHORT_PKL` to an absolute or repository-
relative path.

Local integrity verification is optional. Set the following environment
variables without committing their values:

- `NEDIS_EXPECTED_EXTENSION_COHORT_SHA256`
- `NEDIS_EXTENSION_ADDENDUM`
- `NEDIS_EXPECTED_EXTENSION_ADDENDUM_SHA256`

The final formatted submission workbook is not stored in this repository. Its
underlying non-patient-level tables and workbook payload are generated locally
by `07_prepare_workbook_data.py`.
