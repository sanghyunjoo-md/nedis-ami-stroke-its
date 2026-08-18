# Public-release audit

## Source verification

The supplied clean-rerun archive passed ZIP integrity testing. The restricted NEDIS workbook and signed locked SAP used for local verification matched the checksums recorded in the original clean-rerun validation report. Those checksums are intentionally omitted from this public release.

## Independent rerun

On 2026-08-11, scripts `01_inventory.py` through `10_validate.py` were re-executed from the restricted source workbook in a clean analysis directory using Python 3.12 and the pinned package versions.

The rerun reproduced:

- 309,924 source rows and four exact duplicate rows;
- a target cohort of 191,667 encounters;
- 189,330 encounters in 59 preinterruption and 44 postinterruption complete weeks;
- all 12 coprimary estimates and Holm-adjusted P values;
- all 156 rows in the 13-specification sensitivity matrix;
- all manuscript-facing CSV tables;
- all manuscript-facing PNG figures.

Every compared CSV and PNG matched the original independently generated clean rerun byte for byte. PDF byte hashes differed because regenerated PDFs contain creation metadata; corresponding PNG rendering and plotted values were identical.

## Preserved diagnostic cautions

The rerun reproduced the original warnings:

- five of 12 preperiod placebo tests had nominal `P < 0.05`;
- hemorrhagic-stroke ED visit-count residuals showed lag-12 autocorrelation;
- case-mix-adjusted transfer models were rank deficient and numerically ill-conditioned.

These findings support restrained temporal-association language and retention of the adjusted models as sensitivity analyses rather than confirmatory evidence.

## Public-release changes

The statistical formulas, cohort logic, prespecified dates, outcomes, model families, sensitivity specifications, and table/figure generation logic were not changed. Public-release edits were limited to:

- replacing local input filenames with configurable private paths;
- making SAP and checksum validation optional and local;
- adding a pipeline runner, documentation, citation metadata, license, tests, and ignore rules;
- excluding all restricted or potentially redistributable inputs and outputs.

## v1.1.0 extension verification

The prespecified analysis extension was independently re-executed from the
unchanged parent complete-week cohort. It reconciled 189,330 encounters across
59 preinterruption and 44 postinterruption weeks. All 24 disease-specific
central models, eight disease-heterogeneity models, and 110 sensitivity models
converged. The validation suite reproduced 120 central estimate rows, 16
heterogeneity tests, 48 supportive pairwise contrasts, 220 sensitivity estimate
rows, and 12 supportive case-mix models; 10 case-mix models were estimable and
two retained their prespecified non-estimable status with diagnostic reasons.

The extension results workbook in the submission package matched the verified
clean-run workbook byte for byte, and Appendix E1 Figure E6 matched the
verified submission figure byte for byte. The public release contains only the
statistical and figure-generation code plus the machine-readable analysis
definitions. It excludes the signed extension addendum, local lock manifest,
restricted cohort, derived inputs, all result files, and restricted-file
checksums. Public-release edits to the extension were limited to configurable
private paths, optional local integrity checks, runners, documentation, and
ignore rules; the derivation and statistical model logic was not changed.

## v1.1.1 Stage 3-D cross-audit

On 2026-08-18, the public repository at v1.1.0 was cross-audited against the
Stage 3-D v1.3 manuscript sources and the final 41-sheet aggregate results
workbook. The audit confirmed the locked cohort flow (309,924 source rows,
four exact duplicates, 191,667 eligible encounters, and 189,330 complete-week
encounters across 59 preinterruption and 44 postinterruption weeks), all 12
coprimary tests with three Holm-adjusted signals, and all 24 extension
care-pathway tests with 14 Benjamini-Hochberg-adjusted signals.

The repository definitions for the interruption date, complete-week windows,
outcomes, negative-binomial and grouped-binomial model families, annual
seasonality, four-lag HAC covariance, and fixed multiplicity families matched
the submission sources. A public-exposure scan of all 39 repository files
found no patient-level or facility-level data, aggregate result files,
restricted-file checksums, signed governance documents, or local analysis
outputs.

Release v1.1.1 changes only citation metadata, the final manuscript title,
journal-neutral comments, release documentation, and the supplemental figure
mapping. It changes no analytic logic, configuration, model specification,
multiplicity procedure, numerical result, restricted-data boundary, or
historical lock identifier.
