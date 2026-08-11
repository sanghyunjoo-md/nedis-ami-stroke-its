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

