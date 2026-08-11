# Analysis workflow

## Input and cohort construction

`01_inventory.py` records workbook structure and optional local file checksums without applying filters. `02_ingest_raw.py` imports the `EMIHPTMI` worksheet as strings and writes a local pickle cache. `03_build_cohort.py` removes exact duplicate rows, applies the study dates, age and ED-level criteria, constructs mutually exclusive AMI, ischemic-stroke and hemorrhagic-stroke strata from principal ED discharge diagnoses, derives outcomes, and aggregates complete calendar weeks.

The primary weekly series contains 59 preinterruption weeks and 44 postinterruption weeks. Partial boundary weeks and the transition week beginning 2024-02-19 are excluded from the primary models. The first postinterruption complete week begins 2024-02-26.

## Models

Weekly ED visit counts use negative-binomial NB2 models. Weekly transfer-out rates use grouped-binomial logistic models. Both model families include calendar time, an immediate postinterruption indicator, postinterruption time, and annual sine/cosine seasonality terms. Newey-West HAC covariance uses lag 4 in the primary specification.

The confirmatory family contains 12 effects:

- three disease strata;
- two outcomes (weekly ED visit count and transfer-out rate);
- two interruption parameters (immediate level and weekly slope change).

Holm adjustment is applied across all 12 confirmatory tests.

## Sensitivity and exploratory analyses

`05_sensitivity.py` implements 13 prespecified variants, including an alternative intervention date, transition-week inclusion, holiday exclusion, a second Fourier harmonic, HAC lags 1 and 8, an alternative count-model family, an approximate autoregressive-error model, stable-facility and center-continuity specifications, a disposition-code exclusion, the prespecified conflict rule, and a preperiod placebo interruption.

`06_secondary_exploratory.py` evaluates secondary clinical outcomes and exploratory patient-flow domains. `07_adjusted_and_stratified.py` performs case-mix-adjusted and ED-level analyses, ICU-definition sensitivity analysis, and regional descriptive summaries. These analyses are not part of the 12-test confirmatory family.

## Tables, figures, and validation

`08_tables.py` generates the manuscript-facing tables. `09_figures.py` produces PDF and 300-dpi PNG figures. `10_validate.py` reconciles cohort counts, weekly numerators and denominators, multiplicity adjustments, model score equations, an independent grouped-binomial estimator, sensitivity-matrix dimensions, exploratory-domain dimensions, and figure properties.

Generated files are written to `analysis_outputs/` and remain local.

## Access, severity, and care-pathway extension

The prespecified extension reads the unchanged 189,330-encounter complete-week
cohort created by the parent pipeline. It evaluates symptom-onset-to-arrival
time, impaired consciousness, shock-range systolic blood pressure, hypoxemia,
KTAS escalation, consulting-specialist involvement, condition-concordant
department, and high-acuity final treatment area.

Disease-specific models use the same interruption timing, annual seasonality,
and Newey-West lag-4 covariance framework as the parent analysis. Multiplicity
is controlled by Benjamini-Hochberg adjustment within fixed access, presenting-
severity, and care-pathway domains. Disease-heterogeneity tests form a separate
fixed family. Prespecified variants assess interruption timing, seasonality,
HAC lag, onset-time bounds, disposition-code exclusion, and a stricter stroke-
department definition. Supportive encounter-level care-pathway models add
case-mix covariates and facility fixed effects with two-way clustered covariance
when estimable.

The extension validation independently recomputes multiplicity adjustments and
model coefficients, reconciles all weekly denominators, checks convergence and
confidence intervals, and generates the extension figures. Generated files are
written to `analysis_extension_outputs/` and remain local.
