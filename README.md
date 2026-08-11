# NEDIS AMI and stroke interrupted time-series analysis

This repository contains the Python analysis code accompanying the manuscript:

> **Changes in Emergency Department Utilization and Patient Flow for Acute Myocardial Infarction and Stroke Following the 2024 Mass Resignation of Junior Physicians in South Korea: A Nationwide Interrupted Time-Series Study**

The code constructs the prespecified cohort, aggregates complete calendar weeks, fits the interrupted time-series models, performs sensitivity and exploratory analyses, and generates manuscript tables and figures.

## Data availability and repository scope

The customized National Emergency Department Information System (NEDIS) extract is restricted third-party data. It is **not included** in this repository. This release also excludes patient-level caches, machine-readable aggregate analysis inputs, facility-level outputs, signed governance documents, and local checksums of restricted files.

Qualified researchers must obtain an authorized NEDIS extract directly from the data provider. Reproduction of the reported numerical results requires an extract with the same structure and eligibility period. The repository contains only code and documentation; it cannot by itself reproduce the manuscript's numerical results without authorized data.

## Repository layout

```text
analysis/          Ordered analysis scripts and shared model utilities
data/raw/          Local location for the restricted workbook (ignored by Git)
private/           Optional local location for the signed locked SAP (ignored by Git)
docs/              Workflow, variable map, provenance, and release-audit notes
tests/             Data-free tests of core multiplicity utilities
analysis_outputs/  Generated locally and ignored by Git
```

## Software environment

- Python 3.12
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- scikit-learn 1.8.0
- Matplotlib 3.10.8
- openpyxl 3.1.5
- Pillow 12.2.0

Install the pinned environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Local input configuration

Keep restricted inputs outside version control. Either place the authorized workbook at:

```text
data/raw/NEDIS_custom_extract.xlsx
```

or set an environment variable to its absolute path:

```bash
export NEDIS_RAW_XLSX=/secure/path/NEDIS_custom_extract.xlsx
```

The signed locked SAP is optional for integrity verification. Place it at `private/SAP_signed_locked.docx` or set `NEDIS_LOCKED_SAP`. Authorized researchers may also set `NEDIS_EXPECTED_RAW_SHA256` and `NEDIS_EXPECTED_SAP_SHA256` locally. Do not commit restricted-file paths or checksums unless the data provider permits this.

## Run the analysis

Run the complete pipeline:

```bash
python analysis/run_all.py
```

The ordered scripts are:

1. `01_inventory.py` — non-analytic input inventory
2. `02_ingest_raw.py` — loss-minimizing string import and local cache
3. `03_build_cohort.py` — eligibility, disease strata, outcomes, and weekly inputs
4. `04_primary_its.py` — six primary models and 12 coprimary effects
5. `05_sensitivity.py` — 13 prespecified sensitivity specifications
6. `06_secondary_exploratory.py` — secondary and exploratory outcomes
7. `07_adjusted_and_stratified.py` — adjusted and stratified sensitivity analyses
8. `08_tables.py` — manuscript-facing tables
9. `09_figures.py` — vector and 300-dpi figures
10. `10_validate.py` — reconciliation and independent numerical checks

Outputs are written to `analysis_outputs/`, which is intentionally ignored by Git.

## Reproducibility status

The complete pipeline was re-executed from the restricted source extract on 2026-08-11 using the pinned environment. All manuscript-facing CSV outputs and PNG figures matched the independently generated clean rerun byte for byte. See [docs/public_release_audit.md](docs/public_release_audit.md).

## Interpretation

This is an uncontrolled interrupted time-series study of temporal changes. The code and manuscript use restrained association language; they do not establish that the mass resignation caused the observed changes. Prespecified placebo and residual-diagnostic findings should be considered when interpreting the estimates.

## Citation

Citation metadata are provided in `CITATION.cff`. After the GitHub release is archived in Zenodo, cite the version-specific Zenodo DOI.

## License

The analysis code is released under the MIT License. The license does not grant rights to NEDIS data, documentation, or other third-party materials.

## AI assistance disclosure

Generative AI tools assisted with statistical code drafting, computational validation, repository documentation, and language editing under investigator supervision. The investigators specified the analysis plan, reviewed the code and outputs, retained responsibility for analytic decisions and interpretation, and verified the reported results.
