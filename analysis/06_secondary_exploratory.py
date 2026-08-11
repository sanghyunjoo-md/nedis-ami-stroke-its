#!/usr/bin/env python3
"""Run prespecified secondary clinical and exploratory patient-flow models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model_utils import (
    bh_adjust,
    effect_row,
    fit_grouped_binomial_hac,
    fit_nb2_hac,
    fit_ols_hac,
    linear_combination,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis_outputs" / "02_cohort" / "weekly_primary_inputs.csv"
OUT = ROOT / "analysis_outputs" / "05_secondary_exploratory"
DISEASES = ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]


def design(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(d)),
            d["calendar_time"],
            d["post"],
            d["post_time"],
            d["sin1"],
            d["cos1"],
        ]
    ).astype(float)


def fit_outcome(d: pd.DataFrame, kind: str, numerator: str, denominator: str | None):
    x = design(d)
    if kind == "count":
        fit = fit_nb2_hac(d[numerator].to_numpy(float), x, lag=4)
    elif kind == "rate":
        fit = fit_grouped_binomial_hac(
            d[numerator].to_numpy(float), d[denominator].to_numpy(float), x, lag=4
        )
    elif kind == "los":
        y = np.log(d[numerator].to_numpy(float))
        if not np.all(np.isfinite(y)):
            raise RuntimeError("Weekly median LOS contains nonfinite values")
        fit = fit_ols_hac(y, x, lag=4)
    else:
        raise ValueError(kind)
    return x, fit


def add_effects(rows: list[dict], disease: str, domain: str, outcome: str, fit, multiplicity_domain: str | None) -> None:
    for parameter, idx in (("Immediate level change", 2), ("Weekly slope change", 3)):
        rows.append(
            {
                "disease": disease,
                "domain": domain,
                "outcome": outcome,
                "parameter": parameter,
                "effect_scale": "Ratio" if fit.family.startswith("OLS") else ("IRR" if fit.family.startswith("Negative") else "OR"),
                "model_family": fit.family,
                "alpha": fit.alpha,
                "multiplicity_domain": multiplicity_domain,
                **effect_row(fit.params, fit.cov_hac, idx, "exp"),
            }
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_csv(INPUT, parse_dates=["week_start"])
    clinical_rows: list[dict] = []
    clinical_combined: list[dict] = []
    exploratory_rows: list[dict] = []

    clinical_specs = [
        ("Hospital admission", "rate", "admission_n", "known_disposition_n"),
        ("ICU admission", "rate", "icu_main_n", "known_disposition_n"),
        ("Index-hospital mortality", "rate", "index_hospital_death_n", "mortality_denominator_n"),
        ("ED length of stay", "los", "los_median_minutes", None),
    ]
    exploratory_specs = [
        ("Transfer reason", "Capacity-related absolute count", "count", "transfer_capacity_n", None, "transfer_reason"),
        ("Transfer reason", "Capacity-related composition", "rate", "transfer_capacity_n", "transfer_out_n", "transfer_reason"),
        ("Transfer reason", "Specialist-care absolute count", "count", "transfer_specialist_n", None, "transfer_reason"),
        ("Transfer reason", "Specialist-care composition", "rate", "transfer_specialist_n", "transfer_out_n", "transfer_reason"),
        ("Transfer destination", "Tertiary-hospital transfer rate", "rate", "destination_tertiary_n", "known_disposition_n", "transfer_destination"),
        ("Transfer destination", "Tertiary-hospital destination composition", "rate", "destination_tertiary_n", "destination_known_n", "transfer_destination"),
        ("Transfer destination", "General-hospital transfer rate", "rate", "destination_general_n", "known_disposition_n", "transfer_destination"),
        ("Transfer destination", "General-hospital destination composition", "rate", "destination_general_n", "destination_known_n", "transfer_destination"),
        ("KTAS", "KTAS 1-2 weekly count", "count", "ktas_high_n", None, "ktas"),
        ("KTAS", "KTAS 3-5 weekly count", "count", "ktas_low_n", None, "ktas"),
        ("KTAS", "High-acuity share", "rate", "ktas_high_n", "ktas_valid_n", "ktas"),
    ]

    for disease in DISEASES:
        d = weekly.loc[weekly["disease"].eq(disease)].sort_values("week_start").reset_index(drop=True)
        for outcome, kind, numerator, denominator in clinical_specs:
            _, fit = fit_outcome(d, kind, numerator, denominator)
            if not fit.converged:
                raise RuntimeError(f"Clinical model failed: {disease} | {outcome}")
            add_effects(clinical_rows, disease, "Clinical secondary", outcome, fit, None)
            for k in (13, 26, 43):
                contrast = np.zeros(6)
                contrast[2], contrast[3] = 1.0, float(k)
                clinical_combined.append(
                    {
                        "disease": disease,
                        "outcome": outcome,
                        "post_week": k,
                        **linear_combination(fit.params, fit.cov_hac, contrast, "exp"),
                    }
                )

        for domain, outcome, kind, numerator, denominator, mult_domain in exploratory_specs:
            _, fit = fit_outcome(d, kind, numerator, denominator)
            if not fit.converged:
                raise RuntimeError(f"Exploratory model failed: {disease} | {outcome}")
            add_effects(exploratory_rows, disease, domain, outcome, fit, mult_domain)

    clinical = pd.DataFrame(clinical_rows)
    exploratory = pd.DataFrame(exploratory_rows)
    exploratory["q_bh"] = np.nan
    for domain, idx in exploratory.groupby("multiplicity_domain").groups.items():
        exploratory.loc[idx, "q_bh"] = bh_adjust(exploratory.loc[idx, "p_raw"].to_numpy())
    exploratory["bh_q_lt_0_05"] = exploratory["q_bh"] < 0.05

    clinical.to_csv(OUT / "secondary_clinical_estimates.csv", index=False)
    pd.DataFrame(clinical_combined).to_csv(OUT / "secondary_combined_effects.csv", index=False)
    exploratory.to_csv(OUT / "exploratory_patient_flow_estimates.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "clinical_tests": int(len(clinical)),
        "transfer_reason_tests": int((exploratory["multiplicity_domain"] == "transfer_reason").sum()),
        "transfer_destination_tests": int((exploratory["multiplicity_domain"] == "transfer_destination").sum()),
        "ktas_tests": int((exploratory["multiplicity_domain"] == "ktas").sum()),
        "exploratory_bh_q_lt_0_05": int(exploratory["bh_q_lt_0_05"].sum()),
    }
    (OUT / "secondary_exploratory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
