#!/usr/bin/env python3
"""Prepare flat, non-patient-level tables for the extension results workbook."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_extension_outputs"
DEST = OUT / "08_workbook_inputs"


def write_json_rows(source: Path, destination: Path, key_column: str) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for key, values in payload.items():
        row = {key_column: key}
        for name, value in values.items():
            if isinstance(value, (list, dict)):
                row[name] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                row[name] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(destination, index=False)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    central = pd.read_csv(OUT / "02_models" / "central_extension_estimates.csv")
    central.loc[central["q_signal_0_05"].eq(True)].to_csv(
        DEST / "central_q_signals.csv", index=False
    )

    heterogeneity = pd.read_csv(OUT / "02_models" / "disease_heterogeneity_tests.csv")
    heterogeneity.loc[heterogeneity["q_signal_0_05"].eq(True)].to_csv(
        DEST / "heterogeneity_q_signals.csv", index=False
    )

    write_json_rows(
        OUT / "02_models" / "central_extension_diagnostics.json",
        DEST / "central_diagnostics.csv",
        "model_key",
    )
    write_json_rows(
        OUT / "04_case_mix_adjusted" / "case_mix_adjusted_diagnostics.json",
        DEST / "case_mix_diagnostics.csv",
        "model_key",
    )

    validation = json.loads(
        (OUT / "05_validation" / "validation_report.json").read_text(encoding="utf-8")
    )
    check_rows = []
    for item in validation["checks"]:
        check_rows.append(
            {
                "check": item["name"],
                "status": "PASS" if bool(item["passed"]) else "FAIL",
                "detail": item.get("detail", ""),
            }
        )
    pd.DataFrame(check_rows).to_csv(DEST / "validation_checks.csv", index=False)
    pd.DataFrame(
        {"warning": validation.get("warnings", [])}
    ).to_csv(DEST / "validation_warnings.csv", index=False)
    combined = [
        {
            "section": "Validation check",
            "item": row["check"],
            "status": row["status"],
            "detail": row["detail"],
        }
        for row in check_rows
    ]
    combined.extend(
        {
            "section": "Diagnostic warning",
            "item": warning,
            "status": "REVIEW",
            "detail": "Retained and reported; not a validation failure",
        }
        for warning in validation.get("warnings", [])
    )
    pd.DataFrame(combined).to_csv(DEST / "validation_combined.csv", index=False)

    inventory = pd.DataFrame(
        [
            ["Central estimates", len(central)],
            ["Central q<0.05 signals", int(central["q_signal_0_05"].eq(True).sum())],
            ["Heterogeneity tests", len(heterogeneity)],
            ["Heterogeneity q<0.05 signals", int(heterogeneity["q_signal_0_05"].eq(True).sum())],
            ["Sensitivity estimates", len(pd.read_csv(OUT / "03_sensitivity" / "extension_sensitivity_estimates.csv"))],
            ["Case-mix estimate rows", len(pd.read_csv(OUT / "04_case_mix_adjusted" / "case_mix_adjusted_estimates.csv"))],
        ],
        columns=["component", "rows_or_signals"],
    )
    inventory.to_csv(DEST / "inventory.csv", index=False)

    workbook_sources = {
        "Central Signals": DEST / "central_q_signals.csv",
        "Central Estimates": OUT / "02_models" / "central_extension_estimates.csv",
        "Heterogeneity": OUT / "02_models" / "disease_heterogeneity_tests.csv",
        "Pairwise Contrasts": OUT / "02_models" / "disease_pairwise_contrasts.csv",
        "Sensitivity": OUT / "03_sensitivity" / "extension_sensitivity_estimates.csv",
        "Sensitivity Diagnostics": OUT / "03_sensitivity" / "extension_sensitivity_diagnostics.csv",
        "CaseMix Adjusted": OUT / "04_case_mix_adjusted" / "case_mix_adjusted_estimates.csv",
        "PrePost Distribution": OUT / "01_derived" / "prepost_outcome_distributions.csv",
        "Weekly Inputs": OUT / "01_derived" / "weekly_extension_inputs.csv",
        "Source Categories": OUT / "01_derived" / "source_category_distributions.csv",
        "Vitals Descriptive": OUT / "01_derived" / "initial_vital_signs_descriptive.csv",
        "Central Diagnostics": DEST / "central_diagnostics.csv",
        "CaseMix Diagnostics": DEST / "case_mix_diagnostics.csv",
        "Validation": DEST / "validation_combined.csv",
    }
    workbook_payload = {}
    for sheet_name, source in workbook_sources.items():
        frame = pd.read_csv(source)
        for column in frame.select_dtypes(include=["bool"]).columns:
            frame[column] = frame[column].astype(int)
        clean = frame.astype(object).where(pd.notna(frame), None)
        workbook_payload[sheet_name] = {
            "headers": list(clean.columns),
            "rows": clean.values.tolist(),
        }
    (DEST / "workbook_data.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
