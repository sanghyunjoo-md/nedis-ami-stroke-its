#!/usr/bin/env python3
"""Independent validation and prespecified completeness visualization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import (
    ADDENDUM_PATH,
    CONFIG_PATH,
    COHORT_PATH,
    DISEASES,
    EXPECTED_ADDENDUM_SHA256,
    EXPECTED_COHORT_SHA256,
    OUT,
    design,
    read_json,
    sha256,
    write_json,
)
from model_utils import bh_adjust


VALID_OUT = OUT / "05_validation"
FIG_OUT = OUT / "06_figures"


def main() -> None:
    VALID_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    warnings: list[str] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    check(
        "Machine-readable configuration present",
        CONFIG_PATH.exists(),
        sha256(CONFIG_PATH) if CONFIG_PATH.exists() else str(CONFIG_PATH),
    )
    if EXPECTED_COHORT_SHA256:
        cohort_digest = sha256(COHORT_PATH) if COHORT_PATH.exists() else "missing"
        check(
            "Optional parent clean cohort checksum",
            cohort_digest == EXPECTED_COHORT_SHA256,
            cohort_digest,
        )
    if EXPECTED_ADDENDUM_SHA256:
        addendum_digest = sha256(ADDENDUM_PATH) if ADDENDUM_PATH.exists() else "missing"
        check(
            "Optional signed Addendum checksum",
            addendum_digest == EXPECTED_ADDENDUM_SHA256,
            addendum_digest,
        )

    derivation = read_json(OUT / "01_derived" / "derivation_qc.json")
    check("Encounter derivation reconciliation", derivation["input_rows"] == derivation["derived_rows"] == 189330, f"{derivation['input_rows']}/{derivation['derived_rows']}")
    check("Weekly skeleton size", derivation["weekly_rows"] == 309, str(derivation["weekly_rows"]))
    check("No zero weekly denominators", derivation["onset_zero_valid_weeks"] == 0 and all(value == 0 for value in derivation["zero_denominator_weeks"].values()), json.dumps(derivation["zero_denominator_weeks"]))
    check("Disease reconciliation", derivation["encounters_by_disease"] == derivation["weekly_reconciliation_by_disease"], json.dumps(derivation["encounters_by_disease"]))

    estimates = pd.read_csv(OUT / "02_models" / "central_extension_estimates.csv")
    central = estimates[estimates["parameter"].isin(["Immediate level change", "Weekly slope change"])].copy()
    check("Central estimate row count", len(estimates) == 120, str(len(estimates)))
    check("Central test count", len(central) == 48, str(len(central)))
    check("Combined estimate count", int((estimates["parameter"] == "Combined postinterruption effect").sum()) == 72, str((estimates["parameter"] == "Combined postinterruption effect").sum()))
    expected_domains = {"Access": 6, "Presenting severity": 18, "Care pathways": 24}
    check("Fixed multiplicity domain sizes", central.groupby("domain").size().to_dict() == expected_domains, str(central.groupby("domain").size().to_dict()))
    max_q_diff = 0.0
    for _, group in central.groupby("domain"):
        max_q_diff = max(max_q_diff, float(np.max(np.abs(bh_adjust(group["p_raw"].to_numpy()) - group["q_domain"].to_numpy()))))
    check("Domain BH recomputation", max_q_diff < 1e-12, f"max abs diff={max_q_diff:.3g}")
    finite = np.isfinite(estimates[["estimate", "ci_low", "ci_high", "p_raw"]].to_numpy()).all()
    check("Finite central estimates", bool(finite), "estimate, CI, and nominal P")
    check("Central CI ordering", bool(((estimates["ci_low"] <= estimates["estimate"]) & (estimates["estimate"] <= estimates["ci_high"])).all()), "all rows")

    heterogeneity = pd.read_csv(OUT / "02_models" / "disease_heterogeneity_tests.csv")
    check("Heterogeneity test count", len(heterogeneity) == 16, str(len(heterogeneity)))
    hetero_q_diff = float(np.max(np.abs(bh_adjust(heterogeneity["p_raw"].to_numpy()) - heterogeneity["q_domain"].to_numpy())))
    check("Heterogeneity BH recomputation", hetero_q_diff < 1e-12, f"max abs diff={hetero_q_diff:.3g}")
    pairwise = pd.read_csv(OUT / "02_models" / "disease_pairwise_contrasts.csv")
    check("Pairwise supportive contrast count", len(pairwise) == 48, str(len(pairwise)))

    model_summary = read_json(OUT / "02_models" / "model_run_summary.json")
    check("All central models converged", model_summary["all_disease_specific_models_converged"] and model_summary["all_heterogeneity_models_converged"], json.dumps(model_summary))
    diagnostics = read_json(OUT / "02_models" / "central_extension_diagnostics.json")
    for key, value in diagnostics.items():
        if value["ljung_box"]["12"]["p"] < 0.05:
            warnings.append(f"{key}: residual autocorrelation at lag 12 (Ljung-Box P={value['ljung_box']['12']['p']:.4g}).")
        if value["dispersion"] > 2:
            warnings.append(f"{key}: Pearson/residual dispersion={value['dispersion']:.3g}.")

    weekly = pd.read_csv(OUT / "01_derived" / "weekly_extension_inputs.csv", parse_dates=["week_start"])
    coefficients = pd.read_csv(OUT / "02_models" / "central_extension_coefficients.csv")
    max_independent_difference = 0.0
    for outcome_key in central["outcome_key"].unique():
        for disease in DISEASES:
            frame = weekly.loc[weekly["disease"].eq(disease)].sort_values("week_start").reset_index(drop=True)
            x, terms = design(frame)
            if outcome_key == "onset_to_arrival":
                keep = frame["onset_median_minutes"].gt(0)
                independent = np.linalg.lstsq(x[keep], np.log(frame.loc[keep, "onset_median_minutes"].to_numpy(float)), rcond=None)[0]
            else:
                success = frame[f"{outcome_key}_event_n"].to_numpy(float)
                total = frame[f"{outcome_key}_valid_n"].to_numpy(float)
                expanded_x = np.vstack([x, x])
                expanded_y = np.r_[np.ones(len(frame)), np.zeros(len(frame))]
                weights = np.r_[success, total - success]
                validator = LogisticRegression(C=np.inf, solver="newton-cholesky", fit_intercept=False, max_iter=2000, tol=1e-12)
                validator.fit(expanded_x, expanded_y, sample_weight=weights)
                independent = validator.coef_.ravel()
            saved = coefficients.loc[(coefficients["outcome_key"] == outcome_key) & (coefficients["disease"] == disease)].set_index("term").loc[terms, "coefficient"].to_numpy()
            max_independent_difference = max(max_independent_difference, float(np.max(np.abs(independent - saved))))
    check("Independent coefficient cross-validation", max_independent_difference < 1e-6, f"max abs diff={max_independent_difference:.3g}")

    sensitivity = pd.read_csv(OUT / "03_sensitivity" / "extension_sensitivity_estimates.csv")
    sensitivity_diag = pd.read_csv(OUT / "03_sensitivity" / "extension_sensitivity_diagnostics.csv")
    check("Sensitivity estimate count", len(sensitivity) == 220, str(len(sensitivity)))
    check("Sensitivity model count and convergence", len(sensitivity_diag) == 110 and bool(sensitivity_diag["converged"].all()), f"models={len(sensitivity_diag)}; all={sensitivity_diag['converged'].all()}")
    check("Sensitivity CI ordering", bool(((sensitivity["ci_low"] <= sensitivity["estimate"]) & (sensitivity["estimate"] <= sensitivity["ci_high"])).all()), "all rows")

    adjusted = pd.read_csv(OUT / "04_case_mix_adjusted" / "case_mix_adjusted_estimates.csv")
    adjusted_diag = read_json(OUT / "04_case_mix_adjusted" / "case_mix_adjusted_diagnostics.json")
    check("Case-mix model and estimate counts", len(adjusted_diag) == 12 and len(adjusted) == 60, f"models={len(adjusted_diag)}; rows={len(adjusted)}")
    estimable = adjusted["status"].eq("estimable")
    check("Finite estimable case-mix results", bool(np.isfinite(adjusted.loc[estimable, ["estimate", "ci_low", "ci_high", "p_raw"]].to_numpy()).all()), str(int(estimable.sum())))
    check("NE rows contain reasons", bool(adjusted.loc[~estimable, "reason"].notna().all()), str(int((~estimable).sum())))
    for key, value in adjusted_diag.items():
        if not value["estimable"]:
            warnings.append(f"{key}: case-mix-adjusted model not estimable ({value['reason']}).")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    colors = {"AMI": "#0072B2", "Ischemic stroke": "#009E73", "Hemorrhagic stroke": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for disease in DISEASES:
        disease_weekly = weekly.loc[weekly["disease"].eq(disease)].sort_values("week_start")
        axes[0].plot(disease_weekly["week_start"], 100 * disease_weekly["onset_completeness"], label=disease, color=colors[disease], linewidth=1.8)
        axes[1].plot(disease_weekly["week_start"], 100 * disease_weekly["spo2_completeness"], label=disease, color=colors[disease], linewidth=1.8)
    for axis, title in zip(axes, ["Symptom-onset timestamp completeness", "Oxygen-saturation completeness"]):
        axis.axvline(pd.Timestamp("2024-02-20"), color="#333333", linestyle="--", linewidth=1)
        axis.set_ylabel("Valid records (%)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=3, frameon=False, loc="lower left")
    axes[1].set_xlabel("Complete calendar week")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "Figure_S_Extension_Completeness_Over_Time.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_OUT / "Figure_S_Extension_Completeness_Over_Time.pdf", bbox_inches="tight")
    plt.close(fig)

    passed = all(item["passed"] for item in checks)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "validation_passed": passed, "checks": checks, "warnings": warnings}
    write_json(VALID_OUT / "validation_report.json", report)
    lines = [f"VALIDATION PASSED: {passed}", ""]
    lines.extend(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}: {item['detail']}" for item in checks)
    lines.extend(["", "WARNINGS:"] + [f"- {warning}" for warning in warnings])
    (VALID_OUT / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    checksum_lines = [f"{sha256(path)}  {path.relative_to(OUT)}" for path in files if path.name != "SHA256SUMS.txt"]
    (VALID_OUT / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
