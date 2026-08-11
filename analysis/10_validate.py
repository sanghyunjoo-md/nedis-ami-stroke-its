#!/usr/bin/env python3
"""Independent reconciliation and numerical validation of the clean rerun."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

from model_utils import fit_nb2_hac, holm_adjust
from settings import (
    EXPECTED_RAW_SHA256,
    EXPECTED_SAP_SHA256,
    RAW_XLSX,
    ROOT,
    SAP_DOCX,
)


OUT = ROOT / "analysis_outputs" / "09_validation"
DISEASES = ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def design(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(d)), d["calendar_time"], d["post"], d["post_time"], d["sin1"], d["cos1"]]).astype(float)


def main() -> None:
    if not RAW_XLSX.is_file():
        raise FileNotFoundError(
            f"Restricted NEDIS workbook not found: {RAW_XLSX}. "
            "Set NEDIS_RAW_XLSX or follow data/README.md."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    warnings_list = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    raw_sha = sha256(RAW_XLSX)
    if EXPECTED_RAW_SHA256:
        check("Raw XLSX checksum", raw_sha == EXPECTED_RAW_SHA256, raw_sha)
    else:
        warnings_list.append(
            "NEDIS_EXPECTED_RAW_SHA256 was not set; the raw workbook checksum was recorded but not compared."
        )

    if SAP_DOCX.is_file():
        sap_sha = sha256(SAP_DOCX)
        if EXPECTED_SAP_SHA256:
            check("Locked SAP checksum", sap_sha == EXPECTED_SAP_SHA256, sap_sha)
        else:
            warnings_list.append(
                "NEDIS_EXPECTED_SAP_SHA256 was not set; the SAP checksum was recorded but not compared."
            )
    else:
        warnings_list.append(
            "No local locked SAP file was supplied; SAP checksum validation was skipped."
        )

    qc = json.loads((ROOT / "analysis_outputs" / "02_cohort" / "cohort_qc.json").read_text())
    check("Source row count", qc["source_rows"] == 309924, str(qc["source_rows"]))
    check("Exact duplicates", qc["exact_duplicates_removed"] == 4, str(qc["exact_duplicates_removed"]))
    check("Target cohort count", qc["target_cohort"] == 191667, str(qc["target_cohort"]))
    check("Primary weekly cohort count", qc["primary_model_encounters"] == 189330, str(qc["primary_model_encounters"]))
    check("Locked week counts", qc["pre_weeks"] == 59 and qc["post_weeks"] == 44, f"{qc['pre_weeks']}/{qc['post_weeks']}")

    weekly = pd.read_csv(ROOT / "analysis_outputs" / "02_cohort" / "weekly_primary_inputs.csv")
    for disease, expected in qc["primary_model_counts"].items():
        observed = int(weekly.loc[weekly["disease"].eq(disease), "visit_count"].sum())
        check(f"Weekly reconciliation: {disease}", observed == expected, f"weekly={observed}; cohort={expected}")
    check("Rate numerator <= denominator", bool((weekly["transfer_out_n"] <= weekly["known_disposition_n"]).all()), "all weeks")

    primary = pd.read_csv(ROOT / "analysis_outputs" / "03_primary_its" / "confirmatory_estimates.csv")
    check("Confirmatory family size", len(primary) == 12, str(len(primary)))
    recomputed = holm_adjust(primary["p_raw"].to_numpy())
    holm_diff = float(np.max(np.abs(recomputed - primary["p_holm"].to_numpy())))
    check("Holm adjustment recomputation", holm_diff < 1e-12, f"max abs diff={holm_diff:.3g}")
    finite = np.isfinite(primary[["estimate", "ci_low", "ci_high", "p_raw", "p_holm"]].to_numpy()).all()
    check("Finite primary estimates", bool(finite), "estimate, CI, and P columns")

    saved_coef = pd.read_csv(ROOT / "analysis_outputs" / "03_primary_its" / "all_primary_model_coefficients.csv")
    max_nb_score = 0.0
    max_nb_coef_diff = 0.0
    max_binomial_coef_diff = 0.0
    for disease in DISEASES:
        d = weekly.loc[weekly["disease"].eq(disease)].sort_values("week_start").reset_index(drop=True)
        x = design(d)
        nb = fit_nb2_hac(d["visit_count"].to_numpy(float), x, lag=4)
        max_nb_score = max(max_nb_score, float(np.max(np.abs(nb.score_obs.sum(axis=0)))))
        saved_nb = saved_coef.loc[saved_coef["disease"].eq(disease) & saved_coef["outcome"].eq("ED visit count")].sort_values("term", key=lambda s: s.map({"intercept":0,"time":1,"post":2,"post_time":3,"sin1":4,"cos1":5}))["coefficient"].to_numpy()
        max_nb_coef_diff = max(max_nb_coef_diff, float(np.max(np.abs(nb.params - saved_nb))))

        success = d["transfer_out_n"].to_numpy(float)
        total = d["known_disposition_n"].to_numpy(float)
        xx = np.vstack([x, x])
        yy = np.r_[np.ones(len(d)), np.zeros(len(d))]
        weights = np.r_[success, total - success]
        independent = LogisticRegression(C=np.inf, solver="newton-cholesky", fit_intercept=False, max_iter=1000, tol=1e-12)
        independent.fit(xx, yy, sample_weight=weights)
        saved_bin = saved_coef.loc[saved_coef["disease"].eq(disease) & saved_coef["outcome"].eq("Transfer-out rate")].sort_values("term", key=lambda s: s.map({"intercept":0,"time":1,"post":2,"post_time":3,"sin1":4,"cos1":5}))["coefficient"].to_numpy()
        max_binomial_coef_diff = max(max_binomial_coef_diff, float(np.max(np.abs(independent.coef_.ravel() - saved_bin))))
    check("NB2 score equations", max_nb_score < 1e-3, f"max absolute beta score={max_nb_score:.3g}")
    check("NB2 saved coefficient reproducibility", max_nb_coef_diff < 1e-9, f"max abs diff={max_nb_coef_diff:.3g}")
    check("Independent grouped-binomial coefficient validation", max_binomial_coef_diff < 1e-6, f"max abs diff={max_binomial_coef_diff:.3g}")

    sensitivity = pd.read_csv(ROOT / "analysis_outputs" / "04_sensitivity" / "complete_coprimary_sensitivity_matrix.csv")
    check("Complete sensitivity matrix", len(sensitivity) == 156 and sensitivity["sensitivity"].nunique() == 13, f"rows={len(sensitivity)}; variants={sensitivity['sensitivity'].nunique()}")
    exploratory = pd.read_csv(ROOT / "analysis_outputs" / "05_secondary_exploratory" / "exploratory_patient_flow_estimates.csv")
    check("Exploratory domain sizes", len(exploratory) == 66, str(exploratory.groupby("multiplicity_domain").size().to_dict()))

    placebo = sensitivity.loc[sensitivity["sensitivity"].str.startswith("Preperiod placebo")]
    placebo_nominal = int((placebo["p_raw"] < 0.05).sum())
    if placebo_nominal:
        warnings_list.append(f"Preperiod falsification produced {placebo_nominal} of 12 nominal P<0.05 results; causal interpretation requires caution.")
    diagnostics = json.loads((ROOT / "analysis_outputs" / "03_primary_its" / "primary_diagnostics.json").read_text())
    lb12 = diagnostics["Hemorrhagic stroke | ED visit count"]["ljung_box"]["12"]["p"]
    if lb12 < 0.05:
        warnings_list.append(f"Hemorrhagic-stroke visit-count residuals show lag-12 autocorrelation (Ljung-Box P={lb12:.4g}).")
    adjusted_diag = json.loads((ROOT / "analysis_outputs" / "06_adjusted_stratified" / "case_mix_adjusted_diagnostics.json").read_text())
    for disease, info in adjusted_diag.items():
        if info["information_rank"] < info["design_columns"] or info["information_condition_number"] > 1e10:
            warnings_list.append(f"{disease} adjusted transfer model is numerically ill-conditioned (rank {info['information_rank']}/{info['design_columns']}; condition number {info['information_condition_number']:.3g}); retain as sensitivity only.")

    fig_dir = ROOT / "analysis_outputs" / "08_figures"
    for png in sorted(fig_dir.glob("*.png")):
        with Image.open(png) as im:
            check(f"Figure width >=1200 px: {png.name}", im.width >= 1200, f"{im.width}x{im.height}")
        check(f"Figure file <10 MB: {png.name}", png.stat().st_size < 10_000_000, str(png.stat().st_size))

    passed = all(x["passed"] for x in checks)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_passed": passed,
        "checks": checks,
        "warnings": warnings_list,
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"VALIDATION PASSED: {passed}", ""]
    lines.extend([f"[{'PASS' if x['passed'] else 'FAIL'}] {x['name']}: {x['detail']}" for x in checks])
    lines.extend(["", "WARNINGS:"] + [f"- {x}" for x in warnings_list])
    (OUT / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
