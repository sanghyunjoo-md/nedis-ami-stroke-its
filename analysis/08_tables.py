#!/usr/bin/env python3
"""Generate journal-ready aggregate tables without patient- or facility-level identifiers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COHORT_FILE = ROOT / "analysis_outputs" / "02_cohort" / "cohort_primary_weeks.pkl"
OUT = ROOT / "analysis_outputs" / "07_tables"
DISEASES = ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]


INSURANCE = {
    "10": "National Health Insurance", "20": "Automobile insurance", "30": "Industrial accident insurance",
    "40": "Private insurance only", "51": "Medical Aid type 1", "52": "Medical Aid type 2",
    "60": "Self-pay/uninsured", "88": "Other insurance", "99": "Unknown",
}
ARRIVAL_ROUTE = {"1": "Direct visit", "2": "Transfer-in from another facility", "3": "Outpatient referral", "8": "Other", "9": "Unknown"}
ARRIVAL_MODE = {"1": "119 ambulance", "2": "Healthcare-facility ambulance", "3": "Other ambulance", "4": "Police/public vehicle", "5": "Air transport", "6": "Other vehicle", "7": "Walk-in", "8": "Other", "9": "Unknown"}
ED_LEVEL = {"A": "Regional emergency medical center", "C": "Local emergency medical center"}
REGION = {
    "11": "Seoul", "26": "Busan", "27": "Daegu", "28": "Incheon", "29": "Gwangju", "30": "Daejeon",
    "31": "Ulsan", "36": "Sejong", "41": "Gyeonggi", "42": "Gangwon", "51": "Gangwon", "43": "Chungbuk",
    "44": "Chungnam", "45": "Jeonbuk", "52": "Jeonbuk", "46": "Jeonnam", "47": "Gyeongbuk", "48": "Gyeongnam", "50": "Jeju",
}


def age_category(code: pd.Series) -> pd.Series:
    x = pd.to_numeric(code, errors="coerce")
    return pd.Series(np.select([x.between(5, 8), x.between(9, 13), x.between(14, 16), x.ge(17)], ["20-39", "40-64", "65-79", "80 or older"], default="Unknown"), index=code.index)


def smd_binary(pre_p: float, post_p: float) -> float:
    denom = np.sqrt((pre_p * (1 - pre_p) + post_p * (1 - post_p)) / 2)
    return float((post_p - pre_p) / denom) if denom > 0 else 0.0


def table1(cohort: pd.DataFrame) -> pd.DataFrame:
    d = cohort.copy()
    d["period"] = np.where(d["post"].eq(1), "Postinterruption", "Preinterruption")
    d["Age group"] = age_category(d["ptmibrtd"])
    d["Sex"] = d["ptmisexx"].map({"M": "Male", "F": "Female"}).fillna("Unknown")
    d["Insurance"] = d["ptmiiukd"].map(INSURANCE).fillna("Unknown/out of code")
    d["Arrival route"] = d["ptmiinrt"].map(ARRIVAL_ROUTE).fillna("Unknown/out of code")
    d["Arrival mode"] = d["ptmiinmn"].map(ARRIVAL_MODE).fillna("Unknown/out of code")
    d["Initial KTAS"] = d["ptmikts1"].map({"1":"Level 1", "2":"Level 2", "3":"Level 3", "4":"Level 4", "5":"Level 5", "8":"Other", "9":"Unknown", "-":"Missing"}).fillna("Missing")
    d["Arrival day"] = d["arrival_date"].dt.day_name()
    hour = pd.to_numeric(d["ptmiintm"].str.zfill(4).str[:2], errors="coerce")
    d["Arrival time"] = pd.Series(np.select([hour.between(0, 7), hour.between(8, 15), hour.between(16, 23)], ["00:00-07:59", "08:00-15:59", "16:00-23:59"], default="Unknown"), index=d.index)
    d["ED level"] = d["ptmiemcl"].map(ED_LEVEL).fillna("Unknown")
    d["Region"] = d["ptmiemar"].str[:2].map(REGION).fillna("Unknown")
    variables = ["Age group", "Sex", "Insurance", "Arrival route", "Arrival mode", "Initial KTAS", "Arrival day", "Arrival time", "ED level", "Region"]
    rows = []
    for disease in DISEASES:
        dd = d.loc[d["disease"].eq(disease)]
        totals = dd["period"].value_counts()
        for variable in variables:
            categories = sorted(dd[variable].dropna().unique())
            for category in categories:
                pre_n = int(((dd["period"] == "Preinterruption") & (dd[variable] == category)).sum())
                post_n = int(((dd["period"] == "Postinterruption") & (dd[variable] == category)).sum())
                pre_total = int(totals.get("Preinterruption", 0))
                post_total = int(totals.get("Postinterruption", 0))
                pre_p = pre_n / pre_total if pre_total else np.nan
                post_p = post_n / post_total if post_total else np.nan
                rows.append({
                    "disease": disease, "variable": variable, "category": category,
                    "pre_n": pre_n, "pre_percent": 100 * pre_p, "post_n": post_n, "post_percent": 100 * post_p,
                    "standardized_difference": smd_binary(pre_p, post_p),
                })
    return pd.DataFrame(rows)


def missingness_table(cohort: pd.DataFrame) -> pd.DataFrame:
    d = cohort.copy()
    d["period"] = np.where(d["post"].eq(1), "Postinterruption", "Preinterruption")
    checks = {
        "Known ED disposition": d["known_disposition"],
        "Initial KTAS 1-5": d["ktas_valid"],
        "Valid ED length of stay": d["ed_los_minutes"].notna(),
        "Sex known": d["ptmisexx"].isin(["M", "F"]),
        "Insurance known": ~d["ptmiiukd"].isin(["", "-", "99"]),
        "Arrival route known": ~d["ptmiinrt"].isin(["", "-", "9"]),
        "Arrival mode known": ~d["ptmiinmn"].isin(["", "-", "9"]),
        "Destination known among transfers": (~d["transfer_out"]) | d["destination_known"],
        "Inpatient result known among admissions": (~d["hospital_admission"]) | d["known_inpatient_result"],
    }
    rows = []
    for disease in DISEASES:
        for period in ("Preinterruption", "Postinterruption"):
            mask = d["disease"].eq(disease) & d["period"].eq(period)
            total = int(mask.sum())
            for label, valid in checks.items():
                missing = int((mask & ~valid).sum())
                rows.append({"disease": disease, "period": period, "field": label, "denominator": total, "missing_n": missing, "missing_percent": 100 * missing / total})
    return pd.DataFrame(rows)


def facility_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    d = cohort.copy()
    d["period"] = np.where(d["post"].eq(1), "Postinterruption", "Preinterruption")
    rows = []
    for disease in DISEASES:
        for period in ("Preinterruption", "Postinterruption"):
            x = d.loc[d["disease"].eq(disease) & d["period"].eq(period)]
            counts = x.groupby("ptmiemnm").size()
            rows.append({"disease": disease, "period": period, "contributing_facilities": int(counts.size), "encounters_per_facility_median": float(counts.median()), "encounters_per_facility_q1": float(counts.quantile(0.25)), "encounters_per_facility_q3": float(counts.quantile(0.75))})
    return pd.DataFrame(rows)


def robustness_summary(primary: pd.DataFrame, sensitivity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    placebo_name = "Preperiod placebo: false interruption 2023-07-03"
    placebo = sensitivity.loc[sensitivity["sensitivity"].eq(placebo_name)].copy()
    robustness = sensitivity.loc[~sensitivity["sensitivity"].eq(placebo_name)].copy()
    rows = []
    for _, p in primary.iterrows():
        s = robustness.loc[
            robustness["disease"].eq(p["disease"])
            & robustness["outcome"].eq(p["outcome"])
            & robustness["parameter"].eq(p["parameter"])
        ]
        direction = np.sign(p["estimate"] - 1)
        rows.append({
            "disease": p["disease"], "outcome": p["outcome"], "parameter": p["parameter"],
            "primary_estimate": p["estimate"], "primary_ci_low": p["ci_low"], "primary_ci_high": p["ci_high"], "primary_p_holm": p["p_holm"],
            "sensitivity_variants": int(len(s)), "sensitivity_min": float(s["estimate"].min()), "sensitivity_max": float(s["estimate"].max()),
            "direction_preserved_n": int((np.sign(s["estimate"] - 1) == direction).sum()), "raw_p_lt_0_05_n": int((s["p_raw"] < 0.05).sum()),
        })
    return pd.DataFrame(rows), placebo


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_pickle(COHORT_FILE)
    t1 = table1(cohort)
    missing = missingness_table(cohort)
    facilities = facility_summary(cohort)
    primary = pd.read_csv(ROOT / "analysis_outputs" / "03_primary_its" / "confirmatory_estimates.csv")
    sensitivity = pd.read_csv(ROOT / "analysis_outputs" / "04_sensitivity" / "complete_coprimary_sensitivity_matrix.csv")
    robust, placebo = robustness_summary(primary, sensitivity)
    t1.to_csv(OUT / "Table_1_case_mix_pre_post.csv", index=False)
    missing.to_csv(OUT / "Table_S_missingness.csv", index=False)
    facilities.to_csv(OUT / "Table_S_facility_summary.csv", index=False)
    primary.to_csv(OUT / "Table_2_confirmatory_ITS.csv", index=False)
    robust.to_csv(OUT / "Table_S_robustness_summary.csv", index=False)
    placebo.to_csv(OUT / "Table_S_placebo_results.csv", index=False)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "table1_rows": int(len(t1)), "missingness_rows": int(len(missing)), "facility_rows": int(len(facilities)), "robustness_rows": int(len(robust)), "placebo_rows": int(len(placebo))}
    (OUT / "table_generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
