#!/usr/bin/env python3
"""Derive locked extension variables and reconciliation inputs without editing parent data."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import COHORT_PATH, DISEASES, OUT, add_time_terms, cleaned_string, numeric, parse_datetime, period_label, sha256, week_skeleton, write_json


DERIVED = OUT / "01_derived"


def binary_summary(group: pd.DataFrame, prefix: str) -> dict:
    valid = group[f"{prefix}_valid"].fillna(False).astype(bool)
    event = group[f"{prefix}_event"].fillna(False).astype(bool) & valid
    return {
        "total_n": int(len(group)),
        "valid_n": int(valid.sum()),
        "missing_n": int((~valid).sum()),
        "event_n": int(event.sum()),
        "event_pct": float(100 * event.sum() / valid.sum()) if valid.sum() else np.nan,
    }


def continuous_summary(values: pd.Series, total_n: int) -> dict:
    x = values.dropna().astype(float)
    return {
        "total_n": int(total_n),
        "valid_n": int(len(x)),
        "missing_n": int(total_n - len(x)),
        "valid_pct": float(100 * len(x) / total_n) if total_n else np.nan,
        "median": float(x.median()) if len(x) else np.nan,
        "q1": float(x.quantile(0.25)) if len(x) else np.nan,
        "q3": float(x.quantile(0.75)) if len(x) else np.nan,
        "min": float(x.min()) if len(x) else np.nan,
        "max": float(x.max()) if len(x) else np.nan,
        "zero_n": int(x.eq(0).sum()),
    }


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    source = pd.read_pickle(COHORT_PATH)
    df = pd.DataFrame(index=source.index)
    for col in ["disease", "week_start", "calendar_time", "post", "post_time"]:
        df[col] = source[col]

    arrival_dt = parse_datetime(source["ptmiindt"], source["ptmiintm"])
    onset_dt = parse_datetime(source["ptmiakdt"], source["ptmiaktm"])
    onset_minutes = (arrival_dt - onset_dt).dt.total_seconds() / 60.0
    df["onset_minutes_raw"] = onset_minutes
    df["onset_minutes"] = onset_minutes.where(onset_minutes.between(0, 30 * 24 * 60, inclusive="both"))

    avpu = cleaned_string(source["ptmiresp"])
    df["impaired_consciousness_valid"] = avpu.isin(["A", "V", "P", "U"])
    df["impaired_consciousness_event"] = avpu.isin(["V", "P", "U"])

    sbp = numeric(source["ptmihibp"])
    df["shock_range_sbp_valid"] = sbp.between(0, 300, inclusive="both")
    df["shock_range_sbp_event"] = sbp.lt(90)
    df["sbp_mmHg"] = sbp.where(df["shock_range_sbp_valid"])

    spo2 = numeric(source["ptmivoxs"])
    df["hypoxemia_valid"] = spo2.between(0, 100, inclusive="both")
    df["hypoxemia_event"] = spo2.lt(90)
    df["spo2_pct"] = spo2.where(df["hypoxemia_valid"])

    ktas1 = numeric(source["ptmikts1"])
    ktas2 = numeric(source["ptmikts2"])
    df["ktas_escalation_valid"] = ktas1.between(1, 5) & ktas2.between(1, 5)
    df["ktas_escalation_event"] = ktas2.lt(ktas1)

    specialist = cleaned_string(source["ptmisdcd"])
    df["specialist_involvement_valid"] = specialist.isin(["1", "2", "3", "4"])
    df["specialist_involvement_event"] = specialist.isin(["3", "4"])

    department = cleaned_string(source["ptmidept"])
    df["concordant_department_valid"] = department.notna() & department.ne("") & ~department.isin(["-", "XX", "ZZ"])
    concordant = pd.Series(False, index=df.index)
    concordant.loc[df["disease"].eq("AMI")] = department.loc[df["disease"].eq("AMI")].eq("AA")
    stroke = df["disease"].isin(["Ischemic stroke", "Hemorrhagic stroke"])
    concordant.loc[stroke] = department.loc[stroke].isin(["FA", "BB"])
    df["concordant_department_event"] = concordant

    area = cleaned_string(source["ptmiarea"])
    regional = cleaned_string(source["ptmiemcl"]).eq("A")
    df["high_acuity_area_valid"] = regional & area.isin(["1", "2", "3", "4", "5", "6"])
    df["high_acuity_area_event"] = area.isin(["3", "5"])

    df["doa"] = numeric(source["ptmiemrt"]).eq(41)
    df["age_group"] = cleaned_string(source["ptmibrtd"])
    df["sex"] = cleaned_string(source["ptmisexx"])
    df["initial_ktas"] = ktas1
    df["avpu"] = avpu
    df["arrival_119"] = cleaned_string(source["ptmiinmn"]).eq("1")
    df["transfer_in"] = cleaned_string(source["ptmiinrt"]).eq("2")
    df["ed_level"] = cleaned_string(source["ptmiemcl"])
    df["facility"] = cleaned_string(source["ptmiemnm"])
    df["revised_ktas_code"] = cleaned_string(source["ptmikts2"])
    df["specialist_code"] = specialist
    df["department_code"] = department
    df["area_code"] = area

    vital_specs = {
        "dbp_mmHg": ("ptmilobp", lambda x: x.between(0, 300, inclusive="both")),
        "pulse_bpm": ("ptmipuls", lambda x: x.between(0, 300, inclusive="both")),
        "respiratory_rate_bpm": ("ptmibrth", lambda x: x.between(0, 99, inclusive="both")),
        "temperature_c": ("ptmibdht", lambda x: x.eq(0) | x.between(20, 45, inclusive="both")),
    }
    for name, (source_col, rule) in vital_specs.items():
        values = numeric(source[source_col])
        df[name] = values.where(rule(values))

    df["period"] = period_label(df["post"])
    df.to_pickle(DERIVED / "extension_patient_derived.pkl")

    skeleton = week_skeleton()
    grouped = df.groupby(["disease", "week_start"], observed=True)
    weekly = grouped.size().rename("disease_total_n").reset_index()
    onset = grouped["onset_minutes"].agg([("onset_valid_n", "count"), ("onset_median_minutes", "median")]).reset_index()
    weekly = weekly.merge(onset, on=["disease", "week_start"], how="left")
    binary_outcomes = [
        "impaired_consciousness",
        "shock_range_sbp",
        "hypoxemia",
        "ktas_escalation",
        "specialist_involvement",
        "concordant_department",
        "high_acuity_area",
    ]
    for outcome in binary_outcomes:
        agg = grouped[[f"{outcome}_valid", f"{outcome}_event"]].sum().reset_index()
        agg = agg.rename(columns={f"{outcome}_valid": f"{outcome}_valid_n", f"{outcome}_event": f"{outcome}_event_n"})
        weekly = weekly.merge(agg, on=["disease", "week_start"], how="left")
    weekly = skeleton.merge(weekly, on=["disease", "week_start"], how="left")
    count_cols = [col for col in weekly if col.endswith("_n")]
    weekly[count_cols] = weekly[count_cols].fillna(0).astype(int)
    weekly = add_time_terms(weekly)
    weekly["onset_completeness"] = weekly["onset_valid_n"] / weekly["disease_total_n"].replace(0, np.nan)
    weekly["spo2_completeness"] = weekly["hypoxemia_valid_n"] / weekly["disease_total_n"].replace(0, np.nan)
    weekly.to_csv(DERIVED / "weekly_extension_inputs.csv", index=False)

    prepost_rows: list[dict] = []
    for (disease, period), group in df.groupby(["disease", "period"], observed=True, sort=True):
        onset_stats = continuous_summary(group["onset_minutes"], len(group))
        prepost_rows.append({"disease": disease, "period": period, "outcome": "Onset-to-arrival time (minutes)", "type": "continuous", **onset_stats})
        for outcome in binary_outcomes:
            prepost_rows.append({"disease": disease, "period": period, "outcome": outcome, "type": "binary", **binary_summary(group, outcome)})
    pd.DataFrame(prepost_rows).to_csv(DERIVED / "prepost_outcome_distributions.csv", index=False)

    category_rows: list[dict] = []
    for variable in ["revised_ktas_code", "specialist_code", "department_code", "area_code"]:
        counts = df.groupby(["disease", "period", variable], observed=True, dropna=False).size().rename("n").reset_index()
        counts = counts.rename(columns={variable: "category"})
        counts["variable"] = variable
        totals = counts.groupby(["disease", "period", "variable"])["n"].transform("sum")
        counts["pct"] = 100 * counts["n"] / totals
        category_rows.extend(counts.to_dict("records"))
    pd.DataFrame(category_rows).to_csv(DERIVED / "source_category_distributions.csv", index=False)

    vital_rows: list[dict] = []
    for (disease, period), group in df.groupby(["disease", "period"], observed=True, sort=True):
        for vital in ["sbp_mmHg", "dbp_mmHg", "pulse_bpm", "respiratory_rate_bpm", "temperature_c", "spo2_pct"]:
            vital_rows.append({"disease": disease, "period": period, "vital": vital, **continuous_summary(group[vital], len(group))})
    pd.DataFrame(vital_rows).to_csv(DERIVED / "initial_vital_signs_descriptive.csv", index=False)

    qc = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(COHORT_PATH.relative_to(COHORT_PATH.parents[2])),
        "input_sha256": sha256(COHORT_PATH),
        "input_rows": int(len(source)),
        "derived_rows": int(len(df)),
        "weekly_rows": int(len(weekly)),
        "weeks_by_disease": {str(k): int(v) for k, v in weekly.groupby("disease")["week_start"].nunique().items()},
        "encounters_by_disease": {str(k): int(v) for k, v in df["disease"].value_counts().reindex(DISEASES).items()},
        "weekly_reconciliation_by_disease": {str(k): int(v) for k, v in weekly.groupby("disease")["disease_total_n"].sum().reindex(DISEASES).items()},
        "zero_denominator_weeks": {outcome: int((weekly[f"{outcome}_valid_n"] == 0).sum()) for outcome in binary_outcomes},
        "onset_zero_valid_weeks": int((weekly["onset_valid_n"] == 0).sum()),
    }
    write_json(DERIVED / "derivation_qc.json", qc)
    print(pd.DataFrame(prepost_rows).head(12).to_string(index=False))
    print(qc)


if __name__ == "__main__":
    main()
