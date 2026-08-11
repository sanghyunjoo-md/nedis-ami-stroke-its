#!/usr/bin/env python3
"""Construct the locked cohort and weekly analysis inputs from the raw cache."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_CACHE = ROOT / "analysis_outputs" / "01_raw_cache" / "EMIHPTMI_raw_strings.pkl"
OUT = ROOT / "analysis_outputs" / "02_cohort"
STUDY_START = pd.Timestamp("2023-01-01")
STUDY_END = pd.Timestamp("2024-12-31")
FIRST_COMPLETE_WEEK = pd.Timestamp("2023-01-02")
TRANSITION_WEEK = pd.Timestamp("2024-02-19")
FIRST_POST_WEEK = pd.Timestamp("2024-02-26")
LAST_COMPLETE_WEEK = pd.Timestamp("2024-12-23")


def normalize_diag(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(" ", "", regex=False)
    )


def parse_datetime(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    d = date_s.fillna("").astype(str).str.strip()
    t = time_s.fillna("").astype(str).str.strip()
    valid_t = t.str.fullmatch(r"\d{1,4}")
    t = t.where(valid_t, "").str.zfill(4)
    return pd.to_datetime(d + t, format="%Y%m%d%H%M", errors="coerce")


def disposition_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.where(s.str.fullmatch(r"\d+"), np.nan), errors="coerce")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(RAW_CACHE)
    source_n = len(raw)
    duplicate_mask = raw.duplicated(keep="first")
    dedup = raw.loc[~duplicate_mask].copy()

    dedup["arrival_date"] = pd.to_datetime(dedup["ptmiindt"], format="%Y%m%d", errors="coerce")
    dedup["age_group_num"] = pd.to_numeric(dedup["ptmibrtd"], errors="coerce")
    valid_date = dedup["arrival_date"].between(STUDY_START, STUDY_END, inclusive="both")
    adult = dedup["age_group_num"].ge(5)
    eligible_level = dedup["ptmiemcl"].isin(["A", "C"])
    eligible_base = valid_date & adult & eligible_level
    base = dedup.loc[eligible_base].copy()

    disease_flags: dict[str, pd.Series] = {
        "AMI": pd.Series(False, index=base.index),
        "Ischemic stroke": pd.Series(False, index=base.index),
        "Hemorrhagic stroke": pd.Series(False, index=base.index),
    }
    principal_counts = pd.Series(0, index=base.index, dtype="int16")
    for i in range(1, 21):
        diag_col = f"dgotdiag{i:02d}"
        type_col = f"dgotdggb{i:02d}"
        principal = base[type_col].eq("1")
        principal_counts = principal_counts + principal.astype("int16")
        code = normalize_diag(base[diag_col])
        disease_flags["AMI"] |= principal & code.str.startswith("I21")
        disease_flags["Ischemic stroke"] |= principal & code.str.startswith("I63")
        disease_flags["Hemorrhagic stroke"] |= principal & code.str.startswith(("I60", "I61", "I62"))

    flag_df = pd.DataFrame(disease_flags)
    disease_count = flag_df.sum(axis=1)
    target_any = disease_count.ge(1)
    cross_stratum = disease_count.gt(1)
    cohort = base.loc[target_any & ~cross_stratum].copy()
    flag_cohort = flag_df.loc[cohort.index]
    cohort["disease"] = flag_cohort.idxmax(axis=1)
    cohort["principal_diagnosis_count"] = principal_counts.loc[cohort.index]

    cohort["week_start"] = cohort["arrival_date"] - pd.to_timedelta(cohort["arrival_date"].dt.weekday, unit="D")
    cohort["calendar_time"] = ((cohort["week_start"] - FIRST_COMPLETE_WEEK).dt.days // 7).astype("int16")
    cohort["is_boundary_week"] = ~cohort["week_start"].between(FIRST_COMPLETE_WEEK, LAST_COMPLETE_WEEK)
    cohort["is_transition_week"] = cohort["week_start"].eq(TRANSITION_WEEK)
    cohort["in_primary_weekly_model"] = ~(cohort["is_boundary_week"] | cohort["is_transition_week"])
    cohort["post"] = cohort["week_start"].ge(FIRST_POST_WEEK).astype("int8")
    cohort["post_time"] = np.where(cohort["post"].eq(1), cohort["calendar_time"] - 59, 0).astype("int16")

    disp = disposition_int(cohort["ptmiemrt"])
    known_disp = (
        disp.between(10, 19)
        | disp.between(20, 29)
        | disp.between(30, 39)
        | disp.between(40, 49)
        | disp.eq(88)
    )
    cohort["known_disposition"] = known_disp
    cohort["transfer_out"] = disp.between(21, 28)
    cohort["hospital_admission"] = disp.isin([31, 32, 33, 34, 38])
    cohort["icu_admission_main"] = disp.isin([32, 34])
    cohort["ed_death"] = disp.isin([41, 42, 43, 44, 45, 48])
    inpatient_result = pd.to_numeric(cohort["ptmidcrt"], errors="coerce")
    cohort["known_inpatient_result"] = inpatient_result.between(1, 6)
    cohort["index_hospital_death"] = cohort["ed_death"] | (
        cohort["hospital_admission"] & inpatient_result.eq(4)
    )
    cohort["mortality_denominator"] = known_disp & (
        ~cohort["hospital_admission"] | cohort["known_inpatient_result"]
    )

    arrival_dt = parse_datetime(cohort["ptmiindt"], cohort["ptmiintm"])
    exit_dt = parse_datetime(cohort["ptmiotdt"], cohort["ptmiottm"])
    cohort["ed_los_minutes_raw"] = (exit_dt - arrival_dt).dt.total_seconds() / 60.0
    cohort["ed_los_minutes"] = cohort["ed_los_minutes_raw"].where(
        cohort["ed_los_minutes_raw"].between(0, 7 * 24 * 60, inclusive="both")
    )

    cohort["transfer_capacity"] = disp.isin([21, 22, 23])
    cohort["transfer_specialist"] = disp.eq(24)
    destination = pd.to_numeric(cohort["ptmidctp"], errors="coerce")
    cohort["destination_known"] = cohort["transfer_out"] & destination.isin([1, 2, 3, 4, 5, 9])
    cohort["destination_tertiary"] = cohort["transfer_out"] & destination.eq(1)
    cohort["destination_general"] = cohort["transfer_out"] & destination.eq(2)
    ktas = pd.to_numeric(cohort["ptmikts1"], errors="coerce")
    cohort["ktas_valid"] = ktas.between(1, 5)
    cohort["ktas_high"] = ktas.isin([1, 2])
    cohort["ktas_low"] = ktas.isin([3, 4, 5])

    cohort.to_pickle(OUT / "cohort_all_study_dates.pkl")
    model_cohort = cohort.loc[cohort["in_primary_weekly_model"]].copy()
    model_cohort.to_pickle(OUT / "cohort_primary_weeks.pkl")

    disease_order = ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]
    all_weeks = pd.date_range(FIRST_COMPLETE_WEEK, LAST_COMPLETE_WEEK, freq="7D")
    all_weeks = all_weeks[all_weeks != TRANSITION_WEEK]
    skeleton = pd.MultiIndex.from_product(
        [disease_order, all_weeks], names=["disease", "week_start"]
    ).to_frame(index=False)
    g = model_cohort.groupby(["disease", "week_start"], observed=True)
    weekly = g.agg(
        visit_count=("disease", "size"),
        known_disposition_n=("known_disposition", "sum"),
        transfer_out_n=("transfer_out", "sum"),
        admission_n=("hospital_admission", "sum"),
        icu_main_n=("icu_admission_main", "sum"),
        mortality_denominator_n=("mortality_denominator", "sum"),
        index_hospital_death_n=("index_hospital_death", "sum"),
        los_nonmissing_n=("ed_los_minutes", "count"),
        los_median_minutes=("ed_los_minutes", "median"),
        transfer_capacity_n=("transfer_capacity", "sum"),
        transfer_specialist_n=("transfer_specialist", "sum"),
        destination_known_n=("destination_known", "sum"),
        destination_tertiary_n=("destination_tertiary", "sum"),
        destination_general_n=("destination_general", "sum"),
        ktas_valid_n=("ktas_valid", "sum"),
        ktas_high_n=("ktas_high", "sum"),
        ktas_low_n=("ktas_low", "sum"),
    ).reset_index()
    weekly = skeleton.merge(weekly, how="left", on=["disease", "week_start"])
    count_cols = [c for c in weekly.columns if c.endswith("_n") or c == "visit_count"]
    weekly[count_cols] = weekly[count_cols].fillna(0).astype(int)
    weekly["calendar_time"] = ((weekly["week_start"] - FIRST_COMPLETE_WEEK).dt.days // 7).astype(int)
    weekly["post"] = weekly["week_start"].ge(FIRST_POST_WEEK).astype(int)
    weekly["post_time"] = np.where(weekly["post"].eq(1), weekly["calendar_time"] - 59, 0).astype(int)
    weekly["sin1"] = np.sin(2 * np.pi * weekly["calendar_time"] / 52.18)
    weekly["cos1"] = np.cos(2 * np.pi * weekly["calendar_time"] / 52.18)
    weekly["transfer_rate"] = weekly["transfer_out_n"] / weekly["known_disposition_n"].replace(0, np.nan)
    weekly.to_csv(OUT / "weekly_primary_inputs.csv", index=False)

    disease_counts = cohort["disease"].value_counts().reindex(disease_order, fill_value=0)
    model_disease_counts = model_cohort["disease"].value_counts().reindex(disease_order, fill_value=0)
    flow = pd.DataFrame(
        [
            ["Customized NEDIS extract", source_n],
            ["After removal of exact duplicate rows", len(dedup)],
            ["Valid date, age >=20 years, and eligible ED level", int(eligible_base.sum())],
            ["Target principal diagnosis, mutually exclusive", len(cohort)],
            ["Complete-week ITS cohort", len(model_cohort)],
        ],
        columns=["stage", "encounters"],
    )
    flow.to_csv(OUT / "cohort_flow.csv", index=False)

    qc = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows": source_n,
        "exact_duplicates_removed": int(duplicate_mask.sum()),
        "after_deduplication": int(len(dedup)),
        "invalid_study_date": int((~valid_date).sum()),
        "age_below_20_or_invalid": int((~adult).sum()),
        "ineligible_ed_level": int((~eligible_level).sum()),
        "eligible_base": int(eligible_base.sum()),
        "outside_target_diagnosis": int((~target_any).sum()),
        "cross_stratum_conflicts_excluded": int(cross_stratum.sum()),
        "target_cohort": int(len(cohort)),
        "target_counts": {k: int(v) for k, v in disease_counts.items()},
        "principal_diagnosis_count_distribution": {
            str(k): int(v) for k, v in principal_counts.loc[cohort.index].value_counts().sort_index().items()
        },
        "boundary_week_excluded": int(cohort["is_boundary_week"].sum()),
        "transition_week_excluded": int(cohort["is_transition_week"].sum()),
        "primary_model_encounters": int(len(model_cohort)),
        "primary_model_counts": {k: int(v) for k, v in model_disease_counts.items()},
        "model_weeks_per_disease": {
            k: int(v) for k, v in weekly.groupby("disease")["week_start"].nunique().items()
        },
        "pre_weeks": int(weekly.loc[weekly["post"].eq(0), "week_start"].nunique()),
        "post_weeks": int(weekly.loc[weekly["post"].eq(1), "week_start"].nunique()),
        "zero_visit_weeks": int((weekly["visit_count"] == 0).sum()),
        "unknown_or_nonstandard_disposition_by_disease": {
            str(k): int(v)
            for k, v in model_cohort.loc[~model_cohort["known_disposition"]]
            .groupby("disease", observed=True)
            .size()
            .reindex(disease_order, fill_value=0)
            .items()
        },
        "los_negative": int((cohort["ed_los_minutes_raw"] < 0).sum()),
        "los_over_7_days": int((cohort["ed_los_minutes_raw"] > 7 * 24 * 60).sum()),
        "los_missing_or_invalid": int(cohort["ed_los_minutes"].isna().sum()),
    }
    (OUT / "cohort_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
