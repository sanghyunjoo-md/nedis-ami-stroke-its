#!/usr/bin/env python3
"""Run the locked case-mix, ED-level, ICU-definition, and regional analyses."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from model_utils import (
    effect_row,
    fit_grouped_binomial_hac,
    fit_nb2_hac,
)


ROOT = Path(__file__).resolve().parents[1]
COHORT_FILE = ROOT / "analysis_outputs" / "02_cohort" / "cohort_primary_weeks.pkl"
OUT = ROOT / "analysis_outputs" / "06_adjusted_stratified"
DISEASES = ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]


def weekly_design(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(d)), d["calendar_time"], d["post"], d["post_time"], d["sin1"], d["cos1"]]
    ).astype(float)


def timing_columns(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["calendar_time"] = ((x["week_start"] - pd.Timestamp("2023-01-02")).dt.days // 7).astype(int)
    x["post"] = x["week_start"].ge("2024-02-26").astype(int)
    x["post_time"] = np.where(x["post"].eq(1), x["calendar_time"] - 59, 0).astype(int)
    x["sin1"] = np.sin(2 * np.pi * x["calendar_time"] / 52.18)
    x["cos1"] = np.cos(2 * np.pi * x["calendar_time"] / 52.18)
    return x


def cluster_score_meat(x: sparse.csr_matrix, resid: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(codes, sort=False)
    g = len(uniques)
    membership = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(g, len(codes)))
    score_rows = x.multiply(resid[:, None])
    cluster_scores = membership @ score_rows
    meat = (cluster_scores.T @ cluster_scores).toarray()
    return meat, g


def twoway_cluster_cov(x: sparse.csr_matrix, y: np.ndarray, pfit: np.ndarray, facility: np.ndarray, week: np.ndarray) -> tuple[np.ndarray, dict]:
    n, p = x.shape
    w = pfit * (1 - pfit)
    info = (x.T @ x.multiply(w[:, None])).toarray()
    bread = np.linalg.pinv(info, rcond=1e-10)
    resid = y - pfit
    mf, gf = cluster_score_meat(x, resid, facility)
    mw, gw = cluster_score_meat(x, resid, week)
    intersection = np.array([f"{a}|{b}" for a, b in zip(facility, week)], dtype=object)
    mi, gi = cluster_score_meat(x, resid, intersection)
    common = (n - 1) / max(n - p, 1)
    cf = (gf / (gf - 1)) * common if gf > 1 else 1.0
    cw = (gw / (gw - 1)) * common if gw > 1 else 1.0
    ci = (gi / (gi - 1)) * common if gi > 1 else 1.0
    meat = cf * mf + cw * mw - ci * mi
    cov = bread @ meat @ bread.T
    cov = (cov + cov.T) / 2.0
    eig = np.linalg.eigvalsh(cov)
    return cov, {
        "facility_clusters": int(gf),
        "week_clusters": int(gw),
        "facility_week_clusters": int(gi),
        "design_columns": int(p),
        "information_rank": int(np.linalg.matrix_rank(info, tol=1e-8)),
        "information_condition_number": float(np.linalg.cond(info)),
        "covariance_min_eigenvalue": float(eig.min()),
    }


def case_mix_models(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    diagnostics = {}
    for disease in DISEASES:
        d = cohort.loc[cohort["disease"].eq(disease) & cohort["known_disposition"]].copy()
        d = d.sort_values(["week_start", "ptmiemnm"]).reset_index(drop=True)
        d["ktas_group"] = np.select(
            [d["ptmikts1"].isin(["1", "2"]), d["ptmikts1"].isin(["3", "4", "5"])],
            ["KTAS1-2", "KTAS3-5"],
            default="Unknown",
        )
        d["age_group_cat"] = d["ptmibrtd"].where(d["ptmibrtd"].ne(""), "Unknown")
        d["sex_cat"] = d["ptmisexx"].where(d["ptmisexx"].isin(["M", "F"]), "Unknown")
        d["ed_level"] = d["ptmiemcl"].where(d["ptmiemcl"].isin(["A", "C"]), "Unknown")
        base = np.column_stack(
            [
                np.ones(len(d)),
                d["calendar_time"].to_numpy(float) / 100.0,
                d["post"].to_numpy(float),
                d["post_time"].to_numpy(float) / 50.0,
                np.sin(2 * np.pi * d["calendar_time"].to_numpy(float) / 52.18),
                np.cos(2 * np.pi * d["calendar_time"].to_numpy(float) / 52.18),
                d["ptmiinmn"].eq("1").to_numpy(float),
                d["ptmiinrt"].eq("2").to_numpy(float),
            ]
        )
        categorical = d[["age_group_cat", "sex_cat", "ktas_group", "ed_level", "ptmiemnm"]].astype(str)
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True, dtype=float)
        cat_x = encoder.fit_transform(categorical)
        x = sparse.hstack([sparse.csr_matrix(base), cat_x], format="csr")
        y = d["transfer_out"].to_numpy(int)
        model = LogisticRegression(
            C=np.inf,
            solver="newton-cholesky",
            fit_intercept=False,
            max_iter=1000,
            tol=1e-10,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        beta_scaled = model.coef_.ravel()
        pfit = model.predict_proba(x)[:, 1]
        cov_scaled, diag = twoway_cluster_cov(
            x,
            y,
            pfit,
            d["ptmiemnm"].to_numpy(str),
            d["week_start"].dt.strftime("%Y-%m-%d").to_numpy(str),
        )
        transform = np.ones(x.shape[1])
        transform[1] = 1.0 / 100.0
        transform[3] = 1.0 / 50.0
        beta = transform * beta_scaled
        cov = transform[:, None] * cov_scaled * transform[None, :]
        convergence_messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        diag.update(
            {
                "encounters": int(len(d)),
                "events": int(y.sum()),
                "facilities": int(d["ptmiemnm"].nunique()),
                "iterations": int(model.n_iter_[0]),
                "convergence_warnings": convergence_messages,
                "target_variances": {"post": float(cov[2, 2]), "post_time": float(cov[3, 3])},
            }
        )
        diagnostics[disease] = diag
        for parameter, idx in (("Adjusted immediate level change", 2), ("Adjusted weekly slope change", 3)):
            rows.append(
                {
                    "disease": disease,
                    "parameter": parameter,
                    "effect_scale": "OR",
                    **effect_row(beta, cov, idx, "exp"),
                }
            )
    return pd.DataFrame(rows), diagnostics


def ed_level_models(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joint_rows = []
    separate_rows = []
    weeks = pd.date_range("2023-01-02", "2024-12-23", freq="7D")
    weeks = weeks[weeks != pd.Timestamp("2024-02-19")]
    for disease in DISEASES:
        cd = cohort.loc[cohort["disease"].eq(disease)]
        skeleton = pd.MultiIndex.from_product([[disease], ["A", "C"], weeks], names=["disease", "ptmiemcl", "week_start"]).to_frame(index=False)
        w = cd.groupby(["disease", "ptmiemcl", "week_start"], observed=True).agg(
            visit_count=("disease", "size"),
            known_disposition_n=("known_disposition", "sum"),
            transfer_out_n=("transfer_out", "sum"),
        ).reset_index()
        w = skeleton.merge(w, how="left", on=["disease", "ptmiemcl", "week_start"])
        for c in ["visit_count", "known_disposition_n", "transfer_out_n"]:
            w[c] = w[c].fillna(0).astype(int)
        w = timing_columns(w).sort_values(["week_start", "ptmiemcl"]).reset_index(drop=True)
        level_c = w["ptmiemcl"].eq("C").to_numpy(float)
        x = np.column_stack(
            [
                np.ones(len(w)), w["calendar_time"], w["post"], w["post_time"], w["sin1"], w["cos1"],
                level_c, level_c * w["calendar_time"], level_c * w["post"], level_c * w["post_time"],
            ]
        ).astype(float)
        week_codes = w["week_start"].dt.strftime("%Y-%m-%d").to_numpy()
        for outcome in ("ED visit count", "Transfer-out rate"):
            if outcome == "ED visit count":
                y = w["visit_count"].to_numpy(float)
                fit = fit_nb2_hac(y, x, lag=4)
            else:
                y = w["transfer_out_n"].to_numpy(float)
                n = w["known_disposition_n"].to_numpy(float)
                fit = fit_grouped_binomial_hac(y, n, x, lag=4)
            codes, groups = pd.factorize(week_codes)
            cluster_scores = np.vstack([fit.score_obs[codes == g].sum(axis=0) for g in range(len(groups))])
            correction = len(groups) / (len(groups) - 1) * (len(w) - 1) / (len(w) - x.shape[1])
            meat = correction * (cluster_scores.T @ cluster_scores)
            cov = fit.cov_model @ meat @ fit.cov_model.T
            for level, post_contrast, slope_contrast in (
                ("A", {2: 1.0}, {3: 1.0}),
                ("C", {2: 1.0, 8: 1.0}, {3: 1.0, 9: 1.0}),
            ):
                for parameter, contrast_map in (("Immediate level change", post_contrast), ("Weekly slope change", slope_contrast)):
                    contrast = np.zeros(x.shape[1])
                    for k, v in contrast_map.items():
                        contrast[k] = v
                    est = float(contrast @ fit.params)
                    se = float(np.sqrt(max(contrast @ cov @ contrast, 0.0)))
                    joint_rows.append(
                        {
                            "disease": disease,
                            "outcome": outcome,
                            "ed_level": level,
                            "parameter": parameter,
                            "estimate": float(np.exp(est)),
                            "ci_low": float(np.exp(est - 1.96 * se)),
                            "ci_high": float(np.exp(est + 1.96 * se)),
                        }
                    )
            for interaction, idx in (("Level C x immediate", 8), ("Level C x weekly slope", 9)):
                joint_rows.append(
                    {
                        "disease": disease,
                        "outcome": outcome,
                        "ed_level": "Interaction",
                        "parameter": interaction,
                        **effect_row(fit.params, cov, idx, "exp"),
                    }
                )

            for level in ("A", "C"):
                dl = w.loc[w["ptmiemcl"].eq(level)].sort_values("week_start").reset_index(drop=True)
                xl = weekly_design(dl)
                if outcome == "ED visit count":
                    fl = fit_nb2_hac(dl["visit_count"].to_numpy(float), xl, lag=4)
                else:
                    fl = fit_grouped_binomial_hac(
                        dl["transfer_out_n"].to_numpy(float), dl["known_disposition_n"].to_numpy(float), xl, lag=4
                    )
                for parameter, idx in (("Immediate level change", 2), ("Weekly slope change", 3)):
                    separate_rows.append(
                        {
                            "disease": disease,
                            "outcome": outcome,
                            "ed_level": level,
                            "parameter": parameter,
                            **effect_row(fl.params, fl.cov_hac, idx, "exp"),
                        }
                    )
    return pd.DataFrame(joint_rows), pd.DataFrame(separate_rows)


def icu_sensitivity(cohort: pd.DataFrame) -> pd.DataFrame:
    c = cohort.copy()
    c["icu_route_sensitivity"] = c["hospital_admission"] & c["ptmihsrt"].isin(["21", "22", "23", "24", "25", "26", "28"])
    rows = []
    for disease in DISEASES:
        d = c.loc[c["disease"].eq(disease)]
        w = d.groupby("week_start", observed=True).agg(
            icu_n=("icu_route_sensitivity", "sum"), admission_n=("hospital_admission", "sum")
        ).reset_index()
        w = timing_columns(w).sort_values("week_start")
        x = weekly_design(w)
        fit = fit_grouped_binomial_hac(w["icu_n"].to_numpy(float), w["admission_n"].to_numpy(float), x, lag=4)
        for parameter, idx in (("Immediate level change", 2), ("Weekly slope change", 3)):
            rows.append({"disease": disease, "parameter": parameter, **effect_row(fit.params, fit.cov_hac, idx, "exp")})
    return pd.DataFrame(rows)


def region_descriptive(cohort: pd.DataFrame) -> pd.DataFrame:
    region_map = {
        "11": "Seoul", "26": "Busan", "27": "Daegu", "28": "Incheon", "29": "Gwangju",
        "30": "Daejeon", "31": "Ulsan", "36": "Sejong", "41": "Gyeonggi", "42": "Gangwon",
        "51": "Gangwon", "43": "Chungbuk", "44": "Chungnam", "45": "Jeonbuk", "52": "Jeonbuk",
        "46": "Jeonnam", "47": "Gyeongbuk", "48": "Gyeongnam", "50": "Jeju",
    }
    c = cohort.copy()
    c["region"] = c["ptmiemar"].str[:2].map(region_map).fillna("Unknown")
    c["period"] = np.where(c["post"].eq(1), "Postinterruption", "Preinterruption")
    out = c.groupby(["disease", "region", "period"], observed=True).agg(
        encounters=("disease", "size"),
        known_disposition_n=("known_disposition", "sum"),
        transfer_out_n=("transfer_out", "sum"),
    ).reset_index()
    out["transfer_rate"] = out["transfer_out_n"] / out["known_disposition_n"]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_pickle(COHORT_FILE)
    adjusted, adjusted_diag = case_mix_models(cohort)
    joint, separate = ed_level_models(cohort)
    icu = icu_sensitivity(cohort)
    region = region_descriptive(cohort)
    adjusted.to_csv(OUT / "case_mix_adjusted_transfer_estimates.csv", index=False)
    (OUT / "case_mix_adjusted_diagnostics.json").write_text(json.dumps(adjusted_diag, ensure_ascii=False, indent=2), encoding="utf-8")
    joint.to_csv(OUT / "ed_level_joint_clustered_estimates.csv", index=False)
    separate.to_csv(OUT / "ed_level_separate_hac_estimates.csv", index=False)
    icu.to_csv(OUT / "icu_definition_sensitivity.csv", index=False)
    region.to_csv(OUT / "regional_descriptive_summary.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adjusted_models": 3,
        "adjusted_all_converged": all(not v["convergence_warnings"] for v in adjusted_diag.values()),
        "ed_level_joint_rows": int(len(joint)),
        "ed_level_separate_rows": int(len(separate)),
        "icu_sensitivity_rows": int(len(icu)),
        "regional_rows": int(len(region)),
    }
    (OUT / "adjusted_stratified_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(adjusted_diag, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
