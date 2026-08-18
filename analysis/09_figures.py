#!/usr/bin/env python3
"""Create vector and 300-dpi manuscript figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "08_figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#7F7F7F"
LIGHT_BLUE = "#D9EEF7"
DISEASES = ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1() -> None:
    qc = json.loads((ROOT / "analysis_outputs" / "02_cohort" / "cohort_qc.json").read_text())
    fig, ax = plt.subplots(figsize=(7.1, 8.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    boxes = [
        (10.7, "Customized NEDIS extract\n309,924 source rows"),
        (8.6, "After removal of 4 exact duplicate rows\n309,920 encounters"),
        (6.35, "Valid study date, age 20 years or older,\nand eligible ED level\n262,561 encounters"),
        (3.75, "Target principal ED discharge diagnosis\nI21.x, I63.x, or I60.x-I62.x\n191,667 encounters\nAMI 44,408 | Ischemic stroke 100,014\nHemorrhagic stroke 47,245"),
        (1.05, "Complete-week interrupted time-series cohort\n189,330 encounters\n59 preinterruption and 44 postinterruption weeks\nAMI 43,877 | Ischemic stroke 98,804\nHemorrhagic stroke 46,649"),
    ]
    heights = [1.15, 1.15, 1.45, 1.8, 1.7]
    for (y, text), h in zip(boxes, heights):
        patch = FancyBboxPatch((0.4, y - h / 2), 6.2, h, boxstyle="round,pad=0.08,rounding_size=0.12", edgecolor="#333333", facecolor=LIGHT_BLUE, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(3.5, y, text, ha="center", va="center", fontsize=9.0)
    for y0, y1 in zip([10.1, 8.0, 5.6, 2.85], [9.2, 7.1, 4.75, 1.9]):
        ax.annotate("", xy=(3.5, y1), xytext=(3.5, y0), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
    side = [
        (6.35, "Excluded: 47,359\nIneligible ED level"),
        (3.75, "Excluded: 70,894\nPrincipal diagnosis\noutside target strata"),
        (1.05, "Excluded: 2,337\nBoundary weeks: 713\nTransition week: 1,624"),
    ]
    for y, text in side:
        h = 1.05 if y == 6.35 else 1.2
        patch = FancyBboxPatch((7.0, y - h / 2), 2.65, h, boxstyle="round,pad=0.07,rounding_size=0.12", edgecolor="#777777", facecolor="white", linewidth=1.0)
        ax.add_patch(patch)
        ax.text(8.325, y, text, ha="center", va="center", fontsize=7.7)
        ax.annotate("", xy=(7.0, y), xytext=(6.6, y), arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0))
    save(fig, "Figure_1_Study_cohort_flow")


def timeseries_figure(outcome: str, stem: str, ylabel: str, percent: bool = False) -> None:
    p = pd.read_csv(ROOT / "analysis_outputs" / "03_primary_its" / "primary_predictions_with_95CI.csv", parse_dates=["week_start"])
    p = p.loc[p["outcome"].eq(outcome)]
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 8.9), sharex=True)
    for i, (ax, disease) in enumerate(zip(axes, DISEASES)):
        d = p.loc[p["disease"].eq(disease)].sort_values("week_start").copy()
        scale = 100 if percent else 1
        ax.scatter(d["week_start"], scale * d["observed"], s=13, color=GREY, alpha=0.75, label="Observed", zorder=3)
        for post_value in (0, 1):
            seg = d.loc[d["post"].eq(post_value)]
            ax.fill_between(seg["week_start"], scale * seg["fitted_ci_low"], scale * seg["fitted_ci_high"], color=BLUE, alpha=0.14, linewidth=0)
            ax.plot(seg["week_start"], scale * seg["fitted"], color=BLUE, lw=1.7, label="Fitted" if post_value == 0 else None)
        post = d.loc[d["post"].eq(1)]
        ax.fill_between(post["week_start"], scale * post["counterfactual_ci_low"], scale * post["counterfactual_ci_high"], color=ORANGE, alpha=0.10, linewidth=0)
        ax.plot(post["week_start"], scale * post["counterfactual"], color=ORANGE, lw=1.7, linestyle=(0, (5, 3)), label="No-interruption counterfactual")
        ax.axvspan(pd.Timestamp("2024-02-19"), pd.Timestamp("2024-02-26"), color="#BBBBBB", alpha=0.22, linewidth=0)
        ax.axvline(pd.Timestamp("2024-02-20"), color="#222222", lw=1.0, linestyle=(0, (2, 2)))
        ax.text(0.01, 0.93, f"{chr(65+i)}  {disease}", transform=ax.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDDDDD", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8.0, handlelength=2.3)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[-1].set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"))
    axes[-1].set_xlabel("Complete calendar week")
    fig.subplots_adjust(hspace=0.18)
    save(fig, stem)


def figure4() -> None:
    d = pd.read_csv(ROOT / "analysis_outputs" / "03_primary_its" / "confirmatory_estimates.csv")
    labels = []
    for disease in DISEASES:
        labels.extend([f"{disease}\nED visits (IRR)", f"{disease}\nTransfer out (OR)"])
    order = []
    for disease in DISEASES:
        order.extend([(disease, "ED visit count"), (disease, "Transfer-out rate")])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 7.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, parameter, panel in zip(axes, ["Immediate level change", "Weekly slope change"], ["A  Immediate level change", "B  Weekly slope change"]):
        for y, (disease, outcome) in enumerate(order[::-1]):
            row = d.loc[d["disease"].eq(disease) & d["outcome"].eq(outcome) & d["parameter"].eq(parameter)].iloc[0]
            marker = "o" if outcome == "ED visit count" else "s"
            face = BLUE if row["holm_significant_0_05"] else "white"
            ax.errorbar(row["estimate"], y, xerr=[[row["estimate"]-row["ci_low"]], [row["ci_high"]-row["estimate"]]], fmt=marker, ms=6.5, mfc=face, mec=BLUE, ecolor=BLUE, elinewidth=1.4, capsize=2.5)
        ax.axvline(1.0, color="#666666", linestyle=(0, (3, 3)), lw=1.0)
        ax.grid(axis="x", color="#E0E0E0", lw=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(panel, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Effect estimate (95% CI)")
    axes[0].set_yticks(np.arange(6), labels[::-1], fontsize=8.8)
    axes[0].set_xlim(0.74, 1.52)
    axes[1].set_xlim(0.984, 1.006)
    axes[1].set_xticks([0.985, 0.990, 0.995, 1.000, 1.005])
    axes[0].plot([], [], "o", mfc="white", mec=BLUE, label="ED visit count")
    axes[0].plot([], [], "s", mfc="white", mec=BLUE, label="Transfer-out rate")
    axes[0].plot([], [], "o", mfc=BLUE, mec=BLUE, label="Holm-adjusted P<0.05")
    fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.01), fontsize=8.5)
    fig.subplots_adjust(bottom=0.12)
    save(fig, "Figure_4_Coprimary_effect_estimates")


def diagnostic_figure() -> None:
    diagnostics = json.loads((ROOT / "analysis_outputs" / "03_primary_its" / "primary_diagnostics.json").read_text())
    keys = [f"{d} | {o}" for d in DISEASES for o in ("ED visit count", "Transfer-out rate")]
    fig, axes = plt.subplots(6, 2, figsize=(7.1, 10.0), sharex=True)
    for i, key in enumerate(keys):
        for j, metric in enumerate(("acf", "pacf")):
            ax = axes[i, j]
            values = np.asarray(diagnostics[key][metric])[1:13]
            lags = np.arange(1, 13)
            ax.axhline(0, color="#555555", lw=0.7)
            ax.vlines(lags, 0, values, color=BLUE, lw=1.4)
            ax.scatter(lags, values, color=BLUE, s=10)
            bound = 1.96 / np.sqrt(103)
            ax.axhline(bound, color=ORANGE, linestyle=(0, (3, 3)), lw=0.8)
            ax.axhline(-bound, color=ORANGE, linestyle=(0, (3, 3)), lw=0.8)
            ax.set_ylim(-0.45, 0.45)
            ax.grid(axis="y", color="#E5E5E5", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(metric.upper(), fontsize=10, fontweight="bold")
        axes[i, 0].set_ylabel(key.replace(" | ", "\n"), fontsize=7.5)
    axes[-1, 0].set_xlabel("Lag (weeks)")
    axes[-1, 1].set_xlabel("Lag (weeks)")
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    save(fig, "Figure_S1_Primary_model_ACF_PACF")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure1()
    timeseries_figure("ED visit count", "Figure_2_Weekly_ED_visit_counts", "Weekly ED visits", percent=False)
    timeseries_figure("Transfer-out rate", "Figure_3_Weekly_transfer_out_rates", "Transfer-out rate (%)", percent=True)
    figure4()
    diagnostic_figure()
    manifest = {p.name: p.stat().st_size for p in sorted(OUT.iterdir()) if p.is_file()}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
