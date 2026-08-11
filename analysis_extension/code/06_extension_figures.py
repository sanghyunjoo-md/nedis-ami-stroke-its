#!/usr/bin/env python3
"""Generate manuscript-facing forest plots for the analysis extension."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT


FIGURE_OUT = OUT / "06_figures"


def build_forest(central: pd.DataFrame, domain_name: str, stem: str) -> None:
    data = central.loc[
        central["domain"].eq(domain_name)
        & central["parameter"].isin(["Immediate level change", "Weekly slope change"])
    ].copy()
    disease_short = {
        "AMI": "AMI",
        "Ischemic stroke": "Ischemic",
        "Hemorrhagic stroke": "Hemorrhagic",
    }
    outcomes = data["outcome"].drop_duplicates().tolist()
    ordered = []
    for outcome in outcomes:
        for disease in ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]:
            row = data.loc[
                data["outcome"].eq(outcome) & data["disease"].eq(disease)
            ].iloc[0]
            ordered.append((outcome, disease, row))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, max(5, len(ordered) * 0.32 + 1.5)),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    y = np.arange(len(ordered))[::-1]
    labels = [
        f"{outcome}\n{disease_short[disease]}"
        for outcome, disease, _ in ordered
    ]
    for axis, parameter, title in zip(
        axes,
        ["Immediate level change", "Weekly slope change"],
        ["A  Immediate level change", "B  Weekly slope change"],
    ):
        subset = [
            central.loc[
                central["outcome"].eq(outcome)
                & central["disease"].eq(disease)
                & central["parameter"].eq(parameter)
            ].iloc[0]
            for outcome, disease, _ in ordered
        ]
        estimates = np.array([row.estimate for row in subset])
        lower = np.array([row.ci_low for row in subset])
        upper = np.array([row.ci_high for row in subset])
        signals = np.array([row.q_domain < 0.05 for row in subset])
        for index in range(len(y)):
            axis.errorbar(
                estimates[index],
                y[index],
                xerr=[
                    [estimates[index] - lower[index]],
                    [upper[index] - estimates[index]],
                ],
                fmt="o",
                color="#0072B2" if signals[index] else "white",
                markeredgecolor="#0072B2",
                ecolor="#0072B2",
                capsize=3,
                markersize=6,
            )
        axis.axvline(1, color="#555555", linestyle="--", linewidth=1)
        if parameter == "Immediate level change":
            axis.set_xscale("log")
            if domain_name == "Presenting severity":
                axis.set_xlim(0.65, 1.35)
                ticks = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
            else:
                axis.set_xlim(0.55, 6.5)
                ticks = [0.6, 0.8, 1.0, 2.0, 4.0, 6.0]
            axis.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
        else:
            if domain_name == "Presenting severity":
                axis.set_xlim(0.990, 1.011)
                ticks = [0.990, 0.995, 1.000, 1.005, 1.010]
            else:
                axis.set_xlim(0.992, 1.021)
                ticks = [0.992, 0.999, 1.006, 1.013, 1.020]
            axis.set_xticks(ticks, [f"{tick:.3f}" for tick in ticks])
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#E0E0E0", linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        suffix = ", log scale" if parameter == "Immediate level change" else ""
        axis.set_xlabel(f"Effect estimate (95% CI){suffix}")
    axes[0].set_yticks(y, labels)
    axes[1].set_yticks(y, [""] * len(y))
    fig.suptitle(domain_name, x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURE_OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    central = pd.read_csv(OUT / "02_models" / "central_extension_estimates.csv")
    build_forest(
        central,
        "Presenting severity",
        "Figure_Extension_Severity_Forest",
    )
    build_forest(
        central,
        "Care pathways",
        "Figure_5_Care_Pathway_Estimates",
    )


if __name__ == "__main__":
    main()
