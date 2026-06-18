"""
more_3d_plots.py — Additional interactive 3D scatter plots coloured by churn.

  1. tenure_charges_3d.html      — tenure × monthly_charges × total_charges
  2. risk_combo_3d.html          — fiber optic × month-to-month × electronic check
  3. addon_tenure_charges_3d.html — addon count × tenure × monthly_charges
  4. senior_charges_contract_3d.html — senior × monthly_charges × contract type
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import plotly.express as px

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed" / "cleaned_telco_customer_churn.csv"

df = pd.read_csv(DATA).dropna(subset=["total_charges"]).fillna(0)
df["Churn"] = df["churn"].map({0: "No Churn", 1: "Churn"})

COLORS = {"No Churn": "#4393c3", "Churn": "#d6604d"}
rng = np.random.default_rng(42)


def save(fig, name: str) -> None:
    out = HERE / name
    fig.write_html(str(out))
    fig.update_traces(marker=dict(size=2.5))
    fig.write_html(str(out))
    print(f"Saved → {out}")


# ── 1. tenure × monthly_charges × total_charges ──────────────────────────────

fig1 = px.scatter_3d(
    df,
    x="tenure",
    y="monthly_charges",
    z="total_charges",
    color="Churn",
    color_discrete_map=COLORS,
    opacity=0.4,
    title=(
        "Tenure × Monthly Charges × Total Charges — coloured by Churn<br>"
        "<sup>Churn clusters at low tenure + high monthly charges</sup>"
    ),
    labels={
        "tenure": "Tenure (months)",
        "monthly_charges": "Monthly Charges ($)",
        "total_charges": "Total Charges ($)",
    },
)
fig1.update_traces(marker=dict(size=2.5))
save(fig1, "tenure_charges_3d.html")


# ── 2. High-risk combo: fiber optic × month-to-month × electronic check ───────
# Jitter binary axes so individual points are visible.

n = len(df)
df_risk = df.copy()
df_risk["x"] = df["internet_service_fiber_optic"] + rng.uniform(-0.08, 0.08, n)
df_risk["y"] = df["contract_month-to-month"]       + rng.uniform(-0.08, 0.08, n)
df_risk["z"] = df["payment_method_electronic_check"] + rng.uniform(-0.08, 0.08, n)

fig2 = px.scatter_3d(
    df_risk,
    x="x", y="y", z="z",
    color="Churn",
    color_discrete_map=COLORS,
    opacity=0.35,
    title=(
        "Fiber Optic × Month-to-Month × Electronic Check — coloured by Churn<br>"
        "<sup>Corner (1,1,1) is the highest-churn zone in the dataset</sup>"
    ),
    labels={"x": "Fiber Optic", "y": "Month-to-Month Contract", "z": "Electronic Check"},
)
fig2.update_traces(marker=dict(size=2.5))
fig2.update_layout(
    scene=dict(
        xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),
        yaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),
        zaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),
    )
)
save(fig2, "risk_combo_3d.html")


# ── 3. Add-on count × tenure × monthly_charges ───────────────────────────────

addon_cols = ["streaming_tv", "streaming_movies", "online_backup",
              "device_protection", "multiple_lines"]
df_addon = df.copy()
df_addon["addon_count"] = df[addon_cols].sum(axis=1).astype(int)
df_addon["addon_jitter"] = df_addon["addon_count"] + rng.uniform(-0.25, 0.25, n)

fig3 = px.scatter_3d(
    df_addon,
    x="addon_jitter",
    y="tenure",
    z="monthly_charges",
    color="Churn",
    color_discrete_map=COLORS,
    opacity=0.4,
    title=(
        "Add-on Service Count × Tenure × Monthly Charges — coloured by Churn<br>"
        "<sup>Low add-ons + low tenure + high charges = churn hotspot</sup>"
    ),
    labels={
        "addon_jitter": "Number of Add-on Services",
        "tenure": "Tenure (months)",
        "monthly_charges": "Monthly Charges ($)",
    },
)
fig3.update_traces(marker=dict(size=2.5))
save(fig3, "addon_tenure_charges_3d.html")


# ── 4. Senior × monthly_charges × contract type (jittered) ───────────────────

contract_map = {
    (1, 0, 0): 0,   # month-to-month
    (0, 1, 0): 1,   # one year
    (0, 0, 1): 2,   # two year
}
def contract_label(row):
    key = (int(row["contract_month-to-month"]),
           int(row["contract_one_year"]),
           int(row["contract_two_year"]))
    return contract_map.get(key, -1)

df_sc = df.copy()
df_sc["contract_num"] = df_sc.apply(contract_label, axis=1)
df_sc["x"] = df_sc["senior_citizen"]  + rng.uniform(-0.08, 0.08, n)
df_sc["z"] = df_sc["contract_num"]    + rng.uniform(-0.12, 0.12, n)

fig4 = px.scatter_3d(
    df_sc,
    x="x",
    y="monthly_charges",
    z="z",
    color="Churn",
    color_discrete_map=COLORS,
    opacity=0.4,
    title=(
        "Senior Citizen × Monthly Charges × Contract Type — coloured by Churn<br>"
        "<sup>Senior + high charges + month-to-month = elevated churn risk</sup>"
    ),
    labels={
        "x": "Senior Citizen",
        "monthly_charges": "Monthly Charges ($)",
        "z": "Contract Type",
    },
)
fig4.update_traces(marker=dict(size=2.5))
fig4.update_layout(
    scene=dict(
        xaxis=dict(tickvals=[0, 1], ticktext=["Non-Senior", "Senior"]),
        zaxis=dict(tickvals=[0, 1, 2], ticktext=["Month-to-Month", "One Year", "Two Year"]),
    )
)
save(fig4, "senior_charges_contract_3d.html")
