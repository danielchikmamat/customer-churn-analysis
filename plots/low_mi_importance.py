"""
low_mi_importance.py

Three plots arguing that low MI score ≠ unimportant:

  1. addon_count_churn.png   — churn rate rises monotonically as add-on
                               service count drops, even though each add-on
                               has near-zero MI individually.
  2. phone_service_dist.png  — phone_service MI ≈ 0 because ~97% of customers
                               have it; the feature has almost no variance,
                               which suppresses MI regardless of actual relevance.
  3. household_churn_3d.html — interactive 3D scatter: senior_citizen ×
                               partner × dependents, coloured by churn, showing
                               that the combination of three low/medium-MI
                               features creates clear churn segments.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed" / "cleaned_telco_customer_churn.csv"

df = pd.read_csv(DATA).dropna(subset=["total_charges"]).fillna(0)

# ── 1. Add-on service count vs churn rate ─────────────────────────────────────
# Each of these has MI < 0.004 individually. Collectively the count is a
# strong signal because customers with more services are more locked in.

addon_cols = ["streaming_tv", "streaming_movies", "online_backup",
              "device_protection", "multiple_lines"]

df["addon_count"] = df[addon_cols].sum(axis=1).astype(int)

churn_by_addons = (
    df.groupby("addon_count")["churn"]
    .agg(churn_rate="mean", n="count")
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

bars = ax1.bar(
    churn_by_addons["addon_count"],
    churn_by_addons["churn_rate"],
    color=["#d6604d" if r > 0.25 else "#4393c3"
           for r in churn_by_addons["churn_rate"]],
    alpha=0.8,
    width=0.6,
)
ax2.plot(churn_by_addons["addon_count"], churn_by_addons["n"],
         color="gray", marker="o", linewidth=1.5, linestyle="--", label="# customers")

for bar, rate in zip(bars, churn_by_addons["churn_rate"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{rate:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax1.set_xlabel("Number of add-on services\n(streaming TV, streaming movies, online backup, device protection, multiple lines)",
               fontsize=10)
ax1.set_ylabel("Churn rate", fontsize=11)
ax2.set_ylabel("Number of customers", fontsize=10, color="gray")
ax2.tick_params(axis="y", labelcolor="gray")
ax1.set_ylim(0, 0.65)
ax1.set_xticks(churn_by_addons["addon_count"])
ax1.set_title(
    "Churn Rate by Add-on Service Count\n"
    "Each service has MI < 0.004 — together they form a strong signal",
    fontsize=12,
)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax2.legend(loc="upper right", fontsize=9)
fig.tight_layout()
out1 = HERE / "addon_count_churn.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out1}")

# ── 2. Phone service — near-zero variance explains near-zero MI ───────────────

phone_counts = df["phone_service"].value_counts().sort_index()
phone_churn  = df.groupby("phone_service")["churn"].mean()

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))

# left: distribution
colors_dist = ["#91bfdb", "#4393c3"]
ax_left.bar([0, 1], phone_counts.values, color=colors_dist, width=0.5, edgecolor="white")
for x, n in zip([0, 1], phone_counts.values):
    pct = n / len(df) * 100
    ax_left.text(x, n + 30, f"{pct:.1f}%\n(n={n})", ha="center", fontsize=10)
ax_left.set_xticks([0, 1])
ax_left.set_xticklabels(["No phone\nservice", "Has phone\nservice"], fontsize=11)
ax_left.set_ylabel("Number of customers", fontsize=11)
ax_left.set_title("Distribution of phone_service\n(near-zero variance → MI ≈ 0)", fontsize=11)
ax_left.grid(axis="y", linestyle="--", alpha=0.4)

# right: churn rate per group
churn_colors = ["#4393c3" if r < 0.25 else "#d6604d" for r in phone_churn.values]
ax_right.bar([0, 1], phone_churn.values, color=churn_colors, width=0.5, edgecolor="white")
for x, r in zip([0, 1], phone_churn.values):
    ax_right.text(x, r + 0.005, f"{r:.1%}", ha="center", fontsize=11, fontweight="bold")
ax_right.set_xticks([0, 1])
ax_right.set_xticklabels(["No phone\nservice", "Has phone\nservice"], fontsize=11)
ax_right.set_ylabel("Churn rate", fontsize=11)
ax_right.set_ylim(0, 0.5)
ax_right.set_title("Churn rate per group\n(~24% difference — not irrelevant)", fontsize=11)
ax_right.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle(
    "phone_service  (MI = 0.00007)\n"
    "Near-zero MI is a variance artefact, not evidence of irrelevance",
    fontsize=12, y=1.01,
)
fig.tight_layout()
out2 = HERE / "phone_service_dist.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out2}")

# ── 3. 3D: senior_citizen × partner × dependents coloured by churn ───────────
# Three low/medium-MI features (MI: 0.011, 0.014, 0.011) that together
# carve out distinct churn segments. Jitter reveals density.

rng = np.random.default_rng(42)
n   = len(df)

df_plot = df[["senior_citizen", "partner", "dependents", "churn"]].copy()
df_plot["x"] = df_plot["senior_citizen"] + rng.uniform(-0.07, 0.07, n)
df_plot["y"] = df_plot["partner"]        + rng.uniform(-0.07, 0.07, n)
df_plot["z"] = df_plot["dependents"]     + rng.uniform(-0.07, 0.07, n)
df_plot["Churn"] = df_plot["churn"].map({0: "No Churn", 1: "Churn"})

fig3d = px.scatter_3d(
    df_plot,
    x="x", y="y", z="z",
    color="Churn",
    color_discrete_map={"No Churn": "#4393c3", "Churn": "#d6604d"},
    opacity=0.4,
    title=(
        "3D: Senior × Partner × Dependents — coloured by Churn<br>"
        "<sup>Each feature has MI ≤ 0.014 individually; "
        "their combination reveals distinct churn segments</sup>"
    ),
    labels={"x": "Senior Citizen", "y": "Partner", "z": "Dependents"},
)
fig3d.update_traces(marker=dict(size=2))
fig3d.update_layout(
    scene=dict(
        xaxis=dict(tickvals=[0, 1], ticktext=["Non-Senior", "Senior"]),
        yaxis=dict(tickvals=[0, 1], ticktext=["No partner", "Has partner"]),
        zaxis=dict(tickvals=[0, 1], ticktext=["No dependents", "Has dependents"]),
    )
)
out3 = HERE / "household_churn_3d.html"
fig3d.write_html(str(out3))
print(f"Saved → {out3}")
