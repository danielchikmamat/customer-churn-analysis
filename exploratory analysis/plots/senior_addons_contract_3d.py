import pathlib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "processed" / "cleaned_telco_customer_churn.csv"

df = pd.read_csv(DATA).dropna(subset=["total_charges"])

ADDON_COLS = [
    "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
]
df["addon_count"] = df[ADDON_COLS].fillna(0).sum(axis=1)

df["contract_type"] = "Month-to-Month"
df.loc[df["contract_one_year"] == 1, "contract_type"] = "Long Contract"
df.loc[df["contract_two_year"] == 1, "contract_type"] = "Long Contract"

rng = np.random.default_rng(42)
df["senior_jitter"] = df["senior_citizen"] + rng.uniform(-0.15, 0.15, len(df))

COLORS = {"No Churn": "#4393c3", "Churn": "#d6604d"}
df["churn_label"] = df["churn"].map({0: "No Churn", 1: "Churn"})

fig = go.Figure()

for churn_label, color in COLORS.items():
    mask = df["churn_label"] == churn_label
    sub = df[mask]
    fig.add_trace(go.Scatter3d(
        x=sub["senior_jitter"],
        y=sub["addon_count"],
        z=sub["contract_type"],
        mode="markers",
        marker=dict(color=color, size=2.5, opacity=0.4, symbol="circle"),
        name=churn_label,
        legendgroup=churn_label,
        hovertemplate=(
            f"Churn={churn_label}<br>"
            "Senior Citizen=%{x}<br>"
            "Number of Add-ons=%{y}<br>"
            "Contract Type=%{z}<extra></extra>"
        ),
    ))

fig.update_layout(
    title="Senior Citizen vs Add-on Count by Contract Type<br>colored by Churn",
    scene=dict(
        xaxis=dict(title="Senior Citizen", tickvals=[0, 1], ticktext=["Non-Senior", "Senior"]),
        yaxis=dict(title="Number of Add-ons"),
        zaxis=dict(title="Contract Type", categoryorder="array",
                   categoryarray=["Month-to-Month", "Long Contract"]),
    ),
    legend=dict(title="Churn"),
    margin=dict(l=0, r=0, b=0, t=60),
)

out = HERE / "senior_addons_contract_3d.html"
fig.write_html(str(out))
print(f"Saved → {out}")
