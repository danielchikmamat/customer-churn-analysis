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
ADDON_LABELS = {
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
}

df["contract_type"] = "Month-to-Month"
df.loc[df["contract_one_year"] == 1, "contract_type"] = "Long Contract"
df.loc[df["contract_two_year"] == 1, "contract_type"] = "Long Contract"

df["churn_label"] = df["churn"].map({0: "No Churn", 1: "Churn"})

rng = np.random.default_rng(42)
df["senior_jitter"] = df["senior_citizen"] + rng.uniform(-0.15, 0.15, len(df))

# Melt: one row per customer-addon pair, only where the customer has that addon
melted = (
    df[["senior_jitter", "contract_type", "churn_label"] + ADDON_COLS]
    .melt(
        id_vars=["senior_jitter", "contract_type", "churn_label"],
        value_vars=ADDON_COLS,
        var_name="addon",
        value_name="has_addon",
    )
    .query("has_addon == 1")
)
melted["addon_label"] = melted["addon"].map(ADDON_LABELS)

COLORS = {"No Churn": "#4393c3", "Churn": "#d6604d"}

fig = go.Figure()

for churn_label, color in COLORS.items():
    sub = melted[melted["churn_label"] == churn_label]
    fig.add_trace(go.Scatter3d(
        x=sub["senior_jitter"],
        y=sub["addon_label"],
        z=sub["contract_type"],
        mode="markers",
        marker=dict(color=color, size=2.5, opacity=0.4, symbol="circle"),
        name=churn_label,
        legendgroup=churn_label,
        hovertemplate=(
            f"Churn={churn_label}<br>"
            "Senior Citizen=%{x}<br>"
            "Add-on=%{y}<br>"
            "Contract Type=%{z}<extra></extra>"
        ),
    ))

fig.update_layout(
    title="Senior Citizen vs Add-on by Contract Type<br>colored by Churn",
    scene=dict(
        xaxis=dict(title="Senior Citizen", tickvals=[0, 1], ticktext=["Non-Senior", "Senior"]),
        yaxis=dict(
            title="Add-on",
            categoryorder="array",
            categoryarray=list(ADDON_LABELS.values()),
        ),
        zaxis=dict(
            title="Contract Type",
            categoryorder="array",
            categoryarray=["Month-to-Month", "Long Contract"],
        ),
    ),
    legend=dict(title="Churn"),
    margin=dict(l=0, r=0, b=0, t=60),
)

out = HERE / "senior_each_addon_contract_3d.html"
fig.write_html(str(out))
print(f"Saved → {out}")
