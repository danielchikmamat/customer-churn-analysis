import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "processed" / "cleaned_telco_customer_churn.csv"

df = pd.read_csv(DATA)

BINARY_COLS = [
    c for c in df.columns
    if c != "churn" and df[c].dropna().isin([0, 1]).all()
]

LABELS = {
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "paperless_billing": "Paperless Billing",
    "gender_female": "Gender: Female",
    "gender_male": "Gender: Male",
    "internet_service_dsl": "Internet: DSL",
    "internet_service_fiber_optic": "Internet: Fiber Optic",
    "internet_service_no": "Internet: None",
    "contract_month-to-month": "Contract: Month-to-Month",
    "contract_one_year": "Contract: One Year",
    "contract_two_year": "Contract: Two Year",
    "payment_method_bank_transfer_(automatic)": "Payment: Bank Transfer",
    "payment_method_credit_card_(automatic)": "Payment: Credit Card",
    "payment_method_electronic_check": "Payment: Electronic Check",
    "payment_method_mailed_check": "Payment: Mailed Check",
}

BAR_COLORS = ["#4393c3", "#d6604d"]

for col in BINARY_COLS:
    label = LABELS.get(col, col)

    groups = df.groupby(col, dropna=True)["churn"]
    no_rate  = groups.get_group(0).mean() * 100
    yes_rate = groups.get_group(1).mean() * 100
    no_n     = groups.get_group(0).count()
    yes_n    = groups.get_group(1).count()

    fig, ax = plt.subplots(figsize=(5, 4))

    bars = ax.bar(
        ["No", "Yes"],
        [no_n, yes_n],
        color=BAR_COLORS,
        width=0.45,
        edgecolor="white",
    )

    for bar, n, rate in zip(bars, [no_n, yes_n], [no_rate, yes_rate]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(no_n, yes_n) * 0.01,
            f"{rate:.1f}% churn",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333333",
        )

    ax.set_title(f"{label}", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Number of Customers", fontsize=10)
    ax.set_ylim(0, max(no_n, yes_n) * 1.2)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = HERE / f"{col}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)
