"""
engineer_features.py  —  Feature engineering step in the churn ML pipeline.

Cluster profile insights (cluster_profile_heatmap.png) drive every engineered
feature below.  Four behavioural segments were identified with K-Means (k=4):

  Cluster 0 – Loyal engaged     : churn=0.10, long tenure (~59 mo), 2-yr
                                   contracts, ~72% auto-pay, many add-ons.
  Cluster 1 – Phone-only        : churn=0.07, no internet, lowest charges
                                   (~$21/mo), mailed-check, often 2-yr contract.
  Cluster 2 – High-risk churners: churn=0.55, fiber optic, month-to-month,
                                   electronic-check, short tenure, minimal add-ons.
  Cluster 3 – DSL moderate-risk : churn=0.26, all DSL, mostly month-to-month,
                                   moderate service adoption.

Three separating axes informed the engineered feature set:
  1. Service engagement depth  (Cluster 0 high, Cluster 2 low, Cluster 1 zero)
  2. Contract / payment risk   (Cluster 2: month-to-month + electronic check)
  3. Charge-to-tenure balance  (Cluster 2 pays high prices early in tenure)

Outputs
-------
  data/processed/features_engineered.csv   full engineered dataset (all splits)
  data/processed/X_train.csv               80% training features
  data/processed/X_test.csv                20% test features
  data/processed/y_train.csv               training labels
  data/processed/y_test.csv                test labels

Usage
-----
  python engineer_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
DATA_IN      = ROOT / "data" / "processed" / "cleaned_telco_customer_churn.csv"
DATA_OUT     = ROOT / "data" / "engineered"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Add-on services whose adoption separates engaged (Cluster 0) from
# at-risk (Cluster 2) customers.
ADD_ON_SERVICES = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "multiple_lines",
]
# ---------------------------------------------------------------------------


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["total_charges"])
    df = df.fillna(0)
    print(f"[load] {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from K-Means cluster-profile analysis."""
    out = df.copy()

    # ── 1. Service engagement depth ─────────────────────────────────────────

    # Total number of active add-on services per customer.
    # Cluster 0 averages ~4.8; Cluster 2 averages ~1.9; Cluster 1 = 0.
    # Low counts correlate strongly with higher churn.
    out["service_count"] = out[ADD_ON_SERVICES].sum(axis=1)

    # Monthly spend divided by number of active services.
    # Cluster 2 customers pay fiber-tier prices (~$86/mo) while subscribing to
    # few services, producing a high ratio that predicts churn.
    # +1 prevents division by zero for customers with no add-ons.
    out["charge_per_service"] = out["monthly_charges"] / (out["service_count"] + 1)

    # Whether the customer uses at least one streaming service.
    # Cluster 0 (churn=0.10) has 76-77% streaming adoption vs
    # Cluster 2 (churn=0.55) at 45% and Cluster 1 (churn=0.07) at 0%.
    out["has_streaming"] = (
        (out["streaming_tv"] > 0) | (out["streaming_movies"] > 0)
    ).astype(int)

    # ── 2. Contract / payment risk ───────────────────────────────────────────

    # 1 if the customer is on a one- or two-year contract.
    # Long-term contracts are the strongest signal of commitment:
    # Cluster 0 has 92% on long contracts (churn=0.10); Cluster 2 has 3%.
    out["has_long_contract"] = (
        (out["contract_one_year"] == 1) | (out["contract_two_year"] == 1)
    ).astype(int)

    # 1 if payment is automatic (bank transfer or credit card).
    # Automatic payers rarely churn: Cluster 0 ~72%, Cluster 2 ~27%.
    # Manual payment (especially electronic check) is a friction point
    # that correlates with higher churn risk.
    out["is_auto_payment"] = (
        (out["payment_method_bank_transfer_(automatic)"] == 1)
        | (out["payment_method_credit_card_(automatic)"] == 1)
    ).astype(int)

    # Interaction: fiber optic AND month-to-month contract.
    # This is the defining combination of Cluster 2 (97% fiber, 97% m2m).
    # The heatmap shows both features in deep red for that cluster,
    # making their product a strong churn indicator.
    out["fiber_month_to_month"] = (
        out["internet_service_fiber_optic"] * out["contract_month-to-month"]
    )

    # Interaction: electronic check AND month-to-month contract.
    # Cluster 2 has 64% electronic-check usage alongside 97% m2m contracts.
    # Combined, these flags identify customers with both contract and payment
    # risk simultaneously.
    out["echeck_month_to_month"] = (
        out["payment_method_electronic_check"] * out["contract_month-to-month"]
    )

    # Fiber optic subscriber who has neither online security nor tech support.
    # Cluster 2 pays premium fiber prices (mean $86/mo) but has only 14%
    # online-security and 14% tech-support adoption. This unprotected-fiber
    # profile is a concentrated churn signal.
    out["fiber_no_protection"] = (
        out["internet_service_fiber_optic"]
        * (1 - out["online_security"].clip(0, 1))
        * (1 - out["tech_support"].clip(0, 1))
    )

    # ── 3. Charge-to-tenure balance ──────────────────────────────────────────

    # Monthly charges divided by months of tenure.
    # Cluster 2: high charges (~$86) over short tenure (~20 mo) → high ratio.
    # Cluster 0: similar charges (~$91) but over long tenure (~59 mo) → low ratio.
    # Captures the "paying a lot but hasn't stayed long" risk pattern.
    # +1 prevents division by zero for brand-new customers (tenure=0).
    out["charge_to_tenure"] = out["monthly_charges"] / (out["tenure"] + 1)

    # Effective tenure proxy: how many months of charges have accumulated.
    # total_charges / monthly_charges ≈ months paid. New (Cluster 2) customers
    # have a ratio near 1; loyal (Cluster 0) customers have ratios of 50+.
    # +1 avoids division by zero on the rare zero-charge row.
    out["total_to_monthly_ratio"] = out["total_charges"] / (out["monthly_charges"] + 1)

    # ── 4. Tenure bins (ordinal) ─────────────────────────────────────────────

    # Discretised tenure: Cluster 2 is concentrated in the 0-24 month range
    # while Cluster 0 sits predominantly in the 48+ month range.
    # Ordinal encoding (0-3) preserves the natural order for tree models.
    #   0 = new        (0-12 months)
    #   1 = developing (13-36 months)
    #   2 = established(37-60 months)
    #   3 = loyal      (>60 months)
    out["tenure_bin"] = pd.cut(
        out["tenure"],
        bins=[-1, 12, 36, 60, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    return out


def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that introduce perfect multicollinearity.

    gender_male = 1 - gender_female so carrying both inflates the feature
    space without adding information. gender_female is kept as the sole
    gender indicator.
    """
    return df.drop(columns=["gender_male"], errors="ignore")


def split_and_save(df: pd.DataFrame) -> None:
    """Stratified 80/20 train/test split; save X/y CSVs for the model step."""
    y = df["churn"]
    X = df.drop(columns=["churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,          # preserves churn ratio (~27%) in both splits
        random_state=RANDOM_STATE,
    )

    print(f"\nTrain : {len(X_train)} rows  |  churn rate {y_train.mean():.3f}")
    print(f"Test  : {len(X_test)}  rows  |  churn rate {y_test.mean():.3f}")

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(DATA_OUT / "X_train.csv", index=False)
    X_test.to_csv(DATA_OUT  / "X_test.csv",  index=False)
    y_train.to_csv(DATA_OUT / "y_train.csv", index=False)
    y_test.to_csv(DATA_OUT  / "y_test.csv",  index=False)

    print("\nSaved:")
    for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"):
        print(f"  {DATA_OUT / name}")


def main() -> None:
    df = load(DATA_IN)

    df = engineer(df)
    print(f"\n[engineer] {df.shape[1]} columns after feature engineering")

    df = drop_redundant(df)
    print(f"[drop]     {df.shape[1]} columns after removing redundant features")

    engineered_path = DATA_OUT / "features_engineered.csv"
    df.to_csv(engineered_path, index=False)
    print(f"\n[save]     Full engineered dataset -> {engineered_path}")

    split_and_save(df)


if __name__ == "__main__":
    main()
