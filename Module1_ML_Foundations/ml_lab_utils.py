"""
ml_lab_utils.py - Shared helpers for every lab in Module 1.

Why a shared utils file?
    In real ML teams, you never copy-paste the same loading / plotting code into
    20 notebooks. You put it in ONE place, test it, and reuse it. We follow the
    same discipline here, even for teaching labs.

What lives here:
    1. Pretty console output helpers   -> banner(), step(), note(), show()
    2. Plot helper                     -> save_fig()
    3. Real dataset loader             -> load_titanic()
    4. Enterprise-style synthetic data -> make_telco_churn(), make_ecommerce_customers()

You do NOT need to understand this file before running the labs.
Treat it like a library: read it later when you are curious.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & global settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# One seed for the whole module -> every run gives the same numbers.
# (Reproducibility is a non-negotiable in enterprise ML.)
RANDOM_STATE = 42

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 3)

# Inside Jupyter / VS Code Interactive we want inline plots.
# When running as a plain script we use a non-GUI backend so nothing blocks.
IN_NOTEBOOK = "ipykernel" in sys.modules
try:
    import matplotlib

    if not IN_NOTEBOOK:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # labs still run without matplotlib, they just skip charts
    plt = None
    HAS_MPL = False


# ---------------------------------------------------------------------------
# 1. Console output helpers (make the terminal readable on video)
# ---------------------------------------------------------------------------
def banner(title: str, level: str | None = None) -> None:
    """Big section header, e.g. banner('Data Leakage', 'HARD')."""
    tag = f"[{level}] " if level else ""
    line = "=" * 78
    print(f"\n{line}\n  {tag}{title}\n{line}")


def step(title: str) -> None:
    """Smaller header for a step inside a section."""
    print(f"\n--- {title} " + "-" * max(0, 72 - len(title)))


def note(text: str, keep_lines: bool = False) -> None:
    """Explanation printed next to results (the 'why' behind the numbers).
    keep_lines=True keeps your own line breaks (for lists and tables)."""
    text = textwrap.dedent(text).strip()
    body = text if keep_lines else textwrap.fill(text, width=74)
    print(textwrap.indent(body, "  >> ", predicate=lambda _: True))


def show(label: str, value) -> None:
    """Print one labelled value, aligned."""
    if isinstance(value, float):
        value = f"{value:,.4f}"
    print(f"  {label:<38}: {value}")


# ---------------------------------------------------------------------------
# 2. Plot helper
# ---------------------------------------------------------------------------
def save_fig(fig, name: str) -> None:
    """Save a figure into outputs/ (and show it inline when in a notebook)."""
    if fig is None:
        return
    path = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  [plot saved] outputs/{path.name}")
    if not IN_NOTEBOOK:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Real dataset: Titanic (1,309 real passengers, from OpenML)
# ---------------------------------------------------------------------------
def load_titanic() -> pd.DataFrame:
    """
    Download Titanic once from OpenML, cache it in data/titanic.csv,
    and load from cache afterwards (so the lab works offline next time).
    """
    path = DATA_DIR / "titanic.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml

        print("  Downloading Titanic from OpenML (first run only)...")
        frame = fetch_openml("titanic", version=1, as_frame=True).frame
        frame.to_csv(path, index=False)
    df = pd.read_csv(path)
    df["survived"] = df["survived"].astype(int)
    return df


# ---------------------------------------------------------------------------
# 4. Enterprise synthetic data: Telecom customer churn
# ---------------------------------------------------------------------------
def make_telco_churn(n: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    A realistic telecom churn dataset where WE KNOW the true cause-and-effect.

    Why synthetic? Because for teaching leakage and causation we need ground
    truth. With real data you never know the "true" reason a customer left.

    Columns
    -------
    Legit features (known BEFORE the churn happens):
        customer_id, signup_date, region, senior_citizen, tenure_months,
        contract_type, internet_service, has_tech_support, payment_method,
        monthly_charges, total_charges, support_tickets_90d
    Target:
        churned (1 = customer left within the next 90 days)
    LEAKY features (only known AFTER churn - must never be used for training):
        cancellation_reason, days_since_last_payment, retention_offer_made
    """
    rng = np.random.default_rng(seed)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    # Long contracts -> longer tenure on average
    base_tenure = np.where(contract == "Month-to-month", 14,
                   np.where(contract == "One year", 30, 45))
    tenure = np.clip(rng.exponential(base_tenure), 1, 72).round().astype(int)

    internet = rng.choice(["DSL", "Fiber optic", "None"], size=n, p=[0.35, 0.45, 0.20])
    base_charge = np.select(
        [internet == "Fiber optic", internet == "DSL"], [85, 55], default=22
    )
    monthly = np.clip(base_charge + rng.normal(0, 12, n), 18, 120).round(2)
    tech_support = np.where(
        internet == "None", "No internet", rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
    )
    payment = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=n, p=[0.35, 0.22, 0.22, 0.21],
    )
    senior = rng.binomial(1, 0.16, n)
    region = rng.choice(["North", "South", "East", "West"], n)
    tickets = rng.poisson(np.where(internet == "Fiber optic", 1.6, 0.9))

    # ---- TRUE churn mechanism (the "law of nature" in this toy world) ----
    logit = (
        -2.0
        + 1.30 * (contract == "Month-to-month")
        - 0.60 * (contract == "Two year")
        - 0.035 * tenure
        + 0.018 * (monthly - 65)
        + 0.35 * tickets
        + 0.45 * (internet == "Fiber optic")
        - 0.70 * (tech_support == "Yes")
        + 0.45 * (payment == "Electronic check")
        + 0.30 * senior
    )
    p_churn = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, p_churn)

    total = (tenure * monthly * rng.uniform(0.95, 1.05, n)).round(2)
    signup = pd.Timestamp("2026-06-30") - pd.to_timedelta(tenure * 30, unit="D")

    # ---- LEAKY columns: generated FROM the outcome ----
    reasons = np.array(["Price", "Competitor", "Service quality", "Moved", "Other"])
    cancellation_reason = np.where(
        churned == 1, rng.choice(reasons, n, p=[0.35, 0.25, 0.2, 0.1, 0.1]), None
    )
    days_since_payment = np.where(
        churned == 1, rng.integers(35, 120, n), rng.integers(0, 32, n)
    )
    retention_offer = np.where(churned == 1, rng.binomial(1, 0.7, n), rng.binomial(1, 0.05, n))

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "signup_date": signup.normalize(),
        "region": region,
        "senior_citizen": senior,
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "has_tech_support": tech_support,
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": total,
        "support_tickets_90d": tickets,
        "cancellation_reason": cancellation_reason,
        "days_since_last_payment": days_since_payment,
        "retention_offer_made": retention_offer,
        "churned": churned,
    })
    # Real-world messiness: a few missing values
    miss = rng.random(n) < 0.02
    df.loc[miss, "total_charges"] = np.nan
    return df


TELCO_LEAKY_COLUMNS = ["cancellation_reason", "days_since_last_payment", "retention_offer_made"]
TELCO_NUMERIC = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges", "support_tickets_90d"]
TELCO_CATEGORICAL = ["region", "contract_type", "internet_service", "has_tech_support", "payment_method"]
TELCO_TARGET = "churned"


# ---------------------------------------------------------------------------
# 5. Enterprise synthetic data: E-commerce customers
# ---------------------------------------------------------------------------
def make_ecommerce_customers(n: int = 3000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    E-commerce customers with 3 hidden behaviour segments
    (Bargain hunters, Loyal regulars, Big spenders).
    Used to show ONE business framed as regression, classification and clustering.
    """
    rng = np.random.default_rng(seed)
    seg = rng.choice([0, 1, 2], size=n, p=[0.45, 0.40, 0.15])
    visits = rng.poisson(np.array([4, 12, 8])[seg]) + 1
    avg_basket = np.clip(rng.normal(np.array([25, 45, 160])[seg], np.array([8, 12, 40])[seg]), 5, None)
    discount_share = np.clip(rng.normal(np.array([0.7, 0.25, 0.1])[seg], 0.1), 0, 1)
    days_since_last = np.clip(rng.exponential(np.array([40, 10, 20])[seg]), 0, 365).round()
    returns_rate = np.clip(rng.normal(0.08, 0.05, n) + 0.08 * (seg == 2), 0, 1)

    spend_last_90d = visits * avg_basket
    next_month_spend = np.clip(
        0.30 * spend_last_90d - 1.2 * days_since_last + rng.normal(0, 40, n) + 30, 0, None
    ).round(2)
    churn_logit = -1.0 + 0.035 * days_since_last - 0.12 * visits + 1.0 * discount_share
    will_churn = rng.binomial(1, 1 / (1 + np.exp(-churn_logit)))

    return pd.DataFrame({
        "visits_90d": visits,
        "avg_basket_value": avg_basket.round(2),
        "discount_share": discount_share.round(3),
        "days_since_last_order": days_since_last,
        "returns_rate": returns_rate.round(3),
        "spend_last_90d": spend_last_90d.round(2),
        "next_month_spend": next_month_spend,  # regression target
        "will_churn": will_churn,              # classification target
        "_true_segment": seg,                  # hidden truth (only for checking clustering)
    })
