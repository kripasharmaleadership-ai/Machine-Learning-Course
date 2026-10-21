"""
ml_lab_utils.py - Shared helpers for every lab in Module 5.

What lives here:
    1. Pretty console output helpers   -> banner(), step(), note(), show()
    2. Plot helper                     -> save_fig()
    3. Real dataset loaders            -> load_adult()          (48,842 US census records, income > $50k)
                                          load_bank_marketing() (45,211 bank telemarketing calls)
    4. A 14-row LOAN TABLE             -> loan_table()          (entropy and Gini by hand)
    5. Synthetic CREDIT APPLICANTS     -> make_credit_applicants() (a KNOWN non-linear default probability:
                                          thresholds, an interaction, a noise feature with 200 levels)
    6. Synthetic BIKE RENTALS          -> make_bike_rentals()   (regression with KNOWN additive components:
                                          the ground truth for PDP and SHAP; two near-duplicate features)
    7. Synthetic MACHINE TELEMETRY     -> make_machine_telemetry() (the assignment: predictive maintenance)

Why synthetic data?
    "Did the partial-dependence plot recover the true effect of temperature?" and "how much of the
    error is bias and how much is variance?" can only be PROVEN when you know the truth.
    Labs 12 and 14 then apply everything to real data.

You do NOT need to read this file before running the labs.
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

RANDOM_STATE = 42

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 3)

IN_NOTEBOOK = "ipykernel" in sys.modules
try:
    import matplotlib

    if not IN_NOTEBOOK:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False


def optional_import(name: str):
    """Return the module if installed (xgboost, shap...), else None. Labs never require them."""
    try:
        return __import__(name)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# 1. Console output helpers
# ---------------------------------------------------------------------------
def banner(title: str, level: str | None = None) -> None:
    """Big section header, e.g. banner('Bagging vs boosting', 'HARD')."""
    tag = f"[{level}] " if level else ""
    line = "=" * 78
    print(f"\n{line}\n  {tag}{title}\n{line}")


def step(title: str) -> None:
    """Smaller header for a step inside a section."""
    print(f"\n--- {title} " + "-" * max(0, 72 - len(title)))


def note(text: str, keep_lines: bool = False) -> None:
    """Explanation printed next to results (the 'why' behind the numbers)."""
    text = textwrap.dedent(text).strip()
    body = text if keep_lines else textwrap.fill(text, width=74)
    print(textwrap.indent(body, "  >> ", predicate=lambda _: True))


def show(label: str, value) -> None:
    """Print one labelled value, aligned."""
    if isinstance(value, float):
        value = f"{value:,.4f}"
    print(f"  {label:<46}: {value}")


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
# 3. Real datasets
# ---------------------------------------------------------------------------
ADULT_TARGET = "income_gt_50k"


def load_adult() -> pd.DataFrame:
    """UCI Adult / Census Income (1994 US census): 48,842 people, 14 attributes; target: income > $50k
    (23.9%). Downloaded once from OpenML (data_id 1590), cached in data/adult.csv.
    Cleaning (documented): '?' / missing categories -> 'Unknown'; target -> 0/1; column names with '_'."""
    path = DATA_DIR / "adult.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml

        print("  Downloading Adult (census income) from OpenML (first run only)...")
        fetch_openml(data_id=1590, as_frame=True).frame.to_csv(path, index=False)
    df = pd.read_csv(path)
    df.columns = [c.replace("-", "_") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object and c != "class":
            df[c] = df[c].fillna("Unknown").astype(str).str.strip().replace({"?": "Unknown", "nan": "Unknown"})
    df[ADULT_TARGET] = (df.pop("class").astype(str).str.strip() == ">50K").astype(int)
    return df


BANK_COLUMNS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
                "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]
BANK_TARGET = "subscribed"


def load_bank_marketing() -> pd.DataFrame:
    """Bank Marketing (Moro et al., 2014): 45,211 phone calls of a Portuguese bank's term-deposit campaign;
    target: the client subscribed (11.7%). OpenML data_id 1461 (anonymous V1..V16 columns renamed here),
    cached in data/bank_marketing.csv. WARNING built into the data: `duration` (call length) is only known
    AFTER the call - the dataset's authors say it should be discarded for a realistic predictive model."""
    path = DATA_DIR / "bank_marketing.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml

        print("  Downloading Bank Marketing from OpenML (first run only)...")
        fetch_openml(data_id=1461, as_frame=True).frame.to_csv(path, index=False)
    df = pd.read_csv(path)
    df.columns = BANK_COLUMNS + ["Class"]
    df[BANK_TARGET] = (df.pop("Class").astype(str) == "2").astype(int)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    return df


# ---------------------------------------------------------------------------
# 4. A tiny table for computing entropy and Gini by hand
# ---------------------------------------------------------------------------
def loan_table() -> pd.DataFrame:
    """14 past loan applications (small enough to compute every split by hand)."""
    rows = [
        ("low", "bad", "no", "rent", 1), ("low", "bad", "yes", "rent", 1), ("low", "good", "yes", "rent", 0),
        ("low", "good", "no", "own", 1), ("medium", "bad", "no", "rent", 1), ("medium", "good", "yes", "own", 0),
        ("medium", "good", "yes", "rent", 0), ("medium", "bad", "yes", "own", 0), ("medium", "good", "no", "rent", 1),
        ("high", "good", "yes", "own", 0), ("high", "bad", "yes", "rent", 0), ("high", "good", "no", "own", 0),
        ("high", "bad", "no", "rent", 1), ("high", "good", "yes", "rent", 0),
    ]
    return pd.DataFrame(rows, columns=["income", "credit_history", "employed", "housing", "defaulted"])


# ---------------------------------------------------------------------------
# 5. Synthetic credit applicants: a KNOWN non-linear default probability
# ---------------------------------------------------------------------------
CREDIT_NUM = ["age", "income_k", "debt_ratio", "late_payments", "credit_years", "open_cards", "utilisation"]
CREDIT_CAT = ["employment", "branch_code"]


def credit_true_logit(df: pd.DataFrame) -> np.ndarray:
    """The 'law of nature' (what a perfect model would learn). Deliberately tree-friendly:
    a threshold (debt_ratio > 0.45), a saturating effect (late payments), an interaction
    (young AND high utilisation), a U-shape (age). open_cards and branch_code have NO effect."""
    emp = df["employment"].map({"salaried": 0.0, "self_employed": 0.5, "contract": 0.9, "unemployed": 1.6})
    return (-3.0 + 1.7 * (df["debt_ratio"] > 0.45) + 1.2 * np.minimum(df["late_payments"], 3) / 3 * 1.6
            - 0.06 * np.minimum(df["credit_years"], 15) - 0.6 * np.log(df["income_k"] / 50)
            + 1.4 * ((df["age"] < 30) & (df["utilisation"] > 0.7)) + 0.9 * (df["utilisation"] > 0.9)
            + 0.0006 * (df["age"] - 45) ** 2 + emp).to_numpy(dtype=float)


def make_credit_applicants(n: int = 10_000, seed: int = RANDOM_STATE, with_truth: bool = False) -> pd.DataFrame:
    """
    Credit-card applicants; target `default` (~19%). The default probability is KNOWN (credit_true_logit).
    Built-in traps for later labs:
        - open_cards has NO effect but is correlated with utilisation (a 'guilty by association' feature)
        - branch_code: 200 random branch codes, pure noise, high cardinality (fools impurity importance)
    with_truth -> adds `_true_prob`.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(42, 13, n), 18, 85).round()
    income = np.clip(rng.lognormal(np.log(48), 0.5, n) * (1 + 0.01 * (age - 40)), 8, 500).round(1)
    debt = np.clip(rng.beta(2.2, 4.2, n), 0, 1).round(3)
    late = np.minimum(rng.poisson(0.3 + 0.9 * (debt > 0.5), n), 9)
    credit_years = np.clip((age - 18) * rng.uniform(0.2, 1.0, n), 0, 50).round(1)
    util = np.clip(rng.beta(2, 3, n) + 0.25 * (debt > 0.45), 0, 1).round(3)
    cards = np.clip(np.round(1 + 10 * util + rng.normal(0, 0.7, n)), 0, 15).astype(int)
    employment = rng.choice(["salaried", "self_employed", "contract", "unemployed"], n, p=[.66, .16, .12, .06])
    branch = rng.choice([f"BR{i:03d}" for i in range(200)], n)
    df = pd.DataFrame({"age": age.astype(int), "income_k": income, "debt_ratio": debt, "late_payments": late,
                       "credit_years": credit_years, "open_cards": cards, "utilisation": util,
                       "employment": employment, "branch_code": branch})
    p = 1 / (1 + np.exp(-credit_true_logit(df)))
    df["default"] = (rng.random(n) < p).astype(int)
    if with_truth:
        df["_true_prob"] = p
    return df


# ---------------------------------------------------------------------------
# 6. Synthetic bike rentals: regression with KNOWN additive components
# ---------------------------------------------------------------------------
BIKE_BASE = 210.0
BIKE_FEATURES = ["hour", "workday", "temp_c", "feels_like_c", "humidity", "wind_kmh", "rain"]


def bike_components(df: pd.DataFrame) -> pd.DataFrame:
    """The true effect of each feature (in rentals per hour), each centred on the population later.
    rentals = 210 + hour x workday pattern + temperature hump + humidity drop + rain drop.
    feels_like_c and wind_kmh have NO true effect (feels_like_c is a near-copy of temp_c)."""
    h, w = df["hour"].to_numpy(), df["workday"].to_numpy()
    commute = 160 * np.exp(-((h - 8) ** 2) / 2) + 190 * np.exp(-((h - 17.5) ** 2) / 3)
    leisure = 140 * np.exp(-((h - 14) ** 2) / 12)
    night = -70 * ((h < 6) | (h > 22))
    hour_eff = np.where(w == 1, commute, leisure) + night
    t = df["temp_c"].to_numpy()
    temp_eff = 110 * np.exp(-((t - 24) ** 2) / 90) - 40
    hum = df["humidity"].to_numpy()
    hum_eff = -1.8 * np.maximum(hum - 65, 0)
    rain_eff = -70 * df["rain"].to_numpy()
    return pd.DataFrame({"hour_x_workday": hour_eff, "temp_c": temp_eff, "humidity": hum_eff, "rain": rain_eff},
                        index=df.index)


def make_bike_rentals(n: int = 8000, seed: int = RANDOM_STATE, with_truth: bool = False) -> pd.DataFrame:
    """Hourly bike rentals in a fictional city (target `rentals`). The data-generating formula is
    additive in known components (bike_components) + Gaussian noise (sd 25).
    with_truth -> adds `_true_rentals` (noise-free) and one `_true_<component>` column per component."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    workday = (rng.random(n) < 0.7).astype(int)
    season_temp = rng.choice([4.0, 13.0, 23.0, 15.0], n)
    temp = np.clip(season_temp + rng.normal(0, 5, n) + 3 * np.sin((hour - 9) / 24 * 2 * np.pi), -8, 38).round(1)
    feels = (temp + rng.normal(0, 1.2, n) - 0.08 * np.maximum(temp - 25, 0) ** 2 / 3).round(1)
    humidity = np.clip(rng.normal(62, 16, n) - 0.6 * (temp - 15), 15, 100).round()
    wind = np.clip(rng.gamma(2, 6, n), 0, 60).round(1)
    rain = (rng.random(n) < 0.08 + 0.004 * np.maximum(humidity - 60, 0)).astype(int)
    df = pd.DataFrame({"hour": hour, "workday": workday, "temp_c": temp, "feels_like_c": feels,
                       "humidity": humidity, "wind_kmh": wind, "rain": rain})
    comp = bike_components(df)
    true = BIKE_BASE + comp.sum(axis=1).to_numpy()
    df["rentals"] = np.clip(true + rng.normal(0, 25, n), 0, None).round()
    if with_truth:
        df["_true_rentals"] = true
        for c in comp.columns:
            df[f"_true_{c}"] = comp[c].to_numpy()
    return df


# ---------------------------------------------------------------------------
# 7. Assignment dataset: predictive maintenance (machine failure in the next 7 days)
# ---------------------------------------------------------------------------
MAINT_TARGET = "failure_next_7d"


def make_machine_telemetry(n_machines: int = 400, n_weeks: int = 40, seed: int = 5, robustness_sd: float = 1.4) -> pd.DataFrame:
    """
    Weekly telemetry snapshots of industrial pumps in 4 plants. Target: the pump fails in the next 7 days.
    Mechanism (known): wear grows with days since service; high vibration AND high temperature together
    are dangerous; type C pumps are fragile; plant P4 runs hotter. Machines differ in a hidden robustness.

    Traps for the assignment:
        - every machine appears in many weeks: a random row split leaks the machine's identity
          (the starter's test set is made of UNSEEN machines; CV must group by machine_id)
        - `downtime_hours_next_7d` is recorded AFTER the week: LEAKY
        - `machine_id` itself is a high-cardinality identifier
    robustness_sd: spread of the hidden per-machine robustness (larger = more 'repeat offender' pumps).
    """
    rng = np.random.default_rng(seed)
    machines = pd.DataFrame({
        "machine_id": [f"PUMP-{i:04d}" for i in range(n_machines)],
        "plant": rng.choice(["P1", "P2", "P3", "P4"], n_machines, p=[.3, .3, .25, .15]),
        "pump_type": rng.choice(["A", "B", "C"], n_machines, p=[.5, .35, .15]),
        "install_age_years": rng.uniform(0.5, 15, n_machines).round(1),
        "_robustness": rng.normal(0, robustness_sd, n_machines),
    })
    rows = []
    for _, m in machines.iterrows():
        since_service = rng.integers(0, 120)
        for wk in range(n_weeks):
            since_service += 7
            if rng.random() < 0.06 or since_service > 300:
                since_service = rng.integers(0, 7)                            # scheduled service
            hotter = 6 if m.plant == "P4" else 0
            temp = rng.normal(62 + hotter + 0.03 * since_service, 6)
            vib = np.abs(rng.normal(2.2 + 0.009 * since_service + 0.4 * (m.pump_type == "C"), 0.9))
            pressure = rng.normal(5.5, 0.8)
            rpm = rng.normal(1450, 120)
            load = np.clip(rng.beta(4, 3), 0, 1)
            logit = (-6.1 + 0.016 * since_service + 2.4 * ((vib > 3.6) & (temp > 70)) + 1.0 * (vib > 4.2)
                     + 0.8 * (m.pump_type == "C") + 0.06 * m.install_age_years + 1.5 * (load > 0.85)
                     + 1.0 * (pressure < 4.3) - m._robustness)
            fail = rng.random() < 1 / (1 + np.exp(-logit))
            rows.append({"machine_id": m.machine_id, "week": wk, "plant": m.plant, "pump_type": m.pump_type,
                         "install_age_years": m.install_age_years, "days_since_service": int(since_service),
                         "temperature_c": round(temp, 1), "vibration_mm_s": round(vib, 2),
                         "pressure_bar": round(pressure, 2), "rpm": round(rpm), "load_factor": round(load, 3),
                         "downtime_hours_next_7d": round(float(fail) * rng.uniform(4, 40) + rng.exponential(0.3), 1),
                         MAINT_TARGET: int(fail)})
            if fail:
                since_service = 0                                              # repaired after failure
    return pd.DataFrame(rows)
