"""
ml_lab_utils.py - Shared helpers for every lab in Module 3.

What lives here:
    1. Pretty console output helpers   -> banner(), step(), note(), show()
    2. Plot helper                     -> save_fig()
    3. Real dataset loaders            -> load_medical_costs()  (1,338 real insurance patients)
                                          load_ames()           (1,460 real houses, used in Lab 11)
    4. Synthetic HOUSE SALES           -> make_house_sales()   (a known pricing formula)
    5. Synthetic ENERGY DEMAND         -> make_energy_demand() (a known U-shaped curve, for
                                          bias-variance experiments: we can draw new samples forever)
    6. Synthetic DELIVERY ORDERS       -> make_delivery_orders() (the assignment dataset)

Why synthetic data?
    "Did the model recover the true coefficient?", "How much of the error is bias and
    how much is variance?" can only be PROVEN when you know the true formula. Real data
    never tells you that. Labs 09 and 11 then apply everything to real data.

You do NOT need to read this file before running the labs.
"""
from __future__ import annotations

import shutil
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

pd.set_option("display.width", 150)
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


# ---------------------------------------------------------------------------
# 1. Console output helpers
# ---------------------------------------------------------------------------
def banner(title: str, level: str | None = None) -> None:
    """Big section header, e.g. banner('Gradient descent', 'HARD')."""
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
    print(f"  {label:<44}: {value}")


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
MEDICAL_TARGET = "charges"


def load_medical_costs() -> pd.DataFrame:
    """Medical Cost Personal dataset (US insurance, 1,338 patients): age, sex, bmi,
    children, smoker, region -> yearly charges billed by health insurance (USD).
    Downloaded once from OpenML (data_id 43463) and cached in data/medical_costs.csv."""
    path = DATA_DIR / "medical_costs.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml

        print("  Downloading Medical Cost dataset from OpenML (first run only)...")
        frame = fetch_openml(data_id=43463, as_frame=True).frame
        frame.to_csv(path, index=False)
    df = pd.read_csv(path)
    for c in ["sex", "smoker", "region"]:
        df[c] = df[c].astype(str)
    return df


def load_ames() -> pd.DataFrame:
    """Ames Housing (De Cock, 2011): 1,460 houses, 80 columns, target SalePrice.
    Re-uses Module 2's cached copy when it exists, otherwise downloads from OpenML.
    Only truly empty cells are missing ("None" is a real category, see Module 2 Lab 10)."""
    path = DATA_DIR / "ames_housing.csv"
    if not path.exists():
        sibling = ROOT.parent / "Module2_Data_Preparation_Feature_Engineering" / "data" / "ames_housing.csv"
        if sibling.exists():
            shutil.copy(sibling, path)
        else:
            from sklearn.datasets import fetch_openml

            print("  Downloading Ames Housing from OpenML (first run only)...")
            fetch_openml(name="house_prices", as_frame=True).frame.to_csv(path, index=False)
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


# ---------------------------------------------------------------------------
# 4. Synthetic house sales with a KNOWN pricing formula
# ---------------------------------------------------------------------------
NEIGHBORHOODS = {"Riverside": 0, "Oakwood": 18_000, "Hillcrest": 42_000, "Downtown": 65_000,
                 "Lakeside": 95_000}

# The "law of nature" in this world: price = intercept + sum(coef * feature) + neighbourhood + noise
HOUSE_TRUTH = {"intercept": 40_000, "sqft": 160, "bedrooms": -9_000, "bathrooms": 12_000,
               "age": -1_100, "garage_spaces": 15_000, "distance_km": -4_000}
HOUSE_NOISE_SD = 25_000
HOUSE_FEATURES = ["sqft", "bedrooms", "bathrooms", "age", "garage_spaces", "distance_km"]


def make_house_sales(n: int = 2000, seed: int = RANDOM_STATE, with_truth: bool = False) -> pd.DataFrame:
    """
    House sales in a fictional city. Price is generated by a KNOWN linear formula
    (HOUSE_TRUTH + NEIGHBORHOODS premium + Gaussian noise with sd = HOUSE_NOISE_SD).

    Built-in realism (and traps):
        - bedrooms are strongly correlated with sqft: big houses have more bedrooms.
          On its own, "bedrooms" looks like it ADDS value; holding sqft fixed, an extra
          bedroom (= smaller rooms) REMOVES value (true coefficient -9,000).
        - expensive neighbourhoods are closer to the centre (distance is confounded).
    with_truth -> adds `_true_price` (the noise-free price; only possible with synthetic data).
    """
    rng = np.random.default_rng(seed)
    hood = rng.choice(list(NEIGHBORHOODS), n, p=[.28, .26, .2, .14, .12])
    hood_rank = pd.Series(hood).map({h: i for i, h in enumerate(NEIGHBORHOODS)}).to_numpy()
    sqft = np.clip(rng.lognormal(np.log(1700), 0.33, n) * (1 + 0.04 * hood_rank), 550, 5200).round(-1)
    bedrooms = np.clip(np.round(sqft / 620 + rng.normal(0, 0.55, n)), 1, 7).astype(int)
    bathrooms = np.clip(np.round((sqft / 950 + rng.normal(0, 0.35, n)) * 2) / 2, 1, 5)
    age = rng.integers(0, 80, n)
    garage = rng.choice([0, 1, 2, 3], n, p=[.15, .35, .42, .08])
    distance = np.clip(rng.gamma(2.2, 3.2, n) * (1.25 - 0.18 * hood_rank), 0.3, 40).round(1)

    t = HOUSE_TRUTH
    true_price = (t["intercept"] + t["sqft"] * sqft + t["bedrooms"] * bedrooms + t["bathrooms"] * bathrooms
                  + t["age"] * age + t["garage_spaces"] * garage + t["distance_km"] * distance
                  + pd.Series(hood).map(NEIGHBORHOODS).to_numpy())
    price = (true_price + rng.normal(0, HOUSE_NOISE_SD, n)).round(-2)

    df = pd.DataFrame({"sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms, "age": age,
                       "garage_spaces": garage, "distance_km": distance, "neighborhood": hood,
                       "price": price})
    if with_truth:
        df["_true_price"] = true_price
    return df


# ---------------------------------------------------------------------------
# 5. Synthetic energy demand: a KNOWN non-linear curve (bias-variance lab)
# ---------------------------------------------------------------------------
ENERGY_NOISE_SD = 12.0


def energy_true_curve(temp_c: np.ndarray) -> np.ndarray:
    """Daily electricity demand (MWh) of a small utility as a function of outdoor temperature.
    U-shaped: heating on cold days, air-conditioning on hot days, a comfort plateau at ~18 C."""
    t = np.asarray(temp_c, dtype=float)
    return 260 + 0.38 * (t - 17) ** 2 + 9 * np.sin(t / 3.0)


def make_energy_demand(n: int = 60, seed: int = RANDOM_STATE, noise_sd: float = ENERGY_NOISE_SD,
                       low: float = -8, high: float = 36) -> pd.DataFrame:
    """Draw n days: temperature uniform in [low, high], demand = true curve + noise.
    Because the true curve is known we can draw as many independent training sets as we like."""
    rng = np.random.default_rng(seed)
    temp = rng.uniform(low, high, n)
    demand = energy_true_curve(temp) + rng.normal(0, noise_sd, n)
    return pd.DataFrame({"temp_c": temp, "demand_mwh": demand})


# ---------------------------------------------------------------------------
# 6. Assignment dataset: food-delivery orders (target = delivery_minutes)
# ---------------------------------------------------------------------------
RESTAURANT_PREP = {"fast_food": 7, "bakery": 5, "pizza": 15, "asian": 12, "burgers": 10, "fine_dining": 24}
DELIVERY_TARGET = "delivery_minutes"


def make_delivery_orders(n: int = 8000, seed: int = 11, with_truth: bool = False) -> pd.DataFrame:
    """
    Orders on a fictional food-delivery platform over 120 days. Target: delivery_minutes
    (from order placed to food at the door). Generated by a KNOWN mechanism:

        prep    = restaurant base + 1.1 * kitchen_queue + 0.4 * items
        travel  = distance * minutes_per_km(vehicle) * traffic(rush hour) * weather
        courier = +3 min for couriers in their first 3 months
        delivery = 4 + prep + travel + courier + right-skewed noise (occasional long delays)

    Traps for the assignment:
        - travel time is MULTIPLICATIVE (distance x rain x rush hour): a plain linear
          model under-predicts long trips in bad weather; residuals reveal it.
        - the noise is right-skewed: MAE and RMSE disagree, and the mean is above the median.
        - pickup_minutes (order -> courier picks up the food) is only known AFTER the ETA is
          promised: it is LEAKY, and very tempting, because it contains the kitchen delays.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-05")
    day = rng.integers(0, 120, n)
    hour = rng.choice(np.arange(10, 24), n, p=np.array([2, 4, 9, 9, 4, 3, 3, 4, 9, 12, 11, 6, 3, 2]) / 81)
    minute = rng.integers(0, 60, n)
    ts = start + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    weekend = ts.dayofweek >= 5
    rush = np.isin(hour, [12, 13, 18, 19, 20])

    rtype = rng.choice(list(RESTAURANT_PREP), n, p=[.24, .07, .2, .2, .19, .1])
    items = np.clip(rng.poisson(2.2, n) + 1, 1, 12)
    queue = rng.poisson(2 + 3 * rush + 1.5 * weekend)
    distance = np.clip(rng.gamma(2.0, 1.6, n), 0.3, 15).round(2)
    vehicle = np.where(distance > 6, rng.choice(["scooter", "car"], n, p=[.5, .5]),
                       rng.choice(["bike", "scooter", "car"], n, p=[.45, .4, .15]))
    weather = rng.choice(["clear", "rain", "snow"], n, p=[.7, .24, .06])
    experience = np.clip(rng.gamma(1.6, 9, n), 0, 72).round(1)
    order_value = (items * rng.uniform(7, 16, n) * np.where(rtype == "fine_dining", 2.4, 1)).round(2)

    prep = pd.Series(rtype).map(RESTAURANT_PREP).to_numpy() + 1.1 * queue + 0.4 * items
    mpk = pd.Series(vehicle).map({"bike": 3.4, "scooter": 2.5, "car": 2.3}).to_numpy()
    traffic = 1 + 0.45 * rush * (vehicle != "bike") + 0.15 * weekend
    wx = pd.Series(weather).map({"clear": 1.0, "rain": 1.3, "snow": 1.75}).to_numpy()
    travel = distance * mpk * traffic * wx
    rookie = 3.0 * (experience < 3)
    true_minutes = 4 + prep + travel + rookie
    kitchen_delay = rng.gamma(2.0, 2.2, n) - 4.4                       # right-skewed: occasional long waits
    minutes = np.clip(true_minutes + kitchen_delay + rng.normal(0, 1.5, n), 6, None).round(1)
    arrival = 2 + rng.uniform(2, 10, n)                                # courier reaches the restaurant
    pickup = (np.maximum(prep + kitchen_delay, arrival) + rng.normal(0, 0.5, n)).clip(1).round(1)

    df = pd.DataFrame({
        "order_id": [f"ORD-{700000 + i}" for i in range(n)],
        "order_time": ts.strftime("%Y-%m-%d %H:%M"),
        "restaurant_type": rtype,
        "items": items,
        "order_value": order_value,
        "kitchen_queue": queue,
        "distance_km": distance,
        "vehicle": vehicle,
        "weather": weather,
        "courier_experience_months": experience,
        "pickup_minutes": pickup,                # LEAKY: measured at pickup, after the ETA is shown
        DELIVERY_TARGET: minutes,
    })
    if with_truth:
        df["_true_minutes"] = true_minutes.round(2)
    return df.sort_values("order_time").reset_index(drop=True)
