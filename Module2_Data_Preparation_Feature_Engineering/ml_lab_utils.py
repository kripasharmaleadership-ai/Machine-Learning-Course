"""
ml_lab_utils.py - Shared helpers for every lab in Module 2.

What lives here:
    1. Pretty console output helpers   -> banner(), step(), note(), show()
    2. Plot helper                     -> save_fig()
    3. Real dataset loader             -> load_ames()  (Ames Housing, 1,460 real houses)
    4. Synthetic LOAN APPLICATIONS     -> make_loan_applications(messy=True/False)
       A credit-risk dataset where we KNOW the true default mechanism, available
       as a clean table or as a realistic, messy raw extract.
    5. Synthetic USED-CAR LISTINGS     -> make_raw_used_cars()   (the assignment dataset)
    6. A small enterprise knowledge base for the GenAI lab -> make_support_kb()

Why synthetic data?
    "Did imputation bias the mean?", "Did cleaning help?", "Which features truly
    matter?" can only be PROVEN when you know the right answer. Real data never
    tells you that. Lab 10 then applies everything to real, messy data.

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
    """Big section header, e.g. banner('Missing data', 'HARD')."""
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
# 3. Real dataset: Ames Housing (De Cock, 2011) - 1,460 houses, 80 columns
# ---------------------------------------------------------------------------
def load_ames(naive_parsing: bool = False) -> pd.DataFrame:
    """Download Ames Housing once from OpenML, cache in data/ames_housing.csv.

    naive_parsing=True reads the CSV with pandas defaults - which silently turns
    the legitimate category "None" (e.g. MasVnrType = no masonry veneer) into NaN.
    The default (False) only treats truly empty cells as missing.
    """
    path = DATA_DIR / "ames_housing.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml

        print("  Downloading Ames Housing from OpenML (first run only)...")
        frame = fetch_openml(name="house_prices", as_frame=True).frame
        frame.to_csv(path, index=False)
    if naive_parsing:
        return pd.read_csv(path)
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


# ---------------------------------------------------------------------------
# 4. Synthetic credit-risk data: loan applications
# ---------------------------------------------------------------------------
EDUCATION_LEVELS = ["High School", "Bachelor", "Master", "PhD"]
EMPLOYMENT_TYPES = ["Salaried", "Self-employed", "Contract", "Unemployed", "Retired"]
STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA"]
PURPOSES = ["debt_consolidation", "home_improvement", "car", "medical", "business",
            "education", "wedding", "other"]

LOAN_TARGET = "defaulted"
LOAN_NUMERIC = ["age", "years_employed", "annual_income", "credit_score", "num_credit_lines",
                "credit_utilization", "delinquencies_2y", "loan_amount", "loan_term_months",
                "interest_rate"]
LOAN_NOMINAL = ["employment_type", "home_ownership", "state", "loan_purpose"]
LOAN_ORDINAL = ["education"]

_PURPOSE_TEXT = {
    "debt_consolidation": ["Consolidate my credit card debt", "paying off 3 credit cards",
                           "Combine my loans into one monthly payment", "refinance card balances"],
    "home_improvement": ["New roof for the house", "kitchen renovation", "Fix basement flooding",
                         "solar panels for home"],
    "car": ["Buying a used car", "car repair after accident", "Down payment on a new truck"],
    "medical": ["Surgery bills", "dental work not covered by insurance", "medical expenses"],
    "business": ["Inventory for my small shop", "Start a food truck business",
                 "equipment for my bakery"],
    "education": ["Tuition for nursing course", "coding bootcamp fees", "MBA tuition"],
    "wedding": ["Wedding venue and catering", "our wedding in June"],
    "other": ["Moving to a new city", "vacation", "personal expenses", "family emergency"],
}


def _employer_names(rng: np.random.Generator, k: int) -> np.ndarray:
    first = ["North", "Pacific", "Atlantic", "Metro", "United", "First", "Prime", "Blue", "Green",
             "Summit", "Silver", "Golden", "Red", "Union", "Coastal"]
    mid = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay", "Soylent", "Cyberdyne",
           "Tyrell", "Wonka", "Oscorp", "Aperture", "Monarch", "Nakatomi", "Gekko", "Stark",
           "Wayne", "Dunder", "Pinnacle", "Zenith"]
    last = ["Inc", "LLC", "Corp", "Group", "Holdings", "Labs", "Partners", "Systems", "Logistics",
            "Foods", "Health", "Bank"]
    names = np.array([f"{a} {b} {c}" for a in first for b in mid for c in last])
    return rng.permutation(names)[:k]


def make_loan_applications(n: int = 6000, seed: int = RANDOM_STATE, messy: bool = False,
                           with_truth: bool = False) -> pd.DataFrame:
    """
    Personal-loan applications at a fictional lender, with a KNOWN default mechanism.

    messy=False -> the clean table (numbers are numbers, categories are consistent).
    messy=True  -> what you actually get from a production database export:
                   income as "$54,300" strings, two date formats, sentinel ages (0, 999),
                   inconsistent category spelling, unit errors (income in cents,
                   utilisation in %), typos x10, exact AND near-duplicate rows,
                   informative (not random) missingness.
    with_truth  -> adds `_true_default_prob` (only possible because the data is synthetic).

    The TRUE drivers of default (the "law of nature" in this world):
        payment-to-income ratio (strongest, and NOT a raw column - must be engineered),
        credit score, utilisation > 80%, delinquencies, employment type, age < 25,
        thin credit file, employer, loan purpose, education (weak).
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(41, 12, n), 18, 75).round().astype(int)
    retired = (age >= 60) & (rng.random(n) < 0.6)
    emp = rng.choice(["Salaried", "Self-employed", "Contract", "Unemployed"], n, p=[.62, .17, .14, .07])
    emp = np.where(retired, "Retired", emp)
    years_emp = np.where(emp == "Unemployed", np.nan, (rng.beta(2, 2.5, n) * (age - 17)).round(1))

    edu_idx = rng.choice(4, n, p=[.35, .40, .20, .05])
    education = np.array(EDUCATION_LEVELS)[edu_idx]
    base_inc = pd.Series(emp).map({"Salaried": 58000, "Self-employed": 62000, "Contract": 50000,
                                   "Unemployed": 14000, "Retired": 36000}).to_numpy()
    sigma = np.where(emp == "Self-employed", 0.75, 0.42)
    income = (base_inc * np.array([1, 1.3, 1.55, 1.85])[edu_idx]
              * (1 + 0.008 * np.nan_to_num(years_emp))
              * np.exp(rng.normal(0, sigma) - sigma ** 2 / 2))
    income = np.clip(income, 6000, None).round(-2)

    older = age >= 45
    home = np.where(older, rng.choice(["Rent", "Mortgage", "Own"], n, p=[.25, .45, .30]),
                    rng.choice(["Rent", "Mortgage", "Own"], n, p=[.60, .33, .07]))
    w = 1 / np.arange(1, len(STATES) + 1) ** 0.6
    state = rng.choice(STATES, n, p=w / w.sum())

    n_emp = 1200
    names = _employer_names(rng, n_emp)
    ew = 1 / np.arange(1, n_emp + 1) ** 0.9
    employer_id = rng.choice(n_emp, n, p=ew / ew.sum())
    employed = np.isin(emp, ["Salaried", "Contract"])
    employer = np.where(employed, names[employer_id],
                        np.where(emp == "Self-employed", "Self", None)).astype(object)
    employer_risk = rng.normal(0, 0.6, n_emp)[employer_id] * employed

    lines = rng.poisson(5.5, n) + (age > 30)
    thin_file = lines <= 1
    score = np.clip(rng.normal(690, 65, n) + 1.2 * (age - 41) - 40 * thin_file, 300, 850).round()
    util = np.clip(rng.beta(2, 3, n) * 1.15 + 0.15 * (score < 620), 0, 1.2).round(3)
    delinq = rng.poisson(0.25 + 0.9 * (score < 620))

    purpose = rng.choice(PURPOSES, n, p=[.32, .14, .14, .07, .08, .07, .05, .13])
    purpose_text = np.array([rng.choice(_PURPOSE_TEXT[p]) for p in purpose], dtype=object)
    loan = np.clip(income * rng.uniform(0.05, 0.55, n), 1000, 75000).round(-2)
    term = rng.choice([36, 60], n, p=[.65, .35])
    rate = np.clip(0.07 + (740 - score) / 1000 * 0.30 + rng.normal(0, .01, n) + 0.01 * (term == 60),
                   0.05, 0.32).round(4)
    r = rate / 12
    payment = loan * r / (1 - (1 + r) ** -term)
    pti = payment / (income / 12)

    emp_eff = pd.Series(emp).map({"Salaried": 0, "Self-employed": .35, "Contract": .5,
                                  "Unemployed": 1.0, "Retired": .1}).to_numpy()
    purpose_eff = pd.Series(purpose).map({"business": .5, "medical": .3, "debt_consolidation": .15,
                                          "wedding": .2}).fillna(0).to_numpy()
    logit = (-3.45 + 11.0 * (pti - 0.06) - 0.011 * (score - 690) + 1.4 * (util > 0.8) + 0.6 * util
             + 0.45 * delinq + emp_eff + 0.55 * (age < 25) + employer_risk + purpose_eff
             + 0.7 * thin_file - 0.12 * edu_idx)
    p_default = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, p_default)

    # Thin-file applicants usually have NO credit score (the bureau has nothing).
    # That is informative missingness: the blank itself predicts default.
    score_obs = np.where((thin_file & (rng.random(n) < 0.85)) | (rng.random(n) < 0.015), np.nan, score)

    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    df = pd.DataFrame({
        "application_id": [f"APP-{100000 + i}" for i in range(n)],
        "application_date": dates,
        "age": age,
        "employment_type": emp,
        "years_employed": years_emp,
        "employer": employer,
        "education": education,
        "annual_income": income,
        "home_ownership": home,
        "state": state,
        "credit_score": score_obs,
        "num_credit_lines": lines,
        "credit_utilization": util,
        "delinquencies_2y": delinq,
        "loan_purpose": purpose,
        "purpose_text": purpose_text,
        "loan_amount": loan,
        "loan_term_months": term,
        "interest_rate": rate,
        LOAN_TARGET: defaulted,
    })
    if with_truth:
        df["_true_default_prob"] = p_default.round(4)
    if messy:
        df = _corrupt_loans(df, rng)
    return df


def _corrupt_loans(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Turn the clean table into a realistic raw export. Every corruption below
    is something the author has seen in a real production extract."""
    raw = df.copy()
    n = len(raw)

    def pick(frac):
        return rng.random(n) < frac

    # 1) Dates: two upstream systems, two formats
    d = pd.to_datetime(raw.application_date)
    raw["application_date"] = np.where(pick(0.2), d.dt.strftime("%m/%d/%Y"), d.dt.strftime("%Y-%m-%d"))

    # 2) Income: unit errors, currency strings, and NOT-at-random refusals
    inc = raw.annual_income.to_numpy(dtype=float)
    cents, monthly = pick(0.004), pick(0.006)
    inc = np.where(cents, inc * 100, np.where(monthly & ~cents, (inc / 12).round(), inc))
    dollar_fmt = pick(0.5)
    inc_str = np.array([f"${v:,.0f}" if fm else f"{v:.1f}" for v, fm in zip(inc, dollar_fmt)], dtype=object)
    refuse = (raw.annual_income.to_numpy() < 25000) & pick(0.30)        # MNAR: low earners skip it
    inc_str[refuse | pick(0.02)] = "N/A"
    inc_str[pick(0.015)] = ""
    raw["annual_income"] = inc_str

    # 3) Age: sentinels typed by humans / legacy defaults
    age = raw.age.astype(float).to_numpy()
    age[pick(0.010)] = 0
    age[pick(0.004)] = 999
    age[pick(0.010)] = np.nan
    raw["age"] = age

    # 4) Free-typed categories
    variants = {"Salaried": ["salaried", "SALARIED", " Salaried"],
                "Self-employed": ["self employed", "Self Employed", "SELF_EMPLOYED"],
                "Contract": ["contract", "Contractor"],
                "Unemployed": ["unemployed", "Not employed"],
                "Retired": ["retired", "RETIRED"]}
    et = raw.employment_type.to_numpy(dtype=object)
    for i in np.where(pick(0.15))[0]:
        et[i] = rng.choice(variants[et[i]])
    raw["employment_type"] = et

    ho = raw.home_ownership.to_numpy(dtype=object)
    m = pick(0.08)
    ho[m] = [s.lower() for s in ho[m]]
    m = pick(0.04)
    ho[m] = [s.upper() for s in ho[m]]
    ho[pick(0.003)] = "OTHER"
    ho[pick(0.002)] = "NONE"
    raw["home_ownership"] = ho

    st = raw.state.to_numpy(dtype=object)
    m = pick(0.05)
    st[m] = [s.lower() for s in st[m]]
    m = pick(0.01)
    st[m] = [s + " " for s in st[m]]
    raw["state"] = st

    emp = raw.employer.to_numpy(dtype=object)
    m = pick(0.10) & pd.notna(emp)
    emp[m] = [rng.choice([e.upper(), e.lower(), f" {e}", e.replace(" ", "  ")]) for e in emp[m]]
    raw["employer"] = emp

    # 5) Numeric typos & unit mix-ups
    loan = raw.loan_amount.to_numpy(dtype=float)
    loan[pick(0.004)] *= 10
    raw["loan_amount"] = loan
    util = raw.credit_utilization.to_numpy(dtype=float)
    m = pick(0.02)
    util[m] = (util[m] * 100).round(1)                                # entered as a percentage
    raw["credit_utilization"] = util
    cs = raw.credit_score.to_numpy(dtype=float)
    cs[pick(0.003)] = 0                                               # legacy "no score" code
    raw["credit_score"] = cs

    pt = raw.purpose_text.to_numpy(dtype=object)
    pt[pick(0.08)] = ""
    m = pick(0.05)
    pt[m] = [s.upper() for s in pt[m]]
    raw["purpose_text"] = pt

    # 6) Duplicates: ETL re-loads (exact) and customers re-submitting (near-duplicates)
    exact = raw.sample(frac=0.03, random_state=int(rng.integers(1e6)))
    near = raw.sample(frac=0.02, random_state=int(rng.integers(1e6))).copy()
    near["application_id"] = [f"APP-{900000 + i}" for i in range(len(near))]
    near["employer"] = [e.upper() if isinstance(e, str) else e for e in near.employer]
    near["purpose_text"] = [s.lower() if isinstance(s, str) else s for s in near.purpose_text]
    raw = pd.concat([raw, exact, near], ignore_index=True)
    return raw.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Assignment dataset: raw used-car listings (regression, price)
# ---------------------------------------------------------------------------
_CAR_MODELS = {
    "Toyota": [("Corolla", 24000), ("Camry", 30000), ("RAV4", 34000)],
    "Honda": [("Civic", 25000), ("Accord", 31000), ("CR-V", 34000)],
    "Ford": [("Focus", 21000), ("F-150", 45000), ("Escape", 30000)],
    "BMW": [("3 Series", 46000), ("X5", 68000)],
    "Mercedes-Benz": [("C-Class", 48000), ("GLE", 70000)],
    "Tesla": [("Model 3", 42000), ("Model Y", 50000)],
    "Hyundai": [("Elantra", 22000), ("Tucson", 30000)],
    "Kia": [("Rio", 18000), ("Sportage", 29000)],
    "Volkswagen": [("Golf", 26000), ("Tiguan", 32000)],
    "Chevrolet": [("Malibu", 25000), ("Silverado", 44000)],
}


def make_raw_used_cars(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """
    Raw used-car listings scraped from a marketplace. Target: `price`.
    Deliberately messy - mixed units (km / mi), synonyms (Gas = Petrol), structural
    missingness (electric cars have no engine size), placeholder prices ($1, $999,999),
    dealer re-posts (duplicates), free text, and a leaky column. Used by ASSIGNMENT.md.
    """
    rng = np.random.default_rng(seed)
    makes = rng.choice(list(_CAR_MODELS), n, p=[.16, .14, .13, .07, .06, .06, .1, .09, .09, .1])
    model_idx = [rng.integers(len(_CAR_MODELS[m])) for m in makes]
    model = np.array([_CAR_MODELS[m][i][0] for m, i in zip(makes, model_idx)])
    msrp = np.array([_CAR_MODELS[m][i][1] for m, i in zip(makes, model_idx)], dtype=float)
    year = rng.integers(2006, 2026, n)
    age = 2026 - year
    km = np.clip(rng.normal(15000, 5000, n) * np.maximum(age, 0.5), 50, 400000).round(-1)
    is_tesla = makes == "Tesla"
    fuel = np.where(is_tesla, "Electric", rng.choice(["Petrol", "Diesel", "Hybrid", "Electric"], n,
                                                       p=[.68, .14, .14, .04]))
    engine = np.where(fuel == "Electric", np.nan, rng.choice([1.4, 1.6, 2.0, 2.5, 3.0, 3.5, 5.0], n))
    trans = rng.choice(["Automatic", "Manual"], n, p=[.82, .18])
    owners = np.clip(rng.poisson(age / 5) + 1, 1, 6)
    accident = rng.random(n) < 0.12 + 0.01 * age
    seller = rng.choice(["Dealer", "Private"], n, p=[.55, .45])
    city = rng.choice(["Seattle", "Austin", "Denver", "Boston", "Chicago", "Miami", "Phoenix"], n)
    color = rng.choice(["White", "Black", "Silver", "Grey", "Blue", "Red", "Green", "Other"], n)

    price = (msrp * 0.86 ** age * np.exp(-km / 400000) * np.where(accident, 0.82, 1.0)
             * np.where(fuel == "Hybrid", 1.08, 1.0) * np.where(trans == "Manual", 0.95, 1.0)
             * (1 - 0.03 * (owners - 1)) * np.where(seller == "Dealer", 1.06, 1.0)
             * np.exp(rng.normal(0, 0.08, n)))
    price = np.clip(price, 800, None).round(-1)

    use_miles = rng.random(n) < 0.4
    mileage = np.where(use_miles, [f"{v / 1.609:,.0f} mi" for v in km], [f"{v:,.0f} km" for v in km])
    mileage = mileage.astype(object)
    mileage[rng.random(n) < 0.03] = None
    fuel_raw = fuel.astype(object)
    syn = {"Petrol": ["Gas", "gasoline", "petrol", "PETROL"], "Diesel": ["diesel", "DSL"],
           "Hybrid": ["hybrid", "HEV"], "Electric": ["EV", "electric"]}
    for i in np.where(rng.random(n) < 0.25)[0]:
        fuel_raw[i] = rng.choice(syn[fuel_raw[i]])
    engine_raw = np.array([f"{e:.1f}L" if not np.isnan(e) else None for e in engine], dtype=object)
    engine_raw[(rng.random(n) < 0.05) & (fuel != "Electric")] = None
    trans_raw = np.where(rng.random(n) < 0.2, np.where(trans == "Automatic", "auto", "man"), trans)
    owners_raw = owners.astype(float)
    owners_raw[rng.random(n) < 0.15] = np.nan
    acc_raw = np.where(accident, "Yes", "No").astype(object)
    acc_raw[rng.random(n) < 0.20] = None

    price_raw = np.array([f"${p:,.0f}" for p in price], dtype=object)
    price_raw[rng.random(n) < 0.010] = "$1"                     # "call for price" placeholder
    price_raw[rng.random(n) < 0.006] = "$999,999"
    price_raw[rng.random(n) < 0.01] = None

    desc_bits = ["One owner", "Full service history", "Clean title", "New tires", "Needs some TLC",
                 "Garage kept", "Minor scratches", "No accidents!", "Leather seats", "Sunroof"]
    desc = np.array([", ".join(rng.choice(desc_bits, rng.integers(1, 4), replace=False)) for _ in range(n)],
                    dtype=object)
    posted = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 240, n), unit="D")
    days_listed = rng.integers(1, 90, n)
    sold_price = np.where(rng.random(n) < 0.5, (price * rng.uniform(0.9, 1.0, n)).round(-1), np.nan)

    df = pd.DataFrame({
        "listing_id": [f"L{500000 + i}" for i in range(n)],
        "posted_date": posted.strftime("%Y-%m-%d"),
        "make": np.where(rng.random(n) < 0.15, pd.Series(makes).str.lower(), makes),
        "model": model,
        "year": year,
        "mileage": mileage,
        "fuel_type": fuel_raw,
        "engine_size": engine_raw,
        "transmission": trans_raw,
        "owners": owners_raw,
        "accident_history": acc_raw,
        "seller_type": seller,
        "city": city,
        "color": color,
        "description": desc,
        "days_listed": days_listed,
        "final_sold_price": sold_price,           # LEAKY: only known after the sale
        "price": price_raw,
    })
    reposts = df.sample(frac=0.04, random_state=seed).copy()
    reposts["listing_id"] = [f"L{900000 + i}" for i in range(len(reposts))]
    reposts["posted_date"] = (pd.to_datetime(reposts.posted_date) + pd.Timedelta(days=14)).dt.strftime("%Y-%m-%d")
    df = pd.concat([df, reposts], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. GenAI lab: a small, realistic (= messy) enterprise knowledge base
# ---------------------------------------------------------------------------
KB_FOOTER = ("\n\n(c) NorthStar Telecom. Internal and customer-facing use. For help, contact the "
             "support desk. Last reviewed by the Knowledge Team.")


def make_support_kb() -> tuple[list[dict], list[dict]]:
    """
    Returns (documents, eval_questions) for a fictional telecom's support knowledge base.

    The documents carry the same problems classical tables do:
        - a STALE version of a policy next to the current one       (~ outdated rows)
        - a near-duplicate page copied into another wiki            (~ duplicate records)
        - an empty page                                             (~ missing values)
        - raw HTML, repeated boilerplate footers                    (~ noise)
        - personal data (an email and a phone number)               (~ sensitive columns)
        - inconsistent / missing metadata                           (~ dirty categoricals)
    """
    docs = [
        {"doc_id": "KB-101", "title": "Device Refund Policy", "source": "policies/refunds_2024.docx",
         "meta": {"updated": "2024-01-15", "product": "devices"},
         "text": "Device Refund Policy (2024 edition). If you are not happy with a phone, tablet or "
                 "router bought from NorthStar, you can request a refund within 14 days of purchase. "
                 "How many days do customers have? Customers have 14 days to request a refund for a "
                 "device. A restocking fee of 10% applies to opened devices. Refunds are paid to the "
                 "original payment method within 5 business days."},
        {"doc_id": "KB-102", "title": "Device Refund Policy", "source": "policies/refunds_2025.docx",
         "meta": {"updated": "2025-06-01", "product": "Devices"},
         "text": "Device Refund Policy (updated June 2025). This version replaces the 2024 policy. "
                 "Refund requests for phones, tablets and routers are now accepted up to 30 days "
                 "after delivery. The restocking fee is waived for unopened devices; opened devices "
                 "carry a 10% fee. Approved refunds are paid to the original payment method within "
                 "5 business days. Devices must be returned with all accessories."},
        {"doc_id": "KB-201", "title": "International Roaming with Travel Pass", "source": "wiki/roaming",
         "meta": {"updated": "2025-03-10", "product": "mobile"},
         "text": "Travel Pass lets you use your plan abroad in more than 120 countries. Travel Pass "
                 "costs $10 per day and is only charged on days you actually use your phone abroad. "
                 "Activate it in the NorthStar app under Plans, then Travel. Each day includes 2GB of "
                 "high-speed data. After you use 2GB in a day, data continues at a reduced speed of "
                 "256 kbps until midnight local time. Calls back home are included. Cruise ships and "
                 "in-flight networks are not covered by Travel Pass and are billed at satellite rates."},
        {"doc_id": "KB-301", "title": "Fiber Outage Credits and Repair Times", "source": "wiki/fiber-sla",
         "meta": {"updated": "2025-02-20", "product": "fiber"},
         "text": "Our fiber service level agreement covers outages caused by our network. Report an "
                 "outage in the app or by calling the support line. In metro areas a technician arrives "
                 "within 4 hours; in rural areas within 24 hours. If your fiber internet is down for "
                 "more than 24 hours in a row, you automatically receive a bill credit of one full "
                 "month of service. Outages caused by damage inside your home, or by your own router "
                 "settings, are not eligible for a credit. Planned maintenance is announced 48 hours "
                 "in advance by SMS."},
        {"doc_id": "KB-401", "title": "Cancelling a Contract", "source": "policies/contracts.pdf",
         "meta": {"updated": "2025-01-05"},
         "text": "You can cancel a fixed-term contract at any time. The early termination fee is 50% "
                 "of the monthly price multiplied by the number of months left on the contract. "
                 "Example: 10 months left on a $40 plan means a fee of $200. No fee applies during the "
                 "first 14 days (cooling-off period) or if you move to an address where we cannot "
                 "provide service. To cancel, call support; cancellations by email are not accepted."},
        {"doc_id": "KB-501", "title": "5G Devices and eSIM Activation", "source": "wiki/5g-esim",
         "meta": {"updated": "2025-05-18", "product": "mobile"},
         "text": "5G is available on all current plans at no extra cost. To use 5G you need a 5G "
                 "capable phone and a version 4 SIM or an eSIM. To activate an eSIM, open the NorthStar "
                 "app, choose Add eSIM, and scan the QR code we email you. Keep Wi-Fi on during "
                 "activation. The QR code can only be used once; if activation fails, request a new "
                 "code in the app instead of scanning the old one again. Physical SIM cards are posted "
                 "free of charge within 3 days."},
        {"doc_id": "KB-601", "title": "Billing FAQ", "source": "wiki/billing-faq",
         "meta": {"updated": "2025-04-02", "product": "billing"},
         "text": "Your bill is issued on the 1st of each month and is due on the 15th. Customers who "
                 "pay by autopay receive a $5 monthly discount. If a bill is not paid within 10 days "
                 "of the due date, a late fee of $8 is added. We accept credit cards, debit cards, "
                 "bank transfer and payments at partner stores. Paper bills cost $2 per month; "
                 "e-bills are free."},
        {"doc_id": "KB-602", "title": "billing faq (copy)", "source": "old-wiki/Billing FAQ - copy",
         "meta": {},
         "text": "Your bill is issued on the 1st of each month and is due on the 15th. Customers who "
                 "pay by autopay receive a $5 monthly discount. If a bill is not paid within 10 days "
                 "of the due date, a late fee of $8 is added. We accept credit cards, debit cards, "
                 "bank transfer and payments at partner stores.  Paper bills cost $2 per month; "
                 "e-bills are free."},
        {"doc_id": "KB-701", "title": "Requesting Your Personal Data", "source": "policies/privacy.docx",
         "meta": {"updated": "2025-03-30", "product": "account"},
         "text": "You can request a copy of the personal data we hold about you. Submit the request "
                 "in the app under Privacy. We respond within 30 days and send an encrypted file to the "
                 "email address on your account. For complex requests contact our data protection "
                 "officer, Jane Doe, at jane.doe@northstar-telecom.example or +1-555-0142. We never "
                 "ask for your password when handling a data request."},
        {"doc_id": "KB-801", "title": "Parental Controls", "source": "export/parental.html",
         "meta": {"updated": "2025-04-22", "product": "mobile"},
         "text": "<div class='article'><h2>Parental Controls</h2><p>Use <b>SafeZone</b> to block adult "
                 "content and set screen-time limits for a child's line.</p><p>Turn it on in the "
                 "NorthStar app: open the child's line, tap <i>SafeZone</i>, then choose a filter "
                 "level.</p><p>SafeZone is free on all family plans.</p></div>"},
        {"doc_id": "KB-901", "title": "Network Upgrade Notes", "source": "wiki/drafts/upgrade",
         "meta": {"updated": "2025-07-01"}, "text": "   "},
    ]
    for d in docs:
        if d["text"].strip():
            d["text"] = d["text"] + KB_FOOTER

    questions = [
        {"q": "How many days do I have to ask for a refund on a phone?", "gold": "KB-102", "answer": "30 days"},
        {"q": "How much does the Travel Pass cost per day?", "gold": "KB-201", "answer": "$10 per day"},
        {"q": "What happens to my speed after 2GB of roaming data in one day?", "gold": "KB-201",
         "answer": "256 kbps"},
        {"q": "Do I get money back if my fiber internet is down for more than a day?", "gold": "KB-301",
         "answer": "one full month"},
        {"q": "How fast does a technician come in a rural area?", "gold": "KB-301", "answer": "within 24 hours"},
        {"q": "How is the early termination fee calculated when I cancel?", "gold": "KB-401",
         "answer": "50% of the monthly price"},
        {"q": "How do I activate an eSIM?", "gold": "KB-501", "answer": "scan the QR code"},
        {"q": "What is the late fee if I pay my bill late?", "gold": "KB-601", "answer": "$8"},
        {"q": "Is there a discount for paying by autopay?", "gold": "KB-601", "answer": "$5 monthly discount"},
        {"q": "How long until I receive a copy of my personal data?", "gold": "KB-701", "answer": "within 30 days"},
        {"q": "How can I block adult content on my kid's phone?", "gold": "KB-801", "answer": "SafeZone"},
        {"q": "Is the restocking fee charged on unopened devices?", "gold": "KB-102", "answer": "waived for unopened"},
    ]
    return docs, questions
