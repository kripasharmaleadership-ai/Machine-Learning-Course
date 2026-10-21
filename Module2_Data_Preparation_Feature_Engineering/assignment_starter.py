"""
ASSIGNMENT STARTER - Prepare the raw used-car listings and document every decision
=================================================================================

Read ASSIGNMENT.md first. Reference solution for a DIFFERENT dataset: 12_assignment_worked_example.py

    python assignment_starter.py

The script runs as-is: PART A (profiling) and PART C (contract + rubric) are done for you.
Your work is PART B - replace each TODO with real cleaning code AND a log.record(...) call.
Re-run until every rubric check prints PASS, then submit the files listed in ASSIGNMENT.md.
"""
# %%
import numpy as np
import pandas as pd

from ml_lab_utils import DATA_DIR, OUT_DIR, banner, make_raw_used_cars, note, show, step
from preprocessing import ColumnRule, DataContract, DecisionLog, fingerprint, normalize_category, parse_money

RAW_PATH = DATA_DIR / "raw_used_cars.csv"
if not RAW_PATH.exists():                         # generated once, then treated as an immutable raw file
    make_raw_used_cars().to_csv(RAW_PATH, index=False)
raw = pd.read_csv(RAW_PATH, keep_default_na=False, na_values=[""])
TARGET = "price"
log = DecisionLog("Used-car listings (raw marketplace scrape)")

# %% =========================================================================
# PART A - Profile (done for you)
# ============================================================================
banner("PART A - Profile the raw data")
show("Rows x columns", raw.shape)
show("Raw fingerprint", fingerprint(raw))
prof = pd.DataFrame({"dtype": raw.dtypes.astype(str), "missing %": (raw.isna().mean() * 100).round(1),
                     "unique": raw.nunique(), "example": raw.apply(lambda s: str(s.dropna().iloc[0])[:24])})
print(prof.to_string())
for c in ["make", "fuel_type", "transmission", "accident_history"]:
    print(f"\n  {c}: {raw[c].value_counts(dropna=False).head(12).to_dict()}")

# %% =========================================================================
# PART B - YOUR WORK: clean step by step, logging every decision
# ============================================================================
banner("PART B - Clean and document (your work)")
df = raw.copy()

step("B1  Exact duplicates and dealer re-posts")
# TODO: drop exact duplicates. Then find RE-POSTS: same car listed again with a new listing_id
#       (hint: which columns identify the same physical car? the posted_date differs!).
#       Record each as a decision with rows_affected and the alternative you rejected.
# Example of the expected style:
# n = df.duplicated().sum(); df = df.drop_duplicates()
# log.record("uniqueness", "all", "drop exact duplicates", "why ...", "alternatives ...", f"{n} rows", int(n))

step("B2  Target: parse price, handle placeholders")
# TODO: "$12,500" -> 12500.0 (parse_money helps). "$1" and "$999,999" are placeholders, not prices.
#       Decide: rows without a usable price - drop them or keep them for something? Why?
df["price"] = parse_money(df["price"])            # <- a start; the placeholders are still in there

step("B3  Mileage: mixed units")
# TODO: "45,000 km" and "28,000 mi" -> one numeric column in km. How many rows were in miles?

step("B4  Engine size: '2.0L' strings, and electric cars")
# TODO: parse to float. What does a missing engine size mean for an Electric car vs a Petrol car?

step("B5  Consistent categories")
# TODO: make casing, fuel synonyms (Gas/gasoline/PETROL = Petrol; EV = Electric; HEV = Hybrid; DSL = Diesel),
#       transmission ('auto' / 'man'). normalize_category(series, mapping) does most of the work.

step("B6  Missing values: owners, accident_history")
# TODO: why might accident_history be missing? Is 'missing' informative? Choose a strategy + indicator?

step("B7  Leakage review")
# TODO: final_sold_price and days_listed - known when a seller asks 'what should I list my car for'?
#       Record a decision for EVERY column you exclude, with the reason.

step("B8  Domain features")
# TODO: e.g. car_age = posted year - year; km_per_year; is_luxury_make. Show evidence that they help
#       (correlation / MI / a quick model comparison).

step("B9  Split strategy")
# TODO: random? by time (posted_date)? Can a re-posted car end up in BOTH train and test?

step("B10 Columns kept as-is")
# TODO: every remaining column still needs ONE line: 'reviewed, kept as-is' + evidence (range, categories).

# %% =========================================================================
# PART C - Contract + rubric (done for you)
# ============================================================================
banner("PART C - Validate and self-check")
contract = DataContract({
    "price": ColumnRule("number", min=500, max=250_000, max_missing=0.0),
    "year": ColumnRule("number", min=1990, max=2026),
    "mileage": ColumnRule("number", min=0, max=600_000, max_missing=0.05),
    "engine_size": ColumnRule("number", min=0.8, max=8.0, max_missing=0.10),
    "fuel_type": ColumnRule("category", allowed=["Petrol", "Diesel", "Hybrid", "Electric"]),
    "transmission": ColumnRule("category", allowed=["Automatic", "Manual"]),
    "make": ColumnRule("category", allowed=["Toyota", "Honda", "Ford", "BMW", "Mercedes-Benz", "Tesla",
                                            "Hyundai", "Kia", "Volkswagen", "Chevrolet"]),
}, unique_key="listing_id")
issues = contract.validate(df)
show("Contract violations", len(issues))
if len(issues):
    print(issues.to_string(index=False))

covered = {c.strip() for e in log.entries for c in e["columns"].split(",")}
changing = [e for e in log.entries if e["rows_affected"] is not None]
checks = {
    "Every raw column has at least one logged decision": all(c in covered for c in raw.columns),
    "Every entry has a rationale (WHY)": bool(log.entries) and all(e["rationale"] for e in log.entries),
    ">= 8 data-changing decisions with rows affected": len(changing) >= 8,
    "Alternatives recorded for data-changing decisions": bool(changing) and all(
        e["alternatives_considered"] for e in changing),
    "Data contract passes on prepared data": len(issues) == 0,
    "Leakage review recorded": any("leak" in e["step"].lower() for e in log.entries),
    "Split strategy recorded": any("split" in e["step"].lower() for e in log.entries),
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
missing_cols = sorted(set(raw.columns) - covered)
show("Raw columns without a decision", missing_cols if missing_cols else "none")

if all(checks.values()):
    df.to_csv(OUT_DIR / "assignment_used_cars_prepared.csv", index=False)
    log.to_markdown(OUT_DIR / "assignment_decision_log.md",
                    header=f"Raw rows {len(raw):,} (fingerprint {fingerprint(raw)}) -> prepared {len(df):,} "
                           f"(fingerprint {fingerprint(df)}).")
    print("  [saved] outputs/assignment_used_cars_prepared.csv + outputs/assignment_decision_log.md")
else:
    note("""Not all checks pass yet - keep going with PART B. The outputs are written
automatically once every check is PASS.""")
