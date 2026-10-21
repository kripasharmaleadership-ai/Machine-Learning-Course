# Module 2 Assignment: Prepare a Raw Dataset and Document Every Preprocessing Decision

> *"If it isn't written down, it didn't happen."* In industry, a preprocessing step that nobody can explain can't be audited, reproduced or safely changed.

## The scenario

You've joined the pricing team of a used-car marketplace. The product: **a seller types in their car's details and gets a suggested listing price.** The data team has handed you a raw scrape of 4,160 listings (`data/raw_used_cars.csv`, created on the first run of `assignment_starter.py`).

Your job isn't to build the best model. Your job is to deliver a **prepared dataset that another engineer could trust, reproduce and extend**, with **every decision documented**.

Reference solution for a *different* dataset: [12_assignment_worked_example.py](12_assignment_worked_example.py) (loan applications). Study it first.

## The raw data

| column | notes |
|---|---|
| `listing_id` | identifier |
| `posted_date` | date the listing went live |
| `make`, `model` | free-typed by sellers |
| `year` | model year |
| `mileage` | free text, **mixed units** (`"45,000 km"`, `"28,000 mi"`) |
| `fuel_type` | free text with **synonyms** (`Gas`, `gasoline`, `PETROL`, `EV`, `HEV`, `DSL` ...) |
| `engine_size` | text like `"2.0L"`; what does missing mean for an electric car? |
| `transmission` | `Automatic` / `Manual` / `auto` / `man` |
| `owners` | number of previous owners (often missing) |
| `accident_history` | `Yes` / `No` / missing |
| `seller_type`, `city`, `color` | categorical |
| `description` | free text written by the seller |
| `days_listed` | days the listing stayed online |
| `final_sold_price` | price at which the car eventually sold (if it sold) |
| `price` | **TARGET**: asking price, text like `"$12,500"`, with placeholder values |

There are traps in this data. Finding them is part of the assignment.

## What to do

Work in **Part B** of [assignment_starter.py](assignment_starter.py). Parts A (profiling) and C (data contract + automated rubric) are written for you. At minimum, make and document decisions for:

1. **Uniqueness:** exact duplicates *and* dealer re-posts (the same car, a new `listing_id`)
2. **Target:** parsing `price`; placeholder prices; rows without a usable price
3. **Parsing and units:** mileage in km *and* miles; engine size as text
4. **Consistency:** make casing, fuel-type synonyms, transmission codes
5. **Missing values:** structural (an electric car has no engine size) vs genuinely unknown (`owners`, `accident_history`). Is missingness informative?
6. **Outliers:** any you find, and whether each is an error or a genuine extreme
7. **Leakage review:** which columns would *not* be known when a seller asks for a price?
8. **Feature engineering:** at least 2 domain features, with evidence that they help
9. **Split strategy:** random or by time? Can a re-posted car leak across the split?
10. **Every other column:** "reviewed, kept as-is", with evidence

Every decision goes through `log.record(step, columns, decision, rationale, alternatives, evidence, rows_affected)`.

## Deliverables

| # | File | Produced by |
|---|---|---|
| 1 | `outputs/assignment_used_cars_prepared.csv` | starter (auto, when all checks pass) |
| 2 | `outputs/assignment_decision_log.md` | starter (auto, when all checks pass) |
| 3 | Your completed `assignment_starter.py` | you |
| 4 | `outputs/assignment_data_card.md`: source, rows in/out, period, target definition, split, known limitations, schema (see `outputs/12_loans_data_card.md`) | you |
| 5 | **Stretch (optional):** a scikit-learn `Pipeline` with a small ablation table showing which of your decisions measurably help (see Lab 10, Part 5) | you |

## Grading rubric (100 points)

**Automated checks (40 pts).** These are printed by `assignment_starter.py`, Part C:

| check | pts |
|---|---|
| Every raw column has at least one logged decision | 8 |
| Every entry has a rationale (WHY) | 6 |
| ≥ 8 data-changing decisions with rows affected | 6 |
| Alternatives recorded for data-changing decisions | 6 |
| Data contract passes on the prepared data | 8 |
| Leakage review recorded | 3 |
| Split strategy recorded | 3 |

**Reviewer judgement (60 pts):**

| criterion | pts | what "excellent" looks like |
|---|---|---|
| Evidence quality | 15 | Each WHY cites a **number** ("17% of rows in miles", "price placeholders: 43 × $1"), not an opinion |
| Correct diagnosis | 15 | Structural vs unknown missingness; errors vs genuine extremes; leakage found and excluded |
| Reproducibility | 10 | Re-running the script on the raw file reproduces the same fingerprint; no manual edits |
| Alternatives | 10 | Rejected options are realistic, and the reason for rejecting them is stated |
| Data card | 10 | A new team member understands the dataset's scope and limits in 2 minutes |

## Hints

- `parse_money`, `normalize_category`, `DataContract` and `DecisionLog` are in [preprocessing.py](preprocessing.py).
- Re-posts have a new `listing_id` and a later `posted_date`, but describe the same car. Which columns identify it?
- 1 mile = 1.609 km. Count how many rows you converted, and log it.
- Ask the leakage question from Module 1 for every column: *"Would I know this value at the moment I need the prediction?"*
- If the contract fails *after* your cleaning, it caught something real (see Lab 12, step 3.1). Fix it, log it, re-run.

## Stretch goals

- **GenAI connection:** turn `description` into features. Try keyword flags (`"No accidents!"`, `"Needs some TLC"`) first, then TF-IDF (Lab 11). Do they add signal beyond `accident_history`?
- **Target encoding** for `model` (Lab 04), cross-fitted inside the pipeline.
- **Behavioural test:** a newer car with fewer km must never be priced lower, all else equal (Lab 09, `t_monotonic_*`).
