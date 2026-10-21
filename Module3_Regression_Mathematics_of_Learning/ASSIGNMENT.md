# Module 3 Assignment: Compare Three Regression Models and Justify the Best One

> *"The best model is the one that meets the business requirements, not the one with the lowest RMSE."* A model choice that can't be explained with numbers and business reasons can't be defended in front of a product owner, a regulator or your future self.

## The scenario

You've joined the logistics team of a food-delivery platform. At checkout the app shows an ETA ("your food arrives in ~32 min"). Late deliveries are the #1 cause of refunds. The product team asks you to **replace the operations team's rule of thumb with a regression model** and to recommend which model to ship.

The data: 8,000 orders over 120 days (`data/delivery_orders.csv`, created on the first run of `assignment_starter.py`). The last 30 days are the sealed **test set**.

Reference solution for a *different* problem: [11_assignment_worked_example.py](11_assignment_worked_example.py) (an automated house-valuation model for a lender). Study it first.

## The data

| column | notes |
|---|---|
| `order_id`, `order_time` | identifier, timestamp (`hour`, `weekday`, `is_weekend` are added for you) |
| `restaurant_type` | fast_food, bakery, pizza, asian, burgers, fine_dining |
| `items`, `order_value` | basket size and value |
| `kitchen_queue` | orders already waiting in the kitchen at checkout |
| `distance_km` | restaurant to customer |
| `vehicle` | bike, scooter, car |
| `weather` | clear, rain, snow |
| `courier_experience_months` | experience of the assigned courier |
| `pickup_minutes` | minutes from order to courier pickup |
| `delivery_minutes` | **TARGET**: minutes from order to delivery |

There are traps in this data. Finding them is part of the assignment.

## The business requirements (fixed by the product team)

| requirement | type | metric | threshold / weight |
|---|---|---|---|
| Accuracy floor | HARD gate | MAE | ≤ 4.0 min |
| Late promises | HARD gate | % of orders arriving > 5 min after the ETA | ≤ 12% |
| Bad-weather robustness | HARD gate | MAE on snow days ÷ MAE on clear days | ≤ 1.5 |
| Latency | HARD gate | one prediction | ≤ 20 ms |
| No post-order information | HARD gate | the model must not use anything unknown at checkout | must hold |
| Typical error | soft | MAE | 30% |
| Within 5 minutes | soft | % of orders within ±5 min | 25% |
| Late promises | soft | late rate | 25% |
| Explainability | soft | 1 = readable per-feature effects, 0.5 = a few rules, 0 = opaque | 20% |

You may **add** requirements (and justify them). You may not loosen these.

## What to do

Work in **Part B** of [assignment_starter.py](assignment_starter.py). Part A (data, requirements, baseline, evaluation harness) and Part C (scorecard, rubric, memo) are written for you.

1. **Three candidates** from at least **two algorithm families** (e.g. Ridge / Lasso / Elastic Net, linear regression with domain features, a decision tree). Each one is a scikit-learn `Pipeline` that takes the raw DataFrame. Register each with `evaluate(name, pipeline, explainability, why)`. The harness runs a **time-series CV** (train on the past, validate on the future), measures latency and tests for leakage.
2. **Leakage review:** for every column, ask *"would I know this at checkout?"*
3. **Residual analysis:** `residual_report(name)` breaks out-of-fold errors down by weather and distance. Where does your model fail, and what did you change because of it?
4. **Sensitivity analysis:** define at least two alternative weightings in `SCENARIOS`. Does the winner change?
5. **Open the test set once**: `open_test_set()`, *after* your choice is made on CV.
6. **Justify:** write `JUSTIFICATION` (≥ 120 words). Name the winner, cite CV and test numbers, say which hard gates decided, what the residuals showed, what the sensitivity analysis showed, and the limitations.

## Deliverables

| # | File | Produced by |
|---|---|---|
| 1 | `outputs/assignment_eta_memo.md` | starter (auto, when all checks pass) |
| 2 | `outputs/assignment_eta_scorecard.json` | starter (auto, when all checks pass) |
| 3 | Your completed `assignment_starter.py` | you |
| 4 | **Stretch (optional):** a fourth candidate trained with a **quantile (pinball) loss** (Lab 02) and an explanation of how it changes the late rate vs MAE trade-off | you |

## Grading rubric (100 points)

**Automated checks (40 pts).** These are printed by `assignment_starter.py`, Part C:

| check | pts |
|---|---|
| ≥ 3 candidates besides the baseline | 6 |
| ≥ 2 algorithm families | 4 |
| No candidate uses post-order information | 6 |
| Requirements not loosened | 4 |
| Residuals analysed for the recommended model | 4 |
| ≥ 2 weight scenarios in the sensitivity analysis | 4 |
| Test set opened exactly once | 4 |
| Justification: ≥ 120 words, names the winner, cites ≥ 3 numbers | 8 |

**Reviewer judgement (60 pts):**

| criterion | pts | what "excellent" looks like |
|---|---|---|
| Use of metrics | 15 | MAE, RMSE, R² *and* the business metrics are reported and interpreted ("RMSE is ~30% above MAE: a few very late orders dominate the squared error"), not just listed |
| Business reasoning | 15 | The choice follows from the gates and weights; trade-offs are named ("M1 is simpler but fails the snow gate") |
| Diagnosis | 15 | Residuals are used to *improve* a model (e.g. a feature for distance × weather), and the leaky column is identified and excluded with a reason |
| Honesty about uncertainty | 10 | The top-2 difference is reported with its confidence interval; ties are called ties |
| Limitations | 5 | What the model can't do (new cities, unseen weather, a trend over time) |

## Hints

- `ColumnTransformer`, `OneHotEncoder`, `StandardScaler` and `Pipeline` work as in Module 2. Put feature engineering in a `FunctionTransformer` *inside* the pipeline.
- Travel time is roughly **distance × minutes-per-km × weather × rush hour**: a product, not a sum. A linear model can't build a product itself (Lab 01, 2.4; Lab 09).
- If one model is "too good to be true" (better than any realistic ETA could be), suspect leakage (Module 1, Lab 11).
- A model that predicts the *mean* is late on about half of the long-tail orders. To cut the late rate, look at the asymmetric losses of Lab 02.
- Ties on accuracy are common. That is exactly when the business criteria must decide (Lab 11, 3.3).

## Stretch goals

- **Quantile regression** (`QuantileRegressor` or `HistGradientBoostingRegressor(loss="quantile")`): promise the 70th percentile instead of the mean. What happens to MAE, and to the late rate?
- **Monotonic constraint:** a longer distance must never give a shorter ETA (Lab 07, 3.3). Test it.
- **Neural-net candidate:** an `MLPRegressor` (Lab 10). Where does it land on explainability, and is it worth it?
