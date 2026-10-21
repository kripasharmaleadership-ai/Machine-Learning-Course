# Module 3: Regression and the Mathematics of Learning

Eleven runnable labs and an assignment. Each lab builds from **EASY → MEDIUM → HARD**, and each HARD level ends with a "plot twist". The scenarios are house pricing, energy purchasing, healthcare costs, drug-response genomics, mortgage valuation and food-delivery ETAs.

The theme of the module: **training a model = minimising a loss with gradients**. The same loop that fits a straight line also trains a neural network and a language model (Lab 10). The *right* model is the one that meets the business requirements, not just the one with the lowest RMSE (Lab 11).

Every lab is a plain Python file with `# %%` cell markers:
- **Script mode:** `python 03_gradient_descent_and_learning_rate.py` prints a narrated walkthrough and saves charts to `outputs/`.
- **Cell mode (best for recording):** open the file in VS Code and click **Run Cell** above each `# %%`. Charts appear inline.

## Setup

```bash
pip install -r requirements.txt
python run_all.py          # smoke-test all 11 labs + the assignment starter (~2 minutes)
```

Lab 09 downloads the real **Medical Cost** dataset from OpenML (data_id 43463) the first time it runs, then caches it in `data/medical_costs.csv`. Lab 11 uses **Ames Housing**: it copies Module 2's cached file if it exists, and otherwise downloads it. Every other lab runs fully offline. All randomness is seeded, so the numbers below come out the same on each run. Small differences can appear with other library versions (tested with scikit-learn 1.9, pandas 2.2, numpy 2.2).

## Folder layout

| File | Purpose |
|---|---|
| `ml_lab_utils.py` | Console/plot helpers; loaders for Medical Cost and Ames; synthetic **house sales** (known pricing formula), **energy demand** (known U-shaped curve, so you can draw new samples forever), **delivery orders** (assignment) |
| `regression_toolkit.py` | A **reusable regression library** built up across the module: `regression_metrics()`, `adjusted_r2()`, `LinearRegressionGD` (from scratch), `numerical_gradient()`, `bias_variance_decomposition()`, `bootstrap_compare()`, `residual_table()`, `ModelScorecard` (business gates + weights → decision memo) |
| `01` … `08` | Core topics |
| `09_hands_on_lab_healthcare_costs.py` | The hands-on lab (real data, end to end) |
| `10_genai_connection_...py` | From regression training to neural-network and LLM training |
| `11_assignment_worked_example.py` | Reference solution: three models compared and justified for a mortgage lender |
| `ASSIGNMENT.md` + `assignment_starter.py` | The student assignment (delivery ETAs) with an automated rubric |
| `run_all.py` | Runs every lab and reports pass/fail |
| `outputs/` | Charts, saved models, metrics JSON, model card, model-selection memo |

> **Why synthetic data in most labs?** "Did regression recover the true coefficient?", "How much of the error is bias and how much is variance?" can only be *proven* when you know the true formula. Labs 09 and 11 then apply everything to real data.

---

## Episode plan with the key on-screen moments

### Part A: Core topics

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 01 | Simple and multiple linear regression | A line by hand (cov/var), then sklearn; reading slope and intercept | Normal equation; "holding everything else fixed"; one-hot; non-linear shapes with r ≈ 0 | Units vs standardised coefficients; **multicollinearity**; bootstrap CIs; extrapolation | `bedrooms` is worth **+$75k** alone and **−$9.6k** holding size fixed (truth: −$9k). With all drivers, test RMSE $24.6k ≈ the $25k noise. A 150 sqft shed is valued at **−$200k** |
| 02 | Cost and loss functions, MSE | Residual → loss → cost; the MSE "bowl" over (b0, b1) | MSE → mean, MAE → median; one typo vs OLS / Huber / LAD | Pinball loss for **asymmetric energy costs**; the **log-target trap** | One typo moves the OLS slope 19%, LAD 0%. The q = 0.67 quantile model has a worse RMSE but saves **$45.8k/year**. exp(log-model) under-budgets healthcare spend by **38%**, and smearing fixes it (+2.5%) |
| 03 | Gradient descent and learning rate | One parameter by hand, step by step | LR sweep and the exact limit 2/λmax; scaling (condition number 30 vs 77M); batch / mini-batch / SGD | **Gradient check** catches a silent bug; LR schedules (warm-up + cosine); momentum | 2% above the limit the loss explodes. Raw features: R² 0.83 after 20,000 epochs; standardised: 0.90 in 57. A silent mini-batch bug leaves R² at **−0.08**. Momentum: 1,341 → **122** steps |
| 04 | Underfitting, overfitting, bias–variance | Polynomial degree 1 / 2 / 4 / 15 on 30 days | Validation curve (CV); learning curves for high bias vs high variance | **bias² + variance + noise measured** from 300 training sets, for degree, tree depth and ridge α | Degree 15: train RMSE 7.0 (below the noise of 12!), test 48. With 12 days the best model is a **quadratic** (RMSE 16) and degree 10 explodes; with 1,000 days degree 8 reaches 12.1 |
| 05 | Regularisation and Ridge | Ridge tames a degree-12 polynomial; ridge by hand = sklearn | Paths; collinearity (bootstrap sd ÷10); CV for α; **scaling is mandatory** | Ridge = OLS on fake data = **weight decay**; effective degrees of freedom; 60 lab tests, 80 patients | Test RMSE 6,658 → 17.5 with α = 1. Storing sqft in thousands changes the learned value from **$171 to $127/sqft** without a scaler. Wide data: OLS test R² 0.54 → Ridge **0.82** |
| 06 | Lasso and Elastic Net | Lasso zeros 21 junk columns, keeps the 6 real drivers | Paths and drop order; soft-thresholding; **one-standard-error rule** | p = 500 genes, n = 100: OLS / Ridge / Lasso / EN; **selection stability** | Ridge can't beat min-norm OLS (R² 0.42); Lasso **0.93**. The 1-SE rule drops a real driver (`bedrooms`). Lasso keeps true gene 6 in only **17%** of bootstraps, Elastic Net in 93% |
| 07 | Decision-tree regression | A step function; a depth-2 tree printed as rules | No scaling; finds a threshold × interaction; depth × leaf tuning; cost-complexity pruning | **Can't extrapolate** a trend; instability; **monotonic constraints**; impurity vs permutation importance | The tree's under-valuation grows from $33k to **$56k** per quarter in a rising market. 12.4% of houses get *cheaper* with +100 sqft → 0% with `monotonic_cst`. A random column ranks #4 by impurity, ~0 by permutation |
| 08 | Evaluation metrics | MAE, MSE, RMSE, R², adjusted R² by hand for 8 houses | Same MAE, 3× the RMSE; **negative R²**; R² depends on the population | Adjusted R² vs 100 junk features; **bootstrap CI** for "B beats A"; the hidden segment | R² falls 0.96 → **0.86** on similar-sized houses with the same errors. Dropping `bathrooms` is measurably worse on 900 test houses but "can't tell" on 60. Large Lakeside homes are under-valued by **$108k** while overall R² is 0.94 |

### Part B: Hands-on lab (`09_hands_on_lab_healthcare_costs.py`)

Build a yearly healthcare-cost prediction model on **real** insurance data (1,338 members): baseline → linear regression → regularisation → comparison → residuals → interpretation.

| Part | Level | What you do |
|---|---|---|
| 1 | EASY | Explore: duplicate record, skewed target (1.51), smokers cost ~4×, **three bands** in the age plot, a jump at BMI 30 for smokers |
| 2 | EASY | Stratified split; baselines: mean, median, **mean per smoker group** (CV R² 0.62 already) |
| 3 | MEDIUM | Linear regression in a `ColumnTransformer` + `Pipeline`; coefficients in $ (CV R² 0.75) |
| 4 | MEDIUM | Ridge / Lasso / Elastic Net change nothing on 8 features (honest result); on **44 interaction features** Lasso finds `bmi × smoker` (CV R² 0.84) |
| 5 | HARD | Tuned tree (0.856); **residuals by segment** reveal two smoking premiums; 2 domain features → **CV R² 0.867**, beating 44 interactions and the tree; coefficients + permutation importance; test opened once; model + metrics JSON + model card saved |

Plot twist: the plain linear model over-charges smokers with BMI < 30 by $8.8k and under-charges obese smokers by $7.9k. One feature (`smoker × BMI ≥ 30`, worth **+$19.4k a year**) removes the bias in every segment. On the test set the champion (RMSE $4,855, R² 0.84) and the tree are statistically tied (CI [−207, 391]). The tie-breaker is that the regulator can audit a formula.

### Part C: Connection to GenAI (`10_genai_connection_regression_to_neural_networks.py`)

| Level | Regression idea → neural-network idea | Key on-screen moment |
|---|---|---|
| EASY | Linear regression = **one neuron**: prediction, loss, chain rule, update, by hand | SGD recovers the OLS weights |
| MEDIUM | A 2-layer network in numpy: **backprop** layer by layer, gradient check, SGD vs **Adam**, weight decay = Ridge, early stopping = validation curve | Gradient check 2.6e-10. One neuron: RMSE 54 MWh; 16 tanh neurons: **12.5** (noise floor 12), with no feature engineering |
| HARD | A 40-line **autograd** engine; a character language model trained with the **same five-step loop** (cross-entropy instead of MSE) | Gradient descent converges to the counting (closed-form) solution, **1.628 = 1.628**, just as it converged to least squares in Lab 03. Plus a regression ↔ LLM translation table |

### Part D: Assignment

- **`11_assignment_worked_example.py`**: an Automated Valuation Model for a mortgage lender on real Ames data.
  - **Method:** requirements are written *before* modelling (4 hard gates: MdAPE ≤ 10%, explainable, no over-valuation of top homes, latency; 4 weighted criteria). Ridge, Lasso and a decision tree are compared with a neighbourhood $/sqft baseline on the same folds.
  - **Result:** the tree and the baseline fail the MdAPE gate. Ridge and Lasso are **statistically tied** on test (CI [−304, 830]), so the business criterion decides: Lasso needs **59 inputs instead of 79**. The choice holds under every weight scenario. Test MdAPE is 6.2%, with 73% of homes valued within ±10% (baseline: 36%).
  - **Residual check:** homes under $130k are over-valued by 6.4%, which becomes a manual-review rule in the memo. The lab writes a decision memo (MD + JSON) and passes 10/10 rubric checks.
- **`ASSIGNMENT.md` + `assignment_starter.py`**: students compare three models for **delivery ETAs** shown at checkout.
  - **Built-in trade-offs:** 5 hard gates, including late-promise rate and snow-day robustness, plus a tempting **leaky column**. The harness detects leakage by shuffling that column.
  - **Grading:** an automated rubric awards 40 points; reviewer judgement awards 60.

---

## Recording tips
- Run `python run_all.py` once before recording, so the charts, the dataset caches and all outputs exist.
- In cell mode, pause after each `>>` note. Those lines are written as talking points.
- Several labs show **honest negative results**: regularisation that changes nothing (Lab 09), Ridge that can't beat OLS on sparse data (Lab 06), the 1-SE rule dropping a real feature, a tie decided by business needs (Lab 11). Lean into them: that's how real model selection goes.
- Lab 10 Level 1 has an ASCII diagram of the neuron; draw it on screen while you narrate the five steps.
- `run_all.py` doubles as a 2-minute pre-flight check after any library upgrade.
