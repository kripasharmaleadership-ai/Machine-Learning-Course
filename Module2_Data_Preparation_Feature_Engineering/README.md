# Module 2: Data Preparation and Feature Engineering

Twelve runnable labs. Each one builds from **EASY → MEDIUM → HARD**, and each HARD level ends with a production-grade "plot twist". The scenarios are credit risk, card fraud, house pricing and a telecom support knowledge base.

The theme of the module, proven with numbers in almost every lab: **data quality and domain knowledge beat algorithm complexity.**

Every lab is a plain Python file with `# %%` cell markers:
- **Script mode:** `python 05_scaling_and_normalization.py` prints a narrated walkthrough and saves charts to `outputs/`.
- **Cell mode (best for recording):** open the file in VS Code and click **Run Cell** above each `# %%`. Charts appear inline.

## Setup

```bash
pip install -r requirements.txt
python run_all.py          # smoke-test all 12 labs (~5 minutes)
```

Lab 10 downloads the real **Ames Housing** dataset from OpenML the first time it runs, then caches it in `data/ames_housing.csv`. Every other lab runs fully offline, including the GenAI lab, which needs no API keys and no model downloads. All randomness is seeded (`RANDOM_STATE = 42`), so the numbers below come out the same on each run. Small differences can appear with other library versions (tested with scikit-learn 1.9, pandas 2.2).

## Folder layout

| File | Purpose |
|---|---|
| `ml_lab_utils.py` | Console/plot helpers; the Ames loader; **synthetic loan applications** (clean *or* realistically messy, with a known default mechanism); raw used-car listings (assignment); a telecom knowledge base (GenAI lab) |
| `preprocessing.py` | A **reusable preprocessing library**, built up across the module: parsers, `Winsorizer`, `LoanRawCleaner`, `LoanFeatureBuilder`, `build_preprocessor()` factory, `DataContract`, `DecisionLog`, `fingerprint()` |
| `01` … `09` | Core topics |
| `10_hands_on_lab_ames_pipeline.py` | The hands-on lab (real data, end to end) |
| `11_genai_connection_...py` | From feature engineering to tokens, embeddings and RAG context |
| `12_assignment_worked_example.py` | Reference solution: a raw dataset prepared with **every decision documented** |
| `ASSIGNMENT.md` + `assignment_starter.py` | The student assignment (used-car listings) with an automated rubric |
| `run_all.py` | Runs every lab and reports pass/fail |
| `outputs/` | Charts, pipeline artifact + metadata, pipeline diagram (HTML), decision logs, data card |

> **Why synthetic data in most labs?** "Did imputation bias the mean?", "Which features truly matter?" and "Did cleaning help?" can only be *proven* when you know the true answer. The messy loan extract is generated from a clean table, so every corruption can be checked against the truth. Lab 10 then applies everything to real, messy data.

---

## Episode plan with the key on-screen moments

### Part A: Core topics

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 01 | Data quality > algorithms | Six-dimension quality scorecard; dirty + boosting vs clean + logistic | Label noise × model grid; **systematic** label errors | 2-week budget: tune hyperparameters or fix labels (out-of-fold label audit) | Logistic on clean data with 1 domain feature (0.771) beats boosting on raw data (0.751). A collections bug: the team sees AUC **0.85**, reality is **0.69**. The label-review queue is 77% precise (random: 20%) |
| 02 | Missing data and duplicates | Disguised missing values ('N/A', 0, 999); mean/median/mode | MCAR / MAR / MNAR with ground truth; six imputation strategies | Exact, conflicting and near-duplicates; duplicates leaking across splits | pandas reports **0** missing incomes, reality 396. No-score applicants: actual default 24.2%, median imputation predicts 17.5%, + missing indicator 21.9%. Duplicate rows score AUC **1.00**, new rows 0.78 |
| 03 | Outliers and noise | z-score vs IQR vs MAD (masking) | Error vs genuine extreme; fix / cap / log / robust model | Impossible combinations (multivariate); noise and regression dilution | 20 typos flatten the OLS slope from 0.21 to **0.004**, and the "robust" Huber regression fails too (x-outliers). Raw-column detectors find 1–2 of 40 impossible rows; one ratio feature finds 27; domain rules find **40** |
| 04 | Numerical and categorical features | Type columns by meaning; the codes-as-numbers trap | One-hot vs ordinal; alphabetical-order bug; unseen and rare categories | 1,200 employers: six encodings compared | `loan_purpose` as a number scores AUC **0.48** (worse than random). One-hot with `min_frequency` keeps the accuracy with 12× fewer columns. Smoothed target encoding shrinks a one-off employer from 0.00 to 0.18 |
| 05 | Scaling and normalization | Nearest neighbours dominated by dollars; three scalers | Which models care (measured); convergence; log / Yeo-Johnson / quantile | Four production bugs | kNN 0.60 → **0.75** once scaled; trees identical. lbfgs needs 435 vs 12 iterations. Per-batch normalisation predicts **16%** default for a batch whose real rate is 59% |
| 06 | Feature engineering | Annuity formula → payment-to-income | Cyclical time (sin/cos), bins/splines for a utilisation cliff, interactions (honest null result) | Behavioural features from a **680k-row event log**, point-in-time | Engineered features lift logistic 0.772 → 0.797 (above boosting). Naive aggregation scores AUC **1.000** offline and 0.778 in production; point-in-time scores **0.870** |
| 07 | Feature selection | Pearson vs Spearman, redundancy, VIF; U-shape correlation ≈ 0 | Mutual information with a **noise floor** (shuffled target) | L1 path, RFECV, permutation importance, stability selection, business case | Correlation 0.01 vs MI 0.49 on a U-shape. Stability selection keeps *both* bureau scores; only human de-duplication drops the second pull: same AUC (0.796), data cost **$1.3M → $0.4M** per million applications |
| 08 | Curse of dimensionality | Distance concentration, empty space, 10^d samples | Add 0–2,000 noise features to a real problem | Polynomial explosion and remedies; the PCA trap | Nearest/farthest contrast 610 → **0.14**. Weak-penalty logistic: train AUC **1.00**, test 0.63. 7 domain features (0.823) beat 285 polynomial ones. PCA scores 0.53 vs supervised selection 0.82 |
| 09 | ML preprocessing pipelines | `get_dummies` crashes on one new applicant | Custom transformers, named outputs, grid-search over preprocessing | Raw record in → contract → versioned artifact → **pipeline test suite** → latency | 9/9 pipeline tests pass (invariances, unknowns, extremes, monotonicity). Trained on raw messy records: AUC 0.786 (same as clean) |

### Part B: Hands-on lab (`10_hands_on_lab_ames_pipeline.py`)

Build a reusable preprocessing pipeline with **pandas, SimpleImputer, OneHotEncoder, StandardScaler, ColumnTransformer and Pipeline** on real data.

| Part | Level | What you do |
|---|---|---|
| 1 | EASY | Audit with pandas, and catch a CSV bug: default parsing turns the category `"None"` into NaN (**872** "missing" vs 8 real) |
| 2 | EASY | Column roles; **structural vs genuine** missingness (PoolQC is missing for exactly the 1,162 houses with no pool) |
| 3 | MEDIUM | Evidence-based decisions: log target (skew 1.74 → 0.12), outliers as a *scope* decision, skewed features, domain features |
| 4 | MEDIUM | Config-driven `ColumnTransformer` + `Pipeline` (80 raw columns → 217 model inputs); CV RMSLE 0.111 vs 0.391 baseline |
| 5 | HARD | **Ablation with paired noise bands** (two v1 decisions get a "REVISIT" verdict from the data), model swap, test set opened once (RMSLE 0.127, median error 5.7%), serving a partial web-form record, artifact + auto-generated decision log |

Plot twist: a web form with empty fields prices a 2004, 1,850 sq ft house at **$100k**, because "missing" means "no basement" in training data but "not filled in" on a form. With stored defaults it prices at $274k.

### Part C: Connection to GenAI (`11_genai_connection_tokens_embeddings_rag.py`)

| Level | Classical idea → GenAI idea | Key on-screen moment |
|---|---|---|
| EASY | One-hot / `handle_unknown` → **tokenization** | A BPE tokenizer trained from scratch: `reactivation → re + activa + tion`; vocabulary size vs tokens-per-text trade-off |
| MEDIUM | TF-IDF / Normalizer / PCA → **vectors and embeddings** | Word vectors fail on `"refnd for my fone"`, character n-grams succeed; LSA embeddings group the two refund policies |
| HARD | Cleaning, dedupe, joins, feature selection → **metadata enrichment and context preparation for RAG** | Ingestion pipeline (boilerplate detection, PII redaction, near-dup removal, version resolution). The stale 2024 refund policy reaches the LLM for **2/2** refund questions, and for **0/2** with an `is_latest` metadata filter. The final cited prompt is packed within a token budget |

### Part D: Assignment

- **`12_assignment_worked_example.py`** prepares the raw loan extract, recording **26 documented decisions**. Each entry records what, which columns, rows affected, why (with evidence) and the alternatives rejected. The data contract goes from 12 violations to 1 (it catches a bug the cleaning missed), and then to 0. Test AUC improves 0.731 → 0.778 on a time split. The lab exports the prepared CSV, a decision log (MD + JSON) and a data card, then passes 8/8 rubric checks.
- **`ASSIGNMENT.md` + `assignment_starter.py`**: students repeat the method on raw used-car listings (mixed km/mi units, fuel synonyms, placeholder prices, dealer re-posts, a leaky column). An automated rubric awards 40 points; reviewer judgement awards 60.

---

## Recording tips
- Run `python run_all.py` once before recording, so charts, the Ames cache and all outputs exist.
- In cell mode, pause after each `>>` note. Those lines are written as talking points.
- Several labs show **honest negative results** (an interaction that doesn't help, a "robust" model that fails, ablation verdicts that overrule the author). Lean into them: that's how real data work goes.
- Open `outputs/09_pipeline_diagram.html` in a browser for a clickable diagram of the production pipeline.
- `run_all.py` doubles as a 5-minute pre-flight check after any library upgrade.
