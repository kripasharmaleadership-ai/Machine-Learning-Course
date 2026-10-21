# Model Comparison Report: Nightly call-list ranking for a term-deposit campaign

Bank Marketing (UCI), 45,211 calls in time order. Models compared with 5-fold CV on the first 36,168 calls; each checked on the most recent 9,043. Requirements fixed before results.

## Requirements

| requirement | type | metric | threshold / weight | why |
|---|---|---|---|---|
| No post-call information | HARD gate | `leak-free` | must be True | The list is built BEFORE the call; anything measured during the call is unavailable |
| Ranking quality floor | HARD gate | `CV AUC` | >= 0.65 | The current segment rules reach ~0.60; a model must be clearly better |
| Nightly batch fits the window | HARD gate | `rows / second (batch)` | >= 2000 | 45,000 customers scored in < 30 s inside the nightly job |
| Ranking quality | soft | `CV AUC` | weight 20% (higher is better) | Overall ranking (5-fold CV) |
| Recent months | soft | `recent AUC` | weight 15% (higher is better) | Behaviour drifts: performance on the latest 20% of calls |
| Top-decile lift | soft | `top-10% lift` | weight 15% (higher is better) | The call centre only reaches the top of the list |
| Explainability | soft | `explainability` | weight 20% (higher is better) | Agents must say why a customer is called; auditors must review the logic |
| Training time | soft | `fit seconds` | weight 5% (lower is better) | Weekly retraining on shared hardware |
| Model size | soft | `model size KB` | weight 10% (lower is better) | Stored and versioned with every release |
| Deployment complexity | soft | `complexity points` | weight 15% (lower is better) | Preprocessing steps + learned numbers + dependencies to maintain |

## Candidates

| model | No post-call information | Ranking quality floor | Nightly batch fits the window | Ranking quality | Recent months | Top-decile lift | Explainability | Training time | Model size | Deployment complexity | eligible | failed gates | weighted score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | True | 0.683 | 503,012 | 0.683 | 0.684 | 1.665 | 0.800 | 0.186 | 3.439 | 5.699 | True | - | 0.558 |
| Gradient boosting | True | 0.710 | 246,673 | 0.710 | 0.664 | 1.714 | 0.500 | 1.930 | 484 | 6.136 | True | - | 0.439 |
| Random forest | True | 0.707 | 30,966 | 0.707 | 0.696 | 1.550 | 0.400 | 8.465 | 38,902 | 8.997 | True | - | 0.155 |
| Decision tree (depth 5) | True | 0.620 | 502,258 | 0.620 | 0.634 | 1.390 | 1.000 | 0.160 | 5.626 | 4.763 | False | Ranking quality floor | - |
| Gradient boosting + call duration | False | 0.953 | 190,214 | 0.953 | 0.781 | 1.987 | 0.500 | 0.554 | 658 | 6.271 | False | No post-call information | - |

## Performance

| model | CV AUC | recent AUC | top-10% lift |
|---|---|---|---|
| Logistic regression | 0.683 | 0.684 | 1.665 |
| Decision tree (depth 5) | 0.620 | 0.634 | 1.390 |
| Random forest | 0.707 | 0.696 | 1.550 |
| Gradient boosting | 0.710 | 0.664 | 1.714 |
| Gradient boosting + call duration | 0.953 | 0.781 | 1.987 |

## Explainability

| model | score | method |
|---|---|---|
| Logistic regression | 0.800 | coefficients per feature (odds ratios) |
| Decision tree (depth 5) | 1.000 | 32 readable rules |
| Random forest | 0.400 | SHAP per customer (post-hoc) |
| Gradient boosting | 0.500 | SHAP per customer (post-hoc) |
| Gradient boosting + call duration | 0.500 | SHAP per customer (post-hoc) |

Winner's top global drivers (AUC drop when shuffled, recent calls): poutcome (0.103), housing (0.028), month (0.013), job (0.011).
Example local explanation (tonight's top customer, score 0.95 vs average 0.07): month & day (+0.56), contact history (pdays, previous, poutcome) (+0.19), marital & education (+0.04).

## Speed

| model | fit seconds | latency ms (1 row) | rows / second (batch) |
|---|---|---|---|
| Logistic regression | 0.186 | 3.585 | 503,012 |
| Decision tree (depth 5) | 0.160 | 3.290 | 502,258 |
| Random forest | 8.465 | 17.263 | 30,966 |
| Gradient boosting | 1.930 | 5.739 | 246,673 |
| Gradient boosting + call duration | 0.554 | 5.569 | 190,214 |

## Deployment complexity

| model | model size KB | learned numbers | complexity points |
|---|---|---|---|
| Logistic regression | 3.439 | 50 | 5.699 |
| Decision tree (depth 5) | 5.626 | 58 | 4.763 |
| Random forest | 38,902 | 992680 | 8.997 |
| Gradient boosting | 484 | 13664 | 6.136 |
| Gradient boosting + call duration | 658 | 18666 | 6.271 |

Complexity points = preprocessing steps + log10(learned numbers) + external dependencies.

## Notes per candidate

- **Logistic regression:** explained by: coefficients per feature (odds ratios)
- **Decision tree (depth 5):** explained by: 32 readable rules
- **Random forest:** explained by: SHAP per customer (post-hoc)
- **Gradient boosting:** explained by: SHAP per customer (post-hoc)
- **Gradient boosting + call duration:** explained by: SHAP per customer (post-hoc)

## Recommendation

**Logistic regression**

**Why Logistic regression:** it passes every hard gate (no post-call information, AUC floor, nightly batch window)
and has the best weighted score. On the most recent calls it reaches AUC 0.684 with a top-decile lift of
1.7x; versus Gradient boosting the recent-AUC difference is +0.020 (95% CI +0.011 to +0.030).

- **Rejected:** 'Gradient boosting + call duration' scores best on paper but uses information only known after the call (leak).
- **Explainability:** coefficients per feature (odds ratios) for auditors, plus exact SHAP reasons per customer, grouped into plain-language
  themes, for call agents. (The depth-5 tree is the most readable model but fails the AUC floor: readable rules cost too much accuracy here.)
- **Sensitivity:** as written: Logistic regression; performance-first: Logistic regression; explainability-first: Logistic regression; ops-first (size, complexity): Logistic regression.
- **Limitations:** strong drift between campaigns (subscription rate 7% -> 32%); every model loses AUC on recent
  months; retrain each campaign and monitor the top-decile conversion rate weekly.
