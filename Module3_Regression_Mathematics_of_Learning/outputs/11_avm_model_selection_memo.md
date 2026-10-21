# Model Selection Memo: Automated Valuation Model for home-equity loans (Ames, Iowa)

Train 1,164 houses (5-fold CV), test 292 houses (opened once). Requirements fixed before modelling.

## Business requirements

| requirement | type | metric | threshold / weight | why |
|---|---|---|---|---|
| Accuracy floor | HARD gate | `MdAPE %` | <= 10.0 | Industry AVMs with MdAPE above ~10% are not accepted for lending decisions |
| Explainable | HARD gate | `explainable` | must be True | Regulator requires per-feature reasons for every adverse decision |
| No over-valuation of top homes | HARD gate | `top-decile bias %` | <= 5.0 | Over-valuing the most expensive 10% creates the largest loan losses |
| Latency | HARD gate | `latency ms` | <= 50.0 | Valuation is called live inside the loan application form |
| Typical error (RMSE $) | soft | `RMSE` | weight 35% (lower is better) | Overall dollar accuracy |
| Hit rate within 10% | soft | `PPE10 %` | weight 25% (higher is better) | Share of homes that need no manual review |
| Inputs to collect | soft | `inputs used` | weight 25% (lower is better) | Every input is a question a loan officer must ask |
| Stability across folds | soft | `CV RMSE sd` | weight 15% (lower is better) | Predictable behaviour on new data |

## Candidates

| model | Accuracy floor | Explainable | No over-valuation of top homes | Latency | Typical error (RMSE $) | Hit rate within 10% | Inputs to collect | Stability across folds | eligible | failed gates | weighted score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Lasso (log price) | 5.578 | True | -3.128 | 9.679 | 20,809 | 73.883 | 59 | 1,634 | True | - | 0.806 |
| Ridge (log price) | 5.714 | True | -3.101 | 9.636 | 20,783 | 73.969 | 79 | 1,455 | True | - | 0.750 |
| Decision tree (depth <= 6) | 11.069 | True | -5.452 | 10.427 | 36,691 | 45.962 | 17 | 3,392 | False | Accuracy floor | - |
| Baseline: neighbourhood $/sqft | 12.804 | True | -11.411 | 0.179 | 41,674 | 41.151 | 2 | 4,759 | False | Accuracy floor | - |

## Recommendation

**Lasso (log price)**

**Why Lasso (log price):** it passes every hard gate (MdAPE, explainability, no over-valuation of
top homes, latency) and has the best weighted score. See the candidates table for the numbers.
On accuracy it is statistically TIED with Ridge (log price) (the test CI of the RMSE difference includes 0), so the business criteria decide: it needs 20 fewer inputs (59 vs 79), i.e. a shorter application form.

- **Metrics:** on the sealed test set it reaches MdAPE 6.2% and values
  73% of homes within +-10% (baseline: 36%).
  Test RMSE difference vs Ridge (log price): $263 (95% CI -304 to 830).
- **Business:** weight sensitivity -> as written: Lasso (log price), accuracy-first (RMSE 60%, PPE10 30%, inputs 5%, stability 5%): Lasso (log price), cost-first (RMSE 20%, PPE10 15%, inputs 55%, stability 10%): Lasso (log price), equal weights: Lasso (log price).
- **Rejected:** the decision tree is readable but too inaccurate to pass the MdAPE gate;
  the baseline is the fallback if the model service is unavailable.
- **Segment check:** over-valued price bands: < $130k (+6.4%) -> manual review for valuations in that band.
- **Scope and limitations:** not for homes > 4,000 sqft (manual appraisal); trained on one city, 2006-2010
  sales; re-validate before use in other markets or after large market moves (a trained model cannot
  extrapolate a price trend, Lab 07).
