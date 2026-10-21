# Decision Memo: Card-fraud decision policy (approve / review / auto-decline)

Model frozen (logistic + night x foreign). Strategies tuned on 30,000 validation transactions; test (30,000) opened once. Requirements fixed before any results.

## Business requirements

| requirement | type | metric | threshold / weight | why |
|---|---|---|---|---|
| Analyst capacity | HARD gate | `review rate %` | <= 2.0 | 20 analysts can review at most 2% of transactions |
| Customer friction | HARD gate | `good customers disturbed %` | <= 2.0 | Customer experience: at most 2% of good customers held or declined |
| Fraud $ stopped | HARD gate | `fraud $ stopped %` | >= 60.0 | Board target: stop at least 60% of fraud losses |
| Net cost | soft | `net cost $` | weight 50% (lower is better) | Fraud losses + review cost + customer friction |
| Fraud $ stopped (soft) | soft | `fraud $ stopped %` | weight 20% (higher is better) | Regulator and press: share of fraud money stopped |
| Review precision | soft | `review precision` | weight 15% (higher is better) | Analyst morale: share of reviews that are real fraud |
| Simplicity | soft | `simplicity` | weight 15% (higher is better) | 1 = one number to explain and monitor; lower = more moving parts |

## Candidates

| candidate | Analyst capacity | Customer friction | Fraud $ stopped | Net cost | Fraud $ stopped (soft) | Review precision | Simplicity | eligible | failed gates | weighted score |
|---|---|---|---|---|---|---|---|---|---|---|
| S9 auto-decline + expected-$ queue (top 2%) | 2.000 | 1.761 | 68.841 | 80,163 | 68.841 | 0.227 | 0.400 | True | - | 0.639 |
| S8 expected-$ ranking, top 2% | 2.000 | 1.368 | 60.400 | 93,966 | 60.400 | 0.328 | 0.600 | True | - | 0.610 |
| S0 default 0.5 | 0.540 | 0.187 | 21.311 | 158,658 | 21.311 | 0.660 | 1.000 | False | Fraud $ stopped | - |
| S1 max F1 | 2.013 | 1.218 | 48.253 | 117,513 | 48.253 | 0.406 | 1.000 | False | Analyst capacity, Fraud $ stopped | - |
| S2 Youden J | 14.783 | 13.425 | 94.163 | 139,748 | 94.163 | 0.108 | 1.000 | False | Analyst capacity, Customer friction | - |
| S3 recall >= 80% | 8.640 | 7.351 | 82.169 | 108,593 | 82.169 | 0.164 | 1.000 | False | Analyst capacity, Customer friction | - |
| S4 Bayes p* (avg fraud $) | 4.703 | 3.573 | 66.428 | 104,823 | 66.428 | 0.254 | 1.000 | False | Analyst capacity, Customer friction | - |
| S5 capacity: top 2% | 2.000 | 1.208 | 47.644 | 118,601 | 47.644 | 0.407 | 1.000 | False | Fraud $ stopped | - |
| S6 p*(amount) per transaction | 3.803 | 2.905 | 76.397 | 77,833 | 76.397 | 0.250 | 0.700 | False | Analyst capacity, Customer friction | - |
| S7 two-tier: p*(amount) review + auto-decline | 3.280 | 2.912 | 76.445 | 76,269 | 76.445 | 0.184 | 0.500 | False | Analyst capacity, Customer friction | - |

## Recommendation

**S9 auto-decline + expected-$ queue (top 2%)**

**Why S9 auto-decline + expected-$ queue (top 2%):** it passes every hard gate (analyst capacity, customer friction, fraud $ stopped)
and has the best weighted score on validation; on the sealed test set it stops 69% of fraud
dollars, reviews 2.00% of transactions and disturbs 1.78% of good customers,
for a net cost of $75,537 per 30,000 transactions.

- **Versus the runner-up (S8 expected-$ ranking, top 2%):** test net-cost difference $4,176 (95% CI 1,313 to 8,028).
- **Stress tests:** as written: S9 auto-decline + expected-$ queue (top 2%); fraud wave (3x fraud share): None; review cost doubles ($40): S7 two-tier: p*(amount) review + auto-decline; cost-only (net cost 100%): S9 auto-decline + expected-$ queue (top 2%); simplicity-heavy: S8 expected-$ ranking, top 2%.
- **Rejected:** the default 0.5 stops too little fraud; statistical rules (max F1, Youden J) ignore the money and the capacity.
- **Segments:** foreign and new-account customers are disturbed more often; monitor both monthly.
- **Monitoring and fallback:** track the review queue, the auto-decline rate and the fraud $ stopped daily.
  The queue rule cannot overflow the analysts. In a fraud wave no policy meets every gate with 20 analysts:
  Lowering the auto-decline threshold (auto-decline p >= 0.4) restores the 60% target within the friction cap: that becomes the documented wave-mode fallback. Re-tune when the review or friction costs change.
