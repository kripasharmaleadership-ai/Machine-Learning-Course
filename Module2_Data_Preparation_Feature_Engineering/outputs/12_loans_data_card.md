# Data Card: loan applications (prepared v1)

- Source: raw production extract, 6,300 rows, fingerprint `b4c9288ae248`
- Prepared: 6,000 rows, fingerprint `cc6e1d7cce4b`
- Period: 2024-01-01 to 2025-12-30
- Target: `defaulted` (rate 17.2%)
- Split: time-based at 2025-07-01 (4,475 train / 1,525 test)
- Decisions logged: 26 (see 12_loan_preprocessing_decision_log.md)
- Known limitations: income missing-not-at-random; interest_rate reflects the previous model; age requires legal review; employer excluded in v1.

## Schema

| column | dtype | missing % | example |
|---|---|---|---|
| application_id | object | 0.0 | APP-102073 |
| application_date | datetime64[ns] | 0.0 | 2024-01-30 00:00:00 |
| age | float64 | 2.5 | 38.0 |
| employment_type | object | 0.0 | Self-employed |
| years_employed | float64 | 6.9 | 6.8 |
| employer | object | 10.4 | Self |
| education | object | 0.0 | High School |
| annual_income | float64 | 6.9 | 64800.0 |
| home_ownership | object | 0.2 | Mortgage |
| state | object | 0.0 | MI |
| credit_score | float64 | 2.3 | 690.0 |
| num_credit_lines | int64 | 0.0 | 3 |
| credit_utilization | float64 | 0.0 | 0.447 |
| delinquencies_2y | int64 | 0.0 | 0 |
| loan_purpose | object | 0.0 | debt_consolidation |
| purpose_text | object | 7.9 | paying off 3 credit cards |
| loan_amount | float64 | 0.3 | 11100.0 |
| loan_term_months | int64 | 0.0 | 60 |
| interest_rate | float64 | 0.0 | 0.1196 |
| defaulted | int64 | 0.0 | 0 |