# Preprocessing Decision Log: Personal-loan applications (raw production extract)

Raw rows: 6,300 (fingerprint b4c9288ae248) -> prepared rows: 6,000 (fingerprint cc6e1d7cce4b). Time-split test AUC: 0.778 (raw baseline 0.731).

## 1. [uniqueness] all
- **Decision:** drop exact duplicate rows
- **Why:** ETL reloads: identical rows incl. application_id
- **Alternatives considered:** keep (would overweight those applicants and leak across splits)
- **Evidence:** 180 exact duplicates
- **Rows affected:** 180

## 2. [parsing] application_date
- **Decision:** parse with format='mixed' (ISO + US formats)
- **Why:** Two upstream systems write different formats
- **Alternatives considered:** drop non-ISO rows (20% of data)
- **Evidence:** 0 unparseable after parsing
- **Rows affected:** 0

## 3. [parsing] annual_income
- **Decision:** strip '$' and ',' -> float; 'N/A' and '' -> NaN
- **Why:** Currency-formatted strings; pd.to_numeric would silently turn them into NaN
- **Alternatives considered:** pd.to_numeric(errors='coerce')
- **Evidence:** 2836 '$' strings, 384 blank/N/A
- **Rows affected:** 3,220

## 4. [validity] age
- **Decision:** values outside [18, 100] -> NaN
- **Why:** 0 / 999 are 'unknown' codes
- **Alternatives considered:** clip to range (invents values)
- **Evidence:** values found: [0.0, 999.0]
- **Rows affected:** 85

## 5. [validity] credit_score
- **Decision:** values outside [300, 850] -> NaN
- **Why:** 0 is the legacy 'no score' code
- **Alternatives considered:** clip to range (invents values)
- **Evidence:** values found: [0.0]
- **Rows affected:** 13

## 6. [validity] credit_utilization
- **Decision:** values > 1.5 divided by 100
- **Why:** Entered as a percentage; utilisation above 150% does not occur, and /100 lands in the normal range
- **Alternatives considered:** NaN + impute
- **Evidence:** max before 113, after 1.20
- **Rows affected:** 125

## 7. [validity] annual_income
- **Decision:** > $2M -> NaN (then imputed)
- **Why:** Consistent with income typed in cents (x100); the true value is ambiguous, so do not 'fix'
- **Alternatives considered:** divide by 100 (guess); keep (destroys linear models, Lab 03)
- **Evidence:** 21 values, max 11,160,000
- **Rows affected:** 21

## 8. [validity] loan_amount
- **Decision:** > product maximum $75k -> NaN
- **Why:** Product policy caps loans at $75k; values are exact x10 typos
- **Alternatives considered:** divide by 10
- **Evidence:** 18 values
- **Rows affected:** 18

## 9. [consistency] employment_type, home_ownership, state, employer, purpose_text
- **Decision:** trim, lower-case, unify separators, map synonyms (e.g. 'Contractor' -> 'Contract'; 'OTHER'/'NONE' -> 'Other')
- **Why:** Free-text entry produced spelling variants of the same category
- **Alternatives considered:** one-hot as-is (17 fake categories)
- **Evidence:** employment_type: 17 spellings -> 5 categories

## 10. [uniqueness] age, annual_income, loan_amount, loan_purpose, application_date, employer, state
- **Decision:** drop later re-submissions with identical applicant + loan fields
- **Why:** Same person re-applied; keeping both double-counts them and leaks across splits
- **Alternatives considered:** fuzzy matching (not needed: exact match after normalisation)
- **Evidence:** 120 re-submissions
- **Rows affected:** 120

## 11. [missing values] years_employed
- **Decision:** median + missing indicator
- **Why:** Structurally missing for the unemployed (not applicable); indicator keeps that fact
- **Alternatives considered:** fill 0

## 12. [missing values] annual_income, credit_score, age
- **Decision:** median + missing indicator
- **Why:** MNAR: low earners skip income; thin files have no score (Lab 02)
- **Alternatives considered:** drop rows; mean
- **Evidence:** income 6.6%, score 2.3%, age 2.5% missing

## 13. [missing values] home_ownership, employment_type, state, loan_purpose
- **Decision:** constant 'missing' category before one-hot
- **Why:** Few rows; keeps them scoreable
- **Alternatives considered:** mode

## 14. [feature selection] employer
- **Decision:** excluded from v1
- **Why:** ~1,200 levels; needs target encoding (Lab 04) - deferred to v2 to keep v1 simple and auditable
- **Alternatives considered:** one-hot; target encoding

## 15. [validity] annual_income
- **Decision:** < $5,000 -> NaN (then imputed)
- **Why:** Consistent with monthly income typed into the annual field; x12 is a guess, so do not 'fix'
- **Alternatives considered:** multiply by 12 (guess); keep
- **Evidence:** caught by the data contract after cleaning; min was 1,617
- **Rows affected:** 17

## 16. [leakage / fairness review] interest_rate
- **Decision:** KEEP, flagged
- **Why:** Set by the CURRENT underwriting model from the credit score: known at decision time, but learning from the old model's output can freeze its biases. Revisit in v2.

## 17. [leakage / fairness review] purpose_text
- **Decision:** EXCLUDE from v1
- **Why:** Free text duplicates loan_purpose; may contain PII. Candidate for embeddings (Lab 11) after a PII review.

## 18. [leakage / fairness review] application_id
- **Decision:** EXCLUDE
- **Why:** Identifier

## 19. [leakage / fairness review] state
- **Decision:** KEEP, monitor
- **Why:** Geography can proxy protected attributes; check approval rates by state.

## 20. [leakage / fairness review] age
- **Decision:** KEEP, legal review
- **Why:** Protected attribute in many jurisdictions (e.g. ECOA in the US) - legal sign-off required before production use.

## 21. [splitting] application_date
- **Decision:** train < 2025-07-01 <= test
- **Why:** The model will score FUTURE applicants; a random split hides drift
- **Alternatives considered:** random 80/20
- **Evidence:** 4475 train / 1525 test rows

## 22. [reviewed - kept as-is] education
- **Decision:** no change
- **Why:** Profiled and validated: values plausible, consistent
- **Evidence:** 4 categories, 0.0% missing: ['Bachelor', 'High School', 'Master', 'PhD']

## 23. [reviewed - kept as-is] num_credit_lines
- **Decision:** no change
- **Why:** Profiled and validated: values plausible, consistent
- **Evidence:** range 0..18, 0.0% missing

## 24. [reviewed - kept as-is] delinquencies_2y
- **Decision:** no change
- **Why:** Profiled and validated: values plausible, consistent
- **Evidence:** range 0..7, 0.0% missing

## 25. [reviewed - kept as-is] loan_term_months
- **Decision:** no change
- **Why:** Profiled and validated: values plausible, consistent
- **Evidence:** range 36..60, 0.0% missing

## 26. [target] defaulted
- **Decision:** kept as-is (0/1)
- **Why:** Defined by collections: 90+ days past due within 12 months
- **Evidence:** default rate 17.2%
