# Pre-mortem: Telco churn model

_Imagine it is 12 months from now and this project FAILED. Why?_

Health score: **9/28** - RED - stop modelling, fix the project setup

| # | Likely cause of failure | Unanswered question | Mitigation owner | Due |
|---|---|---|---|---|
| 1 | Unclear problem | Is the decision + action written down (who does what with the score)? | _TBD_ | _TBD_ |
| 2 | Wrong metric | Is the success metric tied to money / a business KPI? | _TBD_ | _TBD_ |
| 3 | Leakage / skew | Were all features checked for availability at prediction time? | _TBD_ | _TBD_ |
| 4 | Noisy labels | Is label quality measured (agreement rate, audit sample)? | _TBD_ | _TBD_ |
| 5 | Optimistic offline results | Is there an evaluation on out-of-time data? | _TBD_ | _TBD_ |
| 6 | Silent decay | Is monitoring (drift, performance) planned before launch? | _TBD_ | _TBD_ |
| 7 | Stuck in notebook | Is there an engineering path to production (API/batch, owner, on-call)? | _TBD_ | _TBD_ |
| 8 | Unproven value | Will a holdout / A-B test measure real business impact? | _TBD_ | _TBD_ |

## Evidence from this lab
- Static model lost $77,490 vs a monitored one over 12 months.
- A systematic label gap in one region was invisible in overall AUC.
- A broken join passed row-count and distribution checks.