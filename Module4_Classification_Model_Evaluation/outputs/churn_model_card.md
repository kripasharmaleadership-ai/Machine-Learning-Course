# Model Card: Customer Churn v1 (retention campaign targeting)

- **Use:** choose which customers marketing calls with a retention offer (25% off for 3 months)
- **Data:** IBM Telco Customer Churn, 7,043 customers; 80/20 split stratified on churn; test set opened once
- **Model:** logistic regression (C = 0.5) on 19 inputs; decision threshold **0.173** tuned by 5-fold CV for campaign profit
- **Test performance:** ROC-AUC 0.842, AP 0.635, recall 0.90,
  precision 0.45, campaign profit $71,565 on 1,409 customers
- **Baselines (test):** call month-to-month $67,167;
  call everybody $52,799; same model at 0.5 $56,165
- **Main drivers:** short tenure, month-to-month contract, fibre optic internet (tenure / charges are collinear: explain as a group)
- **Known blind spot:** churners on two-year contracts are almost never called, one-year about half (low predicted risk) -> separate renewal campaign
- **Fairness note:** seniors are called at a higher rate (higher churn); review the offer policy before launch
- **Assumptions to re-check:** 35% offer success rate, 12-month value horizon. Re-tune the threshold if they change
- **Do not use for:** pricing decisions, credit decisions, or any customer-level decision other than outreach
