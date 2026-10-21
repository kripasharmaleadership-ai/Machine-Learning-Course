# Model Card: Income > $50k estimator v1 (outreach sizing)

- **Use:** estimate the share of people likely to earn > $50k, to size an outreach campaign. NOT for individual
  credit, hiring or pricing decisions.
- **Data:** UCI Adult (1994 US census, 48,842 people); 80/20 split stratified on the target; test opened once.
- **Model:** HistGradientBoosting (tuned by random search, early stopping), native categorical handling.
- **Features:** age, education_num, capital_gain, capital_loss, hours_per_week, workclass, marital_status, occupation, relationship, native_country. Excluded: fnlwgt (sampling weight), education (duplicate), sex and race (sensitive).
- **Test performance:** ROC-AUC 0.930, log loss 0.274, accuracy 0.875 at threshold 0.5.
- **Explainable fallback:** depth-3 decision tree, test AUC 0.860.
- **Fairness:** sex not used, but relationship / marital status act as proxies. At 0.5: recall for women
  59% vs men 66%; calibrated within each group. Policy decision required before any individual use.
- **Known limitations:** 1994 data (incomes and occupations have changed); no job title, employer or region; errors concentrate
  in mid-level occupations where these fields can't separate the outcome.
- **Explanations:** exact SHAP per person (grouped features), relative to the average training person.
