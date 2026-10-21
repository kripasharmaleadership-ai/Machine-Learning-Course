# Model Card: Yearly Medical Charges v1

- **Use:** estimate yearly charges per member at enrolment (pricing, budgeting, outreach)
- **Data:** Medical Cost Personal dataset (1,338 US members; 1 duplicate removed); 80/20 split stratified on target quintiles
- **Model:** linear regression on age, age^2, BMI, children, sex, smoker, region, smoker x (BMI >= 30)
- **Test performance:** RMSE $4,855, MAE $2,650, R^2 0.842
- **Baseline (mean per smoker group):** RMSE $7,777, R^2 0.594
- **Main drivers:** smoking (+$13,576), obese smoker (+$19,371 more), age (accelerating)
- **Known limitations:** no medical history columns -> large unexplained costs for some non-smokers;
  1,338 members from one source, may not represent other populations; linear in age^2 beyond 64 is untested
- **Fairness note:** `sex` has a small coefficient; check local regulation before using it in pricing
- **Do not use for:** individual clinical decisions; members outside age 18-64
