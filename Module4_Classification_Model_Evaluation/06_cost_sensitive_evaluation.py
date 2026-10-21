"""
LAB 06 - Cost-Sensitive Model Evaluation
========================================

THE IDEA
    Metrics like F1 silently assume that a false alarm and a miss are equally bad.
    The business knows better: put a PRICE on every cell of the confusion matrix and
    evaluate the model in money.

        cost matrix (card fraud)     predicted legit          predicted FRAUD (alert)
        actual legit                 TN: $0                   FP: $30 (analyst review + angry customer)
        actual FRAUD                 FN: $ amount stolen      TP: $20 (analyst review)

        total cost = sum over transactions of the cost of the cell they land in

DECISION THEORY (needs CALIBRATED probabilities, Lab 04)
    Flag a transaction when the expected cost of flagging < expected cost of not flagging:
        p * C_TP + (1 - p) * C_FP  <  p * C_FN + (1 - p) * C_TN
        <=>  p  >  p* = (C_FP - C_TN) / ((C_FP - C_TN) + (C_FN - C_TP))
    If C_FN depends on the transaction (its amount), so does the threshold.

LEVELS
    EASY   : The cost matrix; total cost at each threshold; the cheapest threshold.
    MEDIUM : The Bayes threshold p* derived and checked; it fails with uncalibrated scores.
    HARD   : Example-dependent costs (a threshold per transaction); a capacity limit
             (rank by expected $, not by probability); how robust is the choice when the
             costs themselves are uncertain?
"""
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (HAS_MPL, RANDOM_STATE, banner, fraud_features, make_transactions, note, plt, save_fig, show,
                          step)
from classification_toolkit import bayes_threshold, binary_metrics, decision_cost, threshold_sweep

COST_FP, COST_TP = 30.0, 20.0             # review cost (+ customer friction for false alarms)

tx = make_transactions(n=60_000, seed=11)
X, y, amount = fraud_features(tx), tx["is_fraud"].to_numpy(), tx["amount"].to_numpy()
idx_tr, idx_tmp = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=RANDOM_STATE)
idx_va, idx_te = train_test_split(idx_tmp, test_size=0.5, stratify=y[idx_tmp], random_state=RANDOM_STATE)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X.iloc[idx_tr], y[idx_tr])
p_va, p_te = model.predict_proba(X.iloc[idx_va])[:, 1], model.predict_proba(X.iloc[idx_te])[:, 1]
y_va, y_te, amt_va, amt_te = y[idx_va], y[idx_te], amount[idx_va], amount[idx_te]
AVG_FRAUD = float(amount[idx_tr][y[idx_tr] == 1].mean())

# %% =========================================================================
# LEVEL 1 (EASY): Evaluate in money
# ============================================================================
banner("Put a price on every cell of the confusion matrix", "EASY")
show("Train / validation / test transactions", f"{len(idx_tr):,} / {len(idx_va):,} / {len(idx_te):,}")
show("Average fraud amount (training data)", f"${AVG_FRAUD:,.0f}")
print(f"""
    cost per transaction        predicted legit        predicted FRAUD
    actual legit                $0                     ${COST_FP:.0f}  (FP)
    actual FRAUD                ${AVG_FRAUD:,.0f} (FN, avg)       ${COST_TP:.0f}  (TP)
""")
sweep = threshold_sweep(y_va, p_va, thresholds=np.r_[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8],
                        cost_fp=COST_FP, cost_fn=AVG_FRAUD, cost_tp=COST_TP)
none_cost = decision_cost(y_va, np.zeros_like(y_va), COST_FP, AVG_FRAUD, COST_TP)
all_cost = decision_cost(y_va, np.ones_like(y_va), COST_FP, AVG_FRAUD, COST_TP)
print(sweep[["threshold", "TP", "FP", "FN", "precision", "recall", "F1", "cost"]].to_string(
    index=False, float_format=lambda v: f"{v:.3f}" if v < 10 else f"{v:,.0f}"))
best = sweep.loc[sweep.cost.idxmin()]
best_f1 = sweep.loc[sweep.F1.idxmax()]
show("Cost of doing nothing (no model)", f"${none_cost:,.0f}")
show("Cost of reviewing everything", f"${all_cost:,.0f}")
show("Cheapest threshold on validation", f"{best.threshold} -> ${best.cost:,.0f}")
show("Threshold with the best F1", f"{best_f1.threshold} -> ${best_f1.cost:,.0f}")
note(f"""Doing nothing lets every fraud through; reviewing everything drowns the
analysts. The cheapest threshold ({best.threshold}) sits far below 0.5, because a missed
fraud (~${AVG_FRAUD:,.0f}) costs about {AVG_FRAUD / COST_FP:.0f}x a false alarm. Choosing by F1 would cost
${best_f1.cost - best.cost:,.0f} more on this validation set alone. The threshold is chosen on VALIDATION
data; the test set is kept for the final, honest number.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): The Bayes threshold
# ============================================================================
banner("Decision theory: the cost-optimal threshold without a search", "MEDIUM")
p_star = bayes_threshold(COST_FP, AVG_FRAUD, COST_TP)
print(f"""  Flag when the expected cost of flagging is lower:
      p * C_TP + (1 - p) * C_FP  <  p * C_FN
      p * {COST_TP:.0f} + (1 - p) * {COST_FP:.0f}  <  p * {AVG_FRAUD:,.0f}
      p  >  {COST_FP:.0f} / ({COST_FP:.0f} + {AVG_FRAUD:,.0f} - {COST_TP:.0f})  =  {p_star:.4f}""")
fine = threshold_sweep(y_va, p_va, thresholds=np.linspace(0.005, 0.3, 300), cost_fp=COST_FP, cost_fn=AVG_FRAUD,
                       cost_tp=COST_TP)
t_emp = float(fine.loc[fine.cost.idxmin(), "threshold"])
rows = {}
for name, t in [("Bayes p* (no search)", p_star), ("searched on validation", t_emp), ("default 0.5", 0.5)]:
    rows[name] = {"threshold": t, "validation cost": decision_cost(y_va, p_va >= t, COST_FP, AVG_FRAUD, COST_TP),
                  "TEST cost": decision_cost(y_te, p_te >= t, COST_FP, AVG_FRAUD, COST_TP)}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}" if v < 1 else f"${v:,.0f}"))
note("""With calibrated probabilities, the formula gives (almost) the same threshold as
an exhaustive search, with no data needed: only the business costs. On the
test set the two are within noise of each other, and both are far cheaper than 0.5.
The searched threshold can over-fit the validation set a little; p* can't.""")

step("2.1  The formula needs honest probabilities")
weighted = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(
    X.iloc[idx_tr], y[idx_tr])
pw_va, pw_te = weighted.predict_proba(X.iloc[idx_va])[:, 1], weighted.predict_proba(X.iloc[idx_te])[:, 1]
fine_w = threshold_sweep(y_va, pw_va, thresholds=np.linspace(0.005, 0.99, 400), cost_fp=COST_FP,
                         cost_fn=AVG_FRAUD, cost_tp=COST_TP)
rows = {
    "calibrated model + p*": decision_cost(y_te, p_te >= p_star, COST_FP, AVG_FRAUD, COST_TP),
    "class_weight='balanced' + p*": decision_cost(y_te, pw_te >= p_star, COST_FP, AVG_FRAUD, COST_TP),
    "class_weight='balanced' + searched threshold": decision_cost(
        y_te, pw_te >= fine_w.loc[fine_w.cost.idxmin(), "threshold"], COST_FP, AVG_FRAUD, COST_TP),
}
print(pd.Series(rows, name="TEST cost").map(lambda v: f"${v:,.0f}").to_string())
note("""The re-weighted model's scores are inflated (Lab 05), so p* flags far too many
transactions and the bill explodes. Searching a threshold on validation data
rescues it, but you lose the clean link between 'probability' and 'cost'.
Rule: p* works only with calibrated probabilities; otherwise, calibrate first
or search the threshold on validation data.""")

# %% =========================================================================
# LEVEL 3 (HARD): Real costs are per-transaction, capacity-limited and uncertain
# ============================================================================
banner("Example-dependent costs, capacity, and cost uncertainty", "HARD")

step("3.1  A missed $2,000 fraud is not a missed $12 fraud: one threshold per transaction")
p_star_each = COST_FP / (COST_FP + amt_te - COST_TP)          # p* with C_FN = this transaction's amount
p_star_each = np.where(amt_te <= COST_TP, 1.01, p_star_each)   # never worth reviewing below the review cost
policies = {
    "default 0.5": p_te >= 0.5,
    "one threshold (p* for the average fraud)": p_te >= p_star,
    "per-transaction threshold p*(amount)": p_te >= p_star_each,
}
rows = {}
for name, flag in policies.items():
    m = binary_metrics(y_te, flag)
    rows[name] = {"alerts": m["TP"] + m["FP"], "recall (count)": m["recall"],
                  "$ fraud stopped": float(amt_te[(y_te == 1) & flag].sum()),
                  "$ fraud missed": float(amt_te[(y_te == 1) & ~flag].sum()),
                  "TEST cost": decision_cost(y_te, flag, COST_FP, amt_te, COST_TP)}
pol = pd.DataFrame(rows).T
print(pol.to_string(float_format=lambda v: f"{v:.3f}" if v < 1 else f"{v:,.0f}"))
saving = pol["TEST cost"].iloc[1] - pol["TEST cost"].iloc[2]
note(f"""Now C_FN is the amount of THIS transaction, so p* = 30 / (30 + amount - 20):
a $2,000 transaction is worth reviewing at a 1.5% fraud probability, a $25
one only above ~86%. The per-transaction rule may even catch FEWER frauds by
count, but it catches the EXPENSIVE ones. It saves ${saving:,.0f} on the test
set versus the single best threshold, using the same model.""")

step("3.2  Capacity: the team can review only 120 alerts on this test set")
CAP = 120
by_prob = np.argsort(-p_te)[:CAP]
by_value = np.argsort(-(p_te * amt_te - COST_FP * (1 - p_te)))[:CAP]
rows = {}
for name, chosen in [("top-120 by fraud probability", by_prob), ("top-120 by expected $ saved", by_value)]:
    flag = np.zeros(len(y_te), bool)
    flag[chosen] = True
    rows[name] = {"frauds caught": int((y_te[flag] == 1).sum()), "precision": float(y_te[flag].mean()),
                  "$ fraud stopped": float(amt_te[flag & (y_te == 1)].sum()),
                  "TEST cost": decision_cost(y_te, flag, COST_FP, amt_te, COST_TP)}
cap = pd.DataFrame(rows).T
print(cap.to_string(float_format=lambda v: f"{v:.3f}" if v < 1 else f"{v:,.0f}"))
note(f"""When the bottleneck is analyst time, the question becomes 'which 120?'. Ranking
by probability maximises frauds CAUGHT and precision; ranking by expected value
(p x amount - false-alarm cost) maximises MONEY saved: ${cap['$ fraud stopped'].iloc[1] - cap['$ fraud stopped'].iloc[0]:,.0f} more fraud stopped
with the same 120 reviews. Precision-at-k and value-at-k answer different
questions; agree with the business which one is the goal.""")

step("3.3  The costs themselves are estimates: how much does a wrong guess hurt?")
assumed = [10, 30, 100]
true_costs = [10, 30, 100]
regret = {}
for a in assumed:
    t_a = bayes_threshold(a, AVG_FRAUD, COST_TP)
    regret[f"assume C_FP=${a} (t={t_a:.3f})"] = {}
    for c in true_costs:
        cost_used = decision_cost(y_te, p_te >= t_a, c, AVG_FRAUD, COST_TP)
        cost_best = decision_cost(y_te, p_te >= bayes_threshold(c, AVG_FRAUD, COST_TP), c, AVG_FRAUD, COST_TP)
        regret[f"assume C_FP=${a} (t={t_a:.3f})"][f"true C_FP=${c}"] = cost_used - cost_best
reg = pd.DataFrame(regret).T
print("  Extra cost (regret, $) versus the threshold you WOULD have picked with the true cost:")
print(reg.to_string(float_format=lambda v: f"${v:,.0f}"))
best_bill = decision_cost(y_te, p_te >= p_star, COST_FP, AVG_FRAUD, COST_TP)
note(f"""The diagonal is zero (right assumption). Off the diagonal you pay for a wrong
cost estimate. Being 3x off costs ${reg.iloc[1, 0]:,.0f}-${reg.iloc[0, 1]:,.0f} (vs a ~${best_bill / 1000:,.0f}k bill); 10x off up to
${reg.values.max():,.0f}. And the regret is ASYMMETRIC: under-estimating the false-alarm
cost (top row: threshold too low, alert flood) hurts far more than over-
estimating it. Practical rule: agree on the order of magnitude of each cost
with the business, show this regret table, lean towards the cheaper mistake,
and re-tune when the costs change. Never let a library default (0.5) decide.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(fine.threshold, fine.cost / 1000, label="calibrated model")
    axes[0].axvline(p_star, color="g", ls="--", label=f"Bayes p* = {p_star:.3f}")
    axes[0].axhline(none_cost / 1000, color="grey", ls=":", label="no model")
    axes[0].set(xlabel="threshold", ylabel="total cost ($k, validation)", title="The cost curve")
    axes[0].legend(fontsize=8)
    grid = np.logspace(0.5, 3.5, 100)
    axes[1].plot(grid, COST_FP / (COST_FP + grid - COST_TP))
    axes[1].set(xscale="log", yscale="log", xlabel="transaction amount ($)", ylabel="flag if P(fraud) >",
                title="Per-transaction threshold p*(amount)")
    axes[1].scatter(amt_te[y_te == 1], np.clip(p_te[y_te == 1], 1e-4, 1), s=6, color="#e45756", alpha=0.6,
                    label="frauds (above the line = flagged)")
    axes[1].legend(fontsize=8)
    axes[2].bar(pol.index.str.replace(" (", "\n(", regex=False), pol["TEST cost"] / 1000, color=["#9d9d9d", "#4c78a8", "#54a24b"])
    axes[2].set(ylabel="test cost ($k)", title="Same model, three decision policies")
    axes[2].tick_params(axis="x", labelsize=7)
    save_fig(fig, "06_hard_cost_sensitive")

banner("KEY TAKEAWAYS")
print("""  1. Price every cell of the confusion matrix and evaluate the model in money.
  2. With calibrated probabilities the optimal threshold is p* = C_FP / (C_FP + C_FN) (minus TP/TN costs).
  3. Uncalibrated scores (re-weighted, re-sampled) break p*: calibrate or tune on validation.
  4. When costs vary per case, use a per-case threshold; under capacity, rank by expected value.
  5. Costs are estimates: show the regret table, and re-tune when the business changes.""")
