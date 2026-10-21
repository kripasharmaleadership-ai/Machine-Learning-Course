"""
LAB 11 - HANDS-ON LAB: Predict Customer Churn and Plan a Retention Campaign
===========================================================================

THE BUSINESS PROBLEM
    A telecom provider loses ~26% of its customers in a month. Marketing can call customers
    and offer a retention deal (25% off for 3 months). Calling everyone is expensive and gives
    discounts to people who would have stayed anyway; calling nobody loses the churners.
    Question: WHO should we call?

THE ECONOMICS (agreed with marketing, per customer with monthly bill m)
    contact cost            $5 per call
    discount if they stay   0.75 * m          (25% off for 3 months)
    offer success rate      35% of would-be churners accept and stay
    value of a saved churner 12 months of revenue = 12 * m
        TP (call a churner)   : 0.35 * (12 m - 0.75 m) - 5        (expected value of the call)
        FP (call a stayer)    : -(5 + 0.75 m)                     (wasted call + unneeded discount)
        FN / TN (no call)     : 0

THE DATA (real): IBM Telco Customer Churn, 7,043 customers (OpenML 42178, cached in data/)

WHAT YOU WILL DO
    Part 1 (EASY)   Explore the data: churn drivers, data-quality issues
    Part 2 (EASY)   Split once (stratified); baselines, including a business rule
    Part 3 (MEDIUM) Train MULTIPLE CLASSIFIERS in pipelines; COMPARE their metrics with CV
    Part 4 (MEDIUM) HANDLE IMBALANCE: class weights vs threshold moving
    Part 5 (HARD)   TUNE THE DECISION THRESHOLD for profit; INTERPRET the confusion matrix;
                    audit segments; explain the model; open the test set once; save everything
"""
# %%
import json

import joblib
import numpy as np
import sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, make_scorer,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, TunedThresholdClassifierCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_lab_utils import (CHURN_TARGET, HAS_MPL, OUT_DIR, RANDOM_STATE, banner, load_telco_churn, note, plot_confusion,
                          plt, save_fig, show, step)
from classification_toolkit import binary_metrics, bootstrap_compare, segment_report, threshold_strategies

TARGET = CHURN_TARGET
CONTACT, DISCOUNT_MONTHS, SUCCESS, HORIZON = 5.0, 0.75, 0.35, 12


def call_value(monthly, churned):
    """Value of CALLING a customer (vs not calling): positive for churners, negative for stayers."""
    monthly, churned = np.asarray(monthly, dtype=float), np.asarray(churned).astype(bool)
    return np.where(churned, SUCCESS * (HORIZON * monthly - DISCOUNT_MONTHS * monthly) - CONTACT,
                    -(CONTACT + DISCOUNT_MONTHS * monthly))


def campaign_profit(y, call, monthly) -> float:
    return float(np.sum(call_value(monthly, y)[np.asarray(call).astype(bool)]))


# %% =========================================================================
# PART 1 (EASY): Explore
# ============================================================================
banner("PART 1 - Explore the data", "EASY")
raw = load_telco_churn()
show("Customers x columns", raw.shape)
print(raw.head(3).T.to_string())
show("Churn rate", f"{raw[TARGET].mean():.1%}")
show("Rows identical in every column", int(raw.duplicated().sum()))
show("Customers with tenure 0 (TotalCharges was blank)", int((raw.tenure == 0).sum()))
df = raw.copy()
NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT = [c for c in df.columns if c not in NUM + [TARGET]]
show("Numeric / categorical inputs", f"{len(NUM)} / {len(CAT)}")
note("""No missing values left: the 11 blank TotalCharges belong to brand-new
customers (tenure 0) and were set to 0 in the loader, a documented decision,
not a silent fill (Module 2). The 22 identical rows are NOT dropped: this copy
of the data has no customer ID, and identical plans + charges for different
customers are plausible (mostly brand-new month-to-month customers). The
positive class (churn) is 26.5%: imbalanced, but far from fraud-level rarity.""")

step("1.1  Who churns?")
for col in ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"]:
    t = df.groupby(col)[TARGET].agg(["count", "mean"]).rename(columns={"mean": "churn rate"})
    print(t.sort_values("churn rate", ascending=False).round(3).to_string(), "\n")
tenure_band = pd.cut(df.tenure, [-1, 6, 12, 24, 48, 72], labels=["0-6 m", "7-12 m", "13-24 m", "25-48 m", "49-72 m"])
print(df.groupby(tenure_band, observed=True)[TARGET].mean().round(3).rename("churn rate by tenure").to_string())
show("Correlation TotalCharges vs tenure x MonthlyCharges",
     f"{np.corrcoef(df.TotalCharges, df.tenure * df.MonthlyCharges)[0, 1]:.4f}")
note("""Month-to-month contracts churn ~15x more than two-year contracts; fibre
customers and electronic-check payers churn most; churn is concentrated in the
first months. TotalCharges is ~tenure x MonthlyCharges (correlation ~1): it is
redundant, not leaky (it is known at prediction time), so we keep it.""")

# %% =========================================================================
# PART 2 (EASY): Split once, then baselines
# ============================================================================
banner("PART 2 - Split, then create baselines", "EASY")
X, y = df[NUM + CAT], df[TARGET].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
m_tr, m_te = X_tr.MonthlyCharges.to_numpy(), X_te.MonthlyCharges.to_numpy()
show("Train / test customers", f"{len(y_tr):,} / {len(y_te):,}")
show("Churn rate train / test", f"{y_tr.mean():.3f} / {y_te.mean():.3f}")
note("""Stratified: both sets have the same churn rate. The TEST SET IS NOW SEALED until
Part 5.6. Every comparison and every threshold is chosen with cross-validation
on the training set.""")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
cv_single = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)


def oof_proba(model, X_=X_tr, y_=y_tr, folds=cv_single):
    """Out-of-fold probabilities: every training customer is scored by a model that never saw them."""
    out = np.zeros(len(y_))
    for a, b in folds.split(X_, y_):
        out[b] = clone(model).fit(X_.iloc[a], y_[a]).predict_proba(X_.iloc[b])[:, 1]
    return out


def evaluate(p, threshold=0.5, y_=y_tr, monthly=m_tr):
    m = binary_metrics(y_, p >= threshold)
    return {"ROC-AUC": roc_auc_score(y_, p), "AP": average_precision_score(y_, p), "accuracy": m["accuracy"],
            "precision": m["precision"], "recall": m["recall"], "F1": m["F1"], "Brier": brier_score_loss(y_, p),
            "campaign profit $": campaign_profit(y_, p >= threshold, monthly)}


class ContractRule:
    """Business-rule baseline: 'call every month-to-month customer'. Score = churn rate of the contract type."""

    def fit(self, X_, y_):
        self.rate_ = pd.Series(y_).groupby(X_["Contract"].to_numpy()).mean()
        return self

    def predict_proba(self, X_):
        p = X_["Contract"].map(self.rate_).to_numpy()
        return np.c_[1 - p, p]

    def get_params(self, deep=True):
        return {}


results = {"B0 call nobody (majority class)": evaluate(np.zeros(len(y_tr))),
           "B0' call everybody": evaluate(np.ones(len(y_tr))),
           "B1 rule: call month-to-month (p >= 0.3)": evaluate(oof_proba(ContractRule()), threshold=0.3)}
print(pd.DataFrame(results).T.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
base_tab = pd.DataFrame(results).T
note(f"""'Call nobody' is 73% accurate and earns $0. 'Call everybody' catches every
churner and earns ${base_tab.iloc[1]['campaign profit $']:,.0f}, but hands discounts to ~{(y_tr == 0).sum():,} loyal customers.
The one-line contract rule earns MORE (${base_tab.iloc[2]['campaign profit $']:,.0f}) with far fewer calls. The bar
for a model is not 'beat 73% accuracy'; it is 'beat the contract rule in dollars'.""")

# %% =========================================================================
# PART 3 (MEDIUM): Train multiple classifiers, compare metrics
# ============================================================================
banner("PART 3 - Train and compare five classifiers", "MEDIUM")


def prep(scale=True):
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
                             verbose_feature_names_out=False)


MODELS = {
    "M1 logistic regression": Pipeline([("prep", prep()), ("model", LogisticRegression(C=0.5, max_iter=3000))]),
    "M2 KNN (k=40)": Pipeline([("prep", prep()), ("model", KNeighborsClassifier(40))]),
    "M3 decision tree (depth 5)": Pipeline([("prep", prep(False)),
                                            ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=30,
                                                                             random_state=0))]),
    "M4 random forest": Pipeline([("prep", prep(False)),
                                  ("model", RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=0))]),
    "M5 gradient boosting": Pipeline([("prep", prep(False)),
                                      ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                               max_leaf_nodes=8, early_stopping=True,
                                                                               random_state=0))]),
}
OOF = {}
for name, model in MODELS.items():
    OOF[name] = oof_proba(model)
    results[name] = evaluate(OOF[name])
board = pd.DataFrame(results).T
print(board.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
note("""Out-of-fold predictions, all at the DEFAULT threshold 0.5. On ranking
(ROC-AUC, AP) logistic regression, the forest and boosting are within about a
point of each other; the single tree and KNN trail. At 0.5 every model has
recall around 50%: half the churners are never called. That isn't the model's
fault; it's the threshold's (Part 5).""")

step("3.1  Is the gap between the top models real? (repeated CV)")
rows = {}
for name in ["M1 logistic regression", "M4 random forest", "M5 gradient boosting"]:
    aucs = []
    for a, b in cv.split(X_tr, y_tr):
        aucs.append(roc_auc_score(y_tr[b], clone(MODELS[name]).fit(X_tr.iloc[a], y_tr[a]).predict_proba(X_tr.iloc[b])[:, 1]))
    rows[name] = {"mean ROC-AUC (10 folds)": np.mean(aucs), "sd across folds": np.std(aucs)}
rep = pd.DataFrame(rows).T
print(rep.round(4).to_string())
note("""The differences between the top three are smaller than the fold-to-fold
spread. On this data (7,000 customers, mostly categorical features with
simple effects) logistic regression is as good as the ensembles, and it is
faster, calibrated and explainable. We carry it forward as the champion and
keep boosting as the challenger.""")

# %% =========================================================================
# PART 4 (MEDIUM): Handle the imbalance
# ============================================================================
banner("PART 4 - Handle imbalanced data: class weights vs threshold moving", "MEDIUM")
champ = MODELS["M1 logistic regression"]
weighted = Pipeline([("prep", prep()), ("model", LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))])
p_plain, p_weighted = OOF["M1 logistic regression"], oof_proba(weighted)
target_recall = binary_metrics(y_tr, p_weighted >= 0.5)["recall"]
t_equiv = float(np.quantile(p_plain[y_tr == 1], 1 - target_recall))
imb = {"plain, threshold 0.5": evaluate(p_plain), "class_weight='balanced', threshold 0.5": evaluate(p_weighted),
       f"plain, threshold {t_equiv:.2f} (same recall)": evaluate(p_plain, t_equiv)}
imb_tab = pd.DataFrame(imb).T
print(imb_tab.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
show("Mean predicted churn: plain / weighted / actual", f"{p_plain.mean():.3f} / {p_weighted.mean():.3f} / {y_tr.mean():.3f}")
note("""Same lesson as Lab 05, on real data: class_weight='balanced' raises recall at
0.5 because it inflates every probability, and the plain model with a lower
threshold lands in the same place, while its probabilities stay honest (Brier).
We keep the plain, calibrated model and handle the imbalance with the
threshold, chosen for PROFIT, not for a recall target someone made up.""")

# %% =========================================================================
# PART 5 (HARD): Threshold for profit, confusion matrix, segments, test, save
# ============================================================================
banner("PART 5 - Tune the threshold, interpret, audit, test once, save", "HARD")

step("5.1  Seven threshold strategies, evaluated in dollars (out-of-fold)")
v_tp, v_fp = call_value(m_tr, np.ones(len(m_tr))), -call_value(m_tr, np.zeros(len(m_tr)))
strategies = threshold_strategies(y_tr, p_plain, cost_fp=v_fp, cost_fn=v_tp, min_recall=0.8, min_precision=0.6,
                                  capacity=int(0.2 * len(y_tr)))
strategies["campaign profit $"] = [campaign_profit(y_tr, p_plain >= t, m_tr) for t in strategies.threshold]
strategies = strategies.drop(columns=["cost"]).sort_values("campaign profit $", ascending=False)
print(strategies.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
p_star = float(np.median(v_fp / (v_fp + v_tp)))
show("Bayes threshold p* = cost_FP / (cost_FP + value_TP) (median customer)", f"{p_star:.3f}")
best_t = float(strategies.threshold.iloc[0])
note(f"""The menu of strategies on the same out-of-fold predictions. 'Default 0.5'
leaves most of the money on the table. 'Max F1' and 'Youden J' are better but
are not about money. The profit-maximising threshold ({best_t:.2f}) lands close to the
Bayes threshold p* ({p_star:.2f}) computed from the economics alone, because a saved
churner is worth ~5x an unneeded discount. Note it calls MORE customers than
the F1-optimal policy: false alarms are cheap here.""")

step("5.2  The same thing the scikit-learn way: TunedThresholdClassifierCV")
with sklearn.config_context(enable_metadata_routing=True):          # lets the scorer receive per-customer bills
    profit_scorer = make_scorer(campaign_profit).set_score_request(monthly=True)
    tuned = TunedThresholdClassifierCV(champ, scoring=profit_scorer, cv=cv_single, thresholds=200)
    tuned.fit(X_tr, y_tr, monthly=m_tr)
show("Threshold chosen by TunedThresholdClassifierCV", f"{tuned.best_threshold_:.3f}")
show("Its CV profit (mean per fold of ~1,127 customers)", f"${tuned.best_score_:,.0f}")
note("""scikit-learn >= 1.5 wraps the procedure: cross-validate the probabilities,
evaluate a business scorer at many thresholds, keep the best, then refit. It
agrees with our manual search, and the threshold becomes part of the saved
model, so production can't silently fall back to 0.5.""")
FINAL_T = float(tuned.best_threshold_)

step("5.3  Interpret the confusion matrix at the chosen threshold (out-of-fold)")
cm = binary_metrics(y_tr, p_plain >= FINAL_T)
print(f"""
                         NOT called                 CALLED
    stayed (no churn)    TN = {cm['TN']:>5,}              FP = {cm['FP']:>5,}   <- discount given to loyal customers
    churned              FN = {cm['FN']:>5,}              TP = {cm['TP']:>5,}   <- churners we get a chance to save
""")
saved = SUCCESS * cm["TP"]
note(f"""In business words, for every {len(y_tr):,} customers (the training base):
- we call {cm['TP'] + cm['FP']:,} ({(cm['TP'] + cm['FP']) / len(y_tr):.0%} of the base); {cm['precision']:.0%} of the calls reach a real churner;
- we reach {cm['recall']:.0%} of all churners; at a 35% success rate that is ~{saved:,.0f} customers kept;
- {cm['FP']:,} loyal customers get a discount they didn't need (the price of the campaign);
- {cm['FN']:,} churners leave without a call: the model's blind spot (look at them
  in 5.4 before blaming the threshold).
Every cell has a business meaning, and a cost.""", keep_lines=True)

step("5.4  Segment audit: who does the campaign reach, and who does it miss?")
seg_contract = segment_report(y_tr, p_plain, FINAL_T, X_tr["Contract"])
seg_senior = segment_report(y_tr, p_plain, FINAL_T, X_tr["SeniorCitizen"].map({"Yes": "senior", "No": "non-senior"}))
seg_tenure = segment_report(y_tr, p_plain, FINAL_T, pd.cut(X_tr.tenure, [-1, 12, 48, 72],
                                                           labels=["0-12 m", "13-48 m", "49-72 m"]).astype(str))
print(pd.concat([seg_contract, seg_senior, seg_tenure]).to_string(float_format=lambda v: f"{v:.3f}" if v < 10 else f"{v:,.0f}"))
note(f"""The campaign reaches {seg_contract.loc['Month-to-month', 'recall']:.0%} of month-to-month churners, {seg_contract.loc['One year', 'recall']:.0%} of one-year churners
and {seg_contract.loc['Two year', 'recall']:.0%} of two-year churners: their predicted risk is low, so they fall
below the threshold even when they do leave. The model is well calibrated in
every segment (mean predicted ~ prevalence); the blind spot comes from the
THRESHOLD being one-size-fits-all. Contract customers are worth more, so this
is a candidate for a separate, renewal-date-based campaign. Seniors are called at a much higher rate than
non-seniors: justified by their higher churn, but worth a fairness review
before launch (who gets discounts is a policy decision, not just a model one).""")

step("5.5  Explain the model: what drives churn?")
champ_full = clone(champ).fit(X_tr, y_tr)
names = champ_full["prep"].get_feature_names_out()
coef = pd.Series(champ_full["model"].coef_[0], index=names)
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)
print(pd.DataFrame({"log-odds": top, "odds ratio": np.exp(top)}).round(3).to_string())
perm = permutation_importance(champ_full, X_tr, y_tr, scoring="roc_auc", n_repeats=5, random_state=0)
print(pd.Series(perm.importances_mean, index=X_tr.columns).sort_values(ascending=False).head(8)
      .round(4).rename("ROC-AUC drop when shuffled").to_string())
note("""Numeric features are standardised (log-odds per standard deviation);
categories are one-hot (odds relative to the average customer, since nothing
is dropped). Longer tenure strongly lowers churn; two-year contracts lower it,
month-to-month raises it; fibre optic raises it. Permutation importance
(shuffle one raw column, measure the AUC loss) gives the same story at the
level of business inputs: tenure >> contract, internet service. Careful with
TotalCharges (+) and MonthlyCharges (-): TotalCharges ~ tenure x MonthlyCharges,
so the three are collinear and their individual signs can't be read in
isolation (Module 3, Lab 01: multicollinearity). Explain with the group.""")

step("5.6  Open the test set ONCE")
challenger = clone(MODELS["M5 gradient boosting"]).fit(X_tr, y_tr)
p_te = tuned.predict_proba(X_te)[:, 1]
p_ch = challenger.predict_proba(X_te)[:, 1]
test_tab = pd.DataFrame({
    "champion: logistic @ tuned threshold": evaluate(p_te, FINAL_T, y_te, m_te),
    "challenger: boosting @ same threshold": evaluate(p_ch, FINAL_T, y_te, m_te),
    "champion @ 0.5": evaluate(p_te, 0.5, y_te, m_te),
    "rule: call month-to-month": evaluate(ContractRule().fit(X_tr, y_tr).predict_proba(X_te)[:, 1], 0.3, y_te, m_te),
    "call everybody": evaluate(np.ones(len(y_te)), 0.5, y_te, m_te),
}).T
print(test_tab.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
ci = bootstrap_compare(y_te, p_te, p_ch, metric=roc_auc_score, n_boot=1000)
show("ROC-AUC(boosting) - ROC-AUC(logistic), 95% CI", f"{ci['diff (B - A)']:+.4f}  [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
uplift = test_tab.loc["champion: logistic @ tuned threshold", "campaign profit $"] - test_tab.loc["champion @ 0.5", "campaign profit $"]
rule_profit = test_tab.loc["rule: call month-to-month", "campaign profit $"]
champ_profit = test_tab.loc["champion: logistic @ tuned threshold", "campaign profit $"]
half_profit = test_tab.loc["champion @ 0.5", "campaign profit $"]
note(f"""One look at the sealed test set ({len(y_te):,} customers). Plot twist: the same good
model at the default 0.5 earns ${half_profit:,.0f}, LESS than the one-line contract rule
(${rule_profit:,.0f}). At the tuned threshold it earns ${champ_profit:,.0f}: ${uplift:,.0f} more from the
threshold alone, and ${champ_profit - rule_profit:,.0f} ({champ_profit / rule_profit - 1:.0%}) above the rule. A good model at the
wrong threshold loses to a simple rule. Boosting vs logistic: the AUC
difference's confidence interval includes 0, a tie; the explainable,
calibrated model wins the tie. Scale: the test set is 20% of the base, so
multiply by ~5 for a monthly campaign.""")

step("5.7  Save the model (threshold included), metrics and model card")
joblib.dump(tuned, OUT_DIR / "churn_model_v1.joblib")
champ_test = test_tab.loc["champion: logistic @ tuned threshold"]
metrics = {"model": "logistic regression (C=0.5) + TunedThresholdClassifierCV (profit)", "threshold": FINAL_T,
           "train_rows": int(len(y_tr)), "test_rows": int(len(y_te)),
           "economics": {"contact_cost": CONTACT, "discount_months": DISCOUNT_MONTHS, "success_rate": SUCCESS,
                         "horizon_months": HORIZON},
           "cv": {k: float(v) for k, v in evaluate(p_plain, FINAL_T).items()},
           "test": {k: float(v) for k, v in champ_test.items()},
           "test_auc_vs_boosting_ci": ci}
(OUT_DIR / "churn_metrics_v1.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
card = f"""# Model Card: Customer Churn v1 (retention campaign targeting)

- **Use:** choose which customers marketing calls with a retention offer (25% off for 3 months)
- **Data:** IBM Telco Customer Churn, 7,043 customers; 80/20 split stratified on churn; test set opened once
- **Model:** logistic regression (C = 0.5) on 19 inputs; decision threshold **{FINAL_T:.3f}** tuned by 5-fold CV for campaign profit
- **Test performance:** ROC-AUC {champ_test['ROC-AUC']:.3f}, AP {champ_test['AP']:.3f}, recall {champ_test['recall']:.2f},
  precision {champ_test['precision']:.2f}, campaign profit ${champ_test['campaign profit $']:,.0f} on {len(y_te):,} customers
- **Baselines (test):** call month-to-month ${test_tab.loc['rule: call month-to-month', 'campaign profit $']:,.0f};
  call everybody ${test_tab.loc['call everybody', 'campaign profit $']:,.0f}; same model at 0.5 ${test_tab.loc['champion @ 0.5', 'campaign profit $']:,.0f}
- **Main drivers:** short tenure, month-to-month contract, fibre optic internet (tenure / charges are collinear: explain as a group)
- **Known blind spot:** churners on two-year contracts are almost never called, one-year about half (low predicted risk) -> separate renewal campaign
- **Fairness note:** seniors are called at a higher rate (higher churn); review the offer policy before launch
- **Assumptions to re-check:** 35% offer success rate, 12-month value horizon. Re-tune the threshold if they change
- **Do not use for:** pricing decisions, credit decisions, or any customer-level decision other than outreach
"""
(OUT_DIR / "churn_model_card.md").write_text(card, encoding="utf-8")
print("  [saved] outputs/churn_model_v1.joblib, churn_metrics_v1.json, churn_model_card.md")

if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    grid = np.linspace(0.02, 0.9, 89)
    axes[0, 0].plot(grid, [campaign_profit(y_tr, p_plain >= t, m_tr) / 1000 for t in grid], label="out-of-fold (train)")
    axes[0, 0].axvline(FINAL_T, color="g", ls="--", label=f"tuned threshold {FINAL_T:.2f}")
    axes[0, 0].axvline(0.5, color="grey", ls=":", label="default 0.5")
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set(xlabel="threshold", ylabel="campaign profit ($k)", title="Profit curve")
    axes[0, 0].legend(fontsize=8)
    b = board.drop(index=[i for i in board.index if i.startswith("B")]).sort_values("ROC-AUC")
    axes[0, 1].barh(b.index, b["ROC-AUC"], color="#4c78a8")
    axes[0, 1].set(xlim=(0.75, 0.86), xlabel="CV ROC-AUC", title="Five classifiers (out-of-fold)")
    axes[0, 1].tick_params(axis="y", labelsize=8)
    plot_confusion(axes[1, 0], confusion_matrix(y_te, p_te >= FINAL_T), ["stayed", "churned"],
                   title=f"Test confusion matrix at {FINAL_T:.2f} (rows: actual)")
    sc = seg_contract
    xs = np.arange(len(sc))
    axes[1, 1].bar(xs - 0.2, sc["recall"], 0.4, color="#e45756", label="recall (churners reached)")
    axes[1, 1].bar(xs + 0.2, sc["alert rate"], 0.4, color="#4c78a8", label="share of segment called")
    axes[1, 1].set_xticks(xs, sc.index)
    axes[1, 1].set(title="Segment audit by contract type")
    axes[1, 1].legend(fontsize=8)
    save_fig(fig, "11_churn_lab")

banner("KEY TAKEAWAYS")
print("""  1. Baselines first: 'call nobody' is 73% accurate and worthless; the contract rule already profits.
  2. Five classifiers on the same folds: logistic regression ties the ensembles on real data.
  3. Imbalance: class weights and threshold moving land in the same place; keep the calibrated model.
  4. The threshold is a business decision: tuned for profit it beats 0.5 by a wide margin.
  5. Read the confusion matrix in business words, audit segments, open the test set once, ship a model card.""")
