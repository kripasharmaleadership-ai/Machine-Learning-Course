"""
LAB 07 - Logistic Regression
============================

THE MODEL
    log-odds:     z = b + w1 x1 + ... + wp xp              (linear, exactly as in Module 3)
    probability:  p = sigmoid(z) = 1 / (1 + e^-z)          (squashed into 0..1)
    loss:         log loss = -mean[ y log p + (1 - y) log(1 - p) ]   (a.k.a. cross-entropy)
    gradient:     d loss / d w = X.T @ (p - y) / n         (SAME form as linear regression!)

READING THE COEFFICIENTS
    w_j = change in LOG-ODDS for +1 unit of x_j, holding the others fixed
    exp(w_j) = ODDS RATIO: "each late payment multiplies the odds of default by 1.7"
    The change in PROBABILITY depends on where you start (the sigmoid is not a straight line).

LEVELS
    EASY   : Fit on loan applicants; coefficients -> odds ratios; did we recover the TRUE formula?
    MEDIUM : From scratch with gradient descent (= scikit-learn); why not linear regression or
             MSE; regularisation strength C and L1 feature selection.
    HARD   : Perfect separation (coefficients run to infinity); odds ratio != risk change;
             class weights move the intercept, not the slopes.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (HAS_MPL, LOAN_EMPLOYMENT, LOAN_FEATURES, LOAN_TRUTH, RANDOM_STATE, banner,
                          make_loan_applicants, note, plt, save_fig, show, step)
from classification_toolkit import LogisticRegressionGD, sigmoid


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features in their natural units + one-hot employment (salaried = reference)."""
    X = df[LOAN_FEATURES].copy()
    for e in list(LOAN_EMPLOYMENT)[1:]:
        X[f"emp_{e}"] = (df["employment"] == e).astype(float)
    return X


loans = make_loan_applicants(n=40_000, seed=3, with_truth=True)
X, y = design(loans), loans["default"].to_numpy()
X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(X, y, loans["_true_prob"].to_numpy(), test_size=0.25,
                                                      stratify=y, random_state=RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): Fit, read, and check against the truth
# ============================================================================
banner("Logistic regression on 30,000 loan applicants", "EASY")
lr = LogisticRegression(C=1e6, max_iter=5000).fit(X_tr, y_tr)          # C huge = (almost) no penalty
truth = {"income_k": LOAN_TRUTH["income_k"], "debt_to_income": LOAN_TRUTH["debt_to_income"],
         "credit_years": LOAN_TRUTH["credit_years"], "late_payments": LOAN_TRUTH["late_payments"],
         "loan_k": LOAN_TRUTH["loan_k"], "emp_self_employed": LOAN_EMPLOYMENT["self_employed"],
         "emp_contract": LOAN_EMPLOYMENT["contract"]}
coef = pd.DataFrame({"learned w (log-odds per unit)": lr.coef_[0], "TRUE w": pd.Series(truth)[X.columns].to_numpy(),
                     "odds ratio exp(w)": np.exp(lr.coef_[0])}, index=X.columns)
print(coef.round(4).to_string())
show("Test ROC-AUC / log loss", f"{roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1]):.3f} / "
                                f"{log_loss(y_te, lr.predict_proba(X_te)[:, 1]):.4f}")
note(f"""The data was generated from a known logistic formula, and the model recovers
it: every learned weight is close to the true one. Read them as odds ratios:
each late payment multiplies the odds of default by {coef.loc['late_payments', 'odds ratio exp(w)']:.2f}; each extra
year of credit history multiplies them by {coef.loc['credit_years', 'odds ratio exp(w)']:.3f} (a {1 - coef.loc['credit_years', 'odds ratio exp(w)']:.0%} drop); a contract
worker has {coef.loc['emp_contract', 'odds ratio exp(w)']:.1f}x the odds of a salaried applicant with the same profile.
'Holding the others fixed' applies exactly as in linear regression.""")

step("1.1  Scale matters for READING coefficients, not for the model")
std = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000)).fit(X_tr, y_tr)
w_std = pd.Series(std[-1].coef_[0], index=X.columns)
print(pd.DataFrame({"per unit": coef["learned w (log-odds per unit)"], "per standard deviation": w_std,
                    "sd of feature": X_tr.std()}).round(3).sort_values("per standard deviation", key=abs,
                                                                         ascending=False).to_string())
note("""Per unit, 'debt_to_income' has the biggest weight, but DTI only spans 0-1.
Per standard deviation (after StandardScaler) we can compare drivers: DTI,
credit history and late payments are the strongest. Same predictions, same
probabilities, just a different unit for the weights.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): How it learns, and why not plain regression
# ============================================================================
banner("Gradient descent on log loss, and why not linear regression", "MEDIUM")
step("2.1  From scratch: the same loop as Module 3, with a sigmoid and log loss")
sc = StandardScaler().fit(X_tr)
Xs_tr, Xs_te = sc.transform(X_tr), sc.transform(X_te)
gd = LogisticRegressionGD(learning_rate=1.0, n_epochs=5000).fit(Xs_tr, y_tr)
print(pd.DataFrame({"gradient descent (ours)": gd.coef_, "scikit-learn": std[-1].coef_[0]}, index=X.columns).round(4).to_string())
show("Epochs to converge", gd.n_iter_)
show("Log loss: first epoch -> last epoch", f"{gd.loss_history_[0]:.4f} -> {gd.loss_history_[-1]:.4f}")
note("""Our 20-line implementation (classification_toolkit.py) gives the same weights.
The gradient of log loss is X.T @ (p - y) / n: 'prediction minus target',
exactly the form of linear regression's gradient in Module 3. Only the
prediction (sigmoid) and the loss (log loss) changed.""")

step("2.2  Why not just run linear regression on the 0/1 target?")
lin = LinearRegression().fit(Xs_tr, y_tr)
p_lin = lin.predict(Xs_te)
show("Linear 'probabilities' below 0", f"{(p_lin < 0).mean():.1%} of applicants")
show("Linear 'probabilities' above 1", f"{(p_lin > 1).mean():.1%}")
show("ROC-AUC linear / logistic", f"{roc_auc_score(y_te, p_lin):.3f} / {roc_auc_score(y_te, gd.predict_proba(Xs_te)[:, 1]):.3f}")
p_demo = np.array([0.5, 0.9, 0.99, 0.999])
print(pd.DataFrame({"P(default) for a defaulter": p_demo, "squared error (1-p)^2": (1 - p_demo) ** 2,
                    "log loss -log p": -np.log(p_demo)}).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(pd.DataFrame({"P(default) for a NON-defaulter": p_demo, "squared error p^2": p_demo ** 2,
                    "log loss -log(1-p)": -np.log(1 - p_demo)}).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
note(f"""Linear regression on a 0/1 target (the 'linear probability model') ranks
just as well here (same AUC), but {(p_lin < 0).mean():.0%} of its 'probabilities' are below 0, and its
straight line keeps climbing where the true probability flattens out. Log loss
also punishes CONFIDENT mistakes far harder than squared error: saying 0.999
for a customer who repays costs 6.9 in log loss, but at most 1.0 in squared
error. That pressure is what makes logistic regression's probabilities honest
(Lab 04: best calibrated of the three models).""")

step("2.3  Regularisation: C = 1 / penalty strength")
rng = np.random.default_rng(0)
junk = pd.DataFrame(rng.normal(size=(len(X), 20)), columns=[f"junk_{i}" for i in range(20)], index=X.index)
Xj = pd.concat([X, junk], axis=1)
small = np.random.default_rng(1).choice(len(Xj), 1500, replace=False)        # a SMALL dataset: 1,500 applicants
Xj_small, y_small = Xj.iloc[small], y[small]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rows = {}
for C in [0.001, 0.01, 0.1, 1, 100]:
    for pen in ["l2", "l1"]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, l1_ratio=1.0 if pen == "l1" else 0.0,
                                                               solver="saga", max_iter=20000))
        ll = -cross_val_score(m, Xj_small, y_small, cv=cv, scoring="neg_log_loss").mean()
        w = m.fit(Xj_small, y_small)[-1].coef_[0]
        rows[(C, pen)] = {"CV log loss": ll, "non-zero weights (of 27)": int((np.abs(w) > 1e-6).sum()),
                          "junk weights non-zero": int((np.abs(w[7:]) > 1e-6).sum()),
                          "sum |w|": float(np.abs(w).sum())}
reg = pd.DataFrame(rows).T
reg.index.names = ["C", "penalty"]
print(reg.astype({"non-zero weights (of 27)": int, "junk weights non-zero": int}).to_string(
    float_format=lambda v: f"{v:.4f}"))
best_reg = reg["CV log loss"].idxmin()
note(f"""1,500 applicants, 7 real features + 20 columns of pure noise. Small C =
strong penalty: weights shrink (L2) or are switched off (L1). The best CV log
loss in the table is C = {best_reg[0]:g} with {best_reg[1].upper()}: it keeps {reg.loc[best_reg, 'non-zero weights (of 27)'] - reg.loc[best_reg, 'junk weights non-zero']:.0f} of the 7 real drivers
and drops {20 - reg.loc[best_reg, 'junk weights non-zero']:.0f} of the 20 junk columns. Ten times stronger (C = 0.01) and only 2 features
survive: under-fitting. It is Ridge and Lasso from Module 3, applied to log
loss. Choose C by cross-validation (LogisticRegressionCV), never by habit.""")

# %% =========================================================================
# LEVEL 3 (HARD): Separation, odds vs risk, and class weights
# ============================================================================
banner("Three things that surprise people about logistic regression", "HARD")

step("3.1  Perfect separation: the weights run to infinity")
Xsep = np.array([[1.0], [2.0], [3.0], [4.0], [6.0], [7.0], [8.0], [9.0]])
ysep = np.array([0, 0, 0, 0, 1, 1, 1, 1])                      # every x > 5 is a default: perfectly separable
rows = {}
for epochs in [100, 1000, 10_000, 100_000]:
    m = LogisticRegressionGD(learning_rate=0.5, n_epochs=epochs, tol=0).fit(Xsep - 5, ysep)
    rows[f"GD, {epochs:,} epochs"] = {"weight w": m.coef_[0], "log loss": m.loss_history_[-1],
                                      "P(default | x = 5.5)": m.predict_proba([[0.5]])[0, 1]}
for C in [1.0, 100.0]:
    m = LogisticRegression(C=C).fit(Xsep - 5, ysep)
    rows[f"scikit-learn, C = {C:g}"] = {"weight w": m.coef_[0, 0], "log loss": log_loss(ysep, m.predict_proba(Xsep - 5)),
                                        "P(default | x = 5.5)": m.predict_proba([[0.5]])[0, 1]}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}"))
note("""If one feature separates the classes perfectly, the log loss can always be
reduced by making the sigmoid steeper: the weight grows forever and the
'probabilities' become 0 or 1, with no evidence for such certainty. It happens
more than you'd think: small data, rare categories, or a leaky feature.
Regularisation (any finite C) keeps the weight finite. Huge coefficients or
convergence warnings are a symptom: look for separation or leakage.""")

step("3.2  An odds ratio is not a change in risk")
base = X_te.iloc[[0]].copy()
rows = {}
for label, dti in [("low-risk applicant (DTI 0.10)", 0.10), ("medium (DTI 0.40)", 0.40), ("high-risk (DTI 0.75)", 0.75)]:
    a = base.assign(debt_to_income=dti, late_payments=0)
    b_ = a.assign(late_payments=1)
    pa, pb = lr.predict_proba(a)[0, 1], lr.predict_proba(b_)[0, 1]
    rows[label] = {"P(default), 0 late payments": pa, "P(default), 1 late payment": pb, "change (points)": (pb - pa) * 100,
                   "odds ratio": (pb / (1 - pb)) / (pa / (1 - pa))}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""The odds ratio of one late payment is the same for everybody, but the change
in PROBABILITY is not: a few points for a low-risk applicant, many more for a
high-risk one. Report 'percentage-point change for a typical applicant' to
business users; 'multiplies the odds by 1.7' is correct but routinely misread
as 'risk 1.7x higher'.""")

step("3.3  Class weights move the intercept, not the slopes")
plain = LogisticRegression(C=1e6, max_iter=5000).fit(Xs_tr, y_tr)
balanced = LogisticRegression(C=1e6, max_iter=5000, class_weight="balanced").fit(Xs_tr, y_tr)
cmp = pd.DataFrame({"plain": np.r_[plain.intercept_, plain.coef_[0]],
                    "class_weight='balanced'": np.r_[balanced.intercept_, balanced.coef_[0]]},
                   index=["INTERCEPT"] + list(X.columns))
cmp["difference"] = cmp.iloc[:, 1] - cmp.iloc[:, 0]
print(cmp.round(3).to_string())
w_ratio = (1 - y_tr.mean()) / y_tr.mean()
show("log(class-weight ratio)", f"{np.log(w_ratio):.3f}")
show("Test ROC-AUC plain / balanced", f"{roc_auc_score(y_te, plain.predict_proba(Xs_te)[:, 1]):.4f} / "
                                     f"{roc_auc_score(y_te, balanced.predict_proba(Xs_te)[:, 1]):.4f}")
note("""For a well-specified logistic model, re-weighting the classes leaves the
slopes (nearly) untouched and shifts the intercept by about log(weight ratio).
That is a pure shift of all log-odds, i.e. a different threshold on the
SAME ranking (identical AUC), which is exactly what Lab 05 found empirically.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].bar(np.arange(len(coef)) - 0.2, coef["TRUE w"], 0.4, label="true", color="#9d9d9d")
    axes[0].bar(np.arange(len(coef)) + 0.2, coef["learned w (log-odds per unit)"], 0.4, label="learned", color="#4c78a8")
    axes[0].set_xticks(range(len(coef)), coef.index, rotation=35, fontsize=7)
    axes[0].set(ylabel="log-odds per unit", title="Logistic regression recovers the true formula")
    axes[0].legend()
    zz = np.linspace(-6, 6, 200)
    axes[1].plot(zz, sigmoid(zz), label="logistic (sigmoid)")
    order = np.argsort(p_lin)
    axes[1].plot(np.linspace(-6, 6, len(order)), np.sort(p_lin), label="linear regression on 0/1 (sorted)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axhline(1, color="k", lw=0.5)
    axes[1].set(xlabel="score", ylabel="predicted 'probability'", title="Linear predictions escape [0, 1]")
    axes[1].legend(fontsize=8)
    pp = np.linspace(0.001, 0.999, 300)
    axes[2].plot(pp, -np.log(pp), label="log loss (true class prob p)")
    axes[2].plot(pp, (1 - pp) ** 2, label="squared error")
    axes[2].set(xlabel="probability given to the TRUE class", ylabel="loss", ylim=(0, 5),
                title="Log loss punishes confident mistakes")
    axes[2].legend()
    save_fig(fig, "07_logistic_regression")

banner("KEY TAKEAWAYS")
print("""  1. Logistic regression = a linear score in log-odds + a sigmoid, trained on log loss.
  2. exp(w) is an odds ratio; the probability change depends on the starting risk.
  3. Its gradient has the same 'prediction minus target' form as linear regression's.
  4. C controls the penalty (L2 shrinks, L1 selects); the default C = 1 is already regularised.
  5. Perfect separation sends weights to infinity; class weights shift only the intercept.""")
