"""
LAB 07 - Training, Validation and Test Datasets
===============================================

THE EXAM ANALOGY
    TRAINING set   = the textbook + practice problems   -> the model LEARNS from it
    VALIDATION set = mock exams                         -> YOU use it to choose settings
                                                           (model type, depth, threshold...)
    TEST set       = the final exam, sealed in an envelope -> opened ONCE, at the very end,
                                                           to report honest performance

    If a student sees the final exam while practising, their grade means nothing.
    Same for models: any data used to make a decision is no longer "unseen".

    Typical split:   |------------ 60% train ------------|---- 20% val ----|---- 20% test ----|
    Or:              |------- 80% train+val (K-fold cross-validation) -------|---- 20% test ----|

LEVELS
    EASY   : Why we split at all (overfitting), and why we stratify.
    MEDIUM : Three-way split for tuning, K-fold cross-validation, and a simulation
             that proves 'peeking' at the test set inflates your results.
    HARD   : Enterprise splits - GROUP split (same customer in many rows) and
             TIME split (predicting the future), and how random splits lie.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_lab_utils import (HAS_MPL, RANDOM_STATE, TELCO_CATEGORICAL, TELCO_TARGET, banner,
                          make_telco_churn, note, plt, save_fig, show, step)

rng = np.random.default_rng(RANDOM_STATE)
telco = make_telco_churn()
NUM = ["senior_citizen", "tenure_months", "monthly_charges", "support_tickets_90d"]
X_all, y_all = telco[NUM + TELCO_CATEGORICAL], telco[TELCO_TARGET]
encode = ColumnTransformer([("cat", OrdinalEncoder(), TELCO_CATEGORICAL)], remainder="passthrough")

# %% =========================================================================
# LEVEL 1 (EASY): why split, and why stratify
# ============================================================================
banner("Why we never grade a model on its own homework", "EASY")
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_STATE,
                                          stratify=y_all)
tree = make_pipeline(encode, DecisionTreeClassifier(random_state=RANDOM_STATE)).fit(X_tr, y_tr)
show("Unlimited-depth tree - TRAIN accuracy", accuracy_score(y_tr, tree.predict(X_tr)))
show("Unlimited-depth tree - TEST accuracy", accuracy_score(y_te, tree.predict(X_te)))
show("Majority-class guess - TEST accuracy", 1 - y_te.mean())
note("""Near-perfect on training data, but on new customers it is WORSE than
blindly guessing 'nobody churns'. The tree memorized every quirk (OVERFITTING). Without a held-out test set we would
have shipped a model we believed was ~100% accurate.""")

step("Stratification: keep the class ratio identical in every split")
rare = pd.Series(rng.binomial(1, 0.03, 600))  # e.g. fraud: 3% positives, small dataset
print("  seed | test positive rate (random) | test positive rate (stratified)")
for seed in range(5):
    _, te_r = train_test_split(rare, test_size=0.2, random_state=seed)
    _, te_s = train_test_split(rare, test_size=0.2, random_state=seed, stratify=rare)
    print(f"  {seed:>4} | {te_r.mean():>27.2%} | {te_s.mean():>30.2%}")
show("True positive rate", f"{rare.mean():.2%}")
note("""With rare classes, a random split can give a test set with half (or
double) the real rate - your metrics then swing wildly between runs.
stratify=y fixes the ratio. Always stratify classification splits.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): tuning with validation, cross-validation, test peeking
# ============================================================================
banner("Three-way split: train -> tune on validation -> report on test", "MEDIUM")
X_trval, X_test, y_trval, y_test = train_test_split(X_all, y_all, test_size=0.2,
                                                    stratify=y_all, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25,  # 0.25*0.8 = 0.2
                                                  stratify=y_trval, random_state=RANDOM_STATE)
show("Train / Val / Test rows", f"{len(X_train)} / {len(X_val)} / {len(X_test)}")

step("Tune max_depth using ONLY the validation set")
rows = []
for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15, None]:
    m = make_pipeline(encode, DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE))
    m.fit(X_train, y_train)
    rows.append({"max_depth": str(depth),
                 "train_auc": roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]),
                 "val_auc": roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])})
tuning = pd.DataFrame(rows)
print(tuning.round(3).to_string(index=False))
best = tuning.loc[tuning.val_auc.idxmax(), "max_depth"]
show("Best depth chosen on validation", best)
note("""Train AUC keeps rising with depth (memorizing). Validation AUC rises,
peaks, then FALLS - that peak is the sweet spot between underfitting and
overfitting.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(tuning.max_depth, tuning.train_auc, marker="o", label="train")
    ax.plot(tuning.max_depth, tuning.val_auc, marker="o", label="validation")
    ax.set(xlabel="max_depth", ylabel="ROC-AUC", title="Underfit -> sweet spot -> overfit")
    ax.legend()
    save_fig(fig, "07_medium_validation_curve")

step("Open the sealed envelope ONCE: retrain on train+val, score on test")
final = make_pipeline(encode, DecisionTreeClassifier(
    max_depth=None if best == "None" else int(best), random_state=RANDOM_STATE)).fit(X_trval, y_trval)
show("Final TEST ROC-AUC (report this)", roc_auc_score(y_test, final.predict_proba(X_test)[:, 1]))

step("K-fold cross-validation: every row gets to be validation once")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, m in [
    ("tree depth=4", make_pipeline(encode, DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE))),
    ("logistic regression", make_pipeline(
        ColumnTransformer([("num", StandardScaler(), NUM),
                           ("cat", OneHotEncoder(handle_unknown="ignore"), TELCO_CATEGORICAL)]),
        LogisticRegression(max_iter=2000))),
]:
    scores = cross_val_score(m, X_trval, y_trval, cv=cv, scoring="roc_auc")
    print(f"  {name:<22} fold AUCs={np.round(scores, 3)}  mean={scores.mean():.3f} +/- {scores.std():.3f}")
note("""One validation split gives ONE noisy number. 5-fold CV gives five, so you
see the spread. If two models differ by less than the +/- spread, you cannot
honestly claim one is better.""")

step("Simulation: what happens if you 'peek' and pick the best model ON THE TEST SET")
note("""Pure noise: 30 random features, random labels. The TRUE accuracy of any
model is 50%. We try 200 model variants and keep the one with the best
score on a 'test' set of 200 rows, then check it on 10,000 fresh rows.""")
Xn = rng.normal(size=(1400, 30))
yn = rng.integers(0, 2, 1400)
Xn_tr, yn_tr, Xn_te, yn_te = Xn[:1200], yn[:1200], Xn[1200:], yn[1200:]
Xn_fresh, yn_fresh = rng.normal(size=(10_000, 30)), rng.integers(0, 2, 10_000)
best_score, best_model = 0, None
for i in range(200):
    cols = rng.choice(30, size=5, replace=False)
    m = DecisionTreeClassifier(max_depth=int(rng.integers(1, 6)), random_state=i).fit(Xn_tr[:, cols], yn_tr)
    s = accuracy_score(yn_te, m.predict(Xn_te[:, cols]))
    if s > best_score:
        best_score, best_model = s, (m, cols)
show("Best 'test' accuracy after peeking", best_score)
show("Same model on fresh data", accuracy_score(yn_fresh, best_model[0].predict(Xn_fresh[:, best_model[1]])))
note("""The 'test' score says we found signal. There is none. Selecting on the
test set turned it into a validation set and the number became optimistic.
This is exactly what happens on Kaggle public leaderboards and in teams that
'just check the test score one more time'.""")

# %% =========================================================================
# LEVEL 3 (HARD): group splits and time splits
# ============================================================================
banner("GROUP leakage: the same customer in train AND test", "HARD")
note("""Real data often has MANY rows per entity: monthly snapshots per customer,
many claims per patient, many transactions per card. A random row split puts
the same customer in train and test, and the model scores well by RECOGNIZING
customers instead of learning patterns. It will fail on NEW customers.""")
n_cust, months = 600, 12
cust_quirk = rng.normal(0, 2.0, n_cust)                      # hidden per-customer behaviour
cust_device = rng.normal(0, 1, (n_cust, 3))                  # stable per-customer 'fingerprint'
panel = pd.DataFrame({
    "customer": np.repeat(np.arange(n_cust), months),
    "usage": rng.normal(0, 1, n_cust * months),
    "dev1": np.repeat(cust_device[:, 0], months),
    "dev2": np.repeat(cust_device[:, 1], months),
    "dev3": np.repeat(cust_device[:, 2], months),
})
logit_p = 0.6 * panel.usage + np.repeat(cust_quirk, months)
panel["complaint"] = rng.binomial(1, 1 / (1 + np.exp(-logit_p)))
feats = ["usage", "dev1", "dev2", "dev3"]
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
random_cv = cross_val_score(rf, panel[feats], panel.complaint, scoring="roc_auc",
                            cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
group_cv = cross_val_score(rf, panel[feats], panel.complaint, scoring="roc_auc",
                           cv=GroupKFold(5), groups=panel.customer)
show("Rows / customers", f"{len(panel)} / {n_cust}")
show("Random-row CV AUC (optimistic)", random_cv.mean())
show("Group CV AUC (honest, new customers)", group_cv.mean())
note("""dev1-3 are a device fingerprint: useless for NEW customers, but in a random
split they let the forest look up 'this customer complained a lot before'.
GroupKFold keeps every customer entirely in train OR test. Rule: split by
the entity you will predict for in production.""")

banner("TIME leakage: training on the future to predict the past", "HARD")
note("""Production models always predict the FUTURE from the PAST. We simulate 24
monthly cohorts. Before month 12, expensive (premium) customers were the MOST
loyal. In month 12 a competitor launches cheaper premium plans: from then on,
expensive customers churn the MOST. The relationship REVERSED.""")
rows = []
for month in range(24):
    n = 400
    price = rng.uniform(20, 120, n)
    tenure = rng.integers(1, 72, n)
    price_effect = -0.02 if month < 12 else 0.04       # concept drift at month 12
    logit_m = -1 + price_effect * (price - 70) - 0.03 * (tenure - 36)
    rows.append(pd.DataFrame({"month": month, "price": price, "tenure": tenure,
                              "churn": rng.binomial(1, 1 / (1 + np.exp(-logit_m)))}))
ts = pd.concat(rows, ignore_index=True)
tf = ["price", "tenure"]

lr = LogisticRegression(max_iter=1000)
tr_r, te_r = train_test_split(ts, test_size=0.25, random_state=RANDOM_STATE, stratify=ts.churn)
random_auc = roc_auc_score(te_r.churn, lr.fit(tr_r[tf], tr_r.churn).predict_proba(te_r[tf])[:, 1])
past, future = ts[ts.month < 18], ts[ts.month >= 18]
oot_auc = roc_auc_score(future.churn, lr.fit(past[tf], past.churn).predict_proba(future[tf])[:, 1])
show("Random split AUC (what you'd promise)", random_auc)
show("Out-of-time AUC (what you'd deliver)", oot_auc)

step("Walk-forward validation: replay history, retraining every quarter")
print("  test window   | all history (expanding) | last 3 months only (sliding)")
for cutoff in [12, 15, 18, 21]:
    te_w = ts[(ts.month >= cutoff) & (ts.month < cutoff + 3)]
    expanding, sliding = ts[ts.month < cutoff], ts[(ts.month >= cutoff - 3) & (ts.month < cutoff)]
    auc_e = roc_auc_score(te_w.churn, lr.fit(expanding[tf], expanding.churn).predict_proba(te_w[tf])[:, 1])
    auc_s = roc_auc_score(te_w.churn, lr.fit(sliding[tf], sliding.churn).predict_proba(te_w[tf])[:, 1])
    print(f"  months {cutoff}-{cutoff + 2}  | {auc_e:>23.3f} | {auc_s:>28.3f}")
note("""Month 12: AUC ~0.40 - WORSE than a coin flip, because the model learned a
relationship that just reversed. A random split would never show you this.
Walk-forward validation also tells you HOW to retrain: after a regime change,
a model trained on recent data only recovers much faster than one dragging
all of history along. The out-of-time score is what stakeholders actually
experience - report THAT one.""")

banner("KEY TAKEAWAYS")
print("""  1. Train to learn, validate to choose, test ONCE to report.
  2. Always stratify classification splits; use K-fold CV to see the uncertainty.
  3. Every decision made by looking at the test set makes it less 'unseen'.
  4. Split by ENTITY (GroupKFold) when entities repeat across rows.
  5. Split by TIME when you predict the future - random splits are optimistic.""")
