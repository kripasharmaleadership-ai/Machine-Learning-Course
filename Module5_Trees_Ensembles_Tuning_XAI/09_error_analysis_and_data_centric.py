"""
LAB 09 - Error Analysis and Data-Centric Model Improvement
==========================================================

MODEL-CENTRIC vs DATA-CENTRIC
    model-centric : keep the data fixed, change the model (tuning, bigger models, new algorithms)
    data-centric  : keep the model fixed, improve the DATA (fix labels, collect the missing cases,
                    remove duplicates, better features)
    Error analysis tells you which one will pay: look at WHERE the model fails before deciding WHAT to change.

STRUCTURED ERROR ANALYSIS (always on out-of-fold / validation predictions, never on the test set)
    1. per-row error  (here: Brier = (y - p)^2, or misclassification at the operating threshold)
    2. slice it       by every categorical value and numeric band: which slices have most EXCESS errors?
    3. look at rows   the most confident mistakes: a wrong model, or a wrong LABEL?
    4. hypothesise, fix the cause, re-measure; confirm once on the test set

THE SETUP (synthetic, so we know the truth): two problems are planted in the TRAINING data
    - an outsourced team in branch region R15 recorded 35% of outcomes wrongly (label noise);
      the branch region is a model feature, so the model can learn 'R15 is risky'
    - young applicants (< 30) are under-represented (80% of them were never sampled)
    The test set is clean and representative. Can error analysis find both problems?

LEVELS
    EASY   : Overall metrics look fine; slice the out-of-fold errors.
    MEDIUM : Confident mistakes reveal the bad labels; the young slice is thin. Fix each.
    HARD   : Model-centric vs data-centric scoreboard; 'cleaning' that hurts.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from scipy.stats import loguniform, randint

from ml_lab_utils import CREDIT_NUM, HAS_MPL, banner, make_credit_applicants, note, plt, save_fig, show, step
from tree_toolkit import slice_report

FEATS = CREDIT_NUM + ["employment_risk", "region_id"]


def prepare(df):
    rid = df["branch_code"].str[2:].astype(int) // 10                # 20 branch regions of 10 branches
    return df.assign(employment_risk=df["employment"].map({"salaried": 0, "self_employed": 1, "contract": 2, "unemployed": 3}),
                     region_id=rid, region="R" + rid.astype(str).str.zfill(2))


pool = prepare(make_credit_applicants(n=20_000, seed=9, with_truth=True))
rng = np.random.default_rng(0)
is_test = rng.random(len(pool)) < 0.25
test = pool[is_test].reset_index(drop=True)                          # clean, representative
train = pool[~is_test]
train = train[(train.age >= 30) | (rng.random(len(train)) < 0.20)].reset_index(drop=True)   # problem 2: few young applicants
train["label_recorded"] = train["default"]
bad = (train.region == "R15") & (rng.random(len(train)) < 0.35)                             # problem 1: label noise in R15
train.loc[bad, "label_recorded"] = 1 - train.loc[bad, "default"]
y_rec, y_te = train["label_recorded"].to_numpy(), test["default"].to_numpy()
model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)


def evaluate(m, X_tr, y_tr):
    p = m.fit(X_tr[FEATS], y_tr).predict_proba(test[FEATS])[:, 1]
    young, r15 = (test.age < 30).to_numpy(), (test.region == "R15").to_numpy()
    return {"test AUC": roc_auc_score(y_te, p), "test Brier": brier_score_loss(y_te, p),
            "R15: predicted - actual default rate": p[r15].mean() - y_te[r15].mean(),
            "AUC: age < 30": roc_auc_score(y_te[young], p[young]), "Brier: age < 30": brier_score_loss(y_te[young], p[young])}


# %% =========================================================================
# LEVEL 1 (EASY): Slice the errors
# ============================================================================
banner("The overall numbers look fine. Where does the model fail?", "EASY")
show("Training / test applicants", f"{len(train):,} / {len(test):,}")
base = evaluate(model, train, y_rec)
print(pd.Series(base).round(4).to_string())
oof = cross_val_predict(model, train[FEATS], y_rec, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                        method="predict_proba")[:, 1]
train["oof_p"] = oof
train["error"] = (y_rec - oof) ** 2                                   # per-row Brier
show("Out-of-fold AUC on training data", f"{roc_auc_score(y_rec, oof):.3f}")
slices = slice_report(train, train["error"], ["region", "employment", "age", "debt_ratio", "late_payments"], min_size=80)
slices["slice"] = slices["slice"].str.replace("(17.999,", "(18,", regex=False)
print(slices.head(8).to_string(float_format=lambda v: f"{v:.3f}"))
top = slices.iloc[0]
note(f"""We only ever look at OUT-OF-FOLD predictions on the TRAINING data (each row
scored by a model that didn't see it); the test set stays sealed. Slicing
the per-row error by every column ranks the slices by EXCESS errors
(error rate above average x size). Top of the list: '{top['slice']}', with {top['lift']:.1f}x the
average error: expected, because high-risk applicants have probabilities near
0.5, where outcomes are hardest to call. The SURPRISE is 'region = R15' at
{slices.set_index('slice').loc['region = R15', 'lift']:.1f}x: a branch region has no reason to be hard to predict. A slice that
stands out like that is a question: is the model wrong there, or the data?""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Diagnose and fix each problem
# ============================================================================
banner("Look at the rows, find the cause, fix the data", "MEDIUM")
step("2.1  Wrong model, or wrong labels? Ask a model that can't see the region")
BLIND = [f for f in FEATS if f != "region_id"]                      # applicant profile only
oof_blind = cross_val_predict(model, train[BLIND], y_rec, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
train["confident_mistake"] = ((oof_blind < 0.08) & (y_rec == 1)) | ((oof_blind > 0.6) & (y_rec == 0))
by_region = train.assign(expected=oof_blind).groupby("region").agg(
    rows=("error", "size"), recorded_default_rate=("label_recorded", "mean"),
    expected_from_profile=("expected", "mean"), confident_mistakes=("confident_mistake", "mean"))
by_region["recorded / expected"] = by_region.recorded_default_rate / by_region.expected_from_profile
by_region = by_region.sort_values("recorded / expected", ascending=False)
print(by_region.head(5).to_string(float_format=lambda v: f"{v:.3f}"))
print("\n  Confident mistakes in R15 (the profile-only model is sure; the recorded label disagrees):")
print(train[(train.region == "R15") & train.confident_mistake][["age", "debt_ratio", "late_payments", "utilisation",
                                                                 "label_recorded"]].assign(
    profile_model_p=oof_blind[((train.region == "R15") & train.confident_mistake).to_numpy()]).head(5).to_string())
r = by_region.loc["R15"]
note(f"""The full model can't help here: region is one of its features, so it simply
learned 'R15 is risky' and agrees with the bad labels (it predicts R15 well
on the training labels). The detective trick is a model that CANNOT see the
source. Judged on their profiles alone, R15's applicants should default at
{r.expected_from_profile:.0%}, like everyone else; their recorded rate is {r.recorded_default_rate:.0%} ({r['recorded / expected']:.1f}x), and {r.confident_mistakes:.0%} of
their rows are confident mistakes vs ~{by_region.confident_mistakes.median():.0%} elsewhere. Applicants who look like
everyone else but are recorded very differently point to the LABELS: here an
outsourced team. The fix is an audit of R15's outcomes, not a smarter
algorithm. ('Confident learning', in its simplest form.)""")
audited = y_rec.copy()
r15 = (train.region == "R15").to_numpy()
audited[r15] = train.loc[r15, "default"].to_numpy()                   # the audit recovers the true outcome
show("Labels corrected by the R15 audit", int((audited != y_rec).sum()))

step("2.2  The thin slice: young applicants")
young_tr, young_te = (train.age < 30).mean(), (test.age < 30).mean()
show("Share of applicants under 30: training / test (= real population)", f"{young_tr:.1%} / {young_te:.1%}")
print(pd.Series({k: v for k, v in base.items() if "30" in k}).round(4).to_string())
note(f"""Young applicants are {young_te:.0%} of real applicants but only {young_tr:.0%} of the training data
(the sampling skipped most of them). With so few examples, the model learns
little about their specific risk: in the true formula, young applicants with
high card utilisation are much riskier (an interaction). Data-centric fix:
collect more young applicants, rather than hoping a bigger model will
extrapolate.""")
extra = prepare(make_credit_applicants(n=40_000, seed=77))
extra = extra[extra.age < 30].head(2500)
show("Newly collected young applicants", len(extra))

# %% =========================================================================
# LEVEL 3 (HARD): Model-centric vs data-centric
# ============================================================================
banner("Model-centric vs data-centric: the scoreboard", "HARD")
search = RandomizedSearchCV(HistGradientBoostingClassifier(max_iter=400, early_stopping=True, random_state=0),
                            {"learning_rate": loguniform(0.02, 0.2), "max_leaf_nodes": randint(4, 48),
                             "min_samples_leaf": randint(5, 150), "l2_regularization": loguniform(1e-3, 5)},
                            n_iter=15, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc",
                            random_state=0, n_jobs=-1)
bigger = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, random_state=0)
with_young = pd.concat([train, extra], ignore_index=True)
y_young = np.r_[audited, extra["default"].to_numpy()]
board = {
    "A baseline (recorded labels, thin young slice)": base,
    "B model-centric: tuned hyperparameters": evaluate(search, train, y_rec),
    "C model-centric: a 4x bigger model": evaluate(bigger, train, y_rec),
    "D data-centric: audit R15 labels": evaluate(model, train, audited),
    "E data-centric: audit + 2,500 young applicants": evaluate(model, with_young, y_young),
}
keep = train["error"] < np.quantile(train["error"], 0.95)
board["F 'cleaning': delete the 5% highest-error rows"] = evaluate(model, train[keep], y_rec[keep])
score = pd.DataFrame(board).T
print(score.to_string(float_format=lambda v: f"{v:.4f}"))
d = score.loc["E data-centric: audit + 2,500 young applicants"] - score.loc["A baseline (recorded labels, thin young slice)"]
m_ = score.loc["B model-centric: tuned hyperparameters"] - score.loc["A baseline (recorded labels, thin young slice)"]
note(f"""Same algorithm, same clean test set:
- Tuning (B) adds {m_['test AUC']:+.3f} AUC and a bigger model (C) {score.iloc[2, 0] - score.iloc[0, 0]:+.3f}. Neither touches the
  real damage: the model over-predicts R15 customers' default rate by
  {score.iloc[0, 2]:+.1%} points, because it learned 'R15 is risky' from wrong labels. Every
  R15 customer would be over-charged or refused: a fairness and business problem.
- Auditing R15 (D) removes that bias completely ({score.iloc[3, 2]:+.1%}). Adding young applicants (E)
  improves the thin slice (AUC {d['AUC: age < 30']:+.3f}, Brier {d['Brier: age < 30']:+.4f}) and the overall AUC ({d['test AUC']:+.3f}).
- 'Cleaning' by deleting the hardest rows (F) also deletes real surprising
  defaulters, keeps most of R15's bias, and worsens the Brier score.
Error analysis pointed to two DATA problems that no model change could fix.
(Real-world caveat: audits and data collection cost money and time; error
analysis tells you whether that spend will pay.)""", keep_lines=True)

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    s8 = slices.head(8).iloc[::-1]
    axes[0].barh(s8["slice"], s8["lift"], color=["#e45756" if "R15" in s_ else "#9d9d9d" for s_ in s8["slice"]])
    axes[0].axvline(1, color="k", lw=0.8)
    axes[0].set(xlabel="error rate / average (lift)", title="Slices with the most excess errors (OOF)")
    axes[0].tick_params(axis="y", labelsize=8)
    cr = by_region.sort_values("recorded / expected")
    axes[1].barh(cr.index, cr["recorded / expected"], color=["#e45756" if r_ == "R15" else "#4c78a8" for r_ in cr.index])
    axes[1].axvline(1, color="k", lw=0.8)
    axes[1].set(xlabel="recorded default rate / expected from profile", title="Labels vs applicant profiles, by region")
    axes[1].tick_params(axis="y", labelsize=6)
    sc = score["test AUC"].iloc[::-1]
    axes[2].barh([i.split(":")[0] if ":" in i else i[:2] for i in sc.index], sc.values,
                 color=["#9d9d9d", "#54a24b", "#54a24b", "#4c78a8", "#4c78a8", "#9d9d9d"][::-1])
    axes[2].set(xlim=(sc.min() - 0.01, sc.max() + 0.005), xlabel="test AUC (clean labels)",
                title="Model-centric (blue) vs data-centric (green)")
    save_fig(fig, "09_error_analysis")

banner("KEY TAKEAWAYS")
print("""  1. Do error analysis on out-of-fold / validation predictions; keep the test set for confirmation.
  2. Slice per-row errors by every column; rank slices by EXCESS errors, not just error rate.
  3. Confident mistakes concentrated in one source point to label problems: audit, don't tune.
  4. Thin slices need more data from that slice, not a bigger model.
  5. Deleting hard rows is not cleaning; fix the cause. Often the data, not the model, is the bottleneck.""")
