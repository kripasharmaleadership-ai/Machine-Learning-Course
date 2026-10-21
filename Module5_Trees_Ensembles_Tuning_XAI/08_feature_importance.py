"""
LAB 08 - Feature Importance: Three Methods, Three Different Questions
=====================================================================

    IMPURITY importance (MDI)   feature_importances_ of a tree ensemble: total Gini decrease of the
                                splits on that feature, on the TRAINING data. Free, but biased towards
                                features with many distinct values, and blind to over-fitting.
    PERMUTATION importance      shuffle one column, measure how much the score drops. Model-agnostic.
                                On held-out data it answers 'how much does the model RELY on this?'
    DROP-COLUMN importance      retrain without the column, measure the drop. Expensive. Answers
                                'how much worse would we be WITHOUT this feature?'

    None of them answers 'does this feature CAUSE the outcome?' (Level 3).

THE DATA: credit applicants with a KNOWN truth. Three features have NO effect on default:
    open_cards (but correlated with utilisation), branch_code (200 random branch numbers),
    and noise (a column of random numbers we add). A good importance method should rank them last.

LEVELS
    EASY   : Impurity importance of a random forest, and what it sums to.
    MEDIUM : Impurity vs permutation (train / test) vs drop-column, against the known truth.
    HARD   : Correlated twins split the credit; proxies look important; the ranking depends on the model.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (CREDIT_NUM, HAS_MPL, RANDOM_STATE, banner, credit_true_logit, make_credit_applicants, note, plt,
                          save_fig, show, step)
from tree_toolkit import drop_column_importance, permutation_importance_df

credit = make_credit_applicants(n=10_000, seed=8, with_truth=True)
rng = np.random.default_rng(0)
data = credit[CREDIT_NUM].copy()
data["employment_risk"] = credit["employment"].map({"salaried": 0, "self_employed": 1, "contract": 2, "unemployed": 3})
data["branch_code"] = credit["branch_code"].str[2:].astype(int)       # 200 levels, pure noise
data["noise"] = rng.normal(size=len(data))                            # pure noise, continuous
y = credit["default"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(data, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
NO_EFFECT = ["open_cards", "branch_code", "noise"]


def auc(model, X, y_):
    return roc_auc_score(y_, model.predict_proba(X)[:, 1])


# %% =========================================================================
# LEVEL 1 (EASY): Impurity importance
# ============================================================================
banner("feature_importances_ of a random forest", "EASY")
rf = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
mdi = pd.Series(rf.feature_importances_, index=data.columns).sort_values(ascending=False)
print(mdi.round(4).rename("impurity importance (sums to 1)").to_string())
show("Sum", f"{mdi.sum():.3f}")
show("Test AUC of this forest", f"{auc(rf, X_te, y_te):.3f}")
note(f"""For every split, the Gini decrease is credited to the split's feature; summed
over all trees and normalised to 1. Look at the ranking: 'branch_code' (pure
noise!) is #{list(mdi.index).index('branch_code') + 1} and 'noise' is #{list(mdi.index).index('noise') + 1}, ahead of real risk drivers. A default forest
grows its trees until the leaves are pure, and noise features with many
distinct values offer lots of thresholds to carve out individual training
rows. Impurity importance measures how the model FIT the training data,
including the noise it memorised.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Three methods vs the truth
# ============================================================================
banner("Impurity vs permutation vs drop-column", "MEDIUM")
perm_tr = permutation_importance_df(rf, X_tr, y_tr, auc, n_repeats=3)
perm_te = permutation_importance_df(rf, X_te, y_te, auc, n_repeats=5)
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=1)
drop = drop_column_importance(lambda cols: RandomForestClassifier(150, min_samples_leaf=10, n_jobs=-1, random_state=0),
                              X_fit, y_fit, X_val, y_val, auc)
comp = pd.DataFrame({"impurity (MDI)": mdi, "permutation on TRAIN": perm_tr, "permutation on TEST": perm_te,
                     "drop-column (validation)": drop})
comp["truly has an effect?"] = ["no" if c in NO_EFFECT else "yes" for c in comp.index]
comp = comp.sort_values("permutation on TEST", ascending=False)
print(comp.to_string(float_format=lambda v: f"{v:.4f}"))
ranks = comp.drop(columns="truly has an effect?").rank(ascending=False)
worst_rank_real = ranks.loc[[c for c in comp.index if c not in NO_EFFECT]].max()
best_rank_fake = ranks.loc[NO_EFFECT].min()
print("\n  Does the method rank ALL no-effect features below ALL real ones?")
print((best_rank_fake > worst_rank_real).rename("clean separation").to_string())
note("""Only impurity importance is fooled: it puts the two pure-noise columns in the
middle of the ranking. Permutation importance ranks all three no-effect
features last, on train and test alike (on this data; with a model that
memorises harder, train-set permutation can be fooled too, so use held-out
data). Drop-column is the noisiest: retraining changes the model, small
validation sets add noise, and correlated features cover for a dropped one, so
it even gives NEGATIVE importance to real drivers. Rule: rank with permutation
importance on held-out data; use drop-column only for a specific question
('can we stop collecting this field?'), with repeated CV.""")

# %% =========================================================================
# LEVEL 3 (HARD): Twins, proxies, and model dependence
# ============================================================================
banner("What importance cannot tell you", "HARD")

step("3.1  Correlated twins split the credit")
twin_tr = X_tr.assign(debt_ratio_v2=X_tr.debt_ratio + rng.normal(0, 0.01, len(X_tr)))   # a second, near-identical source
twin_te = X_te.assign(debt_ratio_v2=X_te.debt_ratio + rng.normal(0, 0.01, len(X_te)))
rf_twin = RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=0).fit(twin_tr, y_tr)
rf_one = RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
p_one = permutation_importance_df(rf_one, X_te, y_te, auc, n_repeats=5)
p_twin = permutation_importance_df(rf_twin, twin_te, y_te, auc, n_repeats=5)
both = twin_te.copy()
perm = rng.permutation(len(both))
both[["debt_ratio", "debt_ratio_v2"]] = both[["debt_ratio", "debt_ratio_v2"]].to_numpy()[perm]
group_drop = auc(rf_twin, twin_te, y_te) - auc(rf_twin, both, y_te)
Xf, Xv, yf, yv = train_test_split(twin_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=1)
drop_twin = drop_column_importance(lambda cols: RandomForestClassifier(150, min_samples_leaf=10, n_jobs=-1, random_state=0),
                                   Xf, yf, Xv, yv, auc, columns=["debt_ratio", "debt_ratio_v2"])
twin_tab = pd.Series({"debt_ratio alone (no twin), permutation": p_one["debt_ratio"],
                      "with twin: debt_ratio, permutation": p_twin["debt_ratio"],
                      "with twin: debt_ratio_v2, permutation": p_twin["debt_ratio_v2"],
                      "with twin: shuffle BOTH together": group_drop,
                      "with twin: drop debt_ratio (retrain)": drop_twin["debt_ratio"],
                      "with twin: drop debt_ratio_v2 (retrain)": drop_twin["debt_ratio_v2"]}, name="AUC drop")
print(twin_tab.to_string(float_format=lambda v: f"{v:.4f}"))
note(f"""Add a second copy of debt_ratio (say, from another data source). Permutation
importance now SPLITS the credit between the twins: each looks less important
than debt_ratio did alone, because when one is shuffled the model leans on the
other. Drop-column is worse: removing EITHER twin costs ~nothing, so 'neither
matters', while dropping both would hurt badly. Shuffle correlated features as
a GROUP ({group_drop:.3f}) to see their joint importance. Check correlations
before reading any importance ranking.""")

step("3.2  Proxies look important; importance is not causation")
no_util = [c for c in X_tr.columns if c != "utilisation"]
rf_proxy = RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=0).fit(X_tr[no_util], y_tr)
p_proxy = permutation_importance_df(rf_proxy, X_te[no_util], y_te, auc, n_repeats=5)
show("open_cards importance, utilisation IN the model", f"{permutation_importance_df(rf_one, X_te, y_te, auc, n_repeats=5)['open_cards']:.4f}")
show("open_cards importance, utilisation LEFT OUT", f"{p_proxy['open_cards']:.4f}  (rank {list(p_proxy.index).index('open_cards') + 1} of {len(p_proxy)})")
probe = credit.loc[X_te.index].copy()
before = rf_proxy.predict_proba(X_te[no_util])[:, 1]
after = rf_proxy.predict_proba(X_te[no_util].assign(open_cards=np.maximum(X_te.open_cards - 3, 0)))[:, 1]
true_before = 1 / (1 + np.exp(-credit_true_logit(probe)))
true_after = 1 / (1 + np.exp(-credit_true_logit(probe.assign(open_cards=np.maximum(probe.open_cards - 3, 0)))))
show("Intervention 'close 3 cards': model's predicted risk change", f"{(after - before).mean():+.4f}")
show("Intervention 'close 3 cards': TRUE risk change", f"{(true_after - true_before).mean():+.4f}")
note("""open_cards has NO effect on default; it is correlated with card utilisation,
which does. With utilisation in the model, open_cards is (rightly) ignored.
Leave utilisation out and open_cards becomes an important feature: a PROXY.
A model-based recommendation 'close 3 cards to lower your risk score' would
move the score (see the intervention above), but not the real risk (truth: zero). Importance says
what the MODEL uses, not what would happen if you changed the feature.
Causal questions need causal methods (experiments, causal inference).""")

step("3.3  The ranking depends on the model")
models = {"random forest (leaf 10)": rf_one,
          "gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X_tr, y_tr),
          "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X_tr, y_tr)}
rank_tab = pd.DataFrame({name: permutation_importance_df(m, X_te, y_te, auc, n_repeats=5).rank(ascending=False)
                         for name, m in models.items()}).astype(int).sort_values("random forest (leaf 10)")
print(rank_tab.to_string())
print("  test AUC: " + ", ".join(f"{name} {auc(m, X_te, y_te):.3f}" for name, m in models.items()))
note("""Three reasonable models, three importance rankings. The top drivers agree, but
the middle of the list moves: logistic regression can only use straight-line
effects (it can't use the 'young AND high utilisation' interaction or the
U-shape in age), so it relies on different features. Feature importance
describes a MODEL, not the world. Report it with the model, the data split
and the method, and treat small rank differences as noise.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    c = comp.drop(columns="truly has an effect?")
    c = c / c.abs().max()                                                 # each method scaled to its own maximum
    order = c.index[::-1]
    for k, col in enumerate(c.columns):
        axes[0].barh(np.arange(len(order)) + (k - 1.5) * 0.2, c.loc[order, col], 0.2, label=col)
    axes[0].set_yticks(range(len(order)), [f"{o}{' (no effect)' if o in NO_EFFECT else ''}" for o in order], fontsize=8)
    axes[0].set(xlabel="importance (each method scaled to its max)", title="Four importance estimates")
    axes[0].legend(fontsize=7)
    axes[1].barh(twin_tab.index, twin_tab.values, color=["#9d9d9d", "#4c78a8", "#4c78a8", "#54a24b", "#e45756", "#e45756"])
    axes[1].set(xlabel="AUC drop", title="Correlated twins split the credit")
    axes[1].tick_params(axis="y", labelsize=8)
    save_fig(fig, "08_feature_importance")

banner("KEY TAKEAWAYS")
print("""  1. Impurity importance (MDI) is free but rewards memorised noise and many-valued features.
  2. Permutation importance on HELD-OUT data shows what the model relies on; drop-column what we'd lose.
  3. Correlated features split importance; shuffle them as a group; drop-column can say 'neither matters'.
  4. Proxies look important: importance is not causation, and not advice ('close your cards').
  5. Importance describes one model: report the model, the data and the method with it.""")
