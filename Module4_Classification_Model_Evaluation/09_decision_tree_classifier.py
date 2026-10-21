"""
LAB 09 - The Decision-Tree Classifier
=====================================

THE ALGORITHM
    Ask a sequence of yes/no questions ("late_payments <= 0.5?"), each one chosen to make
    the groups it creates as PURE as possible (mostly one class). Stop at a depth / size limit.
    A leaf predicts the class mix of the training rows that ended up in it:
        P(default | leaf) = defaulters in the leaf / rows in the leaf

    Purity is measured with GINI IMPURITY:  G = 1 - sum_c p_c^2
        a pure node (all one class) -> 0;  a 50/50 node -> 0.5
    The best split maximises:  G(parent) - weighted average G(children)

    (Module 5 goes deeper: entropy, information gain, pruning, ensembles. Here: trees as
     CLASSIFIERS - rules, probabilities, and how they behave on imbalanced data.)

LEVELS
    EASY   : A depth-2 tree on loan applicants, printed as rules; one split's Gini by hand.
    MEDIUM : A fraud tree: depth vs over-fitting, min_samples_leaf, cost-complexity pruning.
    HARD   : Trees find interactions on their own; deep-tree probabilities are 0/1 and
             over-confident; small data changes -> a different tree.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from ml_lab_utils import (HAS_MPL, LOAN_FEATURES, RANDOM_STATE, banner, fraud_features, make_loan_applicants,
                          make_transactions, note, plt, save_fig, show, step)


def gini(labels) -> float:
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    p = labels.mean()
    return 1 - p ** 2 - (1 - p) ** 2


# %% =========================================================================
# LEVEL 1 (EASY): A tree you can read
# ============================================================================
banner("A depth-2 tree for loan defaults", "EASY")
loans = make_loan_applicants(n=8000, seed=21)
Xl, yl = loans[LOAN_FEATURES], loans["default"].to_numpy()
small_tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xl, yl)
print(export_text(small_tree, feature_names=LOAN_FEATURES, show_weights=True, decimals=3))
leaf_rates = small_tree.tree_.value[small_tree.apply(Xl), 0, 1]
note(f"""Read it top-down like an underwriting checklist. 'weights: [repaid, defaulted]'
are the training counts in each leaf, and P(default) for a new applicant is
the defaulted share of the leaf they fall into. The whole model fits on an
index card: that is the main appeal of a shallow tree. Look at 'class': EVERY
leaf predicts 0. Even the riskiest leaf ({leaf_rates.max():.0%} defaulters) is below 50%, so at the
default threshold this tree approves everyone. The probabilities are useful;
the 0.5 decision is not (Labs 04-06).""")

step("1.1  Why THAT first question? Gini impurity, by hand")
root_feat = LOAN_FEATURES[small_tree.tree_.feature[0]]
root_thr = small_tree.tree_.threshold[0]
left, right = yl[Xl[root_feat] <= root_thr], yl[Xl[root_feat] > root_thr]
g_parent = gini(yl)
g_split = len(left) / len(yl) * gini(left) + len(right) / len(yl) * gini(right)
show("Gini of all applicants (default rate %.3f)" % yl.mean(), f"{g_parent:.4f}")
show(f"Left  ({root_feat} <= {root_thr:.3f}): n, default rate", f"{len(left)}, {left.mean():.3f}  -> Gini {gini(left):.4f}")
show(f"Right ({root_feat} >  {root_thr:.3f}): n, default rate", f"{len(right)}, {right.mean():.3f}  -> Gini {gini(right):.4f}")
show("Gini decrease of the chosen split", f"{g_parent - g_split:.5f}")
rows = {}
for f in LOAN_FEATURES:
    vals = np.unique(np.quantile(Xl[f], np.linspace(0.02, 0.98, 97)))
    best = max(vals, key=lambda t: g_parent - (np.mean(Xl[f] <= t) * gini(yl[Xl[f] <= t])
                                               + np.mean(Xl[f] > t) * gini(yl[Xl[f] > t])))
    l_, r_ = yl[Xl[f] <= best], yl[Xl[f] > best]
    rows[f] = {"best threshold": best, "Gini decrease": g_parent - (len(l_) * gini(l_) + len(r_) * gini(r_)) / len(yl)}
print(pd.DataFrame(rows).T.sort_values("Gini decrease", ascending=False).round(5).to_string())
note("""The tree tries every feature and every threshold and keeps the split with the
largest drop in impurity. Our brute-force scan finds the same winner. That
greedy search is the whole training algorithm; it is then repeated inside
each child node.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): A tree for fraud, and how to stop it over-fitting
# ============================================================================
banner("Growing, limiting and pruning a fraud tree", "MEDIUM")
tx = make_transactions(n=60_000, seed=1)
X, y = fraud_features(tx), tx["is_fraud"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

step("2.1  Depth: from a stump to memorisation")
rows = {}
for depth in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    rows[str(depth)] = {"leaves": t.get_n_leaves(),
                        "train AP": average_precision_score(y_tr, t.predict_proba(X_tr)[:, 1]),
                        "CV AP": cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=0), X_tr, y_tr,
                                                 cv=cv, scoring="average_precision").mean()}
dsweep = pd.DataFrame(rows).T.astype({"leaves": int})
dsweep.index.name = "max_depth"
print(dsweep.round(3).to_string())
note("""Unlimited depth grows a leaf for almost every fraud: train AP near 1, CV AP
collapses. That's over-fitting (Module 3). The best depth is moderate.
Average precision (not accuracy) is the yardstick, because frauds are rare.""")

step("2.2  Two better brakes: minimum leaf size and cost-complexity pruning")
rows = {}
for leaf in [1, 10, 50, 200]:
    m = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0)
    rows[f"min_samples_leaf={leaf}"] = {"leaves": m.fit(X_tr, y_tr).get_n_leaves(),
                                        "CV AP": cross_val_score(m, X_tr, y_tr, cv=cv, scoring="average_precision").mean()}
path = DecisionTreeClassifier(min_samples_leaf=10, random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
for a in np.quantile(path.ccp_alphas, [0.5, 0.9, 0.97, 0.99]):
    m = DecisionTreeClassifier(min_samples_leaf=10, ccp_alpha=a, random_state=0)
    rows[f"pruned, ccp_alpha={a:.1e}"] = {"leaves": m.fit(X_tr, y_tr).get_n_leaves(),
                                          "CV AP": cross_val_score(m, X_tr, y_tr, cv=cv, scoring="average_precision").mean()}
brakes = pd.DataFrame(rows).T.astype({"leaves": int})
print(brakes.round(3).to_string())
note("""min_samples_leaf forbids leaves too small to estimate a fraud rate from.
Cost-complexity pruning grows the full tree, then removes the branches whose
impurity gain doesn't pay for the extra leaves (penalty alpha per leaf). Both
give much smaller trees that generalise better than an unlimited one.""")

# %% =========================================================================
# LEVEL 3 (HARD): Interactions, probabilities, instability
# ============================================================================
banner("What trees do that lines can't, and what they do badly", "HARD")

step("3.1  A tree finds an interaction on its own")
seg = pd.DataFrame({"night": X_te.is_night.to_numpy(), "foreign": X_te.is_foreign.to_numpy(), "y": y_te})
tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=0).fit(X_tr, y_tr)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X_tr, y_tr)
seg["tree"] = tree.predict_proba(X_te)[:, 1]
seg["logistic"] = lr.predict_proba(X_te)[:, 1]
tab = seg.groupby(["night", "foreign"]).agg(n=("y", "size"), actual_fraud_rate=("y", "mean"),
                                            tree_predicts=("tree", "mean"), logistic_predicts=("logistic", "mean"))
print(tab.round(4).to_string())
nf = tab.loc[(1, 1)]
note(f"""The data contains an INTERACTION: a foreign transaction at night is far riskier
than 'night' plus 'foreign' would suggest. Logistic regression adds the two
effects in log-odds and under-predicts that segment ({nf.logistic_predicts:.1%} vs {nf.actual_fraud_rate:.1%}
actual); the tree simply asks 'night?' then 'foreign?' and estimates the rate
of that leaf ({nf.tree_predicts:.1%}). Trees discover interactions without being told; a
linear model needs the product feature engineered by hand (Lab 02, 3.1).""")

step("3.2  Deep trees give 0/1 'probabilities'")
rows = {}
for name, m in {"unlimited depth": DecisionTreeClassifier(random_state=0),
                "min_samples_leaf=100": DecisionTreeClassifier(min_samples_leaf=100, random_state=0),
                "logistic regression": lr}.items():
    p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    rows[name] = {"distinct probabilities": len(np.unique(p)), "share exactly 0 or 1": np.isin(p, [0, 1]).mean(),
                  "ROC-AUC": roc_auc_score(y_te, p), "Brier": brier_score_loss(y_te, p),
                  "log loss": log_loss(y_te, np.clip(p, 1e-15, 1 - 1e-15))}
print(pd.DataFrame(rows).T.astype({"distinct probabilities": int}).round(4).to_string())
note("""An unlimited tree puts each training fraud in its own pure leaf, so almost
every prediction is exactly 0 or exactly 1. Every fraud it scores 0 is a
'certain' mistake, and log loss explodes. Bigger leaves average more rows and
give usable probabilities; forests (Lab 10) average many trees for smoother
ones still.""")

step("3.3  Instability: a slightly different sample, a different tree")
rng = np.random.default_rng(0)
roots, preds = [], []
for i in range(30):
    idx = rng.choice(len(X_tr), int(0.9 * len(X_tr)), replace=False)        # 90% of the same training data
    t = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=0).fit(X_tr.iloc[idx], y_tr[idx])
    roots.append(f"{X.columns[t.tree_.feature[0]]} | then {X.columns[t.tree_.feature[1]]}")
    preds.append(t.predict_proba(X_te)[:, 1] >= 0.2)
preds = np.array(preds)
flagged_any = preds.any(axis=0)
unanimous = (preds.all(axis=0) | ~preds.any(axis=0))
print(pd.Series(roots).value_counts().rename("first two questions (30 trees)").to_string())
show("Test transactions flagged by at least one tree", int(flagged_any.sum()))
show("... of which ALL 30 trees agree on the flag", int(preds.all(axis=0).sum()))
show("Decisions that depend on which 90% we sampled", f"{(~unanimous).sum()} transactions")
note(f"""Thirty trees, each trained on a random 90% of the SAME data. The top of the
tree is stable (same first two questions every time), but the details are
not: of the {int(flagged_any.sum())} transactions flagged by at least one tree, {(~unanimous).sum()} are flagged by
some trees and cleared by others. A single tree's exact rules should not be
trusted as 'the' truth. Averaging many such trees cancels this variance: that
is the random forest (Lab 10).""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    plot_tree(small_tree, feature_names=LOAN_FEATURES, class_names=["repay", "default"], filled=True,
              impurity=True, proportion=True, rounded=True, fontsize=9, ax=axes[0])
    axes[0].set_title("Depth-2 loan tree (value = class proportions)")
    axes[1].plot(range(len(dsweep)), dsweep["train AP"], "o-", label="train AP")
    axes[1].plot(range(len(dsweep)), dsweep["CV AP"], "o-", label="CV AP")
    axes[1].set_xticks(range(len(dsweep)), dsweep.index)
    axes[1].set(xlabel="max_depth", ylabel="average precision", title="Fraud tree: depth vs over-fitting")
    axes[1].legend()
    save_fig(fig, "09_decision_tree")

banner("KEY TAKEAWAYS")
print("""  1. A tree asks greedy yes/no questions that make the groups purest (largest Gini decrease).
  2. P(class) = class share in the leaf: shallow trees are readable rule lists.
  3. Unlimited trees memorise; limit depth / leaf size or prune, and tune on AP for rare classes.
  4. Trees find interactions on their own; linear models need them engineered.
  5. Deep trees give 0/1 probabilities and change with small data changes: average many (Lab 10).""")
