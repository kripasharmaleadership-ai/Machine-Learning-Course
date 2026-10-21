"""
LAB 02 - Tree Depth, Pruning, and Visualising a Tree
====================================================

THE PROBLEM
    Left alone, a tree keeps splitting until every leaf is pure: it memorises the training data
    (Lab 01, 1.3). Each extra level halves the data in a node, so deep leaves are estimated from
    a handful of rows: high variance (Module 3, Lab 04).

TWO WAYS TO STOP IT
    PRE-pruning (stop early)       max_depth, min_samples_leaf, min_samples_split, max_leaf_nodes,
                                   min_impurity_decrease
    POST-pruning (grow, then cut)  cost-complexity pruning: minimise  impurity + alpha x (number of leaves)
                                   alpha = 0 keeps the full tree; larger alpha removes the weakest branches.
                                   scikit-learn: cost_complexity_pruning_path() + ccp_alpha

LEVELS
    EASY   : Depth vs train / validation AUC; draw and read a tree.
    MEDIUM : Pre-pruning knobs compared; the cost-complexity path; choose alpha by CV and the
             one-standard-error rule; turn the tree into ranked business rules.
    HARD   : Probabilities vs the TRUE default probability; how unstable the chosen tree is;
             the price of interpretability (small tree vs forest).
"""
# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from ml_lab_utils import (CREDIT_NUM, HAS_MPL, RANDOM_STATE, banner, make_credit_applicants, note, plt, save_fig, show,
                          step)

credit = make_credit_applicants(n=8000, seed=2, with_truth=True)
X = pd.get_dummies(credit[CREDIT_NUM + ["employment"]], columns=["employment"], dtype=int)
y, p_true = credit["default"].to_numpy(), credit["_true_prob"].to_numpy()
X_tr, X_te, y_tr, y_te, pt_tr, pt_te = train_test_split(X, y, p_true, test_size=0.3, stratify=y,
                                                        random_state=RANDOM_STATE)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
show("Train / test applicants", f"{len(y_tr):,} / {len(y_te):,}")
show("Best possible test ROC-AUC (the TRUE probabilities)", f"{roc_auc_score(y_te, pt_te):.3f}")


def cv_auc(model):
    return cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()


# %% =========================================================================
# LEVEL 1 (EASY): Depth, and reading a tree
# ============================================================================
banner("Depth: from one question to memorisation", "EASY")
rows = {}
for d in [1, 2, 3, 4, 5, 6, 8, 10, 14, None]:
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
    rows[str(d)] = {"leaves": m.get_n_leaves(), "train AUC": roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1]),
                    "CV AUC": cv_auc(DecisionTreeClassifier(max_depth=d, random_state=0))}
depth_tab = pd.DataFrame(rows).T.astype({"leaves": int})
depth_tab.index.name = "max_depth"
print(depth_tab.to_string(float_format=lambda v: f"{v:.3f}"))
best_d = depth_tab["CV AUC"].idxmax()
note(f"""Train AUC climbs to 1.0 as leaves multiply (the unlimited tree has
{depth_tab.loc['None', 'leaves']:,} leaves for {len(y_tr):,} applicants). CV AUC peaks at depth {best_d} and then falls:
past that point every new split fits noise. The classic over-fitting curve,
drawn with a tree.""")

step("1.1  Draw a depth-3 tree, and read one path")
t3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
print(export_text(t3, feature_names=list(X.columns), show_weights=True, decimals=3))
node_path = t3.decision_path(X_te.iloc[[0]]).indices
print("  Path of the first test applicant:")
for nd in node_path:
    f = t3.tree_.feature[nd]
    if f >= 0:
        val = X_te.iloc[0, f]
        go = "<=" if val <= t3.tree_.threshold[nd] else ">"
        print(f"    node {nd}: {X.columns[f]} = {val} {go} {t3.tree_.threshold[nd]:.3f}")
    else:
        print(f"    leaf {nd}: P(default) = {t3.tree_.value[nd, 0, 1]:.3f}  (from {int(t3.tree_.n_node_samples[nd])} training applicants)")
note("""The printed path IS the explanation of this prediction: a local explanation for
free (Lab 10). In plot_tree (outputs/02_tree_and_pruning.png) each box shows the
question, the Gini, the number of samples and the class mix; colour = majority
class, intensity = purity.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Pre-pruning, post-pruning, rules
# ============================================================================
banner("Pre-pruning vs cost-complexity post-pruning", "MEDIUM")
step("2.1  The pre-pruning knobs")
knobs = {
    "max_depth=5": dict(max_depth=5), "min_samples_leaf=50": dict(min_samples_leaf=50),
    "min_samples_leaf=150": dict(min_samples_leaf=150), "min_samples_split=300": dict(min_samples_split=300),
    "max_leaf_nodes=16": dict(max_leaf_nodes=16), "min_impurity_decrease=0.0005": dict(min_impurity_decrease=0.0005),
    "max_depth=8 + min_samples_leaf=40": dict(max_depth=8, min_samples_leaf=40),
}
rows = {}
for name, kw in knobs.items():
    m = DecisionTreeClassifier(random_state=0, **kw)
    rows[name] = {"leaves": m.fit(X_tr, y_tr).get_n_leaves(), "CV AUC": cv_auc(m)}
pre = pd.DataFrame(rows).T.astype({"leaves": int}).sort_values("CV AUC", ascending=False)
print(pre.to_string(float_format=lambda v: f"{v:.3f}"))
note("""max_depth is a blunt tool: it cuts every branch at the same level, even where
there is plenty of data. min_samples_leaf asks 'is there enough evidence in
this leaf?', which adapts to the data, and usually does better. In practice
combine two knobs and tune them together (Lab 07).""")

step("2.2  Cost-complexity pruning: grow the full tree, then cut back")
path = DecisionTreeClassifier(min_samples_leaf=5, random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
alphas = np.unique(np.quantile(path.ccp_alphas[:-1], np.linspace(0, 1, 30)))
rows = []
for a in alphas:
    m = DecisionTreeClassifier(min_samples_leaf=5, ccp_alpha=a, random_state=0)
    scores = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")
    rows.append({"alpha": a, "leaves": m.fit(X_tr, y_tr).get_n_leaves(), "CV AUC": scores.mean(),
                 "CV AUC se": scores.std(ddof=1) / np.sqrt(len(scores))})
prune = pd.DataFrame(rows)
best = prune.loc[prune["CV AUC"].idxmax()]
one_se = prune[prune["CV AUC"] >= best["CV AUC"] - best["CV AUC se"]].sort_values("leaves").iloc[0]
print(prune.iloc[::3].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
show("Best alpha by CV", f"{best.alpha:.5f} -> {int(best.leaves)} leaves, CV AUC {best['CV AUC']:.3f}")
show("One-standard-error rule: smallest tree within 1 se", f"{one_se.alpha:.5f} -> {int(one_se.leaves)} leaves, "
                                                            f"CV AUC {one_se['CV AUC']:.3f}")
note(f"""The pruning path lists, for increasing alpha, which weakest branch disappears
next: from hundreds of leaves down to the root alone. Cross-validation picks
the alpha with the best AUC; the ONE-STANDARD-ERROR rule then prefers the
SMALLEST tree whose score is statistically indistinguishable from the best
(Module 3, Lab 06): {int(one_se.leaves)} leaves instead of {int(best.leaves)}. Honest result: on this data the
best pruned tree (CV AUC {best['CV AUC']:.3f}) does NOT beat simple pre-pruning
({pre.index[0]}: {pre['CV AUC'].iloc[0]:.3f}). Pruning removes branches that don't reduce GINI
enough, which is not the same as 'don't help the ranking'. Try both; let
cross-validation decide.""")

step("2.3  The pruned tree as ranked business rules")
pruned = DecisionTreeClassifier(min_samples_leaf=5, ccp_alpha=one_se.alpha, random_state=0).fit(X_tr, y_tr)
leaf_te = pruned.apply(X_te)
leaves = pd.DataFrame({"leaf": leaf_te, "y": y_te}).groupby("leaf").agg(test_n=("y", "size"), test_rate=("y", "mean"))


def rule_text(tree, leaf, names):
    """Walk from the root to `leaf` and collect the conditions."""
    parent = {}
    for nd in range(tree.tree_.node_count):
        for child, side in [(tree.tree_.children_left[nd], "<="), (tree.tree_.children_right[nd], ">")]:
            if child >= 0:
                parent[child] = (nd, side)
    conds, nd = [], leaf
    while nd in parent:
        p, side = parent[nd]
        conds.append(f"{names[tree.tree_.feature[p]]} {side} {tree.tree_.threshold[p]:.3g}")
        nd = p
    return " AND ".join(reversed(conds))


leaves["train P(default)"] = [pruned.tree_.value[lf, 0, 1] for lf in leaves.index]
leaves["rule"] = [rule_text(pruned, lf, list(X.columns)) for lf in leaves.index]
top = leaves.sort_values("train P(default)", ascending=False).head(4)
print(top[["train P(default)", "test_n", "test_rate", "rule"]].to_string(float_format=lambda v: f"{v:.3f}"))
show("Pruned tree test ROC-AUC", f"{roc_auc_score(y_te, pruned.predict_proba(X_te)[:, 1]):.3f}")
note("""A risk team can review these rules one by one: the rule text, how many
applicants it covers, and its default rate on NEW (test) applicants. Rules
whose test rate holds up are trustworthy; rules from tiny leaves are not. This
is why a pruned tree is still the most transparent model in this module.""")

# %% =========================================================================
# LEVEL 3 (HARD): Probabilities, stability, and the price of interpretability
# ============================================================================
banner("What pruning does to probabilities, stability and accuracy", "HARD")

step("3.1  Compare each tree's probabilities with the TRUE default probability")
cands = {"depth 3": DecisionTreeClassifier(max_depth=3, random_state=0),
         f"depth {best_d} (best CV)": DecisionTreeClassifier(max_depth=None if best_d == "None" else int(best_d), random_state=0),
         "unlimited depth": DecisionTreeClassifier(random_state=0),
         "pruned (1-SE alpha)": DecisionTreeClassifier(min_samples_leaf=5, ccp_alpha=one_se.alpha, random_state=0),
         "random forest (reference)": RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=0)}
rows = {}
for name, m in cands.items():
    pr = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    rows[name] = {"test AUC": roc_auc_score(y_te, pr), "mean |p - true p|": np.mean(np.abs(pr - pt_te)),
                  "share of 0 or 1 probabilities": np.isin(pr, [0, 1]).mean(),
                  "leaves": m.get_n_leaves() if hasattr(m, "get_n_leaves") else np.nan}
prob_tab = pd.DataFrame(rows).T
print(prob_tab.to_string(float_format=lambda v: f"{v:.0f}" if v >= 2 else f"{v:.3f}"))
note("""Because we know each applicant's true default probability, we can measure how
wrong the trees' NUMBERS are. The unlimited tree says 0% or 100% for most
applicants: no applicant's true risk is anywhere near that. Pruning brings the
numbers much closer to the truth; the forest, averaging hundreds of trees, is
closest of all (Labs 03-04).""")

step("3.2  Stability: retrain the pruned tree on 30 bootstrap samples")
rng = np.random.default_rng(0)
roots, n_leaves, preds = [], [], []
for b in range(30):
    idx = rng.integers(0, len(y_tr), len(y_tr))
    m = DecisionTreeClassifier(min_samples_leaf=5, ccp_alpha=one_se.alpha, random_state=0).fit(X_tr.iloc[idx], y_tr[idx])
    roots.append(f"{X.columns[m.tree_.feature[0]]} <= {m.tree_.threshold[0]:.2f}")
    n_leaves.append(m.get_n_leaves())
    preds.append(m.predict_proba(X_te)[:, 1])
preds = np.array(preds)
print(pd.Series(roots).value_counts().rename("root question (30 retrained trees)").to_string())
show("Leaves: min / median / max", f"{min(n_leaves)} / {int(np.median(n_leaves))} / {max(n_leaves)}")
show("Average sd of an applicant's P(default)", f"{preds.std(axis=0).mean():.3f}")
show("Applicants flagged (P >= 0.3) by some trees but not all", f"{((preds >= 0.3).any(0) & ~(preds >= 0.3).all(0)).mean():.1%}")
note("""The root question is stable (debt ratio is the dominant driver), but the size
of the tree and the lower branches change from sample to sample, and with them
many applicants' decisions. 'The' tree is one draw from a distribution of
plausible trees. Present its rules as 'a good summary', not 'the truth'.""")

step("3.3  The price of interpretability")
rows = {}
for name, m in {"depth 2 (4 rules)": DecisionTreeClassifier(max_depth=2, random_state=0),
                "depth 3 (8 rules)": DecisionTreeClassifier(max_depth=3, random_state=0),
                "pruned tree": cands["pruned (1-SE alpha)"], "random forest": cands["random forest (reference)"]}.items():
    pr = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    rows[name] = {"test AUC": roc_auc_score(y_te, pr),
                  "share of the achievable AUC gain": (roc_auc_score(y_te, pr) - 0.5) / (roc_auc_score(y_te, pt_te) - 0.5)}
price = pd.DataFrame(rows).T
print(price.round(3).to_string())
note(f"""How much accuracy do we give up for a model a person can read? (Notice the
1-SE pruned tree is not better than a plain depth-3 tree here: the smallest
tree within noise of the best on CV can land slightly lower on test.) A tree with 8 rules captures {price.loc['depth 3 (8 rules)', 'share of the achievable AUC gain']:.0%} of the achievable AUC gain (truth = 100%), the
pruned tree {price.loc['pruned tree', 'share of the achievable AUC gain']:.0%}, the forest {price.loc['random forest', 'share of the achievable AUC gain']:.0%}. Whether that gap matters is a business question
(regulation, the cost of errors, who must understand the model). Module 4's
lesson stands: measure the price, then decide.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    plot_tree(t3, feature_names=list(X.columns), class_names=["repay", "default"], filled=True, rounded=True,
              proportion=True, fontsize=8, ax=axes[0])
    axes[0].set_title("A depth-3 tree (value = class proportions)")
    axes[1].plot(depth_tab["leaves"], depth_tab["train AUC"], "o-", label="train")
    axes[1].plot(depth_tab["leaves"], depth_tab["CV AUC"], "o-", label="5-fold CV")
    axes[1].axhline(roc_auc_score(y_te, pt_te), color="grey", ls="--", label="truth (best possible)")
    axes[1].set(xscale="log", xlabel="leaves (log)", ylabel="ROC-AUC", title="Bigger trees over-fit")
    axes[1].legend()
    axes[2].errorbar(prune["leaves"], prune["CV AUC"], yerr=prune["CV AUC se"], fmt="o-", ms=3)
    axes[2].axvline(one_se.leaves, color="g", ls="--", label=f"1-SE rule: {int(one_se.leaves)} leaves")
    axes[2].axvline(best.leaves, color="r", ls=":", label=f"best CV: {int(best.leaves)} leaves")
    axes[2].set(xscale="log", xlabel="leaves after pruning (log)", ylabel="CV ROC-AUC", title="Cost-complexity path")
    axes[2].legend()
    save_fig(fig, "02_tree_and_pruning")

banner("KEY TAKEAWAYS")
print("""  1. Unlimited trees memorise; validation AUC peaks at a moderate size.
  2. Pre-prune with min_samples_leaf (adaptive) rather than only max_depth.
  3. Cost-complexity pruning + CV + the 1-SE rule gives a small tree with near-best accuracy.
  4. A pruned tree reads as ranked rules with coverage and test rates: the most transparent model.
  5. Deep trees give 0/1 probabilities and unstable structure; pricing interpretability is a business call.""")
