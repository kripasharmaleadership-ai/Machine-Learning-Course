"""
LAB 01 - How Decision Trees Make Decisions: Entropy, Information Gain and Gini
=============================================================================

A TREE IS A GAME OF 20 QUESTIONS
    At every node: pick the yes/no question that best SEPARATES the classes, split the data,
    repeat inside each half. A new case falls down the questions to a leaf; the leaf's
    class mix is the prediction. Learning = choosing the questions.

HOW "BEST" IS MEASURED (impurity of a node with class shares p_c)
    entropy          H = -sum p_c log2 p_c     bits of "surprise"; 0 = pure, 1 = 50/50 (binary)
    Gini impurity    G = 1 - sum p_c^2         chance two random rows disagree; 0 = pure, 0.5 = 50/50
    information gain IG = H(parent) - sum_k (n_k / n) H(child_k)
    Gini decrease    same formula with G           (scikit-learn's default 'gini')

LEVELS
    EASY   : 14 loan applications: entropy, information gain and Gini for every question, by hand.
    MEDIUM : A CART tree written from scratch reproduces scikit-learn's tree exactly; entropy vs
             Gini; how numeric thresholds are searched.
    HARD   : Where greedy trees fail: XOR (no single question helps), and the bias of information
             gain towards many-valued features (an ID column wins!) - and gain ratio, the fix.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from ml_lab_utils import (HAS_MPL, RANDOM_STATE, banner, loan_table, make_credit_applicants, note, plt, save_fig,
                          show, step)
from tree_toolkit import ScratchTree, entropy, gain_ratio, gini, information_gain

# %% =========================================================================
# LEVEL 1 (EASY): Every question, by hand
# ============================================================================
banner("Which question should the first node ask?", "EASY")
loans = loan_table()
print(loans.to_string())
y = loans["defaulted"].to_numpy()
show("Defaulted / repaid", f"{y.sum()} / {len(y) - y.sum()}")
p = y.mean()
show("Entropy of the root  -p log2 p - (1-p) log2(1-p)", f"-{p:.3f} log2 {p:.3f} - {1 - p:.3f} log2 {1 - p:.3f} = {entropy(y):.3f} bits")
show("Gini of the root  1 - p^2 - (1-p)^2", f"1 - {p:.3f}^2 - {1 - p:.3f}^2 = {gini(y):.3f}")

step("1.1  One candidate question, worked out in full: 'credit_history = bad?'")
for v in ["bad", "good"]:
    yy = y[loans.credit_history == v]
    show(f"credit_history = {v}: n, defaulted", f"{len(yy)}, {yy.sum()}  -> entropy {entropy(yy):.3f}, Gini {gini(yy):.3f}")
ig = information_gain(y, loans.credit_history)
show("Information gain = H(root) - weighted H(children)", f"{entropy(y):.3f} - "
     f"({(loans.credit_history == 'bad').mean():.3f} x {entropy(y[loans.credit_history == 'bad']):.3f} + "
     f"{(loans.credit_history == 'good').mean():.3f} x {entropy(y[loans.credit_history == 'good']):.3f}) = {ig:.3f} bits")

step("1.2  All four questions")
rows = {}
for c in ["income", "credit_history", "employed", "housing"]:
    rows[c] = {"values": loans[c].nunique(), "information gain (bits)": information_gain(y, loans[c]),
               "Gini decrease": information_gain(y, loans[c], criterion=gini)}
ig_tab = pd.DataFrame(rows).T.sort_values("information gain (bits)", ascending=False)
print(ig_tab.round(4).to_string())
note(f"""The root asks about '{ig_tab.index[0]}': it removes the most uncertainty. Read the
split in words: almost everyone employed repays, and the unemployed mostly
default. Entropy and Gini give the same ranking here: they usually agree,
because both are 0 for a pure node and maximal for a 50/50 one.""")

step("1.3  The full tree on 14 rows (scikit-learn, entropy)")
X14 = pd.get_dummies(loans.drop(columns="defaulted"), dtype=int)
small = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X14, y)
print(export_text(small, feature_names=list(X14.columns), show_weights=True))
note("""Each level repeats the same recipe inside its own subset. With 14 rows the
tree keeps splitting until every leaf is pure: it has memorised the table.
That is the over-fitting problem Lab 02 is about.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): A tree from scratch = scikit-learn
# ============================================================================
banner("CART from scratch, on 4,000 credit applicants", "MEDIUM")
credit = make_credit_applicants(n=4000, seed=1)
NUM = ["age", "income_k", "debt_ratio", "late_payments", "credit_years", "utilisation"]
X, yc = credit[NUM], credit["default"].to_numpy()
scratch = ScratchTree(max_depth=3, criterion="gini").fit(X, yc)
sk = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, yc)
show("Max difference in predicted P(default)", float(np.abs(scratch.predict_proba(X)[:, 1] - sk.predict_proba(X)[:, 1]).max()))
print("\n  Our tree as rules:")
for r in scratch.rules():
    print("   ", r)
note("""Our 60-line tree (tree_toolkit.ScratchTree) makes exactly the same splits as
scikit-learn. The algorithm: for every feature, sort the values, try a cut
BETWEEN every pair of different neighbouring values (the midpoint), keep the
cut with the largest impurity decrease, recurse. For a numeric feature with
n distinct values that is n - 1 candidate questions; the search is
exhaustive, and greedy (it never looks two questions ahead).""")

step("2.1  Entropy vs Gini on real-size data")
tr, te = train_test_split(credit, test_size=0.3, stratify=credit["default"], random_state=RANDOM_STATE)
rows = {}
for crit in ["gini", "entropy", "log_loss"]:
    for depth in [3, 6]:
        m = DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=0).fit(tr[NUM], tr["default"])
        rows[(crit, depth)] = {"CV ROC-AUC": cross_val_score(DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=0),
                                                            tr[NUM], tr["default"], cv=5, scoring="roc_auc").mean(),
                               "root question": f"{NUM[m.tree_.feature[0]]} <= {m.tree_.threshold[0]:.3f}"}
crit_tab = pd.DataFrame(rows).T
crit_tab.index.names = ["criterion", "max_depth"]
print(crit_tab.to_string(float_format=lambda v: f"{v:.4f}"))
gap_crit = crit_tab.loc[("entropy", 3), "CV ROC-AUC"] - crit_tab.loc[("gini", 3), "CV ROC-AUC"]
gap_depth = crit_tab.loc[("gini", 3), "CV ROC-AUC"] - crit_tab.loc[("gini", 6), "CV ROC-AUC"]
note(f"""Gini and entropy choose the same root FEATURE with a slightly different cut
('log_loss' is scikit-learn's name for entropy), and the trees then diverge a
little: {gap_crit:+.3f} AUC between criteria at depth 3. Going from depth 3 to 6 costs
{gap_depth:.3f} AUC, a bigger effect. Gini is a little cheaper (no logarithm), which
is why it is the default. Treat the criterion as a minor hyperparameter; depth
is the one that makes or breaks a tree (Lab 02).""")

# %% =========================================================================
# LEVEL 3 (HARD): Where greedy splitting goes wrong
# ============================================================================
banner("Two blind spots of greedy, impurity-based splitting", "HARD")

step("3.1  XOR: no single question helps, but two together are perfect")
rng = np.random.default_rng(0)
A, B = rng.integers(0, 2, 2000), rng.integers(0, 2, 2000)
noise = rng.normal(size=(2000, 3))
y_xor = (A ^ B).astype(int)
flip = rng.random(2000) < 0.05
y_xor[flip] = 1 - y_xor[flip]                                   # 5% label noise
Xx = pd.DataFrame({"A": A, "B": B, "noise1": noise[:, 0], "noise2": noise[:, 1], "noise3": noise[:, 2]})
gains = {c: information_gain(y_xor, (Xx[c] > Xx[c].median()) if c.startswith("noise") else Xx[c]) for c in Xx}
print(pd.Series(gains, name="information gain of the root split (bits)").to_string(float_format=lambda v: f"{v:.5f}"))
Xa, Xb, ya, yb = train_test_split(Xx, y_xor, test_size=0.5, random_state=0)
from sklearn.ensemble import RandomForestClassifier

rows = {}
for depth in [1, 2, 4, 6, 8, None]:
    m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=0).fit(Xa, ya)
    rows[f"tree, max_depth {depth}"] = {"root question": f"{m.feature_names_in_[m.tree_.feature[0]]}",
                                        "test accuracy": m.score(Xb, yb)}
rf = RandomForestClassifier(300, random_state=0).fit(Xa, ya)
rows["random forest (300 trees)"] = {"root question": "(many trees)", "test accuracy": rf.score(Xb, yb)}
fe = DecisionTreeClassifier(max_depth=1).fit(Xa.assign(A_ne_B=(Xa.A != Xa.B).astype(int)), ya)
rows["depth-1 tree + feature 'A != B'"] = {"root question": "A_ne_B",
                                           "test accuracy": fe.score(Xb.assign(A_ne_B=(Xb.A != Xb.B).astype(int)), yb)}
xor_tab = pd.DataFrame(rows).T
print(xor_tab.to_string(float_format=lambda v: f"{v:.3f}"))
note(f"""y = A XOR B (default if exactly one of two risk flags is set; 5% noise, so 95%
is the best possible). Alone, A and B carry ZERO information: knowing A leaves
the default rate at 50%. The greedy root split lands on a NOISE column, and
up to depth 6 the tree is a coin flip. Only at depth 8+, after many useless
noise splits, does it stumble into part of the structure ({xor_tab.iloc[4, 1]:.0%}). A random
forest ({xor_tab.iloc[6, 1]:.0%}) does better because its random feature subsets sometimes force
a split on A or B first. One engineered feature, 'A != B', makes a single
question almost perfect ({xor_tab.iloc[7, 1]:.0%}). Greedy splitting is short-sighted: interactions
without main effects are hard for trees (and for people writing rules).""")

step("3.2  Information gain loves many-valued features")
loans_id = loans.assign(application_id=[f"APP-{i:02d}" for i in range(len(loans))])
rows = {}
for c in ["income", "credit_history", "employed", "housing", "application_id"]:
    rows[c] = {"distinct values": loans_id[c].nunique(), "information gain": information_gain(y, loans_id[c]),
               "gain ratio": gain_ratio(y, loans_id[c])}
gr = pd.DataFrame(rows).T
print(gr.round(3).to_string())
note(f"""A unique application ID puts every row in its own pure child: information
gain is maximal ({gr.loc['application_id', 'information gain']:.3f} bits, the entire entropy) and it would be chosen
as the root, while predicting nothing for new applicants. C4.5's GAIN RATIO
divides by the entropy of the split itself and pushes the ID down the ranking
(from 1st place to {['1st', '2nd', '3rd', '4th', '5th'][int(gr['gain ratio'].rank(ascending=False)['application_id']) - 1]}). scikit-learn's binary splits limit
the damage but do not remove it: high-cardinality columns still win too many
splits, and they fool impurity-based importance (Lab 08).""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    pp = np.linspace(0.001, 0.999, 200)
    axes[0].plot(pp, -pp * np.log2(pp) - (1 - pp) * np.log2(1 - pp), label="entropy (bits)")
    axes[0].plot(pp, 1 - pp ** 2 - (1 - pp) ** 2, label="Gini")
    axes[0].plot(pp, 2 * (1 - pp ** 2 - (1 - pp) ** 2), "--", label="2 x Gini (same shape)")
    axes[0].set(xlabel="share of class 1 in the node", ylabel="impurity", title="Impurity of a node")
    axes[0].legend()
    axes[1].barh(ig_tab.index, ig_tab["information gain (bits)"], color="#4c78a8")
    axes[1].set(xlabel="information gain (bits)", title="14 loans: which question first?")
    axes[2].barh(gr.index, gr["information gain"], color="#e45756", label="information gain")
    axes[2].barh(gr.index, gr["gain ratio"], color="#4c78a8", alpha=0.7, label="gain ratio")
    axes[2].set(title="An ID column wins on information gain")
    axes[2].legend()
    save_fig(fig, "01_how_trees_decide")

banner("KEY TAKEAWAYS")
print("""  1. A tree asks the question that most reduces impurity, then repeats inside each child.
  2. Entropy (bits) and Gini both measure impurity; they almost always pick the same split.
  3. CART tries every midpoint of every numeric feature: exhaustive and greedy (one step ahead).
  4. Greedy splitting misses interactions without main effects (XOR).
  5. Information gain favours many-valued features (IDs); gain ratio, or dropping IDs, fixes it.""")
