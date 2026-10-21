"""
LAB 01 - Learning Without Labelled Data
=======================================

SUPERVISED vs UNSUPERVISED
    supervised     : rows come with an answer (default yes/no, price) -> learn to predict it
    unsupervised   : rows come WITHOUT an answer -> find STRUCTURE in the inputs themselves

    What structure?                         Method (this module)
    groups of similar rows                  clustering (k-means, hierarchical, DBSCAN)       Labs 03-04
    a few directions that explain most      dimensionality reduction (PCA)                    Lab 05
    rows that don't fit any pattern         anomaly detection (Isolation Forest)              Lab 06
    which rows / items are similar          distances, similarity, embeddings                 Labs 02, 07, 08

THE CATCH
    There is no answer key, so there is no accuracy. "Is this clustering right?" has no single answer:
    it depends on the features, the scaling, the number of clusters, and what the business will DO with it.
    In this module the synthetic data carries a HIDDEN truth (column `_segment`) that the algorithms never
    see, so for once we can check what they found.

LEVELS
    EASY   : Segment 3,000 customers without labels; describe the groups; peek at the hidden truth.
    MEDIUM : Judging a result without labels: inertia, silhouette, stability, business usefulness.
    HARD   : The answer depends on your choices (scaling, features); algorithms find 'clusters' in pure noise.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import HAS_MPL, SPEND_COLS, banner, make_customers, note, plt, save_fig, show, step

customers = make_customers(n=3000, seed=1)
FEATURES = ["age", "income_k"] + SPEND_COLS + ["visits_per_month", "online_share"]
X_raw = customers[FEATURES]
X = StandardScaler().fit_transform(np.log1p(X_raw))                  # log tames skewed spend; scaling equalises units

# %% =========================================================================
# LEVEL 1 (EASY): Find groups without being told what they are
# ============================================================================
banner("3,000 customers, no labels: what groups exist?", "EASY")
print(customers.drop(columns="_segment").head(4).to_string())
note("""A loyalty-card table: age, income, monthly spend in five categories, visits
and online share. No column says 'what kind of customer is this'. Marketing
wants a handful of segments to design campaigns for. That is unsupervised
learning: the algorithm must discover the groups from the inputs alone.""")
km = KMeans(5, n_init=10, random_state=0).fit(X)
customers["cluster"] = km.labels_
profile = customers.groupby("cluster")[FEATURES].median()
profile.insert(0, "customers", customers.cluster.value_counts().sort_index())
print("\n  Median customer per cluster:")
print(profile.round(1).to_string())
note("""Read each row like a persona: one cluster is young and mostly online, one spends
heavily on groceries with many visits (families), one has a high income and big
travel spend, one is older with low spend, one spends on home goods. Naming
and interpreting clusters is the human half of unsupervised learning.""")

step("1.1  Peek at the hidden truth (impossible in real life)")
print(pd.crosstab(customers["_segment"], customers["cluster"]).to_string())
show("Adjusted Rand index vs the hidden personas (1 = perfect, 0 = random)", f"{adjusted_rand_score(customers._segment, km.labels_):.3f}")
note("""Each hidden persona lands almost entirely in one cluster: k-means rediscovered
the segments without ever seeing them. In real projects this table cannot be
built, which is exactly why the rest of this lab is about judging clusters
WITHOUT the truth.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): How do you judge a result without labels?
# ============================================================================
banner("Judging clusters without an answer key", "MEDIUM")
step("2.1  Internal metrics for k = 2..8")
rows = {}
for k in range(2, 9):
    m = KMeans(k, n_init=10, random_state=0).fit(X)
    rows[k] = {"inertia (within-cluster squared distance)": m.inertia_, "silhouette": silhouette_score(X, m.labels_),
               "ARI vs hidden truth (normally unknown)": adjusted_rand_score(customers._segment, m.labels_)}
ktab = pd.DataFrame(rows).T
ktab.index.name = "k"
print(ktab.round(3).to_string())
best_sil = int(ktab["silhouette"].idxmax())
note(f"""Inertia ALWAYS falls as k grows (more centres = shorter distances), so its
minimum is meaningless; look for the 'elbow' where it stops falling fast. The
silhouette (how much closer each point is to its own cluster than to the next
one) peaks at k = {best_sil}. Plot twist: the hidden truth has 5 segments, and k = 5
matches it best ({ktab.loc[5, 'ARI vs hidden truth (normally unknown)']:.2f}). Internal metrics measure geometric neatness, not business meaning:
use them to shortlist k, then let the people who will USE the segments decide.""")

step("2.2  Stability: do we find the same clusters on different samples?")
rng = np.random.default_rng(0)
ref = KMeans(5, n_init=10, random_state=0).fit(X)
stab = {}
for k in [3, 4, 5, 6, 8]:
    base_k = KMeans(k, n_init=10, random_state=0).fit(X)
    aris = []
    for b in range(10):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        m = KMeans(k, n_init=10, random_state=b).fit(X[idx])
        aris.append(adjusted_rand_score(base_k.labels_[idx], m.labels_))
    stab[k] = {"agreement with the full-data clustering (mean ARI, 10 subsamples)": np.mean(aris), "worst": np.min(aris)}
stab_tab = pd.DataFrame(stab).T
print(stab_tab.round(3).to_string())
note(f"""Refit on random 80% subsamples and compare with the full-data clustering. Up
to k = 6 the same clusters come back every time (agreement ~0.99+); at k = 8 the
extra splits become arbitrary (worst run {stab_tab.loc[8, 'worst']:.2f}). Stability rules OUT values of k
that are too large, but it can't choose between 4 and 5: both are real,
stable structure. Combine label-free checks, and expect a judgement call.""")

step("2.3  Usefulness: do the clusters differ on something that matters?")
customers["monthly_spend"] = customers[SPEND_COLS].sum(axis=1)
use = customers.groupby("cluster").agg(customers=("monthly_spend", "size"), median_monthly_spend=("monthly_spend", "median"),
                                       share_of_revenue=("monthly_spend", "sum"), top_category=(SPEND_COLS[0], "size"))
use["share_of_revenue"] = use["share_of_revenue"] / use["share_of_revenue"].sum()
use["top_category"] = [profile.loc[c, SPEND_COLS].idxmax() for c in use.index]
print(use.round(3).to_string())
note("""The ultimate test is business use: segments must be big enough to act on,
different in what they buy and what they are worth, and stable over time.
Here one cluster brings a large share of revenue from a small share of
customers, and each has a different top category: campaigns can differ.""")

# %% =========================================================================
# LEVEL 3 (HARD): The answer depends on your choices
# ============================================================================
banner("Unsupervised results reflect YOUR choices", "HARD")
step("3.1  Scaling and transformation change the clusters")
choices = {
    "raw values (no scaling)": X_raw.to_numpy(float),
    "scaled, no log": StandardScaler().fit_transform(X_raw),
    "log + scaled (our choice)": X,
    "spend MIX only (shares of total spend)": StandardScaler().fit_transform(
        X_raw[SPEND_COLS].div(X_raw[SPEND_COLS].sum(axis=1), axis=0)),
}
rows = {}
for name, Z in choices.items():
    m = KMeans(5, n_init=10, random_state=0).fit(Z)
    sizes = np.bincount(m.labels_)
    rows[name] = {"ARI vs hidden personas": adjusted_rand_score(customers._segment, m.labels_),
                  "smallest / largest cluster": f"{sizes.min()} / {sizes.max()}"}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}"))
note("""Without scaling, distance is dominated by whatever has the biggest numbers
(dollars of travel spend), and the clusters are basically 'spend bands'. Without
the log, a few big spenders pull clusters around. Clustering only the spend
MIX finds 'what people buy' and ignores age and income: a different, equally
legitimate segmentation for a different question. Clustering has no single
right answer; the feature and scaling choices ARE the question you ask.""")

step("3.2  Clusters in pure noise")
noise = np.random.default_rng(1).uniform(size=(3000, len(FEATURES)))
nz = KMeans(5, n_init=10, random_state=0).fit(noise)
show("k-means on uniform random data: cluster sizes", np.bincount(nz.labels_).tolist())
show("Silhouette on noise / on real customers", f"{silhouette_score(noise, nz.labels_):.3f} / {silhouette_score(X, km.labels_):.3f}")
note("""k-means always returns k clusters, even when there is no structure at all: it
partitions noise into tidy-looking groups. The silhouette is much lower, but
nothing in the output says 'there are no clusters here'. Always compare with
a structureless baseline (shuffle each column independently) before presenting
segments as real.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(ktab.index, ktab.iloc[:, 0], "o-")
    axes[0].set(xlabel="k", ylabel="inertia", title="Inertia always falls (look for the elbow)")
    axes[1].plot(ktab.index, ktab["silhouette"], "o-", label="silhouette (label-free)")
    axes[1].plot(ktab.index, ktab.iloc[:, 2], "s--", label="ARI vs hidden truth")
    axes[1].axvline(5, color="grey", ls=":")
    axes[1].set(xlabel="k", title="The label-free metric and the truth disagree")
    axes[1].legend(fontsize=8)
    for c in sorted(customers.cluster.unique()):
        mk = customers.cluster == c
        axes[2].scatter(customers.age[mk], customers.monthly_spend[mk], s=4, alpha=0.5, label=f"cluster {c}")
    axes[2].set(xlabel="age", ylabel="monthly spend", yscale="log", title="Clusters found without labels")
    axes[2].legend(fontsize=7, markerscale=3)
    save_fig(fig, "01_learning_without_labels")

banner("KEY TAKEAWAYS")
print("""  1. Unsupervised learning finds structure in inputs alone: groups, directions, outliers, similarity.
  2. There is no accuracy: judge results with internal metrics, stability and business usefulness.
  3. Label-free metrics shortlist k; they don't know the business (silhouette preferred 4, truth was 5).
  4. Features, transformations and scaling decide what 'similar' means: choose them for the question.
  5. Algorithms find clusters even in noise: always compare with a structureless baseline.""")
