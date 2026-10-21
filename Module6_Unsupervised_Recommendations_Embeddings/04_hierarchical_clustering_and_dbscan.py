"""
LAB 04 - Hierarchical Clustering and DBSCAN
===========================================

HIERARCHICAL (agglomerative) CLUSTERING
    start with every point as its own cluster; repeatedly MERGE the two closest clusters; record every
    merge in a tree (the DENDROGRAM). Cut the tree at any height to get any number of clusters.
    'Closest clusters' needs a LINKAGE rule:
        single    distance between the two closest members     (finds chains, sensitive to noise)
        complete  distance between the two farthest members    (compact clusters)
        average   average distance between all pairs
        ward      the merge that increases within-cluster variance least (k-means-like, the default)

DBSCAN (density-based)
    a CORE point has at least min_samples neighbours within distance eps; clusters are chains of core
    points; points in no dense region are NOISE (label -1). No k to choose; any shape; built-in outliers.
    HDBSCAN (in scikit-learn since 1.3) removes the single eps by trying all densities.

LEVELS
    EASY   : Ten customers merged step by step; the dendrogram; cutting it.
    MEDIUM : Linkages compared on three shapes; DBSCAN on moons; choosing eps with the k-distance curve.
    HARD   : DBSCAN as an outlier finder on real-looking customers; varying densities (HDBSCAN);
             the segment hierarchy; the cost of hierarchical clustering at scale.
"""
# %%
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import HAS_MPL, SPEND_COLS, banner, make_customers, note, plt, save_fig, show, step

# %% =========================================================================
# LEVEL 1 (EASY): Merge, merge, merge
# ============================================================================
banner("Agglomerative clustering on ten customers", "EASY")
customers = make_customers(n=3000, seed=4)
FEATS = ["age", "income_k"] + SPEND_COLS + ["visits_per_month", "online_share"]
X = StandardScaler().fit_transform(np.log1p(customers[FEATS]))
ten_idx = customers.groupby("_segment").head(2).index[:10]
ten = X[ten_idx][:, [0, 1]]                                           # age and income only, to keep it readable
names = [f"{customers._segment[i].split()[0][:6]}-{k}" for k, i in enumerate(ten_idx)]
Z = linkage(ten, method="ward")
print("  The merge history (scipy's linkage matrix, one merge per row):")
for step_no, (a, b, d, n) in enumerate(Z, start=1):
    label = lambda j: names[int(j)] if j < 10 else f"group{int(j) - 10 + 1}"
    print(f"    merge {step_no}: {label(a):<12} + {label(b):<12} at height {d:5.2f} -> group{step_no} ({int(n)} customers)")
cut3 = fcluster(Z, 3, criterion="maxclust")
print("\n  Cut the tree into 3 groups:", dict(zip(names, cut3.tolist())))
sk = AgglomerativeClustering(3, linkage="ward").fit_predict(ten)
show("scikit-learn AgglomerativeClustering agrees (ARI)", f"{adjusted_rand_score(cut3, sk):.3f}")
note("""Every point starts alone; the two closest are merged, then the next closest, and
so on until one cluster remains. The HEIGHT of each merge (how far apart the
groups were) is the y-axis of the dendrogram. Cutting the tree at a height gives
a clustering, and you can choose the number of clusters AFTER looking at the tree:
long vertical gaps mean well-separated groups.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Linkages and DBSCAN on different shapes
# ============================================================================
banner("Which method sees which shape?", "MEDIUM")
Xm, ym = make_moons(800, noise=0.06, random_state=0)
Xb, yb = make_blobs(800, centers=3, cluster_std=1.0, random_state=1)
Xu, yu = make_blobs(n_samples=[600, 150, 50], centers=[[0, 0], [6, 0], [3, 5]], cluster_std=[1.6, 0.5, 0.3], random_state=2)
datasets = {"blobs": (Xb, yb, 3), "moons": (Xm, ym, 2), "uneven blobs": (Xu, yu, 3)}
rows = {}
for dname, (D, yt, k) in datasets.items():
    for link in ["single", "complete", "average", "ward"]:
        rows.setdefault(f"hierarchical, {link}", {})[dname] = adjusted_rand_score(yt, AgglomerativeClustering(k, linkage=link).fit_predict(D))
    rows.setdefault("k-means", {})[dname] = adjusted_rand_score(yt, KMeans(k, n_init=5, random_state=0).fit_predict(D))
    rows.setdefault("DBSCAN (eps 0.3 on scaled data)", {})[dname] = adjusted_rand_score(
        yt, DBSCAN(eps=0.3, min_samples=8).fit_predict(StandardScaler().fit_transform(D)))
linkage_tab = pd.DataFrame(rows).T
print("  Adjusted Rand index vs the true groups:")
print(linkage_tab.round(3).to_string())
note("""Ward linkage and k-means win on round blobs. On the moons, only SINGLE linkage
and DBSCAN follow the curved shapes, because both connect points through chains
of close neighbours. No method wins everywhere: each encodes a different idea
of what a cluster is (a blob around a centre, a chain of neighbours, a dense
region). Look at the data before choosing.""")

step("2.1  DBSCAN: choosing eps with the k-distance curve")
Dm = StandardScaler().fit_transform(Xm)
k_dist = np.sort(NearestNeighbors(n_neighbors=8).fit(Dm).kneighbors(Dm)[0][:, -1])
show("8th-nearest-neighbour distance: 50th / 90th / 99th percentile", f"{np.percentile(k_dist, 50):.3f} / "
     f"{np.percentile(k_dist, 90):.3f} / {np.percentile(k_dist, 99):.3f}")
rows = {}
for eps in [0.08, 0.15, 0.25, 0.4, 0.8]:
    lab = DBSCAN(eps=eps, min_samples=8).fit_predict(Dm)
    rows[eps] = {"clusters found": len(set(lab) - {-1}), "noise share": np.mean(lab == -1), "ARI vs truth": adjusted_rand_score(ym, lab)}
eps_tab = pd.DataFrame(rows).T
eps_tab.index.name = "eps"
print(eps_tab.round(3).to_string())
note("""Sort every point's distance to its min_samples-th neighbour: dense points have
small distances, isolated ones large. eps should sit just above the bulk (the
'knee' of that curve, here ~the 90th-99th percentile). Too small and most points
are 'noise' split into fragments; too large and everything merges into one
cluster. DBSCAN has no k, but eps is just as important.""")

# %% =========================================================================
# LEVEL 3 (HARD): Outliers, varying densities, hierarchy, scale
# ============================================================================
banner("DBSCAN as an outlier finder, HDBSCAN, and the cost of hierarchy", "HARD")
step("3.1  DBSCAN flags the resellers as noise")
cust_o = make_customers(n=3000, seed=4, outliers=15)
Xo = StandardScaler().fit_transform(np.log1p(cust_o[FEATS]))
k_o = np.sort(NearestNeighbors(n_neighbors=10).fit(Xo).kneighbors(Xo)[0][:, -1])
eps_o = float(np.percentile(k_o, 98))
lab_o = DBSCAN(eps=eps_o, min_samples=10).fit_predict(Xo)
is_reseller = (cust_o._segment == "outlier (reseller)").to_numpy()
show("eps chosen at the 98th percentile of the 10-NN distance", f"{eps_o:.2f}")
caught = int((is_reseller & (lab_o == -1)).sum())
show("Points labelled noise / resellers among them", f"{(lab_o == -1).sum()} / {caught} of {is_reseller.sum()}")
show("Clusters found among the rest", len(set(lab_o) - {-1}))
note(f"""With eps set so that 98% of customers live in a dense neighbourhood, DBSCAN
labels only {(lab_o == -1).sum()} customers as noise, and {caught} of the 15 resellers are among them:
DBSCAN doubles as an anomaly detector. But it puts ALL the normal customers in
{len(set(lab_o) - {-1})} cluster(s): the personas' regions touch, and density-based methods only
separate groups divided by a sparse GAP, not neighbouring blobs. Use DBSCAN
for 'find the outliers', k-means / GMM for 'split the crowd into segments'.""")

step("3.2  Clusters of different density: one eps can't fit all")
Xd, yd = make_blobs(n_samples=[400, 400, 300], centers=[[0, 0], [1.6, 0], [6, 4]], cluster_std=[0.25, 0.25, 1.4],
                    random_state=3)                                  # two tight neighbours + one loose cluster
Xd = StandardScaler().fit_transform(Xd)
rows = {}
for eps in [0.05, 0.08, 0.12, 0.25]:
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Xd)
    rows[f"DBSCAN eps {eps}"] = {"clusters": len(set(lab) - {-1}), "noise share": np.mean(lab == -1), "ARI": adjusted_rand_score(yd, lab)}
lab_h = HDBSCAN(min_cluster_size=20, copy=True).fit_predict(Xd)
rows["HDBSCAN (min_cluster_size 20)"] = {"clusters": len(set(lab_h) - {-1}), "noise share": np.mean(lab_h == -1), "ARI": adjusted_rand_score(yd, lab_h)}
dens = pd.DataFrame(rows).T
dens["clusters"] = dens["clusters"].astype(int)
print(dens.round(3).to_string())
note(f"""Two tight clusters side by side and one loose cluster further away (3 true
groups). No eps finds 3: a small eps separates the tight pair but calls much of
the loose cluster noise; a larger eps merges the tight pair (or shatters the
loose one). The best single eps throws away {dens.iloc[1, 1]:.0%} of the points as noise. HDBSCAN
builds the whole hierarchy of densities and keeps the clusters that persist
longest, so each cluster gets its own effective eps: 3 clusters, {dens.iloc[-1, 1]:.0%} noise.
When densities vary, prefer HDBSCAN.""")

step("3.3  The segment hierarchy, and what hierarchy costs")
sample = np.random.default_rng(0).choice(len(X), 400, replace=False)
Zc = linkage(X[sample], method="ward")
for k in [2, 3, 5]:
    cut = fcluster(Zc, k, criterion="maxclust")
    ct = pd.crosstab(customers._segment.to_numpy()[sample], cut)
    print(f"  cut into {k} groups -> each persona's main group: "
          + ", ".join(f"{p.split()[0]}:{ct.loc[p].idxmax()}" for p in ct.index))
rows = {}
for n in [1000, 2000, 4000, 8000]:
    Xn = np.vstack([X] * 3)[:n]
    t0 = time.perf_counter()
    AgglomerativeClustering(5, linkage="ward").fit(Xn)
    t_h = time.perf_counter() - t0
    t0 = time.perf_counter()
    KMeans(5, n_init=3, random_state=0).fit(Xn)
    rows[n] = {"hierarchical seconds": t_h, "k-means seconds": time.perf_counter() - t0, "distance matrix MB": n * n * 8 / 1e6}
cost = pd.DataFrame(rows).T
cost.index.name = "customers"
print(cost.round(3).to_string())
note("""Cutting one tree at different heights gives a family of nested segmentations:
2 super-groups (the value seniors vs everyone else), then 3, then the 5 personas.
That hierarchy is free information a flat k-means doesn't give you. The price:
hierarchical clustering needs all pairwise distances, so time and memory grow
with n^2 (8,000 customers already need ~500 MB of distances). For large data:
k-means, or hierarchical clustering on a sample / on k-means centres.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    dendrogram(Zc, ax=axes[0], no_labels=True, color_threshold=Zc[-5, 2])
    axes[0].set(title="Dendrogram of 400 customers (ward)", ylabel="merge height")
    lab_m = DBSCAN(eps=0.25, min_samples=8).fit_predict(Dm)
    axes[1].scatter(*Dm.T, c=lab_m, s=5, cmap="tab10")
    axes[1].set(title="DBSCAN follows the moons")
    axes[2].plot(k_dist)
    axes[2].set(xlabel="points sorted by distance", ylabel="distance to 8th neighbour", title="k-distance curve: eps at the knee")
    save_fig(fig, "04_hierarchical_dbscan")

banner("KEY TAKEAWAYS")
print("""  1. Agglomerative clustering merges the closest clusters step by step; the dendrogram shows every level.
  2. The linkage defines 'closest': ward ~ k-means blobs, single follows chains, complete keeps groups compact.
  3. DBSCAN finds dense regions of any shape and labels the rest noise; choose eps with the k-distance curve.
  4. One eps can't fit clusters of different density: HDBSCAN adapts.
  5. Hierarchical clustering costs O(n^2): use it on samples or summaries for big data.""")
