"""
LAB 03 - K-Means and Customer Segmentation
==========================================

K-MEANS (Lloyd's algorithm)
    1. place k centres (k-means++: spread them out, far points more likely)
    2. ASSIGN every point to its nearest centre
    3. UPDATE every centre to the mean of its points
    4. repeat 2-3 until nothing moves
    It minimises INERTIA = total squared distance from points to their centre.

WHAT IT ASSUMES (and where it breaks)
    clusters are round blobs of similar size, measured on comparable scales; every point belongs
    to exactly one cluster; k is given. Outliers drag centres; elongated or uneven clusters get cut wrongly.

LEVELS
    EASY   : Lloyd's algorithm step by step; our version = scikit-learn; name the segments.
    MEDIUM : Choosing k; why k-means++ initialisation matters; outliers drag centres.
    HARD   : Shapes k-means can't see (Gaussian mixtures can); soft membership; segments as a
             business asset: action table, new customers, drift monitoring.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import HAS_MPL, SPEND_COLS, banner, make_customers, note, plt, save_fig, show, step
from unsup_toolkit import KMeansScratch

customers = make_customers(n=3000, seed=3)
FEATS = ["age", "income_k"] + SPEND_COLS + ["visits_per_month", "online_share"]
scaler = StandardScaler().fit(np.log1p(customers[FEATS]))
X = scaler.transform(np.log1p(customers[FEATS]))

# %% =========================================================================
# LEVEL 1 (EASY): Lloyd's algorithm, step by step
# ============================================================================
banner("K-means, one iteration at a time", "EASY")
two = StandardScaler().fit_transform(np.log1p(customers[["income_k", "groceries"]]))
rng = np.random.default_rng(4)
C = two[rng.choice(len(two), 3, replace=False)]
for it in range(1, 7):
    d = ((two[:, None, :] - C[None]) ** 2).sum(-1)
    labels = d.argmin(1)                                                  # ASSIGN
    inertia = d[np.arange(len(two)), labels].sum()
    newC = np.array([two[labels == j].mean(0) for j in range(3)])        # UPDATE
    moved = np.abs(newC - C).max()
    print(f"    iteration {it}: inertia {inertia:9.1f}   cluster sizes {np.bincount(labels, minlength=3).tolist()}   "
          f"largest centre move {moved:.3f}")
    C = newC
    if moved < 1e-6:
        break
note("""Assign every customer to the nearest centre, move every centre to the middle of
its customers, repeat. Inertia can only go down at each step, and the
algorithm stops when the centres stop moving, usually after a handful of
iterations. That's all k-means is.""")

step("1.1  Our implementation vs scikit-learn on all 9 features")
mine = KMeansScratch(5, n_init=5, seed=0).fit(X)
sk = KMeans(5, n_init=5, random_state=0).fit(X)
show("Inertia: ours / scikit-learn", f"{mine.inertia_:,.2f} / {sk.inertia_:,.2f}")
show("Agreement between the two clusterings (ARI)", f"{adjusted_rand_score(mine.labels_, sk.labels_):.3f}")

step("1.2  Profile and NAME the segments")
customers["segment"] = sk.labels_
centres = pd.DataFrame(np.expm1(scaler.inverse_transform(sk.cluster_centers_)), columns=FEATS)
centres["customers"] = np.bincount(sk.labels_)


def name_segment(r):
    if r.travel > 300:
        return "Affluent travellers"
    if r.home > 300:
        return "Home improvers"
    if r.groceries > 450:
        return "Family shoppers"
    if r.age > 60:
        return "Value seniors"
    return "Young digitals"


centres["name"] = centres.apply(name_segment, axis=1)
print(centres.round(1).set_index("name").to_string())
note("""Centres live in the scaled space; transforming them back (inverse scaling, then
undoing the log) gives a 'typical customer' per segment in real units. Naming
the segments from their centres (rules a marketer can check) turns cluster
numbers into personas the business can use.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): k, initialisation, outliers
# ============================================================================
banner("Choosing k, starting well, and the pull of outliers", "MEDIUM")
step("2.1  Elbow and silhouette, plus a business constraint")
rows = {}
for k in range(2, 10):
    m = KMeans(k, n_init=5, random_state=0).fit(X)
    rows[k] = {"inertia": m.inertia_, "drop vs k-1": np.nan, "silhouette": silhouette_score(X, m.labels_, sample_size=2000, random_state=0),
               "smallest segment share": np.bincount(m.labels_).min() / len(X)}
ks = pd.DataFrame(rows).T
ks["drop vs k-1"] = -ks.inertia.diff() / ks.inertia.shift()
print(ks.round(3).to_string())
note("""Inertia drops fast until k = 5 and slowly after (the 'elbow'); the silhouette
prefers 4 or 5; and a business constraint (every segment >= 5% of customers,
or campaigns aren't worth running) rules out the large k. Use all three, then
check the profiles: 5 segments with distinct, nameable personas is the answer
a marketing team can act on.""")

step("2.2  Initialisation: random starts vs k-means++")
rows = {}
for init in ["random", "k-means++"]:
    inert = [KMeans(5, init=init, n_init=1, random_state=s).fit(X).inertia_ for s in range(30)]
    rows[init] = {"best inertia": min(inert), "median": np.median(inert), "worst": max(inert),
                  "runs within 0.1% of the best": np.mean(np.array(inert) <= min(inert) * 1.001)}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""k-means only finds a LOCAL minimum: a bad start can merge two real segments
and split another. k-means++ spreads the starting centres out and lands at the
best solution much more often; n_init (several starts, keep the best) does the
rest. scikit-learn's defaults (k-means++, several inits) exist for this reason.""")

step("2.3  Outliers distort the scale")
with_out = make_customers(n=3000, seed=3, outliers=15)
Xo = StandardScaler().fit_transform(with_out[FEATS])                   # no log: extreme values stay extreme
m_out = KMeans(5, n_init=5, random_state=0).fit(Xo)
sizes = np.bincount(m_out.labels_)
show("Cluster sizes with 15 resellers, no log transform", sizes.tolist())
ari_raw = adjusted_rand_score(with_out._segment, m_out.labels_)
ari_log = adjusted_rand_score(with_out._segment, KMeans(5, n_init=5, random_state=0).fit(
    StandardScaler().fit_transform(np.log1p(with_out[FEATS]))).labels_)
show("ARI vs hidden personas: no log / with log", f"{ari_raw:.3f} / {ari_log:.3f}")
res_mask = (with_out._segment == "outlier (reseller)").to_numpy()
show("Electronics spend: resellers' median / everyone else's", f"${with_out.electronics[res_mask].median():,.0f} / "
     f"${with_out.electronics[~res_mask].median():,.0f}")
show("Standard deviation of electronics: with / without the 15 resellers",
     f"{with_out.electronics.std():,.0f} / {with_out.electronics[~res_mask].std():,.0f}")
note(f"""15 resellers (0.5% of customers) spend ~{with_out.electronics[res_mask].median() / with_out.electronics[~res_mask].median():.0f}x the normal electronics budget. They
don't get a cluster of their own here, but they inflate the standard deviation
of electronics several-fold, so after standardising, the REAL differences in
electronics spend between normal customers shrink to almost nothing and that
feature stops helping: ARI {ari_log:.2f} -> {ari_raw:.2f}. With a few more extreme values a
centre gets dragged off entirely. Fixes: log-transform skewed features, cap
extreme values, or find and remove outliers FIRST (Lab 06).""")

# %% =========================================================================
# LEVEL 3 (HARD): Shapes, soft membership, segments in production
# ============================================================================
banner("Beyond round blobs, and segments as a business asset", "HARD")
step("3.1  Shapes k-means can't see")
Xb, yb = make_blobs(n_samples=1500, centers=3, random_state=170)
Xb_stretched = Xb @ np.array([[0.6, -0.64], [-0.4, 0.85]])            # elongated, tilted clusters
Xb_uneven, yb_uneven = make_blobs(n_samples=[1000, 100, 60], cluster_std=[3.0, 0.5, 0.5], centers=[[0, 0], [7, 7], [8, -6]],
                                  random_state=0)
rows = {}
for name, (Z, yt) in {"round, equal blobs": (Xb, yb), "stretched blobs": (Xb_stretched, yb),
                      "very different sizes / spreads": (Xb_uneven, yb_uneven)}.items():
    rows[name] = {"k-means ARI": adjusted_rand_score(yt, KMeans(3, n_init=10, random_state=0).fit_predict(Z)),
                  "Gaussian mixture ARI": adjusted_rand_score(yt, GaussianMixture(3, n_init=3, random_state=0).fit(Z).predict(Z))}
shapes = pd.DataFrame(rows).T
print(shapes.round(3).to_string())
note("""k-means draws straight borders halfway between centres, so it assumes round
clusters of similar spread. On stretched or very uneven clusters it cuts the
big ones in pieces. A Gaussian mixture model learns each cluster's own shape
(a covariance matrix) and size (a weight) and recovers them. Check cluster
shapes (pairs plots, PCA) before trusting k-means.""")

step("3.2  Soft membership: who is BETWEEN segments?")
gmm = GaussianMixture(5, covariance_type="full", n_init=3, random_state=0).fit(X)
proba = gmm.predict_proba(X)
top = proba.max(axis=1)
show("Customers with a clear segment (top probability > 0.9)", f"{np.mean(top > 0.9):.0%}")
show("Customers between segments (top probability < 0.6)", f"{np.mean(top < 0.6):.1%}")
between = customers.loc[np.argsort(top)[:3], FEATS].assign(top_probability=np.sort(top)[:3])
print(between.round(2).to_string())
note(f"""A Gaussian mixture gives each customer a PROBABILITY of belonging to each
segment. Here {np.mean(top > 0.9):.0%} are clear-cut and {np.mean(top < 0.6):.1%} sit between two personas (a
family that also renovates, a well-off young customer...). Those are worth flagging: a hard label would
send them the wrong campaign with full confidence.""")

step("3.3  Segments in production: actions, new customers, drift")
customers["monthly_spend"] = customers[SPEND_COLS].sum(axis=1)
customers["segment_name"] = centres["name"].to_numpy()[customers.segment]
actions = {"Affluent travellers": "premium travel card offer", "Home improvers": "DIY season bundle",
           "Family shoppers": "grocery loyalty multiplier", "Value seniors": "in-store discount day",
           "Young digitals": "app-only flash sales"}
plan = customers.groupby("segment_name").agg(customers=("monthly_spend", "size"), revenue_share=("monthly_spend", "sum"))
plan["revenue_share"] = plan.revenue_share / plan.revenue_share.sum()
plan["action"] = [actions[n] for n in plan.index]
print(plan.round(3).to_string())
new_month = make_customers(n=2000, seed=99)                            # next month's new sign-ups
new_month = pd.concat([new_month, make_customers(n=1500, seed=100).query("_segment == 'young digital'")])   # a TikTok campaign
assigned = sk.predict(scaler.transform(np.log1p(new_month[FEATS])))
shares = pd.DataFrame({"training share": pd.Series(customers.segment).value_counts(normalize=True).sort_index(),
                       "new month share": pd.Series(assigned).value_counts(normalize=True).sort_index()})
shares.index = centres["name"].to_numpy()[shares.index]
shares["change (points)"] = (shares["new month share"] - shares["training share"]) * 100
print(shares.round(3).to_string())
note("""A segmentation becomes a product: an action per segment, new customers assigned
with predict() (the nearest existing centre), and a monitor on segment SHARES.
This month a social-media campaign brought in many young digital customers:
that segment's share jumps. A big shift means either the business changed (good
to know) or the segments are going stale: re-fit and re-name them on a schedule.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].plot(ks.index, ks.inertia, "o-")
    axes[0].set(xlabel="k", ylabel="inertia", title="The elbow")
    ax2 = axes[0].twinx()
    ax2.plot(ks.index, ks.silhouette, "s--", color="#e45756")
    ax2.set_ylabel("silhouette", color="#e45756")
    km_s = KMeans(3, n_init=10, random_state=0).fit_predict(Xb_stretched)
    gm_s = GaussianMixture(3, n_init=3, random_state=0).fit(Xb_stretched).predict(Xb_stretched)
    axes[1].scatter(*Xb_stretched.T, c=km_s, s=4, cmap="tab10")
    axes[1].set(title=f"k-means on stretched blobs (ARI {shapes.iloc[1, 0]:.2f})")
    axes[2].scatter(*Xb_stretched.T, c=gm_s, s=4, cmap="tab10")
    axes[2].set(title=f"Gaussian mixture (ARI {shapes.iloc[1, 1]:.2f})")
    save_fig(fig, "03_kmeans_segmentation")

banner("KEY TAKEAWAYS")
print("""  1. K-means alternates ASSIGN (nearest centre) and UPDATE (mean of members) to minimise inertia.
  2. Choose k with the elbow, silhouette, stability AND business constraints; then name the segments.
  3. k-means++ and several inits avoid bad local minima; logs / outlier removal stop centres being dragged.
  4. k-means assumes round, similar clusters; Gaussian mixtures learn shapes and give soft membership.
  5. In production: assign new customers to existing centres and monitor segment shares for drift.""")
