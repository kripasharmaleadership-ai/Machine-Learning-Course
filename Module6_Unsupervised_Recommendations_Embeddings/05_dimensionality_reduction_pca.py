"""
LAB 05 - Dimensionality Reduction: Principal Component Analysis (PCA)
=====================================================================

THE IDEA
    Many features are correlated (big spenders spend more on everything). PCA finds new axes, the
    PRINCIPAL COMPONENTS, ordered by how much of the data's variance they capture:
        PC1 = the direction along which the data varies most
        PC2 = the next most varied direction, at right angles to PC1, and so on.
    Keep the first few components: most of the information in far fewer numbers.

THE MATHS (one line)
    centre the data, take its singular value decomposition  Xc = U S V^T
    rows of V^T = component directions (loadings);  S^2 / (n - 1) = variance along each;
    scores = Xc V  (the data in the new coordinates);  reconstruction = scores V^T + mean.

USES: visualise high-dimensional data in 2-D; compress; denoise; speed up and stabilise other models.

LEVELS
    EASY   : PCA by hand = scikit-learn on 9 customer features; read the components as business concepts.
    MEDIUM : How many components? Compress and reconstruct 8x8 images of digits; a 2-D map of 64-D data.
    HARD   : Scaling decides the answer; PCA is linear (t-SNE for curved structure); the directions of
             largest variance are not always the ones that predict.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import HAS_MPL, SPEND_COLS, banner, make_customers, note, plt, save_fig, show, step
from unsup_toolkit import pca_svd

customers = make_customers(n=3000, seed=5)
FEATS = ["age", "income_k"] + SPEND_COLS + ["visits_per_month", "online_share"]
X = StandardScaler().fit_transform(np.log1p(customers[FEATS]))

# %% =========================================================================
# LEVEL 1 (EASY): PCA by hand, and what the components mean
# ============================================================================
banner("PCA on 9 customer features", "EASY")
scores, comps, evr = pca_svd(X, 9)
sk = PCA(9).fit(X)
show("Explained variance ratio, by hand", np.round(evr, 3).tolist())
show("Explained variance ratio, scikit-learn", np.round(sk.explained_variance_ratio_, 3).tolist())
show("Cumulative: first 2 / first 3 / first 5 components", f"{evr[:2].sum():.0%} / {evr[:3].sum():.0%} / {evr[:5].sum():.0%}")
loadings = pd.DataFrame(comps[:3].T, index=FEATS, columns=["PC1", "PC2", "PC3"])
print("\n  Loadings (how much each original feature contributes to each component):")
print(loadings.round(2).to_string())
top_pos = lambda c: ", ".join(loadings[c].sort_values(ascending=False).index[:3])
top_neg = lambda c: ", ".join(loadings[c].sort_values().index[:2])
note(f"""The SVD gives the same components as scikit-learn (up to a sign, which is
arbitrary). Read the loadings like a recipe. PC1 = ({top_pos('PC1')}) against
({top_neg('PC1')}): a 'young digital vs older in-store' axis. PC2 = ({top_pos('PC2')}): an
'affluent household spending' axis. A handful of components summarise nine
correlated measurements, and each one can often be named.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): How many components? Compression and visualisation
# ============================================================================
banner("Compressing 64-pixel images of handwritten digits", "MEDIUM")
digits = load_digits()
Xd, yd = digits.data, digits.target
pca_full = PCA().fit(Xd)
cum = np.cumsum(pca_full.explained_variance_ratio_)
for target in [0.5, 0.8, 0.9, 0.95, 0.99]:
    show(f"Components needed for {target:.0%} of the variance (of 64)", int(np.searchsorted(cum, target) + 1))
rows = {}
for k in [2, 5, 10, 20, 40, 64]:
    p = PCA(k).fit(Xd)
    recon = p.inverse_transform(p.transform(Xd))
    acc = cross_val_score(KNeighborsClassifier(5), p.transform(Xd), yd, cv=5).mean()
    rows[k] = {"numbers stored per image": k, "reconstruction error (RMSE, pixel units 0-16)": np.sqrt(np.mean((Xd - recon) ** 2)),
               "KNN digit accuracy on the compressed data": acc}
comp_tab = pd.DataFrame(rows).T
comp_tab.index.name = "components"
print(comp_tab.round(3).to_string())
note(f"""Each image is 64 numbers (8x8 pixels), but neighbouring pixels are strongly
correlated. 20 components keep ~{cum[19]:.0%} of the variance; the reconstruction is
close to the original, and a KNN classifier on just 20 numbers is almost as accurate
as on all 64 ({comp_tab.loc[20].iloc[2]:.3f} vs {comp_tab.loc[64].iloc[2]:.3f}) with 3x less data per image. That is PCA as
compression: the discarded components carry little beyond noise.""")

step("2.1  A 2-D map of 64-dimensional data")
two = PCA(2).fit_transform(Xd)
centroids = pd.DataFrame(two).groupby(yd).mean()
sep = pd.Series({d: np.linalg.norm(centroids.loc[d] - centroids.drop(d), axis=1).min() for d in range(10)})
show("Digits whose centre is closest to another digit's in 2-D", sep.sort_values().head(3).round(2).to_dict())
note(f"""Two components keep only {pca_full.explained_variance_ratio_[:2].sum():.0%} of the variance, but plotted in 2-D (saved in
outputs/05_pca.png) the digits already form visible groups: the best separated
are {', '.join(str(d) for d in sep.sort_values(ascending=False).index[:3])}, while {', '.join(str(d) for d in sep.sort_values().index[:3])} crowd each other. PCA is the first tool for LOOKING at
high-dimensional data, including embeddings (Lab 09 visualises product clusters
this way).""")

# %% =========================================================================
# LEVEL 3 (HARD): Scaling, curves, and prediction
# ============================================================================
banner("Three ways PCA can mislead", "HARD")
step("3.1  Without scaling, PCA finds the feature with the biggest numbers")
raw = customers[FEATS].to_numpy(float)
p_raw = PCA(2).fit(raw)
p_std = PCA(2).fit(X)
print(pd.DataFrame({"PC1 loadings, raw units": p_raw.components_[0], "PC1 loadings, standardised": p_std.components_[0]},
                   index=FEATS).round(2).to_string())
show("PC1 variance share: raw / standardised", f"{p_raw.explained_variance_ratio_[0]:.0%} / {p_std.explained_variance_ratio_[0]:.0%}")
note("""On raw values PC1 is basically 'the spend column with the largest dollar
spread': variance is measured in squared dollars, so dollars beat years and
shares. It 'explains' a huge share of the variance and means little. Standardise
first (unless all features share one unit, like pixels) so PCA finds patterns
across features instead of differences in units.""")

step("3.2  PCA is linear: curved structure needs other tools")
roll_X, roll_t = make_swiss_roll(1000, noise=0.3, random_state=0)   # a 2-D sheet rolled up in 3-D; t = position along it
views = {"PCA to 2-D": PCA(2).fit_transform(roll_X),
         "t-SNE to 2-D": TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(roll_X)}
res = {}
for name, Z in views.items():
    nbr = NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    res[name] = {"trustworthiness (1 = neighbours kept)": trustworthiness(roll_X, Z, n_neighbors=10),
                 "avg distance ALONG the roll to 2-D neighbours": np.abs(roll_t[nbr] - roll_t[:, None]).mean()}
roll_tab = pd.DataFrame(res).T
print(roll_tab.round(3).to_string())
note(f"""A 'Swiss roll': a flat sheet rolled up in 3-D. PCA's best straight projection
squashes the layers on top of each other: points that look like neighbours in
the 2-D picture are on different layers ({roll_tab.iloc[0, 1]:.2f} apart along the sheet). t-SNE, a
NON-linear method that preserves local neighbourhoods, unrolls it ({roll_tab.iloc[1, 1]:.2f}). t-SNE
(and UMAP) are great for PICTURES, but their axes, and distances between
far-apart groups, mean nothing: use them to look, PCA to compute.""")

step("3.3  Largest variance is not the same as most predictive")
rng = np.random.default_rng(2)
n = 2000
big_noise = rng.normal(0, 10, (n, 5))                                  # five high-variance, irrelevant columns
signal = rng.normal(0, 1, n)                                           # one low-variance column that decides the label
y_sig = (signal + rng.normal(0, 0.3, n) > 0).astype(int)
Xs = np.c_[big_noise, signal]
rows = {}
for name, model in {"logistic regression on all 6 raw features": LogisticRegression(max_iter=2000),
                    "PCA to 2 components, then logistic": make_pipeline(PCA(2), LogisticRegression(max_iter=2000)),
                    "standardise, PCA to 2, then logistic": make_pipeline(StandardScaler(), PCA(2), LogisticRegression(max_iter=2000))}.items():
    rows[name] = {"CV accuracy": cross_val_score(model, Xs, y_sig, cv=5).mean()}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""PCA never looks at the label. It keeps the directions with the most VARIANCE,
and here the label lives in a quiet, low-variance feature, so PCA throws it
away and the classifier falls to a coin flip. Even standardising doesn't save
it: all six features then have equal variance, and PCA keeps 2 of 6 arbitrary
directions. Never assume PCA before a supervised model helps: cross-validate
with and without it.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    axes[0].plot(range(1, 65), cum, "-")
    axes[0].axhline(0.9, color="grey", ls=":")
    axes[0].set(xlabel="components", ylabel="cumulative variance explained", title="Digits: how many components?")
    sc_ = axes[1].scatter(two[:, 0], two[:, 1], c=yd, s=5, cmap="tab10")
    axes[1].set(title="64-D digits in 2-D (PCA)", xlabel="PC1", ylabel="PC2")
    fig.colorbar(sc_, ax=axes[1])
    orig = Xd[0].reshape(8, 8)
    recon10 = PCA(10).fit(Xd).inverse_transform(PCA(10).fit(Xd).transform(Xd[:1])).reshape(8, 8)
    axes[2].imshow(np.c_[orig, np.full((8, 1), 16), recon10], cmap="gray_r")
    axes[2].set(title="A digit: original | from 10 numbers", xticks=[], yticks=[])
    axes[3].scatter(*views["t-SNE to 2-D"].T, c=roll_t, s=5, cmap="viridis")
    axes[3].set(title="Swiss roll unrolled by t-SNE (colour = position)")
    save_fig(fig, "05_pca")

banner("KEY TAKEAWAYS")
print("""  1. PCA = SVD of the centred data: new axes ordered by variance; loadings say what each axis means.
  2. A few components often keep most of the information: compress, denoise, speed up, visualise.
  3. Standardise first, or PCA just finds the feature with the biggest units.
  4. PCA is linear; t-SNE / UMAP reveal curved structure for pictures (not for distances).
  5. PCA ignores the target: the high-variance directions may not be the predictive ones.""")
